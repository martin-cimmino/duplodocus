
use crate::utils::json_set;
use serde_json::json;
use serde_json::Value;
use std::io::BufRead;
use std::collections::HashMap;
use std::fs::create_dir_all;
use std::os::unix::fs::OpenOptionsExt;
use std::fs::OpenOptions;
use std::io::Write;
use std::sync::Arc;
use std::io::BufWriter;
use std::sync::Mutex;
use std::cmp::Reverse;
use std::io::Read;
use std::io::BufReader;
use std::fs::File;
use glob::glob;
use dashmap::DashMap;
use std::sync::atomic::Ordering;
use std::sync::atomic::AtomicUsize;
use std::time::Instant;
use crate::minhash_base::FileMap;

use anyhow::{Error, Result, Context};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use rayon;
use rayon::prelude::*;
use mj_io::{read_pathbuf, build_pbar, write_mem_to_pathbuf, read_pathbuf_to_mem, get_output_filename};
use crate::table_old::SuffixTable;
use gjson;

use std::collections::BinaryHeap;
use std::cmp::Ordering as StdOrdering;
use smallvec::SmallVec;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{__m256i, _mm256_loadu_si256, _mm256_cmpeq_epi8, _mm256_movemask_epi8};

#[cfg(target_arch = "x86")]
use std::arch::x86::{__m256i, _mm256_loadu_si256, _mm256_cmpeq_epi8, _mm256_movemask_epi8};



/*====================================================
=                    CONFIGURATIONS                  =
====================================================*/



const DEFAULT_MAX_LINES_PER_PATH: usize = 1_000_000_000;
const DEFAULT_TOKENIZER: &str = "bytes";//.to_string();
const DEFAULT_TEXT_KEY: &str = "text";
const DEFAULT_RANDOM_SEED: usize = 42;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SAConfig {
	pub tokenizer: String, /// Defaults to using bytes
	pub max_lines_per_path: usize, /// Defaults to 1_000_000_000
	pub text_key: String,
	pub random_seed: usize,
}


impl Default for SAConfig {
	fn default() -> Self {
		Self {
			tokenizer: DEFAULT_TOKENIZER.to_string(),
			max_lines_per_path: DEFAULT_MAX_LINES_PER_PATH,
			text_key: DEFAULT_TEXT_KEY.to_string(),
			random_seed: DEFAULT_RANDOM_SEED,
		}
	}
}

pub struct SAConfigOverrides {
	pub tokenizer: Option<String>,
	pub max_lines_per_path: Option<usize>,
	pub text_key: Option<String>,
	pub random_seed: Option<usize>
}

impl SAConfigOverrides {
	fn apply_to(self, config: &mut SAConfig) {
		if let Some(val) = self.tokenizer {
			config.tokenizer = String::from(val.clone());
		}

		if let Some(val) = self.max_lines_per_path {
			config.max_lines_per_path = val;
		}

		if let Some(val) = self.text_key {
			config.text_key = String::from(val);
		}

		if let Some(val) = self.random_seed {
			config.random_seed = val;
		}
	}
}


impl SAConfig {
	pub fn load_with_overrides(
		config_path: Option<PathBuf>,
		overrides: SAConfigOverrides,
		) -> Result<Self, Error> {
		let mut config = Self::default();
		if let Some(path) = config_path {
			let config_content = fs::read_to_string(path)?;
			let file_config: SAConfig = serde_yaml::from_str(&config_content)?;
			config = file_config;
		}

		overrides.apply_to(&mut config);
		Ok(config)
	}
}


/*======================================================
=                    MAKE TABLE                        =
======================================================*/

pub fn make_sa_tables_cmd(
	input_dir: &PathBuf,
	output_dir: &PathBuf,
	config_opt: Option<PathBuf>,
	file_map_opt: Option<PathBuf>,
	tokenizer: Option<String>,
	max_lines_per_path: Option<usize>,
	text_key: Option<String>
	) -> Result<(), Error> {
	let file_map = if let Some(file_map_name) = file_map_opt {
		FileMap::load(&file_map_name).unwrap()
	} else {
		let file_map = FileMap::new(input_dir, &None).unwrap();
		file_map.save(&(output_dir.join("file_map.json.gz"))).unwrap();
		file_map
	};

	let config_overrides = SAConfigOverrides {
		tokenizer: tokenizer,
		max_lines_per_path: max_lines_per_path,
		text_key: text_key,
		random_seed: None
	};

	let config_obj = SAConfig::load_with_overrides(config_opt, config_overrides).unwrap();

	make_sa_tables(&config_obj, &file_map, input_dir, output_dir)
}

pub fn make_sa_tables(
	config_obj: &SAConfig,
	file_map: &FileMap,
	input_dir: &PathBuf,
	output_dir: &PathBuf
	) -> Result<(), Error> {
    assert_eq!(1u16.to_ne_bytes(), 1u16.to_le_bytes(), "System must be little-endian");

	let start_main = Instant::now();
	println!("Building SA tables...");

	// Step 1: load all data into memory and then maybe tokenize if needed?
	let start_read = Instant::now();
	println!("Reading all documents texts...");
	let mut data_docs = load_data_docs(file_map, config_obj, input_dir).unwrap();
	println!(" {:?} DOCS TOTAL", data_docs.len());
	println!("TOTAL IDX {:?}", data_docs.iter().map(|tup| tup.0).sum::<usize>());
	println!("...Read all documents in {:?} secs", start_read.elapsed().as_secs());

	// Step 2 (optional): Tokenize if needed
	// TODO: tokenizer thing

	// Step 3: Break into chunks and make SA table for each of them
	let thread_count = rayon::current_num_threads();
	let chunk_size = (data_docs.len() - 1) / thread_count + 1;
	let owned_chunks: DashMap<usize, Vec<(usize, usize, String)>> = DashMap::new();
	let mut chunk_idx = 0;
	while data_docs.len() >= chunk_size {
		let chunk = data_docs.split_off(data_docs.len() - chunk_size);		
		println!("LENCHUNK {:?}", chunk.len());
		owned_chunks.insert(chunk_idx, chunk);
		chunk_idx += 1;
	}
	println!("DD {:?} | {:?}", data_docs.len(), chunk_size);
	let total_bytes = AtomicUsize::new(0);
	let mut global_offsets: Vec<u64> = Vec::new();
	global_offsets.push(0 as u64);
	for i in 0..(owned_chunks.len() - 1) {
		let allocate_size = owned_chunks.get(&i).unwrap().par_iter().map(|(_, _, s)| {s.len() + 2 + 2 * 8}).sum::<usize>();
		global_offsets.push(global_offsets.last().unwrap() + allocate_size as u64);
	}
	let global_offset_filename = output_dir.clone().join("offsets").join("global_offset.bin");
	let global_offset_bytes: &[u8] = bytemuck::cast_slice(&global_offsets);
	write_mem_to_pathbuf(global_offset_bytes, &global_offset_filename).unwrap();


	let total_pbar = build_pbar(owned_chunks.len() * 4, "Total steps");
	owned_chunks.into_par_iter().for_each(|(table_idx, chunk)| { 
		let allocate_size = chunk.iter().map(|(_, _, s)| {s.len() + 2 + 2 * 8}).sum::<usize>();
		let mut document_offsets: Vec<u64> = Vec::new();
		document_offsets.extend(vec!(u64::MAX, u64::MAX, 0));
		let mut buf = Vec::with_capacity(allocate_size);

		chunk.into_iter().for_each(|(u1, u2, s)| {
			buf.extend_from_slice(s.as_bytes());
			buf.extend_from_slice(&[0xff, 0xff]);
			buf.extend_from_slice(&(u1 as u64).to_le_bytes());
			buf.extend_from_slice(&(u2 as u64).to_le_bytes());				
			document_offsets.extend(vec![u1 as u64, u2 as u64, buf.len() as u64]); // Offsets are trips of (path_id, line_num, current_offset)

		});
		assert_eq!(document_offsets.len() % 3, 0);

		total_bytes.fetch_add(allocate_size, Ordering::SeqCst);
		total_pbar.inc(1);
		// And save concatenated text just for safekeeping
		let output_text_filename = output_dir.clone().join("text").join(format!("text_part_{:04}.bin", table_idx));
		write_mem_to_pathbuf(&buf, &output_text_filename).unwrap();
		total_pbar.inc(1);

		// And save the document offsets
		let output_offset_filename = output_dir.clone().join("offsets").join(format!("offset_part_{:04}.bin", table_idx));
		let offset_vec: &[u8] = bytemuck::cast_slice(&document_offsets);
		write_mem_to_pathbuf(offset_vec, &output_offset_filename).unwrap();


		// And then make the SA table too
		let table = SuffixTable::new(buf);
		total_pbar.inc(1);

		let table_to_write: &[u8] = bytemuck::cast_slice(table.table());
		let output_table_filename = output_dir.clone().join("table").join(format!("table_part_{:04}.bin", table_idx));
		write_mem_to_pathbuf(table_to_write, &output_table_filename).unwrap();
	});

	println!("Made {:?} tables of {:?} bytes total in {:?} seconds", thread_count, total_bytes.into_inner(), start_main.elapsed().as_secs());
	Ok(())
}

pub fn load_data_docs(file_map: &FileMap, config: &SAConfig, local_input: &PathBuf) -> Result<Vec<(usize, usize, String)>, Error> {
	let text_key = config.text_key.clone();
	let extant_idxs: Vec<(&PathBuf, &usize)> = file_map.indices.par_iter().filter(|(p, _idx)| {
		local_input.clone().join(p).exists()
	}).collect();

	let pbar = build_pbar(extant_idxs.len(), "Paths");
	let data_docs: Vec<(usize, usize, String)> = extant_idxs.par_iter().flat_map(|(p,idx)| {
		let mut sub_docs: Vec<(usize, usize, String)> = Vec::new();
		let contents = read_pathbuf(&(local_input.clone().join(p)), true).unwrap();
		for (line_num, line) in contents.lines().enumerate() {
			let line = line.unwrap();
			let value = gjson::get(&line, &text_key).str().to_string();
			sub_docs.push((**idx, line_num, value));
		}	
		pbar.inc(1);
		sub_docs
	}).collect();
	Ok(data_docs)
}


fn get_part_num(path: &PathBuf) -> Result<usize, Error> {
	Ok(path.file_stem()
		.unwrap()
		.to_str()
		.unwrap()
		.split("_")
		.last()
		.unwrap()
		.parse::<usize>()
		.unwrap()
		)
}

/*============================================================
=                            PQ LOOPS                        =
============================================================*/

pub fn pq_serial(storage_dir: &PathBuf,
				 match_length: usize) -> Result<(), Error> {
	/* Step 1: 
	Load into memory:
		- file map
		- global offset file 
		- all text files
		- all offset files
	And open stream readers from SA tables
	*/
	let start_main = Instant::now();
	let text_pattern = storage_dir.clone().join("text").join("text_part_*");
	let text_objects: Vec<PathBuf> = glob(&text_pattern.to_str().unwrap()).unwrap().into_iter().map(|p| p.unwrap()).collect();
	let max_text_idx = AtomicUsize::new(0);
	let text_data: Vec<(usize, Vec<u8>)> = text_objects.into_iter().map(|p| { //PAR
		let text_idx = get_part_num(&p).unwrap();
		max_text_idx.fetch_max(text_idx, Ordering::SeqCst);
		(text_idx, read_pathbuf_to_mem(&p).unwrap().into_inner().into_inner())
	}).collect();
	let mut text_lookup: Vec<Vec<u8>> = vec![Vec::new(); max_text_idx.into_inner() + 1];
	text_data.into_iter().for_each(|(idx, v)| {text_lookup[idx] = v});


	let table_pattern = storage_dir.clone().join("table").join("table_part_*");
	let table_objects: Vec<PathBuf> = glob(&table_pattern.to_str().unwrap()).unwrap().into_iter().map(|p| p.unwrap()).collect();
	let max_table_idx = AtomicUsize::new(0);
	let table_data: Vec<(usize, FileStream)> = table_objects.into_iter().map(|p| { //PAR
		let table_idx = get_part_num(&p).unwrap();		
		max_table_idx.fetch_max(table_idx, Ordering::SeqCst);
		(table_idx, FileStream::new(&p, 16384).unwrap())
	}).collect();
	let mut table_lookup: Vec<Option<FileStream>> =(0..(max_table_idx.into_inner() + 1)).into_iter().map(|_| None).collect();
	table_data.into_iter().for_each(|(idx,v)| {table_lookup[idx] = Some(v)});

	/* Step 2:
	Create priority queue to have 2 indices from each stream 
	*/
	let total_counts:usize = text_lookup.iter().map(|v| v.len()).sum();
	//let pbar = build_pbar(total_counts, "Items");


	let mut pq: BinaryHeap<Reverse<QueueElement>> = BinaryHeap::new();
	for (idx, table_opt) in table_lookup.iter_mut().enumerate() {

		if let Some(table) = table_opt {
			for _ in 0..2 {
				let cur_text = &text_lookup[idx];
				if let Some((table_value, sv)) = table.get_next_text(cur_text, match_length).unwrap() {
					let pqe = QueueElement{stream: idx, 
										   idx: table_value,
										   cmp_bytes: sv};
					pq.push(std::cmp::Reverse(pqe));
				}				
			}
		}
	}

	/* Step 3:
	While the PQ is not empty: 
		- look at the minimum two elements
		- look up their strings and count how many indices they have in common
		- if they have >= match_length in common, write to output streamer
	*/
	let match_writer = MatchWriter::new(&storage_dir.clone().join("matches")).unwrap();
	let mut last_written: (usize, u64) = (usize::MAX, u64::MAX);
	let match_count = AtomicUsize::new(0);
	while pq.len() > 1 {

		let min_el = pq.pop().unwrap();
		//pbar.inc(1);
		let peek_min = pq.peek().unwrap();
		if min_el == *peek_min {
			if last_written != (min_el.0.stream, min_el.0.idx) {
				let min_idx_bytes = min_el.0.idx.to_le_bytes();
				match_writer.write(min_el.0.stream, &min_idx_bytes).unwrap();
				match_count.fetch_add(1, Ordering::SeqCst);
			} 
			let peek_min_idx_bytes = peek_min.0.idx.to_le_bytes();
			match_writer.write(peek_min.0.stream, &peek_min_idx_bytes).unwrap();
			match_count.fetch_add(1, Ordering::SeqCst);

			last_written = (peek_min.0.stream, peek_min.0.idx);
		}

		let min_stream = min_el.0.stream;
		if let Some(ref mut table) = table_lookup[min_stream] {
			let cur_text = &text_lookup[min_stream];
			if let Some((table_value, sv)) = table.get_next_text(cur_text, match_length).unwrap() {
				let pqe = QueueElement{stream: min_stream, 
									   idx: table_value,
									   cmp_bytes: sv};
				pq.push(std::cmp::Reverse(pqe));				
			}
		}
	}
	match_writer.finish().unwrap();

	println!("Finished PQ in {:?} secs | found {:?} matches", start_main.elapsed().as_secs(), match_count.into_inner());
	Ok(())
}


/*=====================================================================
=                            ALTERNATIVE FLOW                         =
=====================================================================*/

pub fn alternative_merge(text_loc: &PathBuf, sa_table_loc: &PathBuf, match_length: usize) -> Result<(), Error> {
	let start_main = Instant::now();
	println!("Starting alt flow");
	let text: Vec<u8> = read_pathbuf_to_mem(text_loc).unwrap().into_inner().into_inner();
	let text_slice: &[u8] = text.as_slice();
	//let sa_table: Vec<u64> = read_u64_vec(sa_table_loc).unwrap();
	//let table_len = sa_table.len();
	let mut matches: Vec<u64> = Vec::new();
	let mut sa_table = FileStream::new(sa_table_loc, 16384).unwrap();
	let (prev_el, prev_text) = sa_table.get_next_text(&text, match_length).unwrap().unwrap();
	loop {
		let result = sa_table.get_next_text(&text, match_length).unwrap();
		if let Some(out_tup) = result {
			let (cur_el, cur_text) = out_tup;
			if cur_text == prev_text {
				if matches.len() > 0 && *matches.last().unwrap() != prev_el as u64 {
					matches.push(prev_el as u64);				
				}
				matches.push(cur_el.try_into().unwrap());
			}
		} else {
			break;
		}
	}

	/*

	let mut matches: Vec<u64> = Vec::new();


	//let pbar = build_pbar(sa_table.len() -1 , "IDXs");

	let mut prev_el = sa_table[0] as usize;
	let prev_end = std::cmp::min(prev_el + match_length, table_len) as usize;
	let mut prev_text = &text_slice[prev_el..prev_end];


	for idx in 1..sa_table.len() {
		let cur_el = sa_table[idx] as usize;
		let cur_end = std::cmp::min(cur_el + match_length, table_len);
		let cur_text = &text_slice[cur_el..cur_end];

		if cur_text == prev_text {
			if matches.len() > 0 && *matches.last().unwrap() != prev_el as u64 {
				matches.push(prev_el as u64);
			}
			matches.push(cur_el.try_into().unwrap());			
		}
		prev_el = cur_el;
		prev_text = cur_text;
		//pbar.inc(1);
	}
	*/


	println!("Finished alt flow in {:?} seconds | found {:?} matches", start_main.elapsed().as_secs(), matches.len());
	Ok(())
}




/*=====================================================================
=                           MERGE MATCHES                             =
=====================================================================*/

pub fn merge_matches(storage_dir: &PathBuf, match_length: usize) -> Result<(), Error> {
	let start_main = Instant::now();
	println!("Starting merging of matches");

	let input_match_dir = storage_dir.clone().join("matches");
	let offset_dir = storage_dir.clone().join("offsets");
	let offset_pattern = offset_dir.clone().join("offset_part_*.bin");

	let match_pattern = input_match_dir.clone().join("match_part*.bin");
	let merged_match_dir = storage_dir.clone().join("merged_matches");

	let matches: DashMap<usize, PathBuf> = glob(&match_pattern.to_str().unwrap()).unwrap().into_iter().map(|p| {
		let path = p.unwrap();
		let part_num = get_part_num(&path).unwrap();
		(part_num, path)
	}).collect();

	let offsets: DashMap<usize, PathBuf> = glob(&offset_pattern.to_str().unwrap()).unwrap().into_iter().map(|p| {
		let path = p.unwrap();
		let part_num = get_part_num(&path).unwrap();
		(part_num, path)
	}).collect();




	let pbar_counter = AtomicUsize::new(0);


	matches.into_par_iter().for_each(|(part_num, match_path)| {
		let offset_path = offsets.get(&part_num).unwrap();	
		merge_match_path(match_path, &offset_path, &merged_match_dir, match_length, pbar_counter.fetch_add(1, Ordering::SeqCst) == 0).unwrap();

	});

	println!("Merged all matches in {:?} secs", start_main.elapsed().as_secs());
	Ok(())


}


pub fn merge_match_path(match_path: PathBuf, offset_path: &PathBuf, merged_match_dir: &PathBuf, match_length: usize, pbar_flag: bool) -> Result<(), Error> {


	let mut match_idxs = read_u64_vec(&match_path).unwrap();
	let offset_trips = read_u64_vec(&offset_path).unwrap();
	let offset_trips: Vec<(u64, u64, u64)> = offset_trips.chunks_exact(3).map(|c| (c[0], c[1], c[2])).collect(); // (path_id, line_num, idx-ceiling)
	

	let pbar_opt = if pbar_flag {
		Some(build_pbar(match_idxs.len(), "Matches"))
	} else {
		None
	};
	match_idxs.sort();
	if match_idxs.len() == 0 {
		return Ok(());
	}
	let match_length = match_length as u64;


	// match_intervals should be packing of (path_id, line_num, start_byte, end_byte) [within the doc!]
	// (but let's just get the interval in the concatenated text file first)
	let mut match_intervals: Vec<u64> = Vec::new();
	let mut cur_start = match_idxs[0];
	let mut cur_end = cur_start + match_length;
	let mut last_branch: bool = false;
	for idx in match_idxs.into_iter().skip(1) {
		if idx <= cur_end {
			cur_end = idx + match_length;
			last_branch = false
		} else {
			let offset_index = offset_trips.partition_point(|&(_, _, third)| third < cur_end);
			let offset_trip = offset_trips.get(offset_index).unwrap();
			let prev_offset_start = offset_trips.get(offset_index - 1).unwrap().2;
			match_intervals.extend(vec!(offset_trip.0, offset_trip.1, cur_start - prev_offset_start, cur_end - prev_offset_start));
			cur_start = idx;
			cur_end = cur_start + match_length;
			last_branch = false
		} 
		if let Some(ref pbar) = pbar_opt {
			pbar.inc(1);
		}
	}
	if !last_branch {
			let offset_index = offset_trips.partition_point(|&(_, _, third)| third < cur_end);
			let offset_trip = offset_trips.get(offset_index).unwrap();
			let prev_offset_start = offset_trips.get(offset_index - 1).unwrap().2;
			match_intervals.extend(vec!(offset_trip.0, offset_trip.1, cur_start - prev_offset_start, cur_end - prev_offset_start));
	}


	let match_interval_bytes: &[u8] = bytemuck::cast_slice(&match_intervals);
	let file_stem = match_path.file_stem().unwrap().to_str().unwrap();
	let output_filename = merged_match_dir.clone().join(format!("merged_{}.bin", file_stem));
	write_mem_to_pathbuf(match_interval_bytes, &output_filename)
}


/*======================================================================
=                            ANNOTATE FILES                            =
======================================================================*/

pub fn sa_annotate_files(input_dir: &PathBuf, storage_dir: &PathBuf, output_dir: &PathBuf,
						 annotate_key: &String) -> Result<(), Error> {
	let start_main = Instant::now();
	println!("Starting annotation...");
	let file_map_loc = storage_dir.clone().join("file_map.json.gz");
	let file_map = FileMap::load(&file_map_loc).unwrap();

	// Then loop through paths
	let merge_match_pattern = storage_dir.clone().join("merged_matches").join("merged_*.bin");
	let merge_match_objects: Vec<PathBuf> = glob(&merge_match_pattern.to_str().unwrap()).unwrap().into_iter().map(|p| p.unwrap()).collect();
	let anno_groups: DashMap<usize, DashMap<usize, Vec<(u64, u64)>>> = DashMap::new();

	merge_match_objects.into_par_iter().for_each(|p| {
		let contents: Vec<u64> = read_u64_vec(&p).unwrap();
		let contents: Vec<(u64, u64, u64, u64)> = contents.chunks_exact(4).map(|c| (c[0], c[1], c[2], c[3])).collect();
		contents.into_iter().for_each(|(path_id, line_num, start, end)| {
			anno_groups.entry(path_id as usize).or_default().entry(line_num as usize).or_default().push((start, end));
		});
	});

	println!("ANNO GROUPS {:?}", anno_groups);

	let extant_idxs: Vec<(&PathBuf, &usize)> = file_map.indices.par_iter().filter(|(p, _idx)| {
		input_dir.clone().join(p).exists()
	}).collect();
	let pbar = build_pbar(extant_idxs.len(), "Pahts");

	extant_idxs.into_par_iter().for_each(|(p, idx)|  {
		let anno_group = anno_groups.entry(*idx).or_default();
		let input_path = input_dir.clone().join(&p);
		let output_path = get_output_filename(&input_path, input_dir, output_dir).unwrap();
		sa_annotate_path(input_path, &anno_group, &output_path, annotate_key).unwrap();
		pbar.inc(1);
	});
	println!("Finished annotation in {:?} secs", start_main.elapsed().as_secs());
	Ok(())
}


fn sa_annotate_path(input_path: PathBuf, anno_group: &DashMap<usize, Vec<(u64, u64)>>, output_path: &PathBuf, annotate_key: &String) -> Result<(), Error> {

	let contents = read_pathbuf_to_mem(&input_path).unwrap();
	let mut output_contents: Vec<u8> = Vec::new();


	for (line_num, line) in contents.lines().enumerate() {
		let line = line.unwrap();
		let anno_val = anno_group.entry(line_num).or_default();
		if anno_val.len() == 0 {
			output_contents.extend(line.as_bytes());
		}  else {
			let mut line_json: Value = serde_json::from_str(&line).unwrap();
			let anno_val: Value = json!(*anno_val);
			json_set(&mut line_json, annotate_key, anno_val).unwrap();
			let output_line = serde_json::to_vec(&line_json).unwrap();
			output_contents.extend(output_line);
		}
		output_contents.push(b'\n')
		
	}

	write_mem_to_pathbuf(&output_contents, output_path).unwrap();
	Ok(())
}




/*======================================================================
=                            FILE STREAM STUFF                         =
======================================================================*/

struct FileStream {
    reader: BufReader<File>,
    buffer: Vec<u64>,
    position: usize,
    chunk_size: usize,
}

impl FileStream {
    fn new(file: &PathBuf, chunk_size: usize) -> Result<Self, Error> {
    	let file = File::open(file).unwrap();
        Ok(Self {        	
            reader: BufReader::new(file),
            buffer: Vec::with_capacity(chunk_size),
            position: 0,
            chunk_size,
        })
    }

    fn refill_buffer(&mut self) -> Result<bool> {
        self.buffer.clear();
        self.position = 0;

        let bytes_to_read = self.chunk_size * 8;
        let mut byte_buffer = vec![0u8; bytes_to_read];
        
        let mut total_read = 0;
        while total_read < bytes_to_read {
            match self.reader.read(&mut byte_buffer[total_read..]) {
                Ok(0) => break, // EOF
                Ok(n) => total_read += n,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e).context("Failed to read from file"),
            }
        }

        // Convert bytes to u64s
        for chunk in byte_buffer[..total_read].chunks_exact(8) {
            let bytes: [u8; 8] = chunk.try_into().unwrap();
            self.buffer.push(u64::from_le_bytes(bytes));
        }

        Ok(!self.buffer.is_empty())
    }


    fn get_next(&mut self) -> Option<Result<u64>> {
    	if self.position >= self.buffer.len() {
    		match self.refill_buffer() {
    			Ok(true) => {},
    			Ok(false) => return None, // EOF 
    			Err(e) => return Some(Err(e))
    		}
    	}

    	if self.position < self.buffer.len() {
    		let value = self.buffer[self.position];
    		self.position += 1;
    		Some(Ok(value))
    	} else {
    		None
    	}
    }

    fn get_next_text<'a>(&mut self, text: &'a Vec<u8>, min_len: usize) -> Result<Option<(u64, &'a [u8])>, Error> {
    	loop {
    		let next_idx_opt = self.get_next();
    		if let Some(next_idx) = next_idx_opt {
    			let next_idx = next_idx.unwrap();
    			if next_idx as usize + min_len >= text.len() {
    				continue;
    			}
    			let text_slice = text.as_slice();
    			let slice = &text_slice[next_idx as usize..next_idx as usize + min_len];
    			//let sv : SmallVec<[u8; 512]> = SmallVec::from_slice(&slice);
    			return Ok(Some((next_idx, slice)));
    		} else {
    			return Ok(None);
    		}
    	}

    	Ok(None)
    }
}



/*======================================================================
=                            PRIORITY QUEUE STUFF                      =
======================================================================*/



struct QueueElement<'a> {
    stream: usize, // which "part" this came from
    idx: u64, // actual index of the suffix table
    cmp_bytes: &'a [u8], // suffix that this points to
}

impl Ord for QueueElement<'_> {
    fn cmp(&self, other: &Self) -> StdOrdering {
    	self.cmp_bytes.cmp(&other.cmp_bytes)
    	//compare_suffix_prefix(self.cmp_bytes, &other.cmp_bytes, self.cmp_bytes.len())
    }
}

impl PartialOrd for QueueElement<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<StdOrdering> {        	
        Some(self.cmp(other))
    }
}

impl PartialEq for QueueElement<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp_bytes == other.cmp_bytes
    }
}

impl Eq for QueueElement<'_> {}





/*====================================================================
=                           MATCH WRITER THING                       =
====================================================================*/

pub struct MatchWriter {
	pub writer: DashMap<usize, Arc<Mutex<BufWriter<File>>>>,
	pub storage_loc: PathBuf,	
}

impl MatchWriter {
	pub fn new(storage_loc: &PathBuf) -> Result<Self, Error> {
		let writer: DashMap<usize, Arc<Mutex<BufWriter<File>>>> = DashMap::new();
		let storage_loc = storage_loc.clone();
		if !storage_loc.exists() {
			create_dir_all(&storage_loc).unwrap();
		}

		Ok(MatchWriter {writer, storage_loc})
	}	

	pub fn make_new_writer(&self, idx: usize) -> Result<Arc<Mutex<BufWriter<File>>>, Error> {
		let filename = &self.get_filename(idx);
		let file = OpenOptions::new().append(true).create(true).mode(0o644).open(filename).unwrap();
		let buf_writer = BufWriter::new(file);
		let output = Arc::new(Mutex::new(buf_writer));
		Ok(output)
	}

	pub fn write(&self, idx: usize, bytes: &[u8]) -> Result<(), Error> {

		let buf_writer_arc = if let Some(buf_writer_arc) = self.writer.get(&idx) {
			buf_writer_arc
		} else {
			let buf_writer_arc = self.make_new_writer(idx).unwrap();
			self.writer.insert(idx, buf_writer_arc);
			self.writer.get(&idx).unwrap()
		};

		buf_writer_arc.lock().unwrap().write(bytes).unwrap();
		Ok(())
	}

	pub fn finish(&self) -> Result<(), Error> {
		self.writer.par_iter().for_each(|e| e.lock().unwrap().flush().unwrap());
		Ok(())
	}

	pub fn get_filename(
		&self,
		idx: usize
		) -> PathBuf {
		self.storage_loc.clone().join(format!("match_part_{:04}.bin", idx))
	}
		
}


/*======================================================
=                           UTILS                      =
======================================================*/

fn read_u64_vec(p: &PathBuf) -> Result<Vec<u64>, Error> {
	let file_size = std::fs::metadata(&p)?.len() as usize;
	let mut output: Vec<u64> = vec![0u64; file_size / 8];
	let bytes_view: &mut [u8] = bytemuck::cast_slice_mut(&mut output);
	std::fs::File::open(&p).unwrap().read_exact(bytes_view).unwrap();	
	Ok(output)	
}



#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline(always)]
unsafe fn compare_suffix_prefix_simd(
    slice1: &[u8],
    slice2: &[u8],
    len: usize,
) -> Option<Ordering> {
    let chunks = len / 32;
    
    for i in 0..chunks {
        let offset = i * 32;
        let a = _mm256_loadu_si256(slice1[offset..].as_ptr() as *const __m256i);
        let b = _mm256_loadu_si256(slice2[offset..].as_ptr() as *const __m256i);
        let cmp = _mm256_cmpeq_epi8(a, b);
        let mask = _mm256_movemask_epi8(cmp);
        
        if mask != -1 {
            let first_diff = mask.trailing_ones() as usize;
            return Some(slice1[offset + first_diff].cmp(&slice2[offset + first_diff]));
        }
    }
    None
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn compare_suffix_prefix_simd(
    slice1: &[u8],
    slice2: &[u8],
    len: usize,
) -> Option<StdOrdering> {
    let chunks = len / 16;
    
    for i in 0..chunks {
        let offset = i * 16;
        let a = vld1q_u8(slice1[offset..].as_ptr());
        let b = vld1q_u8(slice2[offset..].as_ptr());
        let cmp = vceqq_u8(a, b);
        let all_equal = vminvq_u8(cmp) == 0xFF;
        
        if !all_equal {
            for j in 0..16 {
                if slice1[offset + j] != slice2[offset + j] {
                    return Some(slice1[offset + j].cmp(&slice2[offset + j]));
                }
            }
        }
    }
    None
}

#[inline(always)]
pub fn compare_suffix_prefix(
	slice1: &[u8],
	slice2: &[u8],
    len: usize
) -> StdOrdering {    
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        if let Some(result) = unsafe { compare_suffix_prefix_simd(slice1, slice2, len) } {
            return result;
        }
    }
    println!("FALLBACK");
    // Fallback for remainder or non-SIMD platforms
    slice1.cmp(slice2)
}
