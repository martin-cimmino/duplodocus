
use ahash::HashMap;
use crate::utils::json_set;
use serde_json::json;
use serde_json::Value;
use std::io::BufRead;
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
	let text_lookup = make_text_lookups(storage_dir).unwrap();
	let total_counts:usize = text_lookup.iter().map(|v| v.len()).sum();	
	let mut table_streams = make_table_streams(storage_dir, &text_lookup).unwrap();

	let mut stream_iterators: HashMap<usize, TextIterator> = HashMap::default();

	table_streams.iter_mut().for_each(|s| {
		stream_iterators.insert(s.source, s.text_iter(match_length));
	});


	/* Step 2: Initialize min-order data structure 
	*/

	let mut pq: BinaryHeap<Reverse<TreeNode>> = BinaryHeap::new();
	for (source, iterator) in stream_iterators.iter_mut() {
		if let Some(node) = iterator.next() {
			let node = node.unwrap();
			pq.push(Reverse(node));
		}
	}
	if pq.len() == 0 {
		return Ok(());
	}

	// Pop one out and put one back from its stream
	let mut prev_min = pq.pop().unwrap().0;
	let prev_min_source = prev_min.source;
	if let Some(node) = stream_iterators.get_mut(&prev_min_source).unwrap().next() {
		let node = node.unwrap();
		pq.push(Reverse(node));
	}


	/* Step 3: Now do the loop 
	*/
	let match_writer = MatchWriter::new(&storage_dir.clone().join("matches")).unwrap();
	let match_count = AtomicUsize::new(0);
	let mut currently_in_a_run = false;
	while pq.len() > 0 {
		let cur_min = pq.pop().unwrap().0;
		if cur_min == prev_min {		
			if !currently_in_a_run {
				match_writer.write(prev_min.source, &prev_min.sa_value.to_le_bytes()).unwrap();
				match_count.fetch_add(1, Ordering::SeqCst);
			}
			match_writer.write(cur_min.source, &cur_min.sa_value.to_le_bytes()).unwrap();
			match_count.fetch_add(1, Ordering::SeqCst);
			currently_in_a_run = true;
		} else {
			currently_in_a_run = false;
		}
		if let Some(node) = stream_iterators.get_mut(&cur_min.source).unwrap().next() {
			pq.push(Reverse(node.unwrap()));
		}
		prev_min = cur_min;		
	}
	
	match_writer.finish().unwrap();

	println!("Finished PQ in {:?} secs | found {:?} matches", start_main.elapsed().as_secs(), match_count.into_inner());
	Ok(())
}


pub fn pq_serial_old(storage_dir: &PathBuf,
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
	let text_lookup = make_text_lookups(storage_dir).unwrap();
	let total_counts:usize = text_lookup.iter().map(|v| v.len()).sum();	
	let mut table_streams = make_table_streams(storage_dir, &text_lookup).unwrap();
	let pbar = build_pbar(total_counts, "Count.len");
	let mut stream_iterators: HashMap<usize, TextIterator> = HashMap::default();

	table_streams.iter_mut().for_each(|s| {
		stream_iterators.insert(s.source, s.text_iter(match_length));
	});


	/* Step 2: Initialize min-order data structure 
	*/

	let mut loser_tree = LoserTree::new(stream_iterators);
	let match_writer = MatchWriter::new(&storage_dir.clone().join("matches")).unwrap();
	let match_count = AtomicUsize::new(0);
	let prev_min = loser_tree.pop().unwrap();
	if prev_min == None {
		return Ok(());
	}

	let mut prev_min = prev_min.unwrap();
	let mut currently_in_a_run = false;
	while !loser_tree.peek().is_none() {
		let cur_min = loser_tree.pop().unwrap().unwrap();
		if cur_min == prev_min {
			if !currently_in_a_run {
				match_writer.write(prev_min.source, &prev_min.sa_value.to_le_bytes()).unwrap();
				match_count.fetch_add(1, Ordering::SeqCst);
			}
			match_writer.write(cur_min.source, &cur_min.sa_value.to_le_bytes()).unwrap();
			match_count.fetch_add(1, Ordering::SeqCst);
			currently_in_a_run = true;
		} else {
			currently_in_a_run = false;
		}
		prev_min = cur_min;
		// pbar.inc(1);
	}
	match_writer.finish().unwrap();
	println!("Finished PQ in {:?} secs | found {:?} matches ", start_main.elapsed().as_secs(), match_count.into_inner());
	Ok(())
}



fn make_text_lookups(storage_dir: &PathBuf) -> Result<Vec<Vec<u8>>, Error> {
	// Creates a lookup from idx -> text
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

	Ok(text_lookup)
}




fn make_table_streams<'stream, 'a>(storage_dir: &PathBuf, text_lookup: &'a Vec<Vec<u8>>) -> Result<Vec<SAStream<'a>>, Error> {
	let table_pattern = storage_dir.clone().join("table").join("table_part_*");
	let table_objects: Vec<PathBuf> = glob(&table_pattern.to_str().unwrap()).unwrap().into_iter().map(|p| p.unwrap()).collect();
		
	let table_streams: Vec<SAStream> = table_objects.into_par_iter().map(|p| {
		let table_idx = get_part_num(&p).unwrap();
		let stream = SAStream::new(&p, text_lookup[table_idx].as_slice(), table_idx, 16384).unwrap();
		stream
	}).collect();

	Ok(table_streams)

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
	let mut stream = SAStream::new(sa_table_loc, text_slice, 0, 16384).unwrap();
	let mut text_iterator = stream.text_iter(match_length);
	let prev_min = text_iterator.next();
	if prev_min.is_none() {
		return Ok(())
	}
	let mut prev_min = prev_min.unwrap().unwrap();
	let mut currently_in_a_run = false;
	for el in text_iterator {
		let cur_node = el.unwrap();
		if cur_node == prev_min {
			if currently_in_a_run {
				matches.push(prev_min.sa_value);
			} 
			matches.push(cur_node.sa_value);
			currently_in_a_run = true;
		} else {
			currently_in_a_run = false;
		}
		prev_min = cur_node;

	}

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
struct SAStream<'a> {
    reader: BufReader<File>,
    text: &'a [u8],
    buffer: Vec<u64>,
    position: usize,
    chunk_size: usize,
    source: usize, 
}

impl<'a> SAStream<'a> {
    fn new(sa_file: &PathBuf, text: &'a [u8], source: usize, chunk_size: usize) -> Result<Self, Error> {
    	let open_file = File::open(sa_file).unwrap();
        Ok(Self {        	
            reader: BufReader::new(open_file),
            text: text,
            buffer: Vec::with_capacity(chunk_size),
            position: 0,
            chunk_size,
            source
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

    fn text_iter<'stream>(&'stream mut self, min_len: usize) -> TextIterator<'stream, 'a> {
    	TextIterator {
    		stream: self,
    		min_len
    	}
    }
}

impl<'a> Iterator for SAStream<'a> {
	type Item = Result<u64, Error>;

	fn next(&mut self) -> Option<Self::Item> {
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
}

pub struct TextIterator<'stream, 'a> {
	stream: &'stream mut SAStream<'a>,
	min_len: usize
}

impl<'stream, 'a> Iterator for TextIterator<'stream, 'a> {
	type Item = Result<TreeNode<'a>, Error>;
	fn next(&mut self) -> Option<Self::Item> {
		loop {
			let next_idx_opt = self.stream.next();
			if let Some(next_idx) = next_idx_opt {
				let next_idx = next_idx.unwrap();
				if next_idx as usize + self.min_len >= self.stream.text.len() {
					continue;
				}
				let slice = &self.stream.text[next_idx as usize..next_idx as usize + self.min_len];
				//let sv : SmallVec<[u8; 512]> = SmallVec::from_slice(&slice);
				return Some(Ok(TreeNode{sa_value: next_idx,
										cmp_bytes: slice,
									    source: self.stream.source,
										}));
			} else {
				return None;
			}
		}
		None
	}
}



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


/*========================================================
=                            LOSER TREE STUFF            =
========================================================*/


/// Represents an element in the loser tree along with its source sequence index
#[derive(Debug, Clone)]
pub struct TreeNode<'a> {
    sa_value: u64,
    cmp_bytes: &'a [u8],
    source: usize, // which input sequence this came from
}

impl Ord for TreeNode<'_> {
    fn cmp(&self, other: &Self) -> StdOrdering {
        self.cmp_bytes.cmp(&other.cmp_bytes)
    }
}

impl PartialOrd for TreeNode<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<StdOrdering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for TreeNode<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp_bytes == other.cmp_bytes
    }
}

impl Eq for TreeNode<'_> {}



/// Loser Tree for efficient k-way merging
pub struct LoserTree<'a, 'stream: 'a>
{
    tree: Vec<Option<TreeNode<'a>>>, // Internal nodes store losers
    winner: Option<TreeNode<'a>>,     // Current minimum element
    iterators: HashMap<usize, TextIterator<'stream, 'a>>,                // Input sequences
    path_idxs: Vec<Vec<usize>>,
    k: usize,                         // Number of input sequences
}

impl<'a, 'stream: 'a> LoserTree<'a, 'stream>
{
    /// Create a new loser tree from k iterators
    pub fn new(mut iterators: HashMap<usize, TextIterator<'stream, 'a>>) -> Self {
        let k = iterators.len();

        // Tree needs k-1 internal nodes for k leaves
        let tree_size = LoserTree::calculate_tree_size(k);
        let mut tree = vec![None; tree_size];
        
        // Initialize leaves with first element from each iterator
	    let mut leaves: Vec<Option<TreeNode<'a>>> = (0..k)
	        .map(|source| {
	            if let Some(iter) = iterators.get_mut(&source) {
	                if let Some(node) = iter.next() {
	                    Some(node.unwrap())
	                } else {
	                    None
	                }
	            } else {
	                None
	            }
	        })
	        .collect();

        let path_idxs: Vec<Vec<usize>> = (0..k).map(|i| LoserTree::get_path_indices(i, k)).collect();


        // Build the initial tree
        let winner = Self::build_tree(&mut tree, &mut leaves);



        LoserTree {
            tree,
            winner,
            iterators,
            path_idxs,
            k,
        }
    }

	fn build_tree(
	    tree: &mut [Option<TreeNode<'a>>],
	    leaves: &mut [Option<TreeNode<'a>>],
	) -> Option<TreeNode<'a>> {
	    let mut level = leaves.to_vec();
	    let mut level_start = 0;
	    
	    while level.len() > 1 {
	        let mut next_level = Vec::new();
	        let pairs_in_level = (level.len() + 1) / 2;
	        
	        for pair_idx in 0..pairs_in_level {
	            let left_idx = pair_idx * 2;
	            let right_idx = left_idx + 1;
	            
	            if right_idx < level.len() {
	                // Compare two elements
	                match (&level[left_idx], &level[right_idx]) {
	                    (Some(a), Some(b)) => {
	                        if a <= b {
	                            tree[level_start + pair_idx] = Some(b.clone());
	                            next_level.push(Some(a.clone()));
	                        } else {
	                            tree[level_start + pair_idx] = Some(a.clone());
	                            next_level.push(Some(b.clone()));
	                        }
	                    }
	                    (Some(a), None) => {
	                        next_level.push(Some(a.clone()));
	                    }
	                    (None, Some(b)) => {
	                        next_level.push(Some(b.clone()));
	                    }
	                    (None, None) => {
	                        next_level.push(None);
	                    }
	                }
	            } else {
	                // Odd one out
	                next_level.push(level[left_idx].clone());
	            }
	        }
	        
	        level_start += pairs_in_level;
	        level = next_level;
	    }
	    
	    level.into_iter().next().flatten()
	}
    /// Get the current minimum element (if any)
    pub fn peek(&self) -> Option<&TreeNode<'a>> {
    	self.winner.as_ref()    
    }

	/// Extract the minimum element and advance the tree
	pub fn pop(&mut self) -> Result<Option<TreeNode<'a>>, Error> {
	    let winner = self.winner.take().unwrap();
	    let source = winner.source;
	    
	    let new_tree_entry = if let Some(node) = self.iterators.get_mut(&source).unwrap().next() {
	        Some(node.unwrap())
	    } else {
	        None
	    };
	    
	    self.winner = self.replay(new_tree_entry, source);
	    Ok(Some(winner))
	}

	pub fn pop_verbose(&mut self) -> Result<Option<TreeNode<'a>>, Error> {
	    let winner = self.winner.take().unwrap();
	    let source = winner.source;
	    
	    let new_tree_entry = if let Some(node) = self.iterators.get_mut(&source).unwrap().next() {
	        Some(node.unwrap())
	    } else {
	        None
	    };
	    
	    println!("Replaying: source={}, new_entry={:?}", source, new_tree_entry.as_ref().map(|n| n.source));
	    println!("Tree before replay: {:?}", self.tree.iter().map(|n| n.as_ref().map(|x| x.source)).collect::<Vec<_>>());
	    
	    self.winner = self.replay(new_tree_entry, source);
	    
	    println!("Winner after replay: {:?}", self.winner.as_ref().map(|n| n.source));
	    println!("Tree after replay: {:?}", self.tree.iter().map(|n| n.as_ref().map(|x| x.source)).collect::<Vec<_>>());
	    println!("---");
	    
	    Ok(Some(winner))
	}	


    pub fn is_empty(&self) -> bool {
        // Winner must be None AND all tree nodes must be None
        self.winner.is_none() && self.tree.iter().all(|node| node.is_none())
    }

    /// Replay comparisons along the path for a new element
    fn replay_og(&mut self, mut current: Option<TreeNode<'a>>, source: usize) -> Option<TreeNode<'a>> {
        // Find path from leaf to root based on source index
        let path = &self.path_idxs[source];
        for &idx in path {
            if let Some(ref mut loser) = self.tree[idx] {
                match &current {
                    Some(curr) if curr <= loser => {
                        // Current wins, swap with loser
                        std::mem::swap(&mut self.tree[idx], &mut current);
                    }
                    None => {
                        // Current is None, loser becomes winner
                        current = self.tree[idx].take();
                    }
                    _ => {
                        // Loser stays loser, current continues
                    }
                }
            } else if current.is_some() {
                // No loser at this node yet (shouldn't happen in normal operation)
                continue;
            }
        }

        current
    }

	fn replay2(&mut self, mut current: Option<TreeNode<'a>>, source: usize) -> Option<TreeNode<'a>> {
	    let path = &self.path_idxs[source];
	    for &idx in path {
	        match (current.as_ref(), self.tree[idx].as_ref()) {
	            (Some(curr), Some(loser)) => {
	                if curr <= loser {
	                    // Current wins
	                    std::mem::swap(&mut self.tree[idx], &mut current);
	                }
	                // else: loser stays loser, current continues
	            }
	            (None, Some(_)) => {
	                // None loses to Some
	                current = self.tree[idx].take();
	            }
	            (Some(_), None) => {
	                // Some wins against None, store None as loser
	                std::mem::swap(&mut self.tree[idx], &mut current);
	            }
	            (None, None) => {
	                // Both None, continue
	            }
	        }
	    }
	    current
	}    


	fn replay(&mut self, mut current: Option<TreeNode<'a>>, source: usize) -> Option<TreeNode<'a>> {
	    let path = &self.path_idxs[source];
	    for &idx in path {
	        match (&current, &self.tree[idx]) {
	            (Some(curr), Some(loser)) if curr <= loser => {
	                // Current wins, swap with loser
	                std::mem::swap(&mut self.tree[idx], &mut current);
	            }
	            (None, Some(_)) => {
	                // None always loses to Some
	                current = self.tree[idx].take();
	            }
	            (Some(_), None) => {
	                // Some always wins against None, store None as loser
	                self.tree[idx] = current.take();
	                // Wait, we need current back!
	                current = self.tree[idx].take();
	                self.tree[idx] = None;
	            }
	            _ => {
	                // Some(curr) > Some(loser) => loser stays, current continues
	                // Both None => current continues
	            }
	        }
	    }
	    current
	}


    /// Get the indices along the path from leaf to root
    fn get_path_indices(leaf: usize, k: usize) -> Vec<usize> {
        let mut path = Vec::new();
        let mut pos = leaf;
        let mut level_size = k;
        let mut level_start = 0;

        while level_size > 1 {
            let pair_idx = pos / 2;
            path.push(level_start + pair_idx);
            
            level_start += (level_size + 1) / 2;
            pos = pair_idx;
            level_size = (level_size + 1) / 2;
        }

        path
    }

    fn calculate_tree_size(k: usize) -> usize {
	    let mut total = 0;
	    let mut level_size = k;
	    
	    while level_size > 1 {
	        total += (level_size + 1) / 2;
	        level_size = (level_size + 1) / 2;
	    }
	    
	    total    	
    }
}




