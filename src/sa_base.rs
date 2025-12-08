use rand::SeedableRng;
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use crate::minhash_base::FileMap;
use crate::utils::json_set;
use ahash::HashMap;
use dashmap::DashMap;
use glob::glob;
use serde_json::json;
use serde_json::Value;
use std::cmp::Reverse;
use std::fs::File;
use std::io::BufRead;
use std::io::Read;
use std::io::Seek;
use std::io::SeekFrom;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::time::Instant;

use crate::table_old::SuffixTable;
use anyhow::{Error, Result};
use gjson;
use mj_io::{
    build_pbar, get_output_filename, read_pathbuf, read_pathbuf_to_mem, write_mem_to_pathbuf,
};
use rayon;
use rayon::prelude::*;
use std::path::PathBuf;

use std::cmp::Ordering as StdOrdering;
use std::collections::BinaryHeap;

use crate::sa_utils::{FileRange, SAStream, TextIterator, MatchWriter, TreeNode, LoserTree, read_u64_vec, sa_safety_check, sa_thread_memory};
use crate::sa_config::{SAConfigOverrides, SAConfig};

/*

Scaling notes:
- assume that all text can fit into main memory
- do NOT assume that all text + SA's can fit into main memory



Notes for mega-scale suffix-array usage:

Step 1: Make suffix array tables.
Output:
	- storage/file_map.json.gz	
	k separate files:
	- storage/text/text_part_XXXX.bin
	- storage/table/table_part_XXXX.bin
	- storage/offsets/offset_part_XXXX.bin

Where: 
	- text is a global concatenation of:
		<document_text>\xff\xff<path_id><line_num>
		where path_id and line_num are both little-ended u64's

	- table is the suffix table for the given text 
	- offsets is 
		<path_id><line_num><doc_len>
		where each are little-ended u64s

- So we:
1) load all text
2) from here, compute the SA sizes
	a) can do the 'large-as-possible', where we have 1 suffix array, but this is non-threaded
	b) can do the 'one SA per thread', where number of SA's are small (pay logK later)
	c) make chunks based on RAM sizes:
		each thread can have THREAD_MEM := SYS_MEM / THREAD_COUNT memory
		so that means each thread can have THREAD_MEM/9 * SAFETY_MARGIN text max
3) save everything
*/


/*======================================================
=                    MAKE TABLE                        =
======================================================*/
const MEMORY_SAFETY_MARGIN: f64 = 0.90;


pub fn make_sa_tables_cmd(
    input_dir: &PathBuf,
    output_dir: &PathBuf,
    config_opt: Option<PathBuf>,
    file_map_opt: Option<PathBuf>,
    tokenizer: Option<String>,
    max_lines_per_path: Option<usize>,
    text_key: Option<String>,
) -> Result<(), Error> {
    let file_map = if let Some(file_map_name) = file_map_opt {
        FileMap::load(&file_map_name).unwrap()
    } else {
        let file_map = FileMap::new(input_dir, &None).unwrap();
        file_map
            .save(&(output_dir.join("file_map.json.gz")))
            .unwrap();
        file_map
    };

    let config_overrides = SAConfigOverrides {
        tokenizer: tokenizer,
        max_lines_per_path: max_lines_per_path,
        text_key: text_key,
        random_seed: None,
    };

    let config_obj = SAConfig::load_with_overrides(config_opt, config_overrides).unwrap();

    make_sa_tables(&config_obj, &file_map, input_dir, output_dir)
}

pub fn make_sa_tables(
    config_obj: &SAConfig,
    file_map: &FileMap,
    input_dir: &PathBuf,
    output_dir: &PathBuf,
) -> Result<(), Error> {
    assert_eq!(
        1u16.to_ne_bytes(),
        1u16.to_le_bytes(),
        "System must be little-endian"
    );

    let start_main = Instant::now();
    println!("Building SA tables...");

    // Step 1: load all data into memory and then maybe tokenize if needed?
    let start_read = Instant::now();
    println!("Reading all documents texts...");
    let data_docs = load_data_docs(file_map, config_obj, input_dir).unwrap();
    println!(" {:?} DOCS TOTAL", data_docs.len());
    println!(
        "...Read all documents in {:?} secs",
        start_read.elapsed().as_secs()
    );

    // Safety check:
    let corpus_len: usize = data_docs.par_iter().map(|(_, _, doc)| doc.len() + 2 * 8).sum();
    assert!(sa_safety_check(corpus_len));
    let thread_count = rayon::current_num_threads();
    let thread_mem = sa_thread_memory(thread_count, MEMORY_SAFETY_MARGIN);



    // Step 2 (optional): Tokenize if needed
    // TODO: tokenizer thing


    // Step 3: Break into chunks and make SA table for each of them
    let owned_chunks: DashMap<usize, Vec<(usize, usize, String)>> = chunk_data_for_sa(config_obj, data_docs, thread_mem, thread_count);
    let total_bytes = AtomicUsize::new(0);

    let total_pbar = build_pbar(owned_chunks.len() * 4, "Total steps");
    owned_chunks.into_par_iter().for_each(|(table_idx, chunk)| {
        let allocate_size = chunk
            .iter()
            .map(|(_, _, s)| s.len() + 2 + 2 * 8)
            .sum::<usize>();
        let mut document_offsets: Vec<u64> = Vec::new();
        document_offsets.extend(vec![u64::MAX, u64::MAX, 0]);
        let mut buf = Vec::with_capacity(allocate_size);

        chunk.into_iter().for_each(|(u1, u2, s)| {
            buf.extend_from_slice(s.as_bytes());
            buf.extend_from_slice(&[0xff, 0xff]);
            buf.extend_from_slice(&(u1 as u64).to_le_bytes());
            buf.extend_from_slice(&(u2 as u64).to_le_bytes());
            document_offsets.extend(vec![u1 as u64, u2 as u64, buf.len() as u64]);
            // Offsets are trips of (path_id, line_num, current_offset)
        });
        assert_eq!(document_offsets.len() % 3, 0);

        total_bytes.fetch_add(allocate_size, Ordering::SeqCst);
        total_pbar.inc(1);
        // And save concatenated text just for safekeeping
        let output_text_filename = output_dir
            .clone()
            .join("text")
            .join(format!("text_part_{:04}.bin", table_idx));
        write_mem_to_pathbuf(&buf, &output_text_filename).unwrap();
        total_pbar.inc(1);

        // And save the document offsets
        let output_offset_filename = output_dir
            .clone()
            .join("offsets")
            .join(format!("offset_part_{:04}.bin", table_idx));
        let offset_vec: &[u8] = bytemuck::cast_slice(&document_offsets);
        write_mem_to_pathbuf(offset_vec, &output_offset_filename).unwrap();

        // And then make the SA table too
        let table = SuffixTable::new(buf);
        total_pbar.inc(1);

        let table_to_write: &[u8] = bytemuck::cast_slice(table.table());
        let output_table_filename = output_dir
            .clone()
            .join("table")
            .join(format!("table_part_{:04}.bin", table_idx));
        write_mem_to_pathbuf(table_to_write, &output_table_filename).unwrap();
    });

    println!(
        "Made {:?} tables of {:?} bytes total in {:?} seconds",
        thread_count,
        total_bytes.into_inner(),
        start_main.elapsed().as_secs()
    );
    Ok(())
}

pub fn load_data_docs(
    file_map: &FileMap,
    config: &SAConfig,
    local_input: &PathBuf,
) -> Result<Vec<(usize, usize, String)>, Error> {
    let text_key = config.text_key.clone();
    let mut extant_idxs: Vec<(&PathBuf, &usize)> = file_map
        .indices
        .par_iter()
        .filter(|(p, _idx)| local_input.clone().join(p).exists())
        .collect();
    extant_idxs.par_sort_unstable_by_key(|(_, &i)| i);

    let pbar = build_pbar(extant_idxs.len(), "Paths");
    let data_docs: Vec<(usize, usize, String)> = extant_idxs
        .par_iter()
        .flat_map(|(p, idx)| {
            let mut sub_docs: Vec<(usize, usize, String)> = Vec::new();
            let contents = read_pathbuf(&(local_input.clone().join(p)), true).unwrap();
            for (line_num, line) in contents.lines().enumerate() {
                let line = line.unwrap();
                let value = gjson::get(&line, &text_key).str().to_string();
                sub_docs.push((**idx, line_num, value));
            }
            pbar.inc(1);
            sub_docs
        })
        .collect();
    Ok(data_docs)
}

fn get_part_num(path: &PathBuf) -> Result<usize, Error> {
    Ok(path
        .file_stem()
        .unwrap()
        .to_str()
        .unwrap()
        .split("_")
        .last()
        .unwrap()
        .parse::<usize>()
        .unwrap())
}

fn chunk_data_for_sa(
    config: &SAConfig, 
    data_docs: Vec<(usize, usize, String)>, 
    thread_memory: usize, 
    thread_count: usize
) -> DashMap<usize, Vec<(usize, usize, String)>> {
    let owned_chunks: DashMap<usize, Vec<(usize, usize, String)>> = DashMap::new();
    let chunk_limit = thread_memory / 9;

    // Step 1: shuffle and index the data
    let mut indexed: Vec<(u64, (usize, usize, String))> = data_docs
        .into_par_iter()
        .enumerate()
        .map(|(i, item)| {
            let mut rng = ChaCha8Rng::seed_from_u64(config.random_seed.wrapping_add(i as u64));
            (rng.gen(), item)
        })
        .collect();
    
    indexed.par_sort_unstable_by_key(|(key, _)| *key);

    // Step 2: distribute into thread buckets by MOVING (no clone!)
    let mut chunks: Vec<Vec<_>> = (0..thread_count).map(|_| Vec::new()).collect();
    
    for (i, item) in indexed.into_iter().enumerate() {
        chunks[i % thread_count].push(item);
    }
    
    // Sort each bucket in parallel
    chunks.par_iter_mut().for_each(|bucket| {
        bucket.par_sort_unstable_by_key(|(r, _)| *r);
    });

    // Step 3: split each bucket into memory-bounded chunks
    chunks.into_par_iter().enumerate().for_each(|(chunk_id, chunk)| {
        let mut current_part = chunk_id;
        let mut current_size = 0;
        let mut current_chunk = Vec::new();

        for (_, (idx1, idx2, text)) in chunk {
            let item_size = text.len() + 16; // approximate overhead
            assert!(item_size < chunk_limit, "Document too large for chunk limit!");
            
            if current_size + item_size > chunk_limit && current_size > 0 {
                // Flush current chunk
                owned_chunks.insert(current_part, current_chunk);
                current_chunk = Vec::new();
                current_size = 0;
                current_part += thread_count;
            }
            
            current_chunk.push((idx1, idx2, text));
            current_size += item_size;
        }
        
        // Don't forget the last chunk
        if !current_chunk.is_empty() {
            owned_chunks.insert(current_part, current_chunk);
        }
    });

    owned_chunks
}

/*============================================================
=                            PQ LOOPS                        =
============================================================*/

pub fn pq_serial(storage_dir: &PathBuf, match_length: usize) -> Result<(), Error> {
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
    let mut table_streams = make_table_streams(storage_dir, &text_lookup).unwrap();

    let mut stream_iterators: HashMap<usize, TextIterator<File>> = HashMap::default();

    table_streams.iter_mut().for_each(|s| {
        stream_iterators.insert(s.source, s.text_iter(match_length));
    });

    /* Step 2: Initialize min-order data structure
    	*/

    let mut pq: BinaryHeap<Reverse<TreeNode>> = BinaryHeap::new();
    for (_source, iterator) in stream_iterators.iter_mut() {
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

    let mut stream_runs = 0;
    let mut total_steps = 0;
    /* Step 3: Now do the loop
    	*/
    let match_writer = MatchWriter::new(&storage_dir.clone().join("matches")).unwrap();
    let match_count = AtomicUsize::new(0);
    let mut currently_in_a_run = false;
    while pq.len() > 0 {
        let cur_min = pq.pop().unwrap().0;
        total_steps += 1;
        if cur_min.source == prev_min.source {
            stream_runs += 1;
        }
        if cur_min == prev_min {
            if !currently_in_a_run {
                match_writer
                    .write(prev_min.source, &prev_min.sa_value.to_le_bytes())
                    .unwrap();
                match_count.fetch_add(1, Ordering::SeqCst);
            }
            match_writer
                .write(cur_min.source, &cur_min.sa_value.to_le_bytes())
                .unwrap();
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

    println!(
        "Finished PQ in {:?} secs | found {:?} matches | {:?}/{:?}",
        start_main.elapsed().as_secs(),
        match_count.into_inner(),
        stream_runs,
        total_steps
    );
    Ok(())
}

pub fn get_matches_serial(storage_dir: &PathBuf, match_length: usize) -> Result<(), Error> {
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
    let total_counts: usize = text_lookup.iter().map(|v| v.len()).sum();
    let mut table_streams = make_table_streams(storage_dir, &text_lookup).unwrap();
    let pbar = build_pbar(total_counts, "Count.len");
    let mut stream_iterators: HashMap<usize, TextIterator<File>> = HashMap::default();

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
    let mut pop_count = 0;
    while !loser_tree.peek().is_none() {
        pop_count += 1;
        let cur_min = loser_tree.pop_check().unwrap().unwrap();
        if cur_min == prev_min {
            if !currently_in_a_run {
                match_writer
                    .write(prev_min.source, &prev_min.sa_value.to_le_bytes())
                    .unwrap();
                match_count.fetch_add(1, Ordering::SeqCst);
            }
            match_writer
                .write(cur_min.source, &cur_min.sa_value.to_le_bytes())
                .unwrap();
            match_count.fetch_add(1, Ordering::SeqCst);
            currently_in_a_run = true;
        } else {
            currently_in_a_run = false;
        }
        prev_min = cur_min;
        pbar.inc(1);
    }
    match_writer.finish().unwrap();
    println!(
        "Finished PQ in {:?} secs | found {:?} matches| {:?} /{:?} ",
        start_main.elapsed().as_secs(),
        match_count.into_inner(),
        loser_tree.shortcut_count,
        pop_count
    );
    Ok(())
}

fn make_text_lookups(storage_dir: &PathBuf) -> Result<Vec<Vec<u8>>, Error> {
    // Creates a lookup from idx -> text
    let text_pattern = storage_dir.clone().join("text").join("text_part_*");
    let text_objects: Vec<PathBuf> = glob(&text_pattern.to_str().unwrap())
        .unwrap()
        .into_iter()
        .map(|p| p.unwrap())
        .collect();
    let max_text_idx = AtomicUsize::new(0);
    let text_data: Vec<(usize, Vec<u8>)> = text_objects
        .into_par_iter()
        .map(|p| {
            //PAR
            let text_idx = get_part_num(&p).unwrap();
            max_text_idx.fetch_max(text_idx, Ordering::SeqCst);
            (
                text_idx,
                read_pathbuf_to_mem(&p).unwrap().into_inner().into_inner(),
            )
        })
        .collect();
    let mut text_lookup: Vec<Vec<u8>> = vec![Vec::new(); max_text_idx.into_inner() + 1];
    text_data
        .into_iter()
        .for_each(|(idx, v)| text_lookup[idx] = v);

    Ok(text_lookup)
}

fn make_table_streams<'stream, 'a>(
    storage_dir: &PathBuf,
    text_lookup: &'a Vec<Vec<u8>>,
) -> Result<Vec<SAStream<'a, File>>, Error> {
    let table_pattern = storage_dir.clone().join("table").join("table_part_*");
    let table_objects: Vec<PathBuf> = glob(&table_pattern.to_str().unwrap())
        .unwrap()
        .into_iter()
        .map(|p| p.unwrap())
        .collect();

    let table_streams: Vec<SAStream<File>> = table_objects
        .into_par_iter()
        .map(|p| {
            let table_idx = get_part_num(&p).unwrap();
            let open_file = File::open(&p).unwrap();
            let stream = SAStream::new(
                open_file,
                text_lookup[table_idx].as_slice(),
                table_idx,
                16384,
            )
            .unwrap();
            stream
        })
        .collect();

    Ok(table_streams)
}

pub fn get_matches_parallel(storage_dir: &PathBuf, match_length: usize) -> Result<(), Error> {
    /* Paralellism strategy : Paralellize across alphabet (based on statistics from one )
    1. Load all texts into memory
    2. Pick one suffix table and scan through it to get some reference suffices to serve as waypoints.
    3. Binary search through all suffix tables to break into start/end slices.
    4. Farm out:
        - Global lookup of all text data
        - Sub-streamers for each of the SA tables
        - Writers for each part
    */

    let start_main = Instant::now();
    println!("Starting match-gathering");
    let text_lookup = make_text_lookups(storage_dir).unwrap();
    let thread_count = rayon::current_num_threads();
    // Get a #threads-1 list of internal boundaries to make #threads streams for each part
    let reference_suffices =
        get_reference_suffices(storage_dir, match_length, thread_count, &text_lookup).unwrap();
    // Get a #Parts x #threads streams
    let substreams =
        get_all_substreams(storage_dir, reference_suffices, match_length, &text_lookup).unwrap();

    // And then transpose into a #threads x #Parts grid
    let mut thread_iters: Vec<HashMap<usize, SAStream<FileRange>>> =
        (0..thread_count).map(|_| HashMap::default()).collect();
    substreams
        .into_iter()
        .enumerate()
        .for_each(|(part_num, part)| {
            part.into_iter()
                .enumerate()
                .for_each(|(thread_num, stream)| {
                    thread_iters[thread_num].insert(part_num, stream);
                })
        });


    // And set up the matches writers
    let match_writer = MatchWriter::new(&storage_dir.clone().join("matches")).unwrap();
    // Finally we can do the parallel merge thing:
    println!("Starting parallel match-finding...");
    thread_iters.par_iter_mut().enumerate().for_each(|i, streams| {
        get_matches_parallel_thread(streams, &match_writer, match_length, ).unwrap()
    });
    match_writer.finish().unwrap();


    println!("Found all matches in {:?} secs", start_main.elapsed().as_secs());
    Ok(())
}

fn get_reference_suffices(
    storage_dir: &PathBuf,
    match_length: usize,
    thread_count: usize,
    text_lookup: &Vec<Vec<u8>>,
) -> Result<Vec<Vec<u8>>, Error> {
    // Output is a vec of size (thread_count-1), with the internal boundaries needed to break SA's into thread_count parts
    // Get just one table and its reference text
    let table_pattern = storage_dir.clone().join("table").join("table_part_*");
    let table_objects: Vec<PathBuf> = glob(&table_pattern.to_str().unwrap())
        .unwrap()
        .into_iter()
        .map(|p| p.unwrap())
        .collect();
    let reference_table_path: PathBuf = table_objects.first().unwrap().clone();
    let mut reference_table_file: File = File::open(&reference_table_path).unwrap();
    let part_num = get_part_num(&reference_table_path).unwrap();
    let reference_text = text_lookup[part_num].as_slice();

    // And get the boundary indices
    let suffix_len = reference_table_file.metadata().unwrap().len() / 8;
    assert!(thread_count > 1);
    assert!(suffix_len > thread_count.try_into().unwrap());
    let el_idxs: Vec<usize> = (1..thread_count)
        .map(|i| (i as f64 * ((suffix_len as f64) / (thread_count as f64))).round() as usize)
        .collect();

    // For boundary indices, get reference suffices:
    let mut reference_suffices: Vec<Vec<u8>> = Vec::new();
    el_idxs.into_iter().for_each(|i| {
        reference_table_file
            .seek(SeekFrom::Start((8 * i).try_into().unwrap()))
            .unwrap();
        let mut buffer = [0u8; 8];
        reference_table_file.read_exact(&mut buffer).unwrap();
        let boundary_start = u64::from_le_bytes(buffer);
        let boundary_end = std::cmp::min(
            reference_text.len(),
            (boundary_start + match_length as u64).try_into().unwrap(),
        );
        let boundary: Vec<u8> =
            reference_text[boundary_start as usize..boundary_end as usize].to_vec();
        reference_suffices.push(boundary);
    });

    Ok(reference_suffices)
}

fn get_all_substreams<'a>(
    storage_dir: &PathBuf,
    boundaries: Vec<Vec<u8>>,
    match_length: usize,
    text_lookup: &'a Vec<Vec<u8>>,
) -> Result<Vec<Vec<SAStream<'a, FileRange>>>, Error> {
    // Creates |thread_count| TextIterators for each SA Table, each offset by a little bit on each end from the boundaries

    // Step 1: Get table/text pairs
    let mut table_text_pairs: Vec<(PathBuf, &Vec<u8>)> = Vec::new();
    let table_pattern = storage_dir.clone().join("table").join("table_part_*");
    let table_objects: Vec<PathBuf> = glob(&table_pattern.to_str().unwrap())
        .unwrap()
        .into_iter()
        .map(|p| p.unwrap())
        .collect();
    table_objects.into_iter().for_each(|p| {
        let part_num = get_part_num(&p).unwrap();
        table_text_pairs.push((p, &text_lookup[part_num]));
    });

    let output: Vec<Vec<SAStream<FileRange>>> = table_text_pairs
        .into_par_iter()
        .map(|(table_path, text)| {
            get_substreams_from_boundaries(&table_path, text, &boundaries, match_length).unwrap()
        })
        .collect();

    Ok(output)
}

fn get_substreams_from_boundaries<'a>(
    table_path: &PathBuf,
    text: &'a Vec<u8>,
    boundaries: &Vec<Vec<u8>>,
    match_length: usize,
) -> Result<Vec<SAStream<'a, FileRange>>, Error> {
    let mut table_file = File::open(table_path).unwrap();
    let sa_len = table_file.metadata().unwrap().len() / 8;

    let mut start_end_idxs: Vec<(usize, usize)> = Vec::new();
    // 0'th element

    start_end_idxs.push((
        0,
        sa_disk_bisect(
            &mut table_file,
            text,
            boundaries.first().unwrap(),
            match_length,
            true,
        )
        .unwrap(),
    ));
    for idx in 0..boundaries.len() - 1 {
        let l_boundary = &boundaries[idx];
        let r_boundary = &boundaries[idx + 1];
        let left_sa_index =
            sa_disk_bisect(&mut table_file, text, &l_boundary, match_length, false).unwrap();
        let right_sa_index =
            sa_disk_bisect(&mut table_file, text, &r_boundary, match_length, true).unwrap();
        start_end_idxs.push((left_sa_index, right_sa_index));
    }
    start_end_idxs.push((
        sa_disk_bisect(
            &mut table_file,
            text,
            boundaries.last().unwrap(),
            match_length,
            false,
        )
        .unwrap(),
        sa_len.try_into().unwrap(),
    ));

    // Dilate by 1 in each direction to ensure some overlap so we don't miss any intra-table duplicates
    let dilated_idxs: Vec<(usize, usize)> = start_end_idxs
        .into_iter()
        .map(|(l, r)| {
            let new_l = if l > 0 { l - 1 } else { l };

            let new_r = if r < sa_len.try_into().unwrap() {
                r + 1
            } else {
                r
            };
            (new_l, new_r)
        })
        .collect();

    // Then create FileRanges and SAStreams and TextIterators
    let text_slice = text.as_slice();
    let part_num = get_part_num(&table_path).unwrap();
    let sa_streams: Vec<SAStream<FileRange>> = dilated_idxs
        .into_iter()
        .map(|(l, r)| {
            let new_file = File::open(table_path).unwrap();
            let file_range = FileRange::new(
                new_file,
                (l * 8).try_into().unwrap(),
                (r * 8).try_into().unwrap(),
            )
            .unwrap();
            let sas = SAStream::new(file_range, text_slice, part_num, 16384).unwrap();
            sas
        })
        .collect();

    Ok(sa_streams)
}

fn sa_disk_bisect(
    table_file: &mut File,
    text: &Vec<u8>,
    boundary_str: &Vec<u8>,
    match_length: usize,
    find_right: bool, // true for bisect_right, false for bisect_left
) -> Result<usize, Error> {
    // Finds the index in the SA (in terms of SA ELEMENTS, not BYTES) such that everything to the left (idx <) is strictly less than boundary_str
    let sa_len = (table_file.metadata().unwrap().len() / 8) as usize;

    let mut left = 0;
    let mut right = sa_len;

    while left < right {
        let mid = left + (right - left) / 2;
        let mut buffer = [0u8; 8];
        table_file.seek(SeekFrom::Start((mid * 8) as u64))?;
        table_file.read_exact(&mut buffer)?;
        let t_start = u64::from_le_bytes(buffer) as usize;

        let t_end = std::cmp::min(text.len(), t_start + match_length);
        let t = &text[t_start..t_end];

        let cmp_len = std::cmp::min(boundary_str.len(), t.len());
        let comparison = boundary_str[..cmp_len].cmp(&t[..cmp_len]);

        let ordering = if comparison == StdOrdering::Equal {
            boundary_str.len().cmp(&t.len())
        } else {
            comparison
        };

        // bisect_left: find first where suffix >= boundary_str
        // bisect_right: find first where suffix > boundary_str
        let should_search_right = if find_right {
            ordering != StdOrdering::Less // boundary_str >= suffix
        } else {
            ordering == StdOrdering::Greater // boundary_str > suffix
        };

        if should_search_right {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    Ok(left)
}

fn get_matches_parallel_thread<'a>(
    streams: &'a mut HashMap<usize, SAStream<'a, FileRange>>,
    match_writer: &MatchWriter,
    match_length: usize,
    use_pbar: bool
) -> Result<(), Error> {

    let pbar_opt = if use_pbar {
    	let total_size: usize = streams.iter().map(|(k,v)| {v.byte_size as usize}).sum::<usize>();
    	let pbar = build_pbar(total_size, "Thread bytes");
    	Some(pbar)
    } else {
    	None
    };

    let stream_iterators: HashMap<usize, TextIterator<FileRange>> = streams
        .iter_mut()
        .map(|(k, v)| (*k, v.text_iter(match_length)))
        .collect();

    let mut loser_tree = LoserTree::new(stream_iterators);
    let match_count = AtomicUsize::new(0);
    let prev_min = loser_tree.pop().unwrap();
    if prev_min == None {
        return Ok(());
    }


    let mut prev_min = prev_min.unwrap();
    let mut currently_in_a_run = false;
    while !loser_tree.peek().is_none() {
        let cur_min = loser_tree.pop_check().unwrap().unwrap();
        if cur_min == prev_min {
            if !currently_in_a_run {
                match_writer
                    .write(prev_min.source, &prev_min.sa_value.to_le_bytes())
                    .unwrap();
                match_count.fetch_add(1, Ordering::SeqCst);
            }
            match_writer
                .write(cur_min.source, &cur_min.sa_value.to_le_bytes())
                .unwrap();
            match_count.fetch_add(1, Ordering::SeqCst);
            currently_in_a_run = true;
        } else {
            currently_in_a_run = false;
        }
        prev_min = cur_min;
        if let Some(ref pbar) = pbar_opt {
        	pbar.inc(1);
        }
    }

    Ok(())
}

/*=====================================================================
=                            ALTERNATIVE FLOW                         =
=====================================================================*/

pub fn alternative_merge(
    text_loc: &PathBuf,
    sa_table_loc: &PathBuf,
    match_length: usize,
) -> Result<(), Error> {
    let start_main = Instant::now();

    println!("Starting alt flow");
    let text: Vec<u8> = read_pathbuf_to_mem(text_loc)
        .unwrap()
        .into_inner()
        .into_inner();
    let text_slice: &[u8] = text.as_slice();
    //let sa_table: Vec<u64> = read_u64_vec(sa_table_loc).unwrap();
    //let table_len = sa_table.len();
    let mut matches: Vec<u64> = Vec::new();
    let open_sa = File::open(sa_table_loc).unwrap();
    let mut stream = SAStream::new(open_sa, text_slice, 0, 16384).unwrap();
    let mut text_iterator = stream.text_iter(match_length);
    let prev_min = text_iterator.next();
    if prev_min.is_none() {
        return Ok(());
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

    println!(
        "Finished alt flow in {:?} seconds | found {:?} matches",
        start_main.elapsed().as_secs(),
        matches.len()
    );
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

    let matches: DashMap<usize, PathBuf> = glob(&match_pattern.to_str().unwrap())
        .unwrap()
        .into_iter()
        .map(|p| {
            let path = p.unwrap();
            let part_num = get_part_num(&path).unwrap();
            (part_num, path)
        })
        .collect();

    let offsets: DashMap<usize, PathBuf> = glob(&offset_pattern.to_str().unwrap())
        .unwrap()
        .into_iter()
        .map(|p| {
            let path = p.unwrap();
            let part_num = get_part_num(&path).unwrap();
            (part_num, path)
        })
        .collect();

    let pbar_counter = AtomicUsize::new(0);

    matches.into_par_iter().for_each(|(part_num, match_path)| {
        let offset_path = offsets.get(&part_num).unwrap();
        merge_match_path(
            match_path,
            &offset_path,
            &merged_match_dir,
            match_length,
            pbar_counter.fetch_add(1, Ordering::SeqCst) == 0,
        )
        .unwrap();
    });

    println!(
        "Merged all matches in {:?} secs",
        start_main.elapsed().as_secs()
    );
    Ok(())
}

pub fn merge_match_path(
    match_path: PathBuf,
    offset_path: &PathBuf,
    merged_match_dir: &PathBuf,
    match_length: usize,
    pbar_flag: bool,
) -> Result<(), Error> {
    let mut match_idxs = read_u64_vec(&match_path).unwrap();
    let offset_trips = read_u64_vec(&offset_path).unwrap();
    let offset_trips: Vec<(u64, u64, u64)> = offset_trips
        .chunks_exact(3)
        .map(|c| (c[0], c[1], c[2]))
        .collect(); // (path_id, line_num, idx-ceiling)

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
            match_intervals.extend(vec![
                offset_trip.0,
                offset_trip.1,
                cur_start - prev_offset_start,
                cur_end - prev_offset_start,
            ]);
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
        match_intervals.extend(vec![
            offset_trip.0,
            offset_trip.1,
            cur_start - prev_offset_start,
            cur_end - prev_offset_start,
        ]);
    }

    let match_interval_bytes: &[u8] = bytemuck::cast_slice(&match_intervals);
    let file_stem = match_path.file_stem().unwrap().to_str().unwrap();
    let output_filename = merged_match_dir
        .clone()
        .join(format!("merged_{}.bin", file_stem));
    write_mem_to_pathbuf(match_interval_bytes, &output_filename)
}

/*======================================================================
=                            ANNOTATE FILES                            =
======================================================================*/

pub fn sa_annotate_files(
    input_dir: &PathBuf,
    storage_dir: &PathBuf,
    output_dir: &PathBuf,
    annotate_key: &String,
) -> Result<(), Error> {
    let start_main = Instant::now();
    println!("Starting annotation...");
    let file_map_loc = storage_dir.clone().join("file_map.json.gz");
    let file_map = FileMap::load(&file_map_loc).unwrap();

    // Then loop through paths
    let merge_match_pattern = storage_dir
        .clone()
        .join("merged_matches")
        .join("merged_*.bin");
    let merge_match_objects: Vec<PathBuf> = glob(&merge_match_pattern.to_str().unwrap())
        .unwrap()
        .into_iter()
        .map(|p| p.unwrap())
        .collect();
    let anno_groups: DashMap<usize, DashMap<usize, Vec<(u64, u64)>>> = DashMap::new();


    merge_match_objects.into_par_iter().for_each(|p| {
        let contents: Vec<u64> = read_u64_vec(&p).unwrap();
        let contents: Vec<(u64, u64, u64, u64)> = contents
            .chunks_exact(4)
            .map(|c| (c[0], c[1], c[2], c[3]))
            .collect();
        contents
            .into_iter()
            .for_each(|(path_id, line_num, start, end)| {
                anno_groups
                    .entry(path_id as usize)
                    .or_default()
                    .entry(line_num as usize)
                    .or_default()
                    .push((start, end));
            });
    });

    let extant_idxs: Vec<(&PathBuf, &usize)> = file_map
        .indices
        .par_iter()
        .filter(|(p, _idx)| input_dir.clone().join(p).exists())
        .collect();
    let pbar = build_pbar(extant_idxs.len(), "Pahts");

    let anno_docs = AtomicUsize::new(0);
    let anno_bytes = AtomicUsize::new(0);

    extant_idxs.into_par_iter().for_each(|(p, idx)| {
        let anno_group = anno_groups.entry(*idx).or_default();
        let input_path = input_dir.clone().join(&p);
        let output_path = get_output_filename(&input_path, input_dir, output_dir).unwrap();
        let (p_anno_docs, p_anno_bytes) = sa_annotate_path(input_path, &anno_group, &output_path, annotate_key).unwrap();
        anno_docs.fetch_add(p_anno_docs, Ordering::SeqCst);
        anno_bytes.fetch_add(p_anno_bytes, Ordering::SeqCst);
        pbar.inc(1);
    });
    println!(
        "Annotated {:?} docs, marked {:?} bytes in {:?} secs",
		anno_docs.into_inner(),
		anno_bytes.into_inner(),        
        start_main.elapsed().as_secs()
    );
    Ok(())
}

fn sa_annotate_path(
    input_path: PathBuf,
    anno_group: &DashMap<usize, Vec<(u64, u64)>>,
    output_path: &PathBuf,
    annotate_key: &String,
) -> Result<(usize, usize), Error> {
    let contents = read_pathbuf_to_mem(&input_path).unwrap();
    let mut output_contents: Vec<u8> = Vec::new();
    let mut p_anno_docs = 0;
    let mut p_anno_bytes = 0;
    for (line_num, line) in contents.lines().enumerate() {
        let line = line.unwrap();
        let anno_val = anno_group.entry(line_num).or_default();
        if anno_val.len() == 0 {
            output_contents.extend(line.as_bytes());
        } else {
        	p_anno_docs += 1;
            let mut line_json: Value = serde_json::from_str(&line).unwrap();
            p_anno_bytes += anno_val.iter().map(|(l, r)| (r - l) as usize).sum::<usize>();
            let anno_val: Value = json!(*anno_val);
            json_set(&mut line_json, annotate_key, anno_val).unwrap();
            let output_line = serde_json::to_vec(&line_json).unwrap();
            output_contents.extend(output_line);
        }
        output_contents.push(b'\n')
    }

    write_mem_to_pathbuf(&output_contents, output_path).unwrap();
    Ok((p_anno_docs, p_anno_bytes))
}

