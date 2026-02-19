use crate::compact_uint::{CompactUint, U40, read_compact_uint_fast};
use ahash::HashMap;
use dashmap::DashMap;
use std::fs::create_dir_all;
use std::fs::File;
use std::fs::OpenOptions;
use std::io::BufWriter;
use std::io::Read;
use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::sync::Arc;
use std::sync::Mutex;

use anyhow::{Error, Result};
use rayon;
use rayon::prelude::*;
use std::path::PathBuf;
use indicatif::ProgressBar;
use std::cmp::Ordering as StdOrdering;
use sysinfo::System;
use memmap2::Mmap;


/*======================================================
=                           UTILS                      =
======================================================*/

pub fn get_byte_size(num_bytes: usize) -> usize {
    // Gets the number of bytes needed for suffix table indices based on the text sizes
    if num_bytes <= u32::MAX as usize {
        4
    } else if num_bytes <= (1u64 <<40 - 1) as usize {
        5
    } else {
        8
    }
}


pub fn read_u64_vec(p: &PathBuf) -> Result<Vec<u64>, Error> {
    let file_size = std::fs::metadata(&p)?.len() as usize;
    let mut output: Vec<u64> = vec![0u64; file_size / 8];
    let bytes_view: &mut [u8] = bytemuck::cast_slice_mut(&mut output);
    std::fs::File::open(&p)
        .unwrap()
        .read_exact(bytes_view)
        .unwrap();
    Ok(output)
}


pub fn adaptive_batch_size(corpus_len: usize, safety_margin: f64) -> usize {
    /*Peak memory usage is roughly the sum of :
       - text + incidentals : 1.25 * corpus_len 
       - suffix_tables: corpus_len * T / B 
            where B is the number of batches and T is the size of each batch
            where T ~~ ceil(log_2(corpus_len / (num_threads * batch_size)))
            and B is decided adaptively here (but is an integer!)
    */

    let text_memory = corpus_len as f64 * 1.25;

    let rayon_threads = rayon::current_num_threads() as f64;
    let mut sys = System::new_all();
    sys.refresh_memory();
    let total_memory = sys.total_memory() as f64;

    let safe_memory = total_memory * safety_margin;
    assert!(text_memory < safe_memory);

    let safe_table_memory = safe_memory - text_memory;
    for batch_size in 1..=32 {
        let table_byte_size = (((corpus_len as f64 / ((rayon_threads * batch_size as f64) as f64)).log2() / 8.0).ceil() as usize).max(4) ;
        let cur_table_memory = corpus_len as f64 * table_byte_size as f64 / batch_size as f64;
        if cur_table_memory < safe_table_memory {
            println!("Okay to use batch size of {:?}", batch_size);
            return batch_size * rayon_threads as usize
        }
    }

    panic!("need a batch size >32! Try a smaller corpus!");
    
}



pub fn sa_safety_check(text_len: usize) -> bool {
	// Checks if we'll run out of RAM trying to make the SA array
    let mut sys = System::new_all();
    sys.refresh_memory();
    let total_memory = sys.total_memory();

	let predicted_ram_usage = text_len * 12;
	println!("Expecting to use at least {:.2}% of the RAM", predicted_ram_usage as f64 / total_memory as f64 * 100.0);


	predicted_ram_usage < total_memory as usize
}


pub fn sa_thread_memory(num_threads: usize, safety_margin: f64) -> usize {
	// Computes maximum allowable memory per thread
	let mut sys = System::new_all();
	sys.refresh_memory();
	let total_memory = sys.total_memory();

	return (total_memory as f64 / num_threads as f64 * safety_margin) as usize;

}


pub fn calculate_bytes_per_chunk(num_threads: usize, safety_margin: f64) -> usize {
	/* Computes number of bytes of text each chunk should handle
	
	Let's set a cap of safety_margin * total_memory 
	and then let M = cap - used_memory // assumes text is already loaded!

	return M / (|num_threads| * 11) // 11 is just a rough "overhead cost per SA tabl"
	*/

	let mut sys = System::new_all();
	sys.refresh_memory();
	let cap: u64 = (sys.total_memory() as f64 * safety_margin) as u64;
	let free_memory = cap - sys.used_memory();

	(free_memory as usize / (num_threads * 11)) as usize
}

/*======================================================================
=                            FILE STREAM STUFF                         =
======================================================================*/
pub trait ByteSize {
    fn byte_size(&self) -> Result<u64, Error>;
}

pub struct MmapRangeReader {
    mmap: Mmap,
    position: usize,
    start: usize, // Offset within the memmap where the range starts
    end: usize, // Offset within the memmap where the range ends (standard slicing format)
}

impl MmapRangeReader {
    pub fn new(file: &File) -> Result<Self, Error> {
        let mmap = unsafe { Mmap::map(file).unwrap() };
        let len = mmap.len();
        Ok(Self {
            mmap,
            position: 0,
            start: 0,
            end: len,
        })
    }

    pub fn from_range(file: &File, offset: u64, length: u64) -> Result<Self, Error> {
        let mmap = unsafe {
            memmap2::MmapOptions::new()
                .offset(offset)
                .len(length as usize)
                .map(file).unwrap()
        };

        let len = mmap.len();
        Ok(Self {
            mmap,
            position: 0,
            start: 0,
            end: len
        })
    }

    pub fn remaining(&self) -> usize {
        self.end.saturating_sub(self.start + self.position)
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.mmap[self.start..self.end]
    }
}


impl Read for MmapRangeReader {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize, std::io::Error> {
        if self.position >= self.end - self.start { 
            return Ok(0); // EOF
        }
        let remaining = &self.mmap[self.start + self.position..self.end];
        let to_read = buf.len().min(remaining.len());
        buf[..to_read].copy_from_slice(&remaining[..to_read]);
        self.position += to_read;
        Ok(to_read)
    }    
}

impl ByteSize for MmapRangeReader {
    fn byte_size(&self) -> Result<u64, Error> {
        Ok((self.end - self.start) as u64)
    }
}



pub enum SAStreamDyn<'a> {
    U32(SAStream<'a, u32>),
    U40(SAStream<'a, U40>),
    U64(SAStream<'a, u64>),
}



pub struct SAStream<'a, T: CompactUint> {
    pub reader: MmapRangeReader,
    pub text: &'a [u8],
    pub offset: &'a [u64], 
    position: usize, // Position in elements (not bytes!)
    pub source: usize,
    pub total_elements: usize,
    next_call_counter: usize,
    pub min_len: usize,
    batch_buffer: Vec<T>,
    batch_pos: usize,

    _phantom: std::marker::PhantomData<T>
}

impl<'a, T: CompactUint> SAStream<'a, T> {

    pub fn new(
        reader: MmapRangeReader,
        text: &'a [u8],
        offset: &'a [u64],
        source: usize,
        min_len: usize,
        ) -> Result<Self, Error> {
        let byte_size = reader.byte_size().unwrap();
        let total_elements = (byte_size as usize) / T::BYTE_SIZE;
        Ok(Self {
            reader,
            text, 
            offset,
            position: 0,
            source,
            total_elements,
            next_call_counter: 0,
            min_len,
            batch_buffer: Vec::new(),
            batch_pos: 0,
            _phantom: std::marker::PhantomData,
        })
    }

    #[inline(always)]
    fn read_next_batch(reader: &MmapRangeReader,
                       position: &mut usize,
                       total_elements: usize,
                       output: &mut [T]) -> usize {
        let available = total_elements - *position;
        let to_read = output.len().min(available);
        if to_read == 0 {
            return 0;            
        }

        let byte_offset = *position * T::BYTE_SIZE;
        let bytes_to_read = to_read * T::BYTE_SIZE;
        let slice = &reader.as_slice()[byte_offset..byte_offset + bytes_to_read];

        for (i, chunk) in slice.chunks_exact(T::BYTE_SIZE).enumerate() {
            output[i] = read_compact_uint_fast::<T>(chunk);
        }

        *position += to_read;
        to_read        
    }

    #[inline(always)]
    fn read_next_el(&mut self) -> Result<Option<T>, Error> {
        if self.batch_pos >= self.batch_buffer.len() {
            self.batch_buffer.clear();
            self.batch_buffer.resize(8192.min(self.total_elements - self.position), T::zero());
            let read = SAStream::<'a, T>::read_next_batch(&self.reader, &mut self.position, self.total_elements, &mut self.batch_buffer);
            if read == 0 {
                return Ok(None)
            }
            self.batch_buffer.truncate(read);
            self.batch_pos = 0;            
        }
        let value = self.batch_buffer[self.batch_pos];
        self.batch_pos += 1;
        Ok(Some(value))
    }



    pub fn next_eos(&self, idx: usize) -> Result<u64, Error> {
        let offset: &[u64] = self.offset;
        // Binary search for the first element >= idx
        let mut left = 0;
        let mut right = offset.len() / 3;

        while left < right {
            let mid = left + (right - left) / 2;
            
            if offset[mid * 3 + 2] <= idx as u64 {
                // Element at mid is too small, search right half
                left = mid + 1;
            } else {
                // Element at mid could be the answer, search left half
                right = mid;
            }
        }
        Ok(*offset.get(left * 3+ 2).unwrap())        
    }

    pub fn prev_eos(&self, idx: usize) -> Result<u64, Error> {
        let offset: &[u64] = self.offset;
        // Binary search for the last element <= idx
        let mut left = 0;
        let mut right = offset.len() / 3;

        while left < right {
            let mid = left + (right - left) / 2;

            if offset[mid * 3 + 2] < idx as u64 {
                // Element at mid could be the answer, search right half
                left = mid + 1;
            } else {
                // Element at mid is too large, search left half
                right = mid;
            }
        }
        // `left` is now the first element > idx, so we want left - 1
        Ok(*offset.get(left.saturating_sub(1) * 3 + 2).unwrap())
    }    

}


impl<'a, T: CompactUint> Iterator for SAStream<'a, T> {    
    type Item = Result<(u64, Option<TreeNode<'a>>), Error>;


    fn next(&mut self) -> Option<Self::Item> {
        let mut loop_counter: u64 = 0;
        loop {
            self.next_call_counter += 1usize;
            loop_counter += 1;            
            let next_idx_val = match self.read_next_el().unwrap() {
                Some(idx) => idx,
                None => return Some(Ok((loop_counter, None)))
            };
            let next_idx = next_idx_val.to_u64() as usize;

            if next_idx + self.min_len > self.text.len() { // overruns endpoint of the stream, skip
                continue
            }

            let next_eos = self.next_eos(next_idx).unwrap() as usize;

            if next_eos < next_idx + self.min_len { // remainder of doc is shorter than min_len, skip
                continue
            }   


            let slice_end = next_idx + self.min_len;
            let slice = &self.text[next_idx..slice_end];
            let rest_of_doc = &self.text[slice_end..next_eos];


            let prev_char: Option<u8> = if next_idx > 0 {
                self.text.get(next_idx - 1).copied()
            } else {
                None
            };
        
            let tree_output = Some(TreeNode {
                sa_idx: (self.next_call_counter - 1usize) as u64,
                sa_value: next_idx_val.to_u64(),
                cmp_bytes: slice,
                source: self.source,
                prev_char, 
                rest_of_doc,
                offset: self.offset,                
            });

            return Some(Ok((loop_counter, tree_output)));    
        }
    }
}





/*====================================================================
=                           MATCH WRITER THING                       =
====================================================================*/

#[derive(Debug)]
pub struct MatchWriterElement {
    pub source: usize,
    pub part_num: u64,
    pub sa_idx: u64,
    pub sa_value: u64, 
    pub lcp: u64, 
    pub first_run_el: bool,
}

impl MatchWriterElement {
    pub const MATCH_EL_SIZE : usize = 33;

    pub fn write_bytes(&self, buffer: &mut [u8]) {
        // Write directly into a pre-allocated buffer
        // No bounds checking needed if buffer is guaranteed to be correct size
        buffer[0..8].copy_from_slice(&self.part_num.to_le_bytes());
        buffer[8..16].copy_from_slice(&self.sa_idx.to_le_bytes());
        buffer[16..24].copy_from_slice(&self.sa_value.to_le_bytes());
        buffer[24..32].copy_from_slice(&self.lcp.to_le_bytes());
        buffer[32] = self.first_run_el as u8;
    }
    
    pub fn into_bytes(self) -> (usize, Vec<u8>) {
        // Returns (source, rest written as bytes)
        let mut packed_bytes = vec![0u8; Self::MATCH_EL_SIZE];
        self.write_bytes(&mut packed_bytes);
        (self.source, packed_bytes)
    }

    pub fn from_bytes(source: usize, bytes: &[u8]) -> Self {
        // Invert the `into_bytes` method
        // Assumes bytes.len() == SERIALIZED_SIZE
        
        // Using unsafe for maximum performance - avoid bounds checks
        unsafe {
            MatchWriterElement {
                source,
                part_num: u64::from_le_bytes(*(bytes.as_ptr() as *const [u8; 8])),
                sa_idx: u64::from_le_bytes(*(bytes.as_ptr().add(8) as *const [u8; 8])),
                sa_value: u64::from_le_bytes(*(bytes.as_ptr().add(16) as *const [u8; 8])),
                lcp: u64::from_le_bytes(*(bytes.as_ptr().add(24) as *const [u8; 8])),
                first_run_el: *bytes.get_unchecked(32) != 0,
            }
        }
    }    
}



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

        Ok(MatchWriter {
            writer,
            storage_loc,
        })
    }

    pub fn make_new_writer(&self, idx: usize) -> Result<Arc<Mutex<BufWriter<File>>>, Error> {
        let filename = &self.get_filename(idx);
        let file = OpenOptions::new()
            .append(true)
            .create(true)
            .mode(0o644)
            .open(filename)
            .unwrap();
        let buf_writer = BufWriter::new(file);
        let output = Arc::new(Mutex::new(buf_writer));
        Ok(output)
    }


    pub fn write_element(&self, element: MatchWriterElement) -> Result<(), Error> {
        let (source, bytes) = element.into_bytes();
        let buf_writer_arc = if let Some(buf_writer_arc) = self.writer.get(&source) {
            buf_writer_arc
        } else {
            let buf_writer_arc = self.make_new_writer(source).unwrap();
            self.writer.insert(source, buf_writer_arc);
            self.writer.get(&source).unwrap()
        };
        buf_writer_arc.lock().unwrap().write(&bytes).unwrap();
        Ok(())
    }


    pub fn finish(&self) -> Result<(), Error> {
        self.writer
            .par_iter()
            .for_each(|e| e.lock().unwrap().flush().unwrap());
        Ok(())
    }

    pub fn get_filename(&self, idx: usize) -> PathBuf {
        self.storage_loc
            .clone()
            .join(format!("match_part_{:04}.bin", idx))
    }
}


pub struct MergedWriterElement {
    pub doc_id: usize,
    pub line_num: usize,
    pub start_idx: usize,
    pub lcp: usize, 
    pub safe: bool,
}


/*========================================================
=                            LOSER TREE STUFF            =
========================================================*/

/// Represents an element in the loser tree along with its source sequence index
#[derive(Debug, Clone)]
pub struct TreeNode<'a> {
    pub sa_idx: u64, // actual index into the suffix array 
    pub sa_value: u64,       // contents of the suffix array at sa_idx
    pub cmp_bytes: &'a [u8], // first min_len bytes of the text pointed to by sa_value
    pub source: usize,       // which input stream this came from
    pub prev_char: Option<u8>,     // the byte that occurs BEFORE this one in the stream (if it exists)
    pub rest_of_doc: &'a [u8], // rest of the document (continuation of cmp_bytes)
    pub offset: &'a [u64],
}
impl TreeNode<'_> {
	pub fn prev_char_same(&self, other: &TreeNode)  -> bool {

		match (&self.prev_char, other.prev_char) {
			(None, None) => false,
			(Some(_), None) => false,
			(None, Some(_)) => false,
			(Some(a), Some(b)) => {
				*a == b
			}
		}		
	}
}

impl Ord for TreeNode<'_> {
    fn cmp(&self, other: &Self) -> StdOrdering {
        let order = self.cmp_bytes.cmp(&other.cmp_bytes);
        if order == StdOrdering::Equal {
            self.source.cmp(&other.source)            
        } else {
            order
        }
    }
}

impl PartialOrd for TreeNode<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<StdOrdering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for TreeNode<'_> {
    fn eq(&self, other: &Self) -> bool {
        (self.cmp_bytes == other.cmp_bytes) && (self.source == other.source)
    }
}

impl Eq for TreeNode<'_> {}


impl <'a> TreeNode<'a> {
    pub fn cmp_eq(&self, other: &TreeNode) -> bool {
        self.cmp_bytes == other.cmp_bytes
    }

    pub fn lcp(&self, other: &TreeNode, min_len: usize) -> u64 {
        (min_len + self.rest_of_doc.iter()
            .zip(other.rest_of_doc.iter())
            .take_while(|(x, y)| x == y)
            .count()) as u64
    }

    pub fn prev_bos(&self) -> Result<u64, Error> {
        let offset: &[u64] = self.offset;
        // Binary search for the last element <= idx
        let mut left = 0;
        let mut right = offset.len() / 3;
        let search_idx = (self.sa_value + 1) as u64;
        while left < right {
            let mid = left + (right - left) / 2;

            if offset[mid * 3 + 2] < search_idx {
                // Element at mid could be the answer, search right half
                left = mid + 1;
            } else {
                // Element at mid is too large, search left half
                right = mid;
            }
        }
        // `left` is now the first element > idx, so we want left - 1
        Ok(*offset.get(left.saturating_sub(1) * 3 + 2).unwrap())
    }

}




/// Loser Tree for efficient k-way merging
pub struct LoserTree<'a, T: CompactUint> {
    tree: Vec<Option<TreeNode<'a>>>, // Internal nodes store losers
    loser: Option<TreeNode<'a>>,     // Current minimum element
    pub shortcut_count: usize,
    iterators: &'a mut HashMap<usize, SAStream<'a, T>>, // Input sequences
    path_idxs: Vec<Vec<usize>>,
    pbar_opt: Option<ProgressBar>,
}

impl<'a, T: CompactUint> LoserTree<'a, T> {
    /// Create a new loser tree from k iterators
    pub fn new(iterators: &'a mut HashMap<usize, SAStream<'a, T>>, pbar_opt: Option<ProgressBar>) -> Self {
        let k = iterators.len();

        // Tree needs k-1 internal nodes for k leaves
        let tree_size = LoserTree::<T>::calculate_loser_tree_size(k);
        let mut tree = vec![None; tree_size];

        // Initialize leaves with first element from each iterator
        let mut leaves: Vec<Option<TreeNode<'a>>> = (0..k)
            .map(|source| {
                if let Some(iter) = iterators.get_mut(&source) {        
                    let (counter, next_el) = iter.next().unwrap().unwrap();
                    if let Some(ref pbar) = pbar_opt {
                        pbar.inc(counter);
                    }
                    if let Some(node) = next_el {
                        Some(node)
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect();

        let path_idxs: Vec<Vec<usize>> = (0..k)
            .map(|i| LoserTree::<T>::get_path_indices(i, k))
            .collect();

        // Build the initial tree
        let loser = Self::build_tree(&mut tree, &mut leaves);

        let shortcut_count = 0;

        LoserTree {
            tree,
            loser,
            iterators,
            path_idxs,
            shortcut_count,
            pbar_opt
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
                                // A loses to B, so B stops here and A advances
                                tree[level_start + pair_idx] = Some(b.clone());
                                next_level.push(Some(a.clone()));
                            } else {
                                // Other way around
                                tree[level_start + pair_idx] = Some(a.clone());
                                next_level.push(Some(b.clone()));
                            }
                        }
                        (Some(a), None) => {
                            // B is None (ultimate winner), so A advances
                            next_level.push(Some(a.clone()));
                        }
                        (None, Some(b)) => {
                            // A is None (ultimate winner), so B advances
                            next_level.push(Some(b.clone()));
                        }
                        (None, None) => {
                            // None must advance
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
        self.loser.as_ref()
    }



    /// Extract the minimum element and advance the tree
    pub fn pop(&mut self) -> Result<Option<TreeNode<'a>>, Error> {
        let loser = self.loser.take().unwrap();
        let source = loser.source;
        let (counter, next_el) = self.iterators.get_mut(&source).unwrap().next().unwrap().unwrap();
        self.inc_pbar_opt(counter);
        let new_tree_entry = if let Some(node) = next_el {
            Some(node)
        } else {
            None
        };

        self.loser = self.replay(new_tree_entry, source);
        Ok(Some(loser))
    }

    pub fn pop_check(&mut self) -> Result<Option<TreeNode<'a>>, Error> {
        let loser = self.loser.take().unwrap();
        let loser_source = loser.source;

        let (counter, next_el) = self.iterators.get_mut(&loser_source).unwrap().next().unwrap().unwrap();
        self.inc_pbar_opt(counter);

        let new_tree_entry =
            if let Some(node) = next_el {
                let new_node = node;
                match self.tree.last() {
                    Some(None) => {
                        self.shortcut_count += 1;
                        self.loser = Some(new_node);
                        return Ok(Some(loser));
                    }
                    Some(Some(second_min)) => {
                        if new_node < *second_min {
                            self.shortcut_count += 1;
                            self.loser = Some(new_node);
                            return Ok(Some(loser));
                        } else {
                            Some(new_node)
                        }
                    }
                    _ => Some(new_node),
                }
            } else {
                None
            };

        self.loser = self.replay(new_tree_entry, loser_source);
        Ok(Some(loser))
    }

    pub fn pop_verbose(&mut self) -> Result<Option<TreeNode<'a>>, Error> {
        let loser = self.loser.take().unwrap();
        let source = loser.source;
        let (counter, next_el) = self.iterators.get_mut(&source).unwrap().next().unwrap().unwrap();
        self.inc_pbar_opt(counter);

        let new_tree_entry = if let Some(node) = next_el {
            Some(node)
        } else {
            None
        };

        println!(
            "Replaying: source={}, new_entry={:?}",
            source,
            new_tree_entry.as_ref().map(|n| n.source)
        );
        println!(
            "Tree before replay: {:?}",
            self.tree
                .iter()
                .map(|n| n.as_ref().map(|x| x.source))
                .collect::<Vec<_>>()
        );

        self.loser = self.replay(new_tree_entry, source);

        println!(
            "Winner after replay: {:?}",
            self.loser.as_ref().map(|n| n.source)
        );
        println!(
            "Tree after replay: {:?}",
            self.tree
                .iter()
                .map(|n| n.as_ref().map(|x| x.source))
                .collect::<Vec<_>>()
        );
        println!("---");

        Ok(Some(loser))
    }

    pub fn is_empty(&self) -> bool {
        // Winner must be None AND all tree nodes must be None
        self.loser.is_none() && self.tree.iter().all(|node| node.is_none())
    }

    fn replay(&mut self, mut current: Option<TreeNode<'a>>, source: usize) -> Option<TreeNode<'a>> {
        let path = &self.path_idxs[source];
        for &idx in path {
            match (&current, &self.tree[idx]) {
                // Both exist, need comparison
                (Some(curr), Some(loser)) => {
                    if curr > loser {
                        // Current WINS (is bigger) against stored loser, so we need to swap and advance stored loser
                        std::mem::swap(&mut self.tree[idx], &mut current);
                    } else {
                        // Current LOSES (is smaller), so current loser stays where it is and current continues up
                    }
                }

                // Current is None (stream is exhausted), but tree has a value
                // None can be though of as MAX, so needs swap
                (None, Some(_)) => {
                    std::mem::swap(&mut self.tree[idx], &mut current);
                }
                // Other two cases are where None is in the tree, so no swap ever needed
                (_, _) => {}
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
    fn calculate_loser_tree_size(k: usize) -> usize {
        let mut total = 0;
        let mut level_size = k;

        while level_size > 1 {
            total += (level_size + 1) / 2;
            level_size = (level_size + 1) / 2;
        }

        total
    }

    fn inc_pbar_opt(&self, inc: u64) -> () {
        if let Some(ref pbar) = self.pbar_opt {
            pbar.inc(inc);
        }
    }
}
