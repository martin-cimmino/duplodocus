use ahash::HashMap;
use dashmap::DashMap;
use std::fs::create_dir_all;
use std::fs::File;
use std::fs::OpenOptions;
use std::io::BufReader;
use std::io::BufWriter;
use std::io::Read;
use std::io::Seek;
use std::io::SeekFrom;
use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::sync::Arc;
use std::sync::Mutex;

use anyhow::{Context, Error, Result};
use rayon;
use rayon::prelude::*;
use std::path::PathBuf;

use std::cmp::Ordering as StdOrdering;
use sysinfo::System;


/*======================================================
=                           UTILS                      =
======================================================*/

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


pub fn sa_safety_check(text_len: usize) -> bool {
	// Checks if we'll run out of RAM trying to make the SA array
	let mut sys = System::new_all();
	sys.refresh_memory();
	let total_memory = sys.total_memory();

	let predicted_ram_usage = text_len * 9;
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

/*======================================================================
=                            FILE STREAM STUFF                         =
======================================================================*/

pub trait ByteSize {
	fn byte_size(&self) -> Result<u64, Error>;
}

impl ByteSize for File {
	fn byte_size(&self) -> Result<u64, Error> {
		Ok(self.metadata().unwrap().len())
	}
}

#[allow(dead_code)]
pub struct FileRange {
    file: File,
    start: u64,
    end: u64,
    current: u64,
}
impl FileRange {
    pub fn new(mut file: File, start: u64, end: u64) -> Result<Self, Error> {
        // Seek to the start position
        file.seek(SeekFrom::Start(start)).unwrap();

        Ok(FileRange {
            file,
            start,
            end,
            current: start,
        })
    }
}

impl Read for FileRange {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize, std::io::Error> {
        // Calculate how many bytes we can still read
        let remaining = self.end.saturating_sub(self.current);

        if remaining == 0 {
            return Ok(0); // EOF
        }

        // Limit the read to not exceed our range
        let to_read = std::cmp::min(buf.len() as u64, remaining) as usize;
        let n = self.file.read(&mut buf[..to_read]).unwrap();

        self.current += n as u64;
        Ok(n)
    }
}

impl ByteSize for FileRange {
	fn byte_size(&self) -> Result<u64, Error> {
		Ok(self.end - self.start)
	}
}


pub struct SAStream<'a, R: Read + ByteSize> {
    pub reader: BufReader<R>,
    pub text: &'a [u8],
    pub buffer: Vec<u64>,
    pub position: usize,
    pub chunk_size: usize,
    pub source: usize,
    pub byte_size: u64,
}

impl<'a, R: Read + ByteSize> SAStream<'a, R> {
    pub fn new(sa_file: R, text: &'a [u8], source: usize, chunk_size: usize) -> Result<Self, Error> {
    	let byte_size = sa_file.byte_size().unwrap();

        Ok(Self {
            reader: BufReader::new(sa_file),
            text: text,
            buffer: Vec::with_capacity(chunk_size),
            position: 0,
            chunk_size,
            source,
            byte_size
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

    pub fn text_iter<'stream>(&'stream mut self, min_len: usize) -> TextIterator<'stream, 'a, R> {
        TextIterator {
            stream: self,
            min_len,
        }
    }
}

impl<'a, R: Read + ByteSize> Iterator for SAStream<'a, R> {
    type Item = Result<u64, Error>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.position >= self.buffer.len() {
            match self.refill_buffer() {
                Ok(true) => {}
                Ok(false) => return None, // EOF
                Err(e) => return Some(Err(e)),
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

pub struct TextIterator<'stream, 'a, R: Read> where R: ByteSize, R: ByteSize {
    stream: &'stream mut SAStream<'a, R>,
    min_len: usize,
}


#[allow(unreachable_code)]
impl<'stream, 'a, R: Read + ByteSize> Iterator for TextIterator<'stream, 'a, R> {
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
                let prev_char: Option<u8> = if next_idx > 0 {
                	let prev_char = (&self.stream.text.get(next_idx as usize - 1)).clone().unwrap();
                	Some(*prev_char)
                } else {
                	None
                };               

                return Some(Ok(TreeNode {
                    sa_value: next_idx,
                    cmp_bytes: slice,
                    source: self.stream.source,
                    prev_char: prev_char
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



/*========================================================
=                            LOSER TREE STUFF            =
========================================================*/

/// Represents an element in the loser tree along with its source sequence index
#[derive(Debug, Clone)]
pub struct TreeNode<'a> {
    pub sa_value: u64,       // value 1 that the object holds
    pub cmp_bytes: &'a [u8], // value 2 that the object holds
    pub source: usize,       // which input stream this came from
    pub prev_char: Option<u8>     // the byte that occurs BEFORE this one in the stream (if it exists)
}
impl TreeNode<'_> {
	pub fn prev_char_same(&self, other: &TreeNode)  -> bool {
		match (&self.prev_char, other.prev_char) {
			(None, None) => false,
			(a, b) => *a == b
		}		
	}
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
pub struct LoserTree<'a, 'stream: 'a, R: Read + ByteSize> {
    tree: Vec<Option<TreeNode<'a>>>, // Internal nodes store losers
    loser: Option<TreeNode<'a>>,     // Current minimum element
    pub shortcut_count: usize,
    iterators: HashMap<usize, TextIterator<'stream, 'a, R>>, // Input sequences
    path_idxs: Vec<Vec<usize>>,
}

impl<'a, 'stream: 'a, R: Read + ByteSize> LoserTree<'a, 'stream, R> {
    /// Create a new loser tree from k iterators
    pub fn new(mut iterators: HashMap<usize, TextIterator<'stream, 'a, R>>) -> Self {
        let k = iterators.len();

        // Tree needs k-1 internal nodes for k leaves
        let tree_size = LoserTree::<R>::calculate_loser_tree_size(k);
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

        let path_idxs: Vec<Vec<usize>> = (0..k)
            .map(|i| LoserTree::<R>::get_path_indices(i, k))
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

        let new_tree_entry = if let Some(node) = self.iterators.get_mut(&source).unwrap().next() {
            Some(node.unwrap())
        } else {
            None
        };

        self.loser = self.replay(new_tree_entry, source);
        Ok(Some(loser))
    }

    pub fn pop_check(&mut self) -> Result<Option<TreeNode<'a>>, Error> {
        let loser = self.loser.take().unwrap();
        let loser_source = loser.source;

        let new_tree_entry =
            if let Some(node) = self.iterators.get_mut(&loser_source).unwrap().next() {
                let new_node = node.unwrap();
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

        let new_tree_entry = if let Some(node) = self.iterators.get_mut(&source).unwrap().next() {
            Some(node.unwrap())
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
}
