use std::marker::PhantomData;

/// Bit-packed suffix array with minimal memory footprint
pub struct SuffixTable<T: TokenType> {
    text: Vec<T::Storage>,      // Packed token storage
    table: BitPackedVec,         // Bit-packed suffix array indices
    _phantom: PhantomData<T>,
}

/// Token type trait
pub trait TokenType: Copy {
    type Storage: Copy;
    const BYTES: usize;
    
    fn pack(value: usize) -> Self::Storage;
    fn unpack(storage: Self::Storage) -> usize;
}

// U8 token (raw bytes)
#[derive(Copy, Clone)]
pub struct U8Token;

impl TokenType for U8Token {
    type Storage = u8;
    const BYTES: usize = 1;
    
    fn pack(value: usize) -> u8 { value as u8 }
    fn unpack(storage: u8) -> usize { storage as usize }
}

// U16 token (standard tokenizer)
#[derive(Copy, Clone)]
pub struct U16Token;

impl TokenType for U16Token {
    type Storage = u16;
    const BYTES: usize = 2;
    
    fn pack(value: usize) -> u16 { value as u16 }
    fn unpack(storage: u16) -> usize { storage as usize }
}

// U24 token (packed 3-byte integer)
#[derive(Copy, Clone)]
pub struct U24Token;

impl TokenType for U24Token {
    type Storage = [u8; 3];
    const BYTES: usize = 3;
    
    fn pack(value: usize) -> [u8; 3] {
        [
            (value >> 16) as u8,
            (value >> 8) as u8,
            value as u8,
        ]
    }
    
    fn unpack(storage: [u8; 3]) -> usize {
        ((storage[0] as usize) << 16) |
        ((storage[1] as usize) << 8) |
        (storage[2] as usize)
    }
}


/// Highly optimized bit-packed vector using u64 backing store
#[derive(Clone, Debug)]
pub struct BitPackedVec {
    data: Vec<u64>,
    len: usize,
    bits_per_entry: u8,
    pub max_val: usize,
}

impl PartialEq for BitPackedVec {
    fn eq(&self, other: &Self) -> bool {
        // Must have same length and bits per entry
        if self.len != other.len || self.bits_per_entry != other.bits_per_entry {
            return false;
        }
        
        // Compare all entries (not raw data, since padding bits may differ)
        for i in 0..self.len {
            if self.get(i) != other.get(i) {
                return false;
            }
        }
        
        true
    }
}

impl Eq for BitPackedVec {}

impl BitPackedVec {
    pub fn new(capacity: usize, max_value: usize) -> Self {
        let bits_per_entry = if max_value == 0 {
            1
        } else {
            (usize::BITS - max_value.leading_zeros()) as u8
        };
        
        let total_bits = capacity * bits_per_entry as usize;
        let num_u64s = (total_bits + 63) / 64;
        
        Self {
            data: vec![0u64; num_u64s],
            len: 0,
            bits_per_entry,
            max_val: max_value
        }
    }
    
    pub fn with_exact_capacity(capacity: usize, bits_per_entry: u8) -> Self {
        let total_bits = capacity * bits_per_entry as usize;
        let num_u64s = (total_bits + 63) / 64;
        
        Self {
            data: vec![0u64; num_u64s],
            len: 0,
            bits_per_entry,
            max_val: (2 << bits_per_entry) - 1
        }
    }
    
    #[inline(always)]
    pub fn push(&mut self, value: usize) {
        debug_assert!(value < (1usize << self.bits_per_entry), "Value too large");
        
        let bit_offset = self.len * self.bits_per_entry as usize;
        let word_index = bit_offset / 64;
        let bit_in_word = bit_offset % 64;
        let bits_per_entry = self.bits_per_entry as usize;
        
        let value_u64 = value as u64;
        
        // Write to first word
        self.data[word_index] |= value_u64 << bit_in_word;
        
        // Handle overflow to next word (only if needed)
        let bits_remaining = 64 - bit_in_word;
        if bits_per_entry > bits_remaining {
            self.data[word_index + 1] |= value_u64 >> bits_remaining;
        }
        
        self.len += 1;
    }
    
    #[inline(always)]
    pub fn get(&self, index: usize) -> usize {
        debug_assert!(index < self.len, "Index out of bounds");
        
        let bit_offset = index * self.bits_per_entry as usize;
        let word_index = bit_offset / 64;
        let bit_in_word = bit_offset % 64;
        let bits_per_entry = self.bits_per_entry as usize;
        
        let mask = if bits_per_entry >= 64 {
            u64::MAX
        } else {
            (1u64 << bits_per_entry) - 1
        };
        
        // Fast path: value entirely within one word
        let bits_remaining = 64 - bit_in_word;
        if bits_per_entry <= bits_remaining {
            ((self.data[word_index] >> bit_in_word) & mask) as usize
        } else {
            // Value spans two words
            let low_bits = self.data[word_index] >> bit_in_word;
            let high_bits = self.data[word_index + 1] << bits_remaining;
            ((low_bits | high_bits) & mask) as usize
        }
    }
    
    #[inline(always)]
    pub fn set(&mut self, index: usize, value: usize) {
        debug_assert!(index < self.len, "Index out of bounds");
        debug_assert!(value < (1usize << self.bits_per_entry), "Value too large");
        
        let bit_offset = index * self.bits_per_entry as usize;
        let word_index = bit_offset / 64;
        let bit_in_word = bit_offset % 64;
        let bits_per_entry = self.bits_per_entry as usize;
        
        let value_u64 = value as u64;
        let mask = if bits_per_entry >= 64 {
            u64::MAX
        } else {
            (1u64 << bits_per_entry) - 1
        };
        
        let bits_remaining = 64 - bit_in_word;
        
        if bits_per_entry <= bits_remaining {
            // Value entirely within one word
            let clear_mask = !(mask << bit_in_word);
            self.data[word_index] = (self.data[word_index] & clear_mask) | ((value_u64 & mask) << bit_in_word);
        } else {
            // Value spans two words
            let overflow_bits = bits_per_entry - bits_remaining;
            
            // Clear and set low bits in first word
            let low_mask = (1u64 << bits_remaining) - 1;
            let clear_mask_low = !(low_mask << bit_in_word);
            self.data[word_index] = (self.data[word_index] & clear_mask_low) | ((value_u64 & low_mask) << bit_in_word);
            
            // Clear and set high bits in second word
            let high_mask = (1u64 << overflow_bits) - 1;
            self.data[word_index + 1] = (self.data[word_index + 1] & !high_mask) | (value_u64 >> bits_remaining);
        }
    }
    
    #[inline(always)]
    pub fn swap(&mut self, a: usize, b: usize) {
        debug_assert!(a < self.len && b < self.len, "Index out of bounds");
        
        if a == b {
            return;
        }
        
        let val_a = self.get(a);
        let val_b = self.get(b);
        self.set(a, val_b);
        self.set(b, val_a);
    }
    
    #[inline(always)]
    pub unsafe fn swap_unchecked(&mut self, a: usize, b: usize) {
        if a == b {
            return;
        }
        
        let val_a = self.get_unchecked(a);
        let val_b = self.get_unchecked(b);
        self.set_unchecked(a, val_b);
        self.set_unchecked(b, val_a);
    }
    
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, index: usize) -> usize {
        let bit_offset = index * self.bits_per_entry as usize;
        let word_index = bit_offset / 64;
        let bit_in_word = bit_offset % 64;
        let bits_per_entry = self.bits_per_entry as usize;
        
        let mask = if bits_per_entry >= 64 {
            u64::MAX
        } else {
            (1u64 << bits_per_entry) - 1
        };
        
        let bits_remaining = 64 - bit_in_word;
        
        if bits_per_entry <= bits_remaining {
            ((*self.data.get_unchecked(word_index) >> bit_in_word) & mask) as usize
        } else {
            let low_bits = *self.data.get_unchecked(word_index) >> bit_in_word;
            let high_bits = *self.data.get_unchecked(word_index + 1) << bits_remaining;
            ((low_bits | high_bits) & mask) as usize
        }
    }
    
    #[inline(always)]
    pub unsafe fn set_unchecked(&mut self, index: usize, value: usize) {
        let bit_offset = index * self.bits_per_entry as usize;
        let word_index = bit_offset / 64;
        let bit_in_word = bit_offset % 64;
        let bits_per_entry = self.bits_per_entry as usize;
        
        let value_u64 = value as u64;
        let mask = if bits_per_entry >= 64 {
            u64::MAX
        } else {
            (1u64 << bits_per_entry) - 1
        };
        
        let bits_remaining = 64 - bit_in_word;
        
        if bits_per_entry <= bits_remaining {
            let clear_mask = !(mask << bit_in_word);
            let word = self.data.get_unchecked_mut(word_index);
            *word = (*word & clear_mask) | ((value_u64 & mask) << bit_in_word);
        } else {
            let overflow_bits = bits_per_entry - bits_remaining;
            let low_mask = (1u64 << bits_remaining) - 1;
            
            let word1 = self.data.get_unchecked_mut(word_index);
            let clear_mask_low = !(low_mask << bit_in_word);
            *word1 = (*word1 & clear_mask_low) | ((value_u64 & low_mask) << bit_in_word);
            
            let word2 = self.data.get_unchecked_mut(word_index + 1);
            let high_mask = (1u64 << overflow_bits) - 1;
            *word2 = (*word2 & !high_mask) | (value_u64 >> bits_remaining);
        }
    }
    
    pub fn len(&self) -> usize {
        self.len
    }
    
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    
    pub fn bits_per_entry(&self) -> u8 {
        self.bits_per_entry
    }
    
    pub fn bytes_used(&self) -> usize {
        self.data.len() * 8
    }

    #[inline]
    pub fn iter(&self) -> BitPackedVecIter {
        BitPackedVecIter {
            vec: self,
            start: 0,
            end: self.len
        }
    }    
}

impl<'a> IntoIterator for &'a BitPackedVec {
    type Item = usize;
    type IntoIter = BitPackedVecIter<'a>;
    
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}


pub struct BitPackedVecIter<'a> {
    vec: &'a BitPackedVec,
    start: usize,
    end: usize,
}

impl<'a> Iterator for BitPackedVecIter<'a> {
    type Item = usize;
    
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.start < self.end {
            let value = self.vec.get(self.start);
            self.start += 1;
            Some(value)
        } else {
            None
        }
    }
    
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.end - self.start;
        (remaining, Some(remaining))
    }
    
    #[inline]
    fn count(self) -> usize {
        self.end - self.start
    }
    
    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.start = self.start.saturating_add(n);
        self.next()
    }

}


