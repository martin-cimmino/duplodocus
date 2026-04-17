
#[derive(Clone, Eq, PartialEq)]
pub struct BitPackedVec {
    data: Vec<u64>,
    len: usize,
    bits_per_entry: u8,
    pub max_val: usize,
}

/// Immutable view into a bit-packed vector
pub struct BitPackedSlice<'a> {
    data: &'a [u64],
    offset: usize,        // offset in entries (not bits)
    len: usize,           // length in entries
    bits_per_entry: u8,
    pub max_val: usize
}

/// Mutable view into a bit-packed vector
pub struct BitPackedSliceMut<'a> {
    data: &'a mut [u64],
    offset: usize,
    len: usize,
    bits_per_entry: u8,
    pub max_val: usize
}

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
        self.as_slice().get(index)
    }
    
    #[inline(always)]
    pub fn set(&mut self, index: usize, value: usize) {
        self.as_mut_slice().set(index, value);
    }
    
    /// Get an immutable slice view of the entire vector
    pub fn as_slice(&self) -> BitPackedSlice {
        BitPackedSlice {
            data: &self.data,
            offset: 0,
            len: self.len,
            bits_per_entry: self.bits_per_entry,
            max_val: self.max_val,
        }
    }
    
    /// Get a mutable slice view of the entire vector
    pub fn as_mut_slice(&mut self) -> BitPackedSliceMut {
        BitPackedSliceMut {
            data: &mut self.data,
            offset: 0,
            len: self.len,
            bits_per_entry: self.bits_per_entry,
            max_val: self.max_val
        }
    }
    
    pub fn len(&self) -> usize {
        self.len
    }
    
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<'a> BitPackedSlice<'a> {
    #[inline(always)]
    pub fn get(&self, index: usize) -> usize {
        assert!(index < self.len, "Index out of bounds");
        
        let actual_index = self.offset + index;
        let bit_offset = actual_index * self.bits_per_entry as usize;
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
            ((self.data[word_index] >> bit_in_word) & mask) as usize
        } else {
            let low_bits = self.data[word_index] >> bit_in_word;
            let high_bits = self.data[word_index + 1] << bits_remaining;
            ((low_bits | high_bits) & mask) as usize
        }
    }
    
    pub fn len(&self) -> usize {
        self.len
    }
    
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    
    /// Get a sub-slice from start..end
    pub fn slice(&self, start: usize, end: usize) -> BitPackedSlice<'a> {
        assert!(start <= end && end <= self.len, "Invalid slice range");
        BitPackedSlice {
            data: self.data,
            offset: self.offset + start,
            len: end - start,
            bits_per_entry: self.bits_per_entry,
            max_val: self.max_val,
        }
    }
    
    /// Get a sub-slice from start..
    pub fn slice_from(&self, start: usize) -> BitPackedSlice<'a> {
        self.slice(start, self.len)
    }
    
    /// Get a sub-slice from ..end
    pub fn slice_to(&self, end: usize) -> BitPackedSlice<'a> {
        self.slice(0, end)
    }
    
    /// Split at index, returning (left, right)
    pub fn split_at(&self, mid: usize) -> (BitPackedSlice<'a>, BitPackedSlice<'a>) {
        assert!(mid <= self.len, "Split index out of bounds");
        (self.slice(0, mid), self.slice(mid, self.len))
    }
    
    /// Iterator over elements
    pub fn iter(&self) -> BitPackedSliceIter<'a> {
        BitPackedSliceIter {
            slice: *self,
            index: 0,
        }
    }
}

impl<'a> BitPackedSliceMut<'a> {
    #[inline(always)]
    pub fn get(&self, index: usize) -> usize {
        assert!(index < self.len, "Index out of bounds");
        
        let actual_index = self.offset + index;
        let bit_offset = actual_index * self.bits_per_entry as usize;
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
            ((self.data[word_index] >> bit_in_word) & mask) as usize
        } else {
            let low_bits = self.data[word_index] >> bit_in_word;
            let high_bits = self.data[word_index + 1] << bits_remaining;
            ((low_bits | high_bits) & mask) as usize
        }
    }
    
    #[inline(always)]
    pub fn set(&mut self, index: usize, value: usize) {
        assert!(index < self.len, "Index out of bounds");
        assert!(value < (1usize << self.bits_per_entry), "Value too large");
        
        let actual_index = self.offset + index;
        let bit_offset = actual_index * self.bits_per_entry as usize;
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
            self.data[word_index] = (self.data[word_index] & clear_mask) | ((value_u64 & mask) << bit_in_word);
        } else {
            let overflow_bits = bits_per_entry - bits_remaining;
            
            let low_mask = (1u64 << bits_remaining) - 1;
            let clear_mask_low = !(low_mask << bit_in_word);
            self.data[word_index] = (self.data[word_index] & clear_mask_low) | ((value_u64 & low_mask) << bit_in_word);
            
            let high_mask = (1u64 << overflow_bits) - 1;
            self.data[word_index + 1] = (self.data[word_index + 1] & !high_mask) | (value_u64 >> bits_remaining);
        }
    }
    
    pub fn len(&self) -> usize {
        self.len
    }
    
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    
    /// Get a sub-slice from start..end
    pub fn slice(&mut self, start: usize, end: usize) -> BitPackedSliceMut {
        assert!(start <= end && end <= self.len, "Invalid slice range");
        BitPackedSliceMut {
            data: self.data,
            offset: self.offset + start,
            len: end - start,
            bits_per_entry: self.bits_per_entry,
            max_val: self.max_val
        }
    }
    
    /// Get a sub-slice from start..
    pub fn slice_from(&mut self, start: usize) -> BitPackedSliceMut {
        let len = self.len;
        self.slice(start, len)
    }
    
    /// Get a sub-slice from ..end
    pub fn slice_to(&mut self, end: usize) -> BitPackedSliceMut {
        self.slice(0, end)
    }
    
    /// Split at index, returning (left, right)
    pub fn split_at_mut(&mut self, mid: usize) -> (BitPackedSliceMut, BitPackedSliceMut) {
        assert!(mid <= self.len, "Split index out of bounds");
        
        // We need to split the mutable reference
        // This is safe because the two slices don't overlap
        let left = BitPackedSliceMut {
            data: unsafe { &mut *(self.data as *mut [u64]) },
            offset: self.offset,
            len: mid,
            bits_per_entry: self.bits_per_entry,
            max_val: self.max_val,
        };
        
        let right = BitPackedSliceMut {
            data: self.data,
            offset: self.offset + mid,
            len: self.len - mid,
            bits_per_entry: self.bits_per_entry,
            max_val: self.max_val
        };
        
        (left, right)
    }

    /// Iterator over elements
    pub fn iter_mut(&'a mut self) -> BitPackedSliceMutIter<'a> {
        BitPackedSliceMutIter {
            slice: self,
            index: 0,
        }
    }    
    
    /// Get an immutable view of this mutable slice
    pub fn as_slice(&self) -> BitPackedSlice {
        BitPackedSlice {
            data: self.data,
            offset: self.offset,
            len: self.len,
            bits_per_entry: self.bits_per_entry,
            max_val: self.max_val
        }
    }
}

// Copy trait for immutable slice since it's just a view
impl<'a> Copy for BitPackedSlice<'a> {}
impl<'a> Clone for BitPackedSlice<'a> {
    fn clone(&self) -> Self {
        *self
    }
}

/// Iterator over a bit-packed slice
pub struct BitPackedSliceIter<'a> {
    slice: BitPackedSlice<'a>,
    index: usize,
}

impl<'a> Iterator for BitPackedSliceIter<'a> {
    type Item = usize;
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.slice.len() {
            let value = self.slice.get(self.index);
            self.index += 1;
            Some(value)
        } else {
            None
        }
    }
    
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.slice.len() - self.index;
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for BitPackedSliceIter<'a> {}


/// Iterator over a MUTABLE bit-packed slice
pub struct BitPackedSliceMutIter<'a> {
    slice: &'a mut BitPackedSliceMut<'a>,
    index: usize,
}

impl<'a> Iterator for BitPackedSliceMutIter<'a> {
    type Item = usize;
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.slice.len() {
            let value = self.slice.get(self.index);
            self.index += 1;
            Some(value)
        } else {
            None
        }
    }
    
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.slice.len() - self.index;
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for BitPackedSliceMutIter<'a> {}
