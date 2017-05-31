use std::ops::{Index, IndexMut};

pub trait IndexLen: Index<usize> {
    fn len(&self) -> usize;
}

impl<'a, T: 'a> IndexLen for [T] {
    fn len(&self) -> usize {
        self.len()
    }
}

pub trait IndexMutLen: IndexMut<usize> + IndexLen {}

impl<'a, T: 'a> IndexMutLen for [T] {}

/// Implementation detail
pub struct Columns<'a, T: 'a> {
    data: &'a [T],
    rowstride: usize,
    curr: usize,
}

impl<'a, T: 'a> Columns<'a, T> {
    /// Implementation detail
    #[inline(always)]
    pub fn from_slice(data: &'a [T], rowstride: usize) -> Self {
        assert_eq!(data.len() % rowstride, 0);

        Columns {
            data: data,
            rowstride: rowstride,
            curr: 0,
        }
    }
}

impl<'a, T: 'a> Iterator for Columns<'a, T> {
    type Item = Column<'a, T>;
    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.curr < self.rowstride {
            let data = &self.data[self.curr..];
            self.curr += 1;
            Some(Column {
                data: data,
                rowstride: self.rowstride,
            })
        } else {
            None
        }
    }
}

/// Implementation detail
pub struct Column<'a, T: 'a> {
    data: &'a [T],
    rowstride: usize,
}

impl<'a, T: 'a> Index<usize> for Column<'a, T> {
    type Output = T;
    #[inline(always)]
    fn index(&self, idx: usize) -> &T {
        &self.data[idx * self.rowstride]
    }
}

impl<'a, T: 'a> IndexLen for Column<'a, T> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.data.len() / self.rowstride
    }
}

/// Implementation detail
pub struct ColumnsMut<'a, T: 'a> {
    data: &'a mut [T],
    rowstride: usize,
    curr: usize,
}

impl<'a, T: 'a> ColumnsMut<'a, T> {
    #[inline(always)]
    pub fn from_slice(data: &'a mut [T], rowstride: usize) -> Self {
        assert_eq!(data.len() % rowstride, 0);

        ColumnsMut {
            data: data,
            rowstride: rowstride,
            curr: 0,
        }
    }
}

impl<'a, T: 'a> Iterator for ColumnsMut<'a, T> {
    type Item = ColumnMut<'a, T>;
    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.curr < self.rowstride {
            let data = unsafe { &mut *(&mut self.data[self.curr..] as *mut [T]) };
            self.curr += 1;
            Some(ColumnMut {
                data: data,
                rowstride: self.rowstride,
            })
        } else {
            None
        }
    }
}

/// Implementation detail
pub struct ColumnMut<'a, T: 'a> {
    data: &'a mut [T],
    rowstride: usize,
}

impl<'a, T: 'a> Index<usize> for ColumnMut<'a, T> {
    type Output = T;
    #[inline(always)]
    fn index(&self, idx: usize) -> &T {
        &self.data[idx * self.rowstride]
    }
}

impl<'a, T: 'a> IndexMut<usize> for ColumnMut<'a, T> {
    #[inline(always)]
    fn index_mut(&mut self, idx: usize) -> &mut T {
        &mut self.data[idx * self.rowstride]
    }
}

impl<'a, T: 'a> IndexLen for ColumnMut<'a, T> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.data.len() / self.rowstride
    }
}

impl<'a, T: 'a> IndexMutLen for ColumnMut<'a, T> {}
