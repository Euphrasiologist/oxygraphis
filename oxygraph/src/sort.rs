//! Not written by myself, but by the great bluss.
//!
//! See https://github.com/rust-ndarray/ndarray/issues/195
//! for discussion and details.

use ndarray::prelude::*;
use ndarray::{Data, RemoveAxis, Zip};

use std::cmp::Ordering;
use std::ptr::copy_nonoverlapping;

/// See https://stackoverflow.com/questions/69764803/how-to-sort-a-vector-by-indices-in-rust
///
/// Sort a vector by another array of indices **in place**.
pub fn sort_by_indices<T>(data: &mut [T], mut indices: Vec<usize>) {
    for idx in 0..data.len() {
        if indices[idx] != idx {
            let mut current_idx = idx;
            loop {
                let target_idx = indices[current_idx];
                indices[current_idx] = current_idx;
                if indices[target_idx] == target_idx {
                    break;
                }
                data.swap(current_idx, target_idx);
                current_idx = target_idx;
            }
        }
    }
}

/// Hold the indices of a permutation.
///
/// Type invariant: Each index appears exactly once.
#[derive(Clone, Debug)]
pub struct Permutation {
    pub indices: Vec<usize>,
}

impl Permutation {
    /// Checks if the permutation is correct
    pub fn from_indices(v: Vec<usize>) -> Result<Self, ()> {
        let perm = Permutation { indices: v };
        if perm.correct() {
            Ok(perm)
        } else {
            Err(())
        }
    }

    /// Inner function to check correctness of the permutation.
    fn correct(&self) -> bool {
        let axis_len = self.indices.len();
        let mut seen = vec![false; axis_len];
        for &i in &self.indices {
            if seen[i] {
                return false;
            }
            seen[i] = true;
        }
        true
    }
}

pub trait SortArray {
    /// ***Panics*** if `axis` is out of bounds.
    fn identity(&self, axis: Axis) -> Permutation;
    fn sort_axis_by<F>(&self, axis: Axis, less_than: F) -> Permutation
    where
        F: FnMut(usize, usize) -> bool;
}

pub trait PermuteArray {
    type Elem;
    type Dim;
    fn permute_axis(self, axis: Axis, perm: &Permutation) -> Array<Self::Elem, Self::Dim>
    where
        Self::Elem: Clone,
        Self::Dim: RemoveAxis;
}

impl<A, S, D> SortArray for ArrayBase<S, D>
where
    S: Data<Elem = A>,
    D: Dimension,
{
    fn identity(&self, axis: Axis) -> Permutation {
        Permutation {
            indices: (0..self.len_of(axis)).collect(),
        }
    }

    fn sort_axis_by<F>(&self, axis: Axis, mut less_than: F) -> Permutation
    where
        F: FnMut(usize, usize) -> bool,
    {
        let mut perm = self.identity(axis);
        perm.indices.sort_by(move |&a, &b| {
            if less_than(a, b) {
                Ordering::Less
            } else if less_than(b, a) {
                Ordering::Greater
            } else {
                Ordering::Equal
            }
        });
        perm
    }
}

impl<A, D> PermuteArray for Array<A, D>
where
    D: Dimension,
{
    type Elem = A;
    type Dim = D;

    fn permute_axis(self, axis: Axis, perm: &Permutation) -> Array<A, D>
    where
        D: RemoveAxis,
    {
        let axis = axis;
        let axis_len = self.len_of(axis);
        assert_eq!(axis_len, perm.indices.len());
        debug_assert!(perm.correct());

        let mut v = Vec::with_capacity(self.len());
        let mut result;

        // panic-critical begin: we must not panic
        unsafe {
            v.set_len(self.len());
            result = Array::from_shape_vec_unchecked(self.dim(), v);
            for i in 0..axis_len {
                let perm_i = perm.indices[i];
                Zip::from(result.index_axis_mut(axis, perm_i))
                    .and(self.index_axis(axis, i))
                    .for_each(|to, from| copy_nonoverlapping(from, to, 1));
            }
            // forget moved array elements but not its vec
            let mut old_storage = self.into_raw_vec();
            old_storage.set_len(0);
            // old_storage drops empty
        }
        // panic-critical end
        result
    }
}
