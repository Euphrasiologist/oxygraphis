//! Not written by myself, but by the great bluss.
//!
//! See <https://github.com/rust-ndarray/ndarray/issues/195>
//! for discussion and details.

use ndarray::prelude::*;
use ndarray::{Data, RemoveAxis, Zip};
use thiserror::Error;

use std::cmp::Ordering;
use std::ptr::copy_nonoverlapping;

/// See <https://stackoverflow.com/questions/69764803/how-to-sort-a-vector-by-indices-in-rust>
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

#[derive(Error, Debug)]
pub enum PermutationError {
    #[error("Permutation index check incorrect in ndarray sort.")]
    FromIndices,
}

/// Type invariant: Each index appears exactly once
#[derive(Clone, Debug)]
pub struct Permutation {
    pub indices: Vec<usize>,
}

impl Permutation {
    /// Checks if the permutation is correct
    pub fn from_indices(v: Vec<usize>) -> Result<Self, PermutationError> {
        let perm = Permutation { indices: v };
        if perm.correct() {
            Ok(perm)
        } else {
            Err(PermutationError::FromIndices)
        }
    }

    /// Not sure what this does.
    fn correct(&self) -> bool {
        let axis_len = self.indices.len();
        let mut seen = vec![false; axis_len];
        for &i in &self.indices {
            match seen.get_mut(i) {
                None => return false,
                Some(s) => {
                    if *s {
                        return false;
                    } else {
                        *s = true;
                    }
                }
            }
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

        if self.is_empty() {
            return self;
        }

        let mut result = Array::uninit(self.dim());

        unsafe {
            // logically move ownership of all elements from self into result
            // the result realizes this ownership at .assume_init() further down
            let mut moved_elements = 0;
            Zip::from(&perm.indices)
                .and(result.axis_iter_mut(axis))
                .for_each(|&perm_i, result_pane| {
                    // possible improvement: use unchecked indexing for `index_axis`
                    Zip::from(result_pane)
                        .and(self.index_axis(axis, perm_i))
                        .for_each(|to, from| {
                            copy_nonoverlapping(from, to.as_mut_ptr(), 1);
                            moved_elements += 1;
                        });
                });
            debug_assert_eq!(result.len(), moved_elements);
            // panic-critical begin: we must not panic
            // forget moved array elements but not its vec
            // old_storage drops empty
            let (mut old_storage, _) = self.into_raw_vec_and_offset();
            old_storage.set_len(0);

            result.assume_init()
            // panic-critical end
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_permute_axis() {
        let a = array![
            [107998.96, 1.],
            [107999.08, 2.],
            [107999.20, 3.],
            [108000.33, 4.],
            [107999.45, 5.],
            [107999.57, 6.],
            [108010.69, 7.],
            [107999.81, 8.],
            [107999.94, 9.],
            [75600.09, 10.],
            [75600.21, 11.],
            [75601.33, 12.],
            [75600.45, 13.],
            [75600.58, 14.],
            [109000.70, 15.],
            [75600.82, 16.],
            [75600.94, 17.],
            [75601.06, 18.],
        ];

        let perm = a.sort_axis_by(Axis(0), |i, j| a[[i, 0]] < a[[j, 0]]);
        let b = a.permute_axis(Axis(0), &perm);
        assert_eq!(
            b,
            array![
                [75600.09, 10.],
                [75600.21, 11.],
                [75600.45, 13.],
                [75600.58, 14.],
                [75600.82, 16.],
                [75600.94, 17.],
                [75601.06, 18.],
                [75601.33, 12.],
                [107998.96, 1.],
                [107999.08, 2.],
                [107999.20, 3.],
                [107999.45, 5.],
                [107999.57, 6.],
                [107999.81, 8.],
                [107999.94, 9.],
                [108000.33, 4.],
                [108010.69, 7.],
                [109000.70, 15.],
            ]
        );
    }
}
