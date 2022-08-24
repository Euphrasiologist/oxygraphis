//! A bipartite graph can be converted to an interaction
//! matrix, which is a binary matrix representing all the
//! possible combinations of hosts/parasites (or sites/species).

use crate::bipartite::BipartiteGraph;
use crate::MARGIN_LR;
use itertools::Itertools;
use permutation;
use std::collections::HashSet;
use thiserror::Error;

/// An error type for the interaction matrix 
/// struct.
#[derive(Error, Debug)]
pub enum InteractionMatrixError {
    #[error("Could not transpose matrix.")]
    TransposeError,
    #[error("Error in NODF calculation.")]
    NODFError,
}

/// A simple representation of a matrix
/// for our purposes. It's a matrix
/// represented as a nested Vec.
pub struct InteractionMatrix {
    pub inner: Vec<Vec<bool>>,
    pub rownames: Vec<String>,
    pub colnames: Vec<String>,
}

impl InteractionMatrix {
    pub fn new(rn: usize, cn: usize) -> Self {
        // outer vec is the number of rows,
        // inner is the number of columns
        let matrix: Vec<Vec<bool>> = vec![Vec::with_capacity(cn); rn];
        InteractionMatrix {
            inner: matrix,
            rownames: Vec::with_capacity(rn),
            colnames: Vec::with_capacity(cn),
        }
    }

    /// Sort an interation matrix
    pub fn sort(&mut self) {
        // sort the rows and the row labels
        let row_sums: &Vec<usize> = &self
            .inner
            .iter()
            .map(|e| e.iter().filter(|f| **f).count())
            .collect();
        let per = permutation::sort(row_sums);
        self.rownames = per.apply_slice(&self.rownames);
        self.rownames.reverse();
        self.inner = per.apply_slice(&self.inner);
        self.inner.reverse();

        // sort the columns and the column labels
        // mod from https://stackoverflow.com/questions/65458789/folding-matrix-rows-element-wise-and-computing-average-in-rust
        fn compute_col_sums(mat: &[Vec<bool>]) -> Vec<usize> {
            assert!(!mat.is_empty());

            let col_len = mat[0].len();
            let col_sums = mat.iter().fold(vec![0usize; col_len], |mut col_sums, row| {
                row.iter()
                    .enumerate()
                    .for_each(|(i, cell)| col_sums[i] += *cell as usize);
                col_sums
            });

            col_sums
        }

        let col_sums = compute_col_sums(&self.inner);
        let per_cols = permutation::sort(col_sums);
        // now we sort each row in the inner vec
        for el in &mut self.inner {
            *el = per_cols.apply_slice(el.clone());
            el.reverse();
        }
        // and sort the column names
        self.colnames = per_cols.apply_slice(&self.colnames);
        self.colnames.reverse();
    }

    /// Make an interaction matrix from a bipartite
    /// graph.
    pub fn from_bipartite(graph: BipartiteGraph) -> Self {
        let (parasites, hosts) = graph.get_parasite_host_from_graph();

        // initialise the matrix
        let mut int_max = InteractionMatrix::new(parasites.len(), hosts.len());

        for (i, (n1, _)) in parasites.iter().enumerate() {
            for (n2, _) in hosts.iter() {
                let is_edge = graph.0.contains_edge(*n1, *n2);
                if is_edge {
                    int_max.inner[i].push(true);
                } else {
                    int_max.inner[i].push(false);
                }
            }
        }

        int_max.rownames = parasites
            .into_iter()
            .map(|(_, s)| s.into())
            .collect::<Vec<String>>();

        int_max.colnames = hosts
            .into_iter()
            .map(|(_, s)| s.into())
            .collect::<Vec<String>>();

        int_max
    }

    /// Make an SVG interaction matrix plot
    ///
    /// TODO: sort out the height on this
    /// and maybe labels.
    pub fn plot(&self, width: i32, height: i32) {
        // space on the x axis and y axis
        let x_spacing = (width as f64 - (MARGIN_LR * 2.0)) / self.colnames.len() as f64;
        let y_spacing = x_spacing; //(height as f64 - (MARGIN_LR * 2.0)) / self.rownames.len() as f64;

        // we want to make as many circles in a row as there are rownames
        let mut assoc_circles = String::new();

        for parasite in 0..self.rownames.len() {
            for host in 0..self.colnames.len() {
                let is_assoc = self.inner[parasite][host];
                let col = if is_assoc { "black" } else { "white" };
                let x = (x_spacing * host as f64) + (x_spacing / 2.0) + MARGIN_LR;
                let y = (y_spacing * parasite as f64) + (y_spacing / 2.0) + MARGIN_LR;
                assoc_circles += &format!(
                    "<circle cx=\"{}\" cy=\"{}\" r=\"{}\" fill=\"{}\" stroke=\"black\"><title>{}</title></circle>\n",
                    x, y, (x_spacing / 2.0), col, &format!("{} x {}", self.rownames[parasite], self.colnames[host])
                );
            }
        }

        let svg = format!(
            r#"<svg version="1.1"
    width="{}" height="{}"
    xmlns="http://www.w3.org/2000/svg">
    {}
</svg>
        "#,
            width, height, assoc_circles
        );

        println!("{}", svg);
    }

    /// Transpose an interaction matrix.
    pub fn transpose(&mut self) -> Result<Self, InteractionMatrixError> {
        let mut matrix = self.inner.clone();
        for inner in &mut matrix {
            inner.reverse();
        }

        let t = (0..matrix[0].len())
            .map(|_| {
                matrix
                    .iter_mut()
                    .map(|inner| inner.pop())
                    .collect::<Option<Vec<bool>>>()
            })
            .collect::<Option<Vec<Vec<bool>>>>();

        let inner = match t {
            Some(m) => m,
            None => return Err(InteractionMatrixError::TransposeError),
        };

        Ok(Self {
            inner,
            rownames: self.colnames.clone(),
            colnames: self.rownames.clone(),
        })
    }

    /// Compute the NODF of an interaction matrix
    /// Can be sorted or unsorted. If you want it to
    /// compute on a sorted matrix, call `.sort()` before
    /// this function.
    /// 
    /// TODO: refactor this code.
    pub fn nodf(&mut self) -> Result<f64, InteractionMatrixError> {
        // following https://nadiah.org/2021/07/16/nodf-nestedness-worked-example
        // rows first
        let mut pos_row_set_vec = Vec::new();
        for row in &self.inner {
            let positions_of_ones_iter = row
                .iter()
                .enumerate()
                .filter(|(_, &r)| r)
                .map(|(index, _)| index)
                .collect::<HashSet<_>>();

            pos_row_set_vec.push(positions_of_ones_iter);
        }
        // make combinations
        let comb_pos_row = pos_row_set_vec.iter().combinations(2);

        // store row number paired
        let mut np_row = Vec::new();
        for upper_lower in comb_pos_row {
            // these should be guaranteed subsets
            let upper = upper_lower[0];
            let lower = upper_lower[1];
            if lower.len() >= upper.len() {
                np_row.push(0.0);
            } else {
                let int: HashSet<_> = upper.intersection(lower).collect();
                np_row.push((100.0 * int.len() as f64) / lower.len() as f64)
            }
        }

        // now columns, just copying above, clean up later
        let t_int = self
            .transpose()
            .map_err(|_| InteractionMatrixError::NODFError)?;

        let mut pos_col_set_vec = Vec::new();
        for row in t_int.inner {
            let positions_of_ones_iter = row
                .iter()
                .enumerate()
                .filter(|(_, &r)| r)
                .map(|(index, _)| index)
                .collect::<HashSet<_>>();

            pos_col_set_vec.push(positions_of_ones_iter);
        }
        // make combinations
        let comb_pos_col = pos_col_set_vec.iter().combinations(2);

        // store row number paired
        let mut np_col = Vec::new();
        for upper_lower in comb_pos_col {
            // these should be guaranteed subsets
            let upper = upper_lower[0];
            let lower = upper_lower[1];
            if lower.len() >= upper.len() {
                np_col.push(0.0);
            } else {
                let int: HashSet<_> = upper.intersection(lower).collect();
                np_col.push((100.0 * int.len() as f64) / lower.len() as f64)
            }
        }

        // append all to np_row
        np_row.append(&mut np_col);

        let mean = np_row.iter().sum::<f64>() / np_row.len() as f64;

        Ok(mean)
    }
}

#[cfg(test)]
mod tests {

    // bring everything in from above
    use super::*;

    #[test]
    fn check_nodf() {
        // create a new interaction matrix
        // copy the example here
        // https://nadiah.org/2021/07/16/nodf-nestedness-worked-example
        let mut int_mat = InteractionMatrix::new(5, 5);
        int_mat.rownames = vec![
            "1r".into(),
            "2r".into(),
            "3r".into(),
            "4r".into(),
            "5r".into(),
        ];
        int_mat.colnames = vec![
            "1c".into(),
            "2c".into(),
            "3c".into(),
            "4c".into(),
            "5c".into(),
        ];

        int_mat.inner = vec![
            vec![true, false, true, true, true],
            vec![true, true, true, false, false],
            vec![false, true, true, true, false],
            vec![true, true, false, false, false],
            vec![true, true, false, false, false],
        ];

        let nodf = int_mat.nodf();

        assert_eq!(nodf.unwrap().floor(), 58.333f64.floor());
    }
}
