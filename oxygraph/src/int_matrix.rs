//! A bipartite graph can be converted to an interaction
//! matrix, which is a binary matrix representing all the
//! possible combinations of hosts/parasites (or sites/species).

use crate::bipartite::BipartiteGraph;
use crate::MARGIN_LR;
use itertools::Itertools;
use permutation;
use std::collections::HashSet;

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
        self.inner = per.apply_slice(&self.inner);

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
        }
        // and sort the column names
        self.colnames = per_cols.apply_slice(&self.colnames);
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

    /// Compute the NODF of an interaction matrix
    /// Can be sorted or unsorted. If you want it to
    /// compute on a sorted matrix, call `.sort()` before
    /// this function.
    pub fn nodf(&self) {
        // following https://nadiah.org/2021/07/16/nodf-nestedness-worked-example
        // rows first
        let mut pos_row_set_vec = Vec::new();
        for row in &self.inner {
            let mut positions_of_ones_iter = row.iter();
            let mut pos_row_hs = HashSet::new();
            while let Some(pos) = positions_of_ones_iter.position(|e| *e == true) {
                pos_row_hs.insert(pos);
            }

            pos_row_set_vec.push(pos_row_hs);
        }
        // make combinations
        let comb_pos_row = pos_row_set_vec.iter().combinations(2);

        // for Vec { upper, lower, .. } in comb_pos_row {}
    }
}
