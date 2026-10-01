//! A bipartite graph can be converted to an interaction
//! matrix, which is a binary matrix representing all the
//! possible combinations of hosts/parasites (or sites/species).

use crate::bipartite::BipartiteGraph;
use crate::bipartite::Partition;
use crate::modularity;
use crate::modularity::PlotData;
use crate::sort::*;
use crate::LpaWbPlus;
use crate::MARGIN_LR;
use calm_io::*;
use itertools::Itertools;
use ndarray::{Array2, ArrayBase, Axis, Dim, OwnedRepr};
use crate::null::{null_matrix, NullModel, NullModelError};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::fmt;
use std::io::Write;
use std::path::PathBuf;



/// A 2D interaction matrix using floating-point weights.
///
/// Internally represented by `ndarray::Array2<f64>`.
pub type Matrix = Array2<f64>;

/// Compute entropy (Shannon) for a frequency matrix
fn entropy(matrix: &Matrix) -> f64 {
    let total = matrix.sum();
    matrix
        .iter()
        .filter(|&&v| v > 0.0)
        .map(|&v| {
            let p = v / total;
            -p * p.ln()
        })
        .sum()
}

/// First (i, j) in column-major order satisfying `pred`, matching R's `which(...)[1]`.
fn first_col_major(nr: usize, nc: usize, pred: impl Fn(usize, usize) -> bool) -> Option<(usize, usize)> {
    for j in 0..nc {
        for i in 0..nr {
            if pred(i, j) {
                return Some((i, j));
            }
        }
    }
    None
}

/// Entropy of the minimum-entropy matrix with the given marginal totals, built greedily as in
/// `bipartite::H2fun`: repeatedly place min(largest remaining row total, largest remaining
/// column total) at their intersection, taking the first maximum on ties.
fn h2_min_entropy(row_sums: &[f64], col_sums: &[f64]) -> f64 {
    let (nr, nc) = (row_sums.len(), col_sums.len());
    let mut web = Array2::<f64>::zeros((nr, nc));
    let first_max = |v: &[f64]| {
        let m = v.iter().cloned().fold(f64::MIN, f64::max);
        v.iter().position(|&x| x == m).unwrap()
    };
    let mut rs_rest = row_sums.to_vec();
    let mut cs_rest = col_sums.to_vec();
    let mut guard = 0;
    while (rs_rest.iter().sum::<f64>() * 1e10).round() != 0.0 && guard < 10 * (nr * nc + 1) {
        let i = first_max(&rs_rest);
        let j = first_max(&cs_rest);
        web[[i, j]] = rs_rest[i].min(cs_rest[j]);
        let rsn = web.sum_axis(Axis(1));
        let csn = web.sum_axis(Axis(0));
        rs_rest = row_sums.iter().zip(rsn.iter()).map(|(a, b)| a - b).collect();
        cs_rest = col_sums.iter().zip(csn.iter()).map(|(a, b)| a - b).collect();
        guard += 1;
    }
    entropy(&web)
}

/// Result structure for the Nested NODF calculation, modeled after the `vegan::nestednodf` output.
#[derive(Debug, Clone)]
pub struct NestedNODFResult {
    /// The binary (presence/absence) or weighted community matrix used.
    pub comm: Matrix,
    /// Proportion of the matrix filled with interactions.
    pub fill: f64,
    /// Nestedness contribution from rows.
    pub n_rows: f64,
    /// Nestedness contribution from columns.
    pub n_cols: f64,
    /// Overall NODF score.
    pub nodf: f64,
}

/// A wrapper around the interaction matrix with labels for rows (parasites) and columns (hosts).
#[derive(Debug, Clone)]
pub struct InteractionMatrix {
    /// The core 2D ndarray matrix (interactions or weights).
    pub inner: Matrix,
    /// Row names, typically parasite species.
    pub rownames: Vec<String>,
    /// Column names, typically host species.
    pub colnames: Vec<String>,
}

// Possibly not necessary at the end but useful for debugging.
impl fmt::Display for InteractionMatrix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut output_string = String::new();
        for el in self.inner.rows() {
            let mut temp_string = String::new();
            for e in el {
                temp_string += &format!("{}\t", *e as usize);
            }
            // remove last \t
            temp_string.pop();
            output_string += &format!("{}\n", temp_string);
        }
        writeln!(f, "# Row names: {}", self.rownames.join(", "))?;
        writeln!(f, "# Columns names: {}", self.colnames.join(", "))?;
        write!(f, "{}", output_string)
    }
}

/// Summary statistics for an interaction matrix.
#[derive(Debug)]
pub struct InteractionMatrixStats {
    /// Is it a weighted matrix?
    pub weighted: bool,
    /// Number of rows in the matrix
    pub no_rows: usize,
    /// Number of columns in the matrix
    pub no_cols: usize,
    /// Number of possible interactions.
    pub no_poss_ints: usize,
    /// Percentage of possible interactions seen.
    pub perc_ints: f64,
    /// Mean number of realized interactions per species (rows + cols).
    pub link_density: f64,
}

/// Result of a permutation significance test on a network metric.
#[derive(Debug, Clone)]
pub struct PermutationTestResult {
    /// Null model used.
    pub null_model: NullModel,
    /// Observed value of the metric.
    pub observed: f64,
    /// Mean of the null distribution.
    pub mean_null: f64,
    /// Standard deviation of the null distribution (sample SD, n - 1).
    pub sd_null: f64,
    /// Standardised effect size: (observed - mean_null) / sd_null.
    pub z: f64,
    /// One-tailed P-value, (k + 1) / (n + 1), where k is the number of null values >= observed.
    pub p_value: f64,
    /// Number of valid (non-NaN) permutations used.
    pub n_permutations: usize,
}

/// Species-level d' with its null distribution.
#[derive(Debug, Clone)]
pub struct DPrimeNullResult {
    /// Species name.
    pub species: String,
    /// Observed d'.
    pub observed: Option<f64>,
    /// Mean of null d' values.
    pub mean_null: f64,
    /// 2.5% quantile of null d' values.
    pub lower_null: f64,
    /// 97.5% quantile of null d' values.
    pub upper_null: f64,
    /// One-tailed P-value, (k + 1) / (n + 1).
    pub p_value: f64,
}

/// Quantile with linear interpolation (R's default, type 7). `sorted` must be ascending.
fn quantile7(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let h = (sorted.len() - 1) as f64 * q;
    let lo = h.floor() as usize;
    let hi = h.ceil() as usize;
    sorted[lo] + (h - lo as f64) * (sorted[hi] - sorted[lo])
}

impl InteractionMatrix {
    /// Write an interaction matrix to a TSV file.
    pub fn write_tsv(&self, filename: PathBuf, kind: &str) -> Result<(), std::io::Error> {
        let mut writer = std::fs::File::create(filename)?;
        // write the header
        let header = format!("# {} edited interaction matrix\n", kind);
        writer.write_all(header.as_bytes())?;
        // write the column names first
        let col_headers = format!("parasite\thost\tweight\n");
        writer.write_all(col_headers.as_bytes())?;

        // iterate over each of the elements and print out host and parasite and weight
        // lengths of the rows and columns
        let parasites = self.rownames.len();
        let hosts = self.colnames.len();

        // write the rows
        for parasite in 0..parasites {
            for host in 0..hosts {
                let line = format!(
                    "{}\t{}\t{}\n",
                    self.rownames[parasite],
                    self.colnames[host],
                    self.inner[[parasite, host]]
                );
                if let Err(e) = writer.write_all(line.as_bytes()) {
                    return Err(e);
                }
            }
        }
        writer.flush()?;
        Ok(())
    }

    /// Compute statistics on the interaction matrix.
    ///
    /// - Counts the number of rows, columns, and possible interactions.
    /// - Calculates the percentage of realized interactions.
    /// - Determines whether the matrix is weighted.
    ///
    /// # Returns
    /// `InteractionMatrixStats` summarizing the matrix.
    pub fn stats(&self) -> InteractionMatrixStats {
        let no_rows = self.rownames.len();
        let no_cols = self.colnames.len();
        let no_poss_ints = no_rows * no_cols;
        // matrix sum here must remove the weights
        // if we want to do this on a weighted matrix
        // we need to change the sum_matrix() function
        let weighted = self.inner.iter().any(|e| *e > 1.0);

        // change matrix to binary matrix
        let bin_mat = self.inner.map(|e| if *e > 0.0 { 1.0 } else { 0.0 });

        let realized = bin_mat.sum();
        let perc_ints = realized / no_poss_ints as f64;
        let link_density = realized / (no_rows + no_cols) as f64;

        InteractionMatrixStats {
            weighted,
            no_rows,
            no_cols,
            no_poss_ints,
            perc_ints,
            link_density,
        }
    }

    /// Create a new empty `InteractionMatrix` with the given dimensions.
    ///
    /// # Arguments
    /// * `rn` - Number of rows (parasites).
    /// * `cn` - Number of columns (hosts).
    ///
    /// # Returns
    /// Empty `InteractionMatrix` with pre-allocated labels.
    pub fn new(rn: usize, cn: usize) -> Self {
        // outer vec is the number of rows,
        // inner is the number of columns
        let matrix: Matrix = Array2::zeros((rn, cn));
        InteractionMatrix {
            inner: matrix,
            rownames: Vec::with_capacity(rn),
            colnames: Vec::with_capacity(cn),
        }
    }

    /// Sort rows and columns of the matrix by decreasing marginal totals (interaction counts).
    ///
    /// Sorts both the matrix and its corresponding row/column labels.
    pub fn sort(&mut self) {
        // sort the rows and the row labels
        // generate the row sums
        let row_sums = &self.inner.sum_axis(Axis(1));
        // generate the permutation order of the row_sums
        let perm = row_sums.sort_axis_by(Axis(0), |i, j| row_sums[i] > row_sums[j]);
        // sort the matrix by row order
        self.inner = self.inner.clone().permute_axis(Axis(0), &perm);
        // sort the row names in place
        sort_by_indices(&mut self.rownames, perm.indices);

        // sort the columns and the column labels
        let col_sums = &self.inner.sum_axis(Axis(0));
        let perm_cols = col_sums.sort_axis_by(Axis(0), |i, j| col_sums[i] > col_sums[j]);
        // now we sort each row in the inner vec
        self.inner = self.inner.clone().permute_axis(Axis(1), &perm_cols);
        // and sort the column names
        sort_by_indices(&mut self.colnames, perm_cols.indices);
    }

    /// Create an `InteractionMatrix` from a [`BipartiteGraph`].
    ///
    /// - Parasites become **rows**.
    /// - Hosts become **columns**.
    /// - Edges are treated as interactions with optional weights.
    ///
    /// # Returns
    /// Populated `InteractionMatrix`.
    ///
    /// # Examples
    /// ```
    /// use oxygraph::bipartite::{BipartiteGraph, Partition, SpeciesNode};
    /// use oxygraph::int_matrix::InteractionMatrix;
    /// use petgraph::Graph;
    ///
    /// // Create a simple bipartite graph manually
    /// let mut graph = Graph::new();
    /// let p1 = graph.add_node(SpeciesNode::new("Parasite1".to_string(), Partition::Parasites));
    /// let h1 = graph.add_node(SpeciesNode::new("Host1".to_string(), Partition::Hosts));
    /// graph.add_edge(p1, h1, 1.0);
    ///
    /// let bp_graph = BipartiteGraph(graph);
    /// let int_matrix = InteractionMatrix::from_bipartite(bp_graph);
    ///
    /// assert_eq!(int_matrix.rownames, vec!["Parasite1"]);
    /// assert_eq!(int_matrix.colnames, vec!["Host1"]);
    /// assert_eq!(int_matrix.inner.shape(), &[1, 1]);
    /// assert_eq!(int_matrix.inner[[0, 0]], 1.0);
    /// ```
    pub fn from_bipartite(graph: BipartiteGraph) -> Self {
        let (mut parasites, mut hosts) = graph.get_parasite_host_from_graph();
        // Graph node order depends on hash ordering when the edge list is read, so sort by
        // name to make the matrix, and anything stochastic computed from it, reproducible.
        parasites.sort_by(|a, b| a.1.name.cmp(&b.1.name));
        hosts.sort_by(|a, b| a.1.name.cmp(&b.1.name));

        // Early return if graph is empty
        if parasites.is_empty() || hosts.is_empty() {
            return InteractionMatrix::new(0, 0);
        }

        let mut int_max = InteractionMatrix::new(parasites.len(), hosts.len());

        for (i, (n1, _)) in parasites.iter().enumerate() {
            for (j, (n2, _)) in hosts.iter().enumerate() {
                if let Some(e) = graph.0.find_edge(*n1, *n2) {
                    // FIXME: is this right?
                    // if an edge is found, default wright is 1.0.
                    let weight = graph.0.edge_weight(e).unwrap_or(&1.0);
                    int_max.inner[[i, j]] = if *weight == 0.0 { 1.0 } else { *weight };
                } else {
                    int_max.inner[[i, j]] = 0.0;
                }
            }
        }

        int_max.rownames = parasites.into_iter().map(|(_, s)| s.name.clone()).collect();

        int_max.colnames = hosts.into_iter().map(|(_, s)| s.name.clone()).collect();

        int_max
    }

    /// Extract modular assignments from the interaction matrix based on [`PlotData`].
    ///
    /// # Arguments
    /// * `modularity_plot_data` - Row and column module assignments.
    ///
    /// # Returns
    /// A mapping of module IDs to lists of `(parasite, host)` pairs.
    pub fn modules(
        &self,
        modularity_plot_data: PlotData,
    ) -> BTreeMap<usize, Vec<(String, String)>> {
        let PlotData {
            rows,
            cols,
            modules,
        } = modularity_plot_data;
        let parasites = self.rownames.len();
        let hosts = self.colnames.len();
        let mut modularity_labels = BTreeMap::new();
        // keep track of cumulative column & row sizes
        let mut cumulative_col_size = 0;
        let mut cumulative_row_size = 0;

        // iterate over the modules
        for module in 0..modules.len() {
            // get this row size and the previous row size information
            let row_size = rows.iter().filter(|e| **e == modules[module]).count();
            let prev_row_size = rows
                .iter()
                .filter(|e| **e == *modules.get(module - 1).unwrap_or(&(module as u32)))
                .count();
            // and the same for the columns
            let col_size = cols.iter().filter(|e| **e == modules[module]).count();
            let prev_col_size = cols
                .iter()
                .filter(|e| **e == *modules.get(module - 1).unwrap_or(&(module as u32)))
                .count();

            // as a by-product of the unwrap_or() on the .get() function above,
            // skip the first iteration in the cumulative sums.
            if module > 0 {
                cumulative_col_size += prev_col_size;
                cumulative_row_size += prev_row_size;
            }

            // add to the modules
            let module_space = (cumulative_col_size..cumulative_col_size + col_size)
                .cartesian_product(cumulative_row_size..cumulative_row_size + row_size)
                .collect::<Vec<_>>();
            let mut zipped = Vec::new();
            for parasite in 0..parasites {
                for host in 0..hosts {
                    let is_assoc = self.inner[[parasite, host]] > 0.0;
                    if module_space.contains(&(host, parasite)) && is_assoc {
                        let p = self.rownames[parasite].clone();
                        let h = self.colnames[host].clone();
                        zipped.push((p, h));
                    }
                }
            }
            modularity_labels.insert(module, zipped);
        }

        modularity_labels
    }

    /// Generate an SVG plot of the interaction matrix, optionally highlighting modularity.
    ///
    /// # Arguments
    /// * `width` - Width of the SVG canvas.
    /// * `modularity_plot_data` - Optional module data for plotting module boundaries.
    ///
    /// # Returns
    /// * `Some` modularity data mapping if modularity data was provided.
    /// * `None` otherwise.
    ///
    /// # Examples
    /// ```no_run
    /// use oxygraph::int_matrix::InteractionMatrix;
    /// use ndarray::array;
    ///
    /// let matrix = InteractionMatrix {
    ///     inner: array![
    ///         [1.0, 0.0, 1.0],
    ///         [0.0, 1.0, 0.0]
    ///     ],
    ///     rownames: vec!["P1".into(), "P2".into()],
    ///     colnames: vec!["H1".into(), "H2".into(), "H3".into()],
    /// };
    ///
    /// // This prints SVG to STDOUT
    /// matrix.plot(500, None);
    /// ```
    pub fn plot(
        &self,
        width: i32,
        modularity_plot_data: Option<PlotData>,
    ) -> Option<BTreeMap<usize, Vec<(String, String)>>> {
        // space on the x axis and y axis
        let x_spacing = (width as f64 - (MARGIN_LR * 2.0)) / self.colnames.len() as f64;
        let y_spacing = x_spacing;

        // we want to make as many circles in a row as there are rownames
        let mut svg_data = String::new();

        // lengths of the rows and columns
        let parasites = self.rownames.len();
        let hosts = self.colnames.len();
        let grid_size = parasites * hosts;

        for parasite in 0..parasites {
            for host in 0..hosts {
                let is_assoc = self.inner[[parasite, host]] > 0.0;
                let col = if is_assoc { "black" } else { "white" };
                let x = (x_spacing * host as f64) + (x_spacing / 2.0) + MARGIN_LR;
                let y = (y_spacing * parasite as f64) + (y_spacing / 2.0) + MARGIN_LR;

                // don't draw every circle if the grid has more than 500 circles
                if grid_size > 500 {
                    match is_assoc {
                        true => {
                            svg_data += &format!(
                                "<circle cx=\"{}\" cy=\"{}\" r=\"{}\" fill=\"{}\" stroke=\"black\"><title>{}</title></circle>\n",
                                x, y, (x_spacing / 2.0), col, &format!("{} x {}", self.rownames[parasite], self.colnames[host])
                            );
                        }
                        // don't plot the white circles.
                        false => (),
                    }
                } else {
                    // plot every circle.
                    svg_data += &format!(
                        "<circle cx=\"{}\" cy=\"{}\" r=\"{}\" fill=\"{}\" stroke=\"black\"><title>{}</title></circle>\n",
                        x, y, (x_spacing / 2.0), col, &format!("{} x {}", self.rownames[parasite], self.colnames[host])
                    );
                }
            }
        }

        // if we have a modularity plot
        let mut return_modules = false;

        if let Some(rects) = modularity_plot_data.clone() {
            return_modules = true;
            // destructure the plot data
            let PlotData {
                rows,
                cols,
                modules,
            } = rects;

            // keep track of cumulative column & row sizes
            let mut cumulative_col_size = 0;
            let mut cumulative_row_size = 0;

            // iterate over the modules
            for module in 0..modules.len() {
                // get this row size and the previous row size information
                let row_size = rows.iter().filter(|e| **e == modules[module]).count();
                let prev_row_size = rows
                    .iter()
                    .filter(|e| **e == *modules.get(module - 1).unwrap_or(&(module as u32)))
                    .count();
                // and the same for the columns
                let col_size = cols.iter().filter(|e| **e == modules[module]).count();
                let prev_col_size = cols
                    .iter()
                    .filter(|e| **e == *modules.get(module - 1).unwrap_or(&(module as u32)))
                    .count();

                // as a by-product of the unwrap_or() on the .get() function above,
                // skip the first iteration in the cumulative sums.
                if module > 0 {
                    cumulative_col_size += prev_col_size;
                    cumulative_row_size += prev_row_size;
                }

                // rect height and widths are multiples of the column and the
                // row lengths.
                let rect_width = col_size as f64 * x_spacing;
                let rect_height = row_size as f64 * y_spacing;

                // we then need to translate the rects the appropriate
                // amount, offset by the cumulative column and row sizes.
                let translate = format!(
                    "translate({} {})",
                    (cumulative_col_size as f64 * x_spacing) + MARGIN_LR,
                    (cumulative_row_size as f64 * y_spacing) + MARGIN_LR
                );

                // append to the SVG data.
                svg_data += &format!("<rect x=\"0\" y=\"0\" width=\"{rect_width}\" height=\"{rect_height}\" style=\"fill: none; stroke: red; stroke-width: 2px;\" transform=\"{translate}\"/>");
            }
        }

        let svg = format!(
            r#"<svg version="1.1"
    width="{}" height="{}"
    xmlns="http://www.w3.org/2000/svg">
    {}
</svg>
        "#,
            width,
            (MARGIN_LR * 2.0) + (y_spacing * self.rownames.len() as f64),
            svg_data
        );

        let _ = stdoutln!("{}", svg);

        if return_modules {
            Some(self.modules(modularity_plot_data.unwrap()))
        } else {
            None
        }
    }

    /// Transpose an interaction matrix.
    ///
    /// # Returns
    /// A new `InteractionMatrix` with rows and columns swapped.
    ///
    /// # Example
    /// ```
    /// use oxygraph::int_matrix::InteractionMatrix;
    ///
    /// let mut matrix = InteractionMatrix::new(2, 3);
    /// matrix.rownames = vec!["A".to_string(), "B".to_string()];
    /// matrix.colnames = vec!["X".to_string(), "Y".to_string(), "Z".to_string()];
    /// let transposed = matrix.transpose();
    /// assert_eq!(transposed.rownames, vec!["X", "Y", "Z"]);
    /// assert_eq!(transposed.colnames, vec!["A", "B"]);
    /// ```
    pub fn transpose(&mut self) -> Self {
        let inner = self.inner.t().to_owned();

        Self {
            inner,
            rownames: self.colnames.clone(),
            colnames: self.rownames.clone(),
        }
    }

    /// Sort the matrix by decreasing fill, optionally weighted.
    ///
    /// # Arguments
    /// * `weighted` - If true, sorts by weighted degree after fill.
    ///
    /// # Examples
    /// ```
    /// use oxygraph::int_matrix::InteractionMatrix;
    /// use ndarray::array;
    ///
    /// let mut matrix = InteractionMatrix {
    ///     inner: array![
    ///         [1.0, 0.0, 1.0], // sum = 2
    ///         [1.0, 1.0, 1.0], // sum = 3
    ///         [0.0, 0.0, 1.0], // sum = 1
    ///     ],
    ///     rownames: vec!["A".into(), "B".into(), "C".into()],
    ///     colnames: vec!["X".into(), "Y".into(), "Z".into()],
    /// };
    ///
    /// matrix.sort_by_decreasing_fill(false);
    ///
    /// // Rownames should now be sorted by row sum (highest to lowest)
    /// assert_eq!(matrix.rownames, vec!["B", "A", "C"]);
    /// ```
    pub fn sort_by_decreasing_fill(&mut self, weighted: bool) {
        let bin_comm = self.inner.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
        let rfill: Vec<usize> = bin_comm
            .axis_iter(Axis(0))
            .map(|r| r.sum() as usize)
            .collect();
        let cfill: Vec<usize> = bin_comm
            .axis_iter(Axis(1))
            .map(|c| c.sum() as usize)
            .collect();

        // Row sorting: fill, then weighted abundance if requested
        let mut row_indices: Vec<_> = if weighted {
            let rgrad: Vec<f64> = self.inner.axis_iter(Axis(0)).map(|r| r.sum()).collect();
            (0..self.inner.nrows())
                .map(|i| (i, rfill[i], rgrad[i]))
                .collect()
        } else {
            (0..self.inner.nrows())
                .map(|i| (i, rfill[i], 0.0))
                .collect()
        };
        row_indices.sort_by(|a, b| b.1.cmp(&a.1).then(b.2.partial_cmp(&a.2).unwrap()));

        let sorted_rows: Vec<_> = row_indices
            .iter()
            .map(|(i, _, _)| self.inner.row(*i).to_owned())
            .collect();
        self.inner = Array2::from_shape_vec(
            (sorted_rows.len(), self.inner.ncols()),
            sorted_rows.iter().flat_map(|r| r.iter().cloned()).collect(),
        )
        .unwrap();
        self.rownames = row_indices
            .iter()
            .map(|(i, _, _)| self.rownames[*i].clone())
            .collect();

        // Column sorting: fill, then weighted abundance if requested
        let mut col_indices: Vec<_> = if weighted {
            let cgrad: Vec<f64> = self.inner.axis_iter(Axis(1)).map(|c| c.sum()).collect();
            (0..self.inner.ncols())
                .map(|i| (i, cfill[i], cgrad[i]))
                .collect()
        } else {
            (0..self.inner.ncols())
                .map(|i| (i, cfill[i], 0.0))
                .collect()
        };
        col_indices.sort_by(|a, b| b.1.cmp(&a.1).then(b.2.partial_cmp(&a.2).unwrap()));

        let sorted_cols: Vec<_> = col_indices
            .iter()
            .map(|(i, _, _)| self.inner.column(*i).to_owned())
            .collect();
        self.inner = Array2::from_shape_vec(
            (self.inner.nrows(), sorted_cols.len()),
            (0..self.inner.nrows())
                .flat_map(|row_idx| sorted_cols.iter().map(move |col| col[row_idx]))
                .collect(),
        )
        .unwrap();
        self.colnames = col_indices
            .iter()
            .map(|(i, _, _)| self.colnames[*i].clone())
            .collect();
    }

    /// Compute the Nested NODF index for the interaction matrix.
    ///
    /// # Arguments
    /// * `order` - If true, sorts rows/columns by decreasing fill before calculation.
    /// * `weighted` - If true, weights are used in the calculation.
    /// * `wbinary` - If true, weights are binarized before calculating nestedness.
    ///
    /// # Returns
    /// A `NestedNODFResult` containing nestedness statistics.
    ///
    /// # Examples
    /// ```
    /// use oxygraph::int_matrix::InteractionMatrix;
    /// use ndarray::array;
    ///
    /// let mut matrix = InteractionMatrix {
    ///     inner: array![
    ///         [1.0, 1.0, 1.0], // nested with all others
    ///         [1.0, 1.0, 0.0],
    ///         [1.0, 0.0, 0.0]
    ///     ],
    ///     rownames: vec!["P1".into(), "P2".into(), "P3".into()],
    ///     colnames: vec!["H1".into(), "H2".into(), "H3".into()],
    /// };
    ///
    /// let nodf = matrix.nodf(true, false, false);
    ///
    /// assert_eq!(nodf.nodf, 100.0);
    /// assert_eq!(nodf.fill, 6.0 / 9.0);
    /// ```
    pub fn nodf(&self, order: bool, weighted: bool, wbinary: bool) -> NestedNODFResult {
        // If ordering is requested, work on a sorted clone so `self` is unchanged.
        let owned;
        let m: &Self = if order {
            owned = {
                let mut tmp = self.clone();
                tmp.sort_by_decreasing_fill(weighted);
                tmp
            };
            &owned
        } else {
            self
        };

        let nr = m.inner.nrows();
        let nc = m.inner.ncols();

        // return early for degenerate matrices — using || prevents usize
        // underflow in the loop bounds below (e.g. 0usize - 1 would panic).
        if nr < 2 || nc < 2 {
            return NestedNODFResult {
                comm: m.inner.clone(),
                fill: 0.0,
                n_rows: 0.0,
                n_cols: 0.0,
                nodf: 0.0,
            };
        }

        let bin_comm = m.inner.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
        let rfill: Vec<usize> = bin_comm
            .axis_iter(Axis(0))
            .map(|r| r.sum() as usize)
            .collect();
        let cfill: Vec<usize> = bin_comm
            .axis_iter(Axis(1))
            .map(|c| c.sum() as usize)
            .collect();

        // avoid divide by zero here
        let total_fill = if nr == 0 || nc == 0 {
            0.0
        } else {
            rfill.iter().sum::<usize>() as f64 / (nr * nc) as f64
        };

        let mut paired_rows = Vec::new();
        let mut valid_row_pairs = 0;

        for i in 0..(nr - 1) {
            let first_row = m.inner.row(i);
            for j in (i + 1)..nr {
                if rfill[i] <= rfill[j] || rfill[i] == 0 || rfill[j] == 0 {
                    continue;
                }

                valid_row_pairs += 1;
                let second_row = m.inner.row(j);

                let overlap = if weighted {
                    if !wbinary {
                        let diff_gt_zero = first_row
                            .iter()
                            .zip(second_row.iter())
                            .filter(|(&a, &b)| (a - b) > 0.0 && b > 0.0)
                            .count();
                        let denom = second_row.iter().filter(|&&v| v > 0.0).count();
                        (diff_gt_zero as f64) / (denom as f64)
                    } else {
                        let diff_ge_zero = first_row
                            .iter()
                            .zip(second_row.iter())
                            .filter(|(&a, &b)| (a - b) >= 0.0 && b > 0.0)
                            .count();
                        let denom = second_row.iter().filter(|&&v| v > 0.0).count();
                        (diff_ge_zero as f64) / (denom as f64)
                    }
                } else {
                    let shared = first_row
                        .iter()
                        .zip(second_row.iter())
                        .filter(|(&a, &b)| a > 0.0 && b > 0.0)
                        .count();
                    shared as f64 / rfill[j] as f64
                };

                paired_rows.push(overlap);
            }
        }

        let mut paired_cols = Vec::new();
        let mut valid_col_pairs = 0;

        for i in 0..(nc - 1) {
            let first_col = m.inner.column(i);
            for j in (i + 1)..nc {
                if cfill[i] <= cfill[j] || cfill[i] == 0 || cfill[j] == 0 {
                    continue;
                }

                valid_col_pairs += 1;
                let second_col = m.inner.column(j);

                let overlap = if weighted {
                    if !wbinary {
                        let diff_gt_zero = first_col
                            .iter()
                            .zip(second_col.iter())
                            .filter(|(&a, &b)| (a - b) > 0.0 && b > 0.0)
                            .count();
                        let denom = second_col.iter().filter(|&&v| v > 0.0).count();
                        (diff_gt_zero as f64) / (denom as f64)
                    } else {
                        let diff_ge_zero = first_col
                            .iter()
                            .zip(second_col.iter())
                            .filter(|(&a, &b)| (a - b) >= 0.0 && b > 0.0)
                            .count();
                        let denom = second_col.iter().filter(|&&v| v > 0.0).count();
                        (diff_ge_zero as f64) / (denom as f64)
                    }
                } else {
                    let shared = first_col
                        .iter()
                        .zip(second_col.iter())
                        .filter(|(&a, &b)| a > 0.0 && b > 0.0)
                        .count();
                    shared as f64 / cfill[j] as f64
                };

                paired_cols.push(overlap);
            }
        }

        let n_rows = if valid_row_pairs > 0 {
            paired_rows.iter().sum::<f64>() * 100.0 / valid_row_pairs as f64
        } else {
            0.0
        };

        let n_cols = if valid_col_pairs > 0 {
            paired_cols.iter().sum::<f64>() * 100.0 / valid_col_pairs as f64
        } else {
            0.0
        };

        let total_pairs = (nr * (nr - 1)) / 2 + (nc * (nc - 1)) / 2;

        let nodf = if total_pairs > 0 {
            (paired_rows.iter().sum::<f64>() + paired_cols.iter().sum::<f64>()) * 100.0
                / total_pairs as f64
        } else {
            0.0
        };

        NestedNODFResult {
            comm: m.inner.clone(),
            fill: total_fill,
            n_rows,
            n_cols,
            nodf,
        }
    }

    /// Compute the H2' specialization index for a bipartite interaction matrix.
    ///
    /// Replicates the R function `bipartite::H2fun`. For matrices of non-negative integers
    /// it follows `H2_integer = TRUE`; for any non-integer weights it follows
    /// `H2_integer = FALSE`, where the maximum entropy is that of the expected matrix
    /// under independence.
    ///
    /// The method compares the observed (uncorrected) Shannon entropy of interaction
    /// frequencies with a maximum entropy matrix and a minimum entropy configuration
    /// derived by greedily filling the matrix while maintaining row and column totals.
    ///
    /// Returned value is:
    /// ```text
    /// H2' = (H2_max - H2_uncorr) / (H2_max - H2_min)
    /// ```
    /// where:
    /// - `H2_uncorr` is the observed entropy of the interaction matrix
    /// - `H2_max` is the entropy of the expected matrix under independence (integer-approximated
    ///   for integer data)
    /// - `H2_min` is the entropy of a maximally specialized (minimum entropy) matrix
    ///
    /// # Returns
    /// `f64` — the H2' value, in the range [0, 1], where 1 indicates maximum specialization.
    ///
    /// # Example
    /// ```rust
    /// use ndarray::array;
    /// use oxygraph::InteractionMatrix;
    ///
    /// let matrix = InteractionMatrix {
    ///     inner: array![[1.0, 0.0, 1.0], [0.0, 2.0, 0.0], [0.0, 1.0, 1.0]],
    ///     rownames: vec!["a".into(), "b".into(), "c".into()],
    ///     colnames: vec!["x".into(), "y".into(), "z".into()],
    /// };
    /// let h2p = matrix.h2_prime();
    /// assert!(h2p >= 0.0 && h2p <= 1.0);
    /// ```
    pub fn h2_prime(&self) -> f64 {
        let matrix = &self.inner;
        let (nr, nc) = matrix.dim();
        let is_integer = matrix.iter().all(|&v| v.fract() == 0.0);

        let total: f64 = matrix.sum();
        if total <= 0.0 {
            return 0.0;
        }
        let row_sums = matrix.sum_axis(Axis(1));
        let col_sums = matrix.sum_axis(Axis(0));
        let h2_uncorr = entropy(matrix);

        // Expected matrix under independence (R: `exexpec`).
        let expected = row_sums.clone().insert_axis(Axis(1))
            * col_sums.clone().insert_axis(Axis(0))
            / total;

        let h2_max = if !is_integer {
            // R: H2_integer = FALSE
            entropy(&expected)
        } else {
            // R: H2_integer = TRUE. Fill an integer matrix towards the expected one.
            let mut newweb = expected.mapv(f64::floor);
            // On the first pass R compares against an all-zero matrix, so `difexp` starts
            // as the expected matrix itself; afterwards it is expected - newweb.
            let mut difexp = expected.clone();
            let mut webfull = Array2::<bool>::from_elem((nr, nc), false);

            while newweb.sum() < total - 1e-9 {
                let rsn = newweb.sum_axis(Axis(1));
                let csn = newweb.sum_axis(Axis(0));
                for i in 0..nr {
                    if rsn[i] == row_sums[i] {
                        webfull.row_mut(i).fill(true);
                    }
                }
                for j in 0..nc {
                    if csn[j] == col_sums[j] {
                        webfull.column_mut(j).fill(true);
                    }
                }
                // smallest current value among open cells
                let min_open = newweb
                    .indexed_iter()
                    .filter(|(ij, _)| !webfull[*ij])
                    .map(|(_, &v)| v)
                    .fold(f64::INFINITY, f64::min);
                if !min_open.is_finite() {
                    break;
                }
                // greatest shortfall among the smallest open cells
                let greatest = newweb
                    .indexed_iter()
                    .filter(|(ij, &v)| !webfull[*ij] && v == min_open)
                    .map(|(ij, _)| difexp[ij])
                    .fold(f64::NEG_INFINITY, f64::max);
                // R samples among ties; take the first in column-major order (R's `which`).
                match first_col_major(nr, nc, |i, j| {
                    !webfull[[i, j]] && newweb[[i, j]] == min_open && difexp[[i, j]] == greatest
                }) {
                    Some((i, j)) => newweb[[i, j]] += 1.0,
                    None => break,
                }
                difexp = &expected - &newweb;
            }
            let h2_max = entropy(&newweb);

            // R's local refinement. Each of R's 500 tries restarts from the same matrix,
            // so the net effect is a single adjustment.
            let max_expected = expected.iter().cloned().fold(f64::MIN, f64::max);
            if max_expected > 0.3679 * total {
                let mut newmx = newweb.clone();
                let difexp = &expected - &newmx;
                let min_dif = difexp.iter().cloned().fold(f64::INFINITY, f64::min);
                let is_min = |i: usize, j: usize| difexp[[i, j]] == min_dif;
                let n_min = difexp.iter().filter(|&&d| d == min_dif).count();
                let first_mask: Array2<bool> = if n_min > 1 {
                    let largest = newmx
                        .indexed_iter()
                        .filter(|((i, j), _)| is_min(*i, *j))
                        .map(|(_, &v)| v)
                        .fold(f64::MIN, f64::max);
                    Array2::from_shape_fn((nr, nc), |(i, j)| is_min(i, j) && newmx[[i, j]] == largest)
                } else {
                    Array2::from_shape_fn((nr, nc), |(i, j)| is_min(i, j))
                };
                if let Some(cell) = first_col_major(nr, nc, |i, j| first_mask[[i, j]]) {
                    newmx[cell] -= 1.0;
                    let throw = (0..nr).find(|&i| first_mask.row(i).iter().any(|&b| b)).unwrap();
                    let thcol = (0..nc).find(|&j| first_mask.column(j).iter().any(|&b| b)).unwrap();
                    let mr = difexp.row(throw).iter().cloned().fold(f64::MIN, f64::max);
                    let mc = difexp.column(thcol).iter().cloned().fold(f64::MIN, f64::max);
                    if mr >= mc {
                        let scnd = (0..nc).find(|&j| difexp[[throw, j]] == mr).unwrap();
                        newmx[[throw, scnd]] += 1.0;
                        let cmin = difexp.column(scnd).iter().cloned().fold(f64::INFINITY, f64::min);
                        let thrd = (0..nr).find(|&i| difexp[[i, scnd]] == cmin).unwrap();
                        newmx[[thrd, scnd]] -= 1.0;
                        newmx[[thrd, thcol]] += 1.0;
                    } else {
                        let scnd = (0..nr).find(|&i| difexp[[i, thcol]] == mc).unwrap();
                        newmx[[scnd, thcol]] += 1.0;
                        let rmin = difexp.row(scnd).iter().cloned().fold(f64::INFINITY, f64::min);
                        let thrd = (0..nc).find(|&j| difexp[[scnd, j]] == rmin).unwrap();
                        newmx[[scnd, thrd]] -= 1.0;
                        newmx[[throw, thrd]] += 1.0;
                    }
                    newweb = newmx;
                }
            }
            h2_max.max(entropy(&newweb))
        };

        let h2_min = h2_min_entropy(&row_sums.to_vec(), &col_sums.to_vec());

        let h2_min = h2_min.min(h2_uncorr);
        let h2_max = h2_max.max(h2_uncorr);
        if (h2_max - h2_min).abs() < 1e-12 {
            return 0.0;
        }
        (h2_max - h2_uncorr) / (h2_max - h2_min)
    }

    /// Calculate d' (d-prime) specialization for each species in a bipartite interaction matrix.
    ///
    /// d′ quantifies how much a species deviates from using partners in proportion to their availability.
    /// It is 0 for complete generalists and 1 for complete specialists.
    ///
    /// # Arguments
    /// * `partition` - Whether to calculate d′ for rows (Parasites) or columns (Hosts).
    /// * `abundances` - Abundance of each species in the partition. If absent, then the background
    /// frequencies q as the column sums (or row sums) of the interaction matrix are calculated,
    /// normalized by the total
    ///
    /// # Returns
    /// A vector of `(species_name, optional<d_prime_value>)` tuples.
    pub fn d_prime(
        &self,
        partition: Partition,
        abundances: Option<&[f64]>,
    ) -> Vec<(String, Option<f64>)> {
        let (mat, names, num, q): (Array2<f64>, &Vec<String>, usize, Vec<f64>) = match partition {
            // parasites == rows
            Partition::Parasites => {
                let num_cols = self.inner.ncols();
                let q = if let Some(abuns) = abundances {
                    assert_eq!(abuns.len(), num_cols);
                    let total: f64 = abuns.iter().sum();
                    abuns.iter().map(|a| a / total).collect()
                } else {
                    let col_sums: Vec<f64> = self
                        .inner
                        .axis_iter(ndarray::Axis(1))
                        .map(|col| col.sum())
                        .collect();
                    let total: f64 = col_sums.iter().sum();
                    col_sums.iter().map(|c| c / total).collect()
                };
                (self.inner.clone(), &self.rownames, self.inner.nrows(), q)
            }
            // hosts == columns
            Partition::Hosts => {
                let transposed = self.inner.t();
                let num_rows = transposed.ncols();
                let q = if let Some(abuns) = abundances {
                    assert_eq!(abuns.len(), num_rows);
                    let total: f64 = abuns.iter().sum();
                    abuns.iter().map(|a| a / total).collect()
                } else {
                    let row_sums: Vec<f64> = transposed
                        .axis_iter(ndarray::Axis(1))
                        .map(|row| row.sum())
                        .collect();
                    let total: f64 = row_sums.iter().sum();
                    row_sums.iter().map(|r| r / total).collect()
                };
                (
                    transposed.into_owned(),
                    &self.colnames,
                    self.inner.ncols(),
                    q,
                )
            }
        };

        let col_sums: Vec<f64> = mat
            .axis_iter(ndarray::Axis(1))
            .map(|col| col.sum())
            .collect();
        let total_matrix_sum = mat.sum();

        (0..num)
            .map(|i| {
                let row = mat.row(i);
                let name = names[i].clone();
                let row_sum: f64 = row.sum();
                if row_sum == 0.0 {
                    return (name, None);
                }

                let d_raw: f64 = row
                    .iter()
                    .zip(q.iter())
                    .filter_map(|(&xj, &qj)| {
                        if xj > 0.0 && qj > 0.0 {
                            let pj = xj / row_sum;
                            Some(pj * (pj / qj).ln())
                        } else {
                            None
                        }
                    })
                    .sum();

                // d_min: greedy redistribution
                let expected: Vec<usize> = q
                    .iter()
                    .map(|&qj| (qj * row_sum).floor() as usize)
                    .collect();
                let mut residual = row_sum as usize - expected.iter().sum::<usize>();
                let mut x_new = expected.clone();

                while residual > 0 {
                    let mut best_idx = None;
                    let mut best_d = f64::INFINITY;

                    for j in 0..q.len() {
                        if abundances.is_none() && x_new[j] >= col_sums[j] as usize {
                            continue;
                        }

                        x_new[j] += 1;
                        let xsum = x_new.iter().sum::<usize>() as f64;
                        let d_candidate: f64 = x_new
                            .iter()
                            .zip(q.iter())
                            .filter_map(|(&xj, &qj)| {
                                if xj > 0 && qj > 0.0 {
                                    let pj = xj as f64 / xsum;
                                    Some(pj * (pj / qj).ln())
                                } else {
                                    None
                                }
                            })
                            .sum();
                        x_new[j] -= 1;

                        if d_candidate < best_d {
                            best_d = d_candidate;
                            best_idx = Some(j);
                        }
                    }

                    if let Some(idx) = best_idx {
                        x_new[idx] += 1;
                    }
                    residual -= 1;
                }

                let xsum = x_new.iter().sum::<usize>() as f64;
                let d_min: f64 = x_new
                    .iter()
                    .zip(q.iter())
                    .filter_map(|(&xj, &qj)| {
                        if xj > 0 && qj > 0.0 {
                            let pj = xj as f64 / xsum;
                            Some(pj * (pj / qj).ln())
                        } else {
                            None
                        }
                    })
                    .sum();

                let d_max = if abundances.is_some() {
                    let min_q = q
                        .iter()
                        .copied()
                        .filter(|&qj| qj > 0.0)
                        .fold(f64::INFINITY, f64::min);
                    if min_q == 0.0 || min_q == f64::INFINITY {
                        return (name, None);
                    }
                    (1.0 / min_q).ln()
                } else {
                    (total_matrix_sum / row_sum).ln()
                };

                if (d_max - d_min).abs() < 1e-12 {
                    return (name, None);
                }

                let d_prime = (d_raw - d_min) / (d_max - d_min);
                (name, Some(d_prime))
            })
            .collect()
    }

    /// Run the `LPAwb+` modularity algorithm on the matrix.
    ///
    /// # Arguments
    /// * `init_module_guess` - Optional initial guess for module assignments.
    ///
    /// # Returns
    /// A `LpaWbPlus` result representing module assignments.
    pub fn lpa_wb_plus(self, init_module_guess: Option<u32>) -> LpaWbPlus {
        modularity::lpa_wb_plus(&self.inner, init_module_guess)
    }

    /// Run the `DIRTLPAwb+` modularity algorithm on the matrix.
    ///
    /// # Arguments
    /// * `mini` - Minimum module size.
    /// * `reps` - Number of replicates to run.
    ///
    /// # Returns
    /// A `LpaWbPlus` result representing module assignments.
    pub fn dirt_lpa_wb_plus(&self, mini: u32, reps: u32) -> LpaWbPlus {
        modularity::dirt_lpa_wb_plus(&self.inner, mini, reps)
    }

    /// Run `DIRTLPAwb+` reproducibly from `seed`.
    pub fn dirt_lpa_wb_plus_seeded(&self, mini: u32, reps: u32, seed: u64) -> LpaWbPlus {
        modularity::dirt_lpa_wb_plus_seeded(&self.inner, mini, reps, seed)
    }

    /// Run `LPAwb+` reproducibly from `seed`.
    pub fn lpa_wb_plus_seeded(&self, init_module_guess: Option<u32>, seed: u64) -> LpaWbPlus {
        modularity::lpa_wb_plus_with_rng(&self.inner, init_module_guess, &mut StdRng::seed_from_u64(seed))
    }

    /// Sum of all values in the matrix.
    ///
    /// Equivalent to counting edges if the matrix is unweighted.
    ///
    /// # Returns
    /// The sum of matrix elements as `f64`.
    pub fn sum_matrix(&self) -> f64 {
        self.inner.sum()
    }

    /// Compute row sums of the interaction matrix.
    ///
    /// # Returns
    /// An array of row sums.
    pub fn row_sums(&self) -> ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>> {
        self.inner.sum_axis(Axis(1))
    }

    /// Compute column sums of the interaction matrix.
    ///
    /// # Returns
    /// An array of column sums.
    pub fn col_sums(&self) -> ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>> {
        self.inner.sum_axis(Axis(0))
    }

    /// Mean number of realized interactions per species (rows + cols combined).
    pub fn link_density(&self) -> f64 {
        let realized = self.inner.iter().filter(|&&v| v > 0.0).count();
        realized as f64 / (self.rownames.len() + self.colnames.len()) as f64
    }

    /// Mean d' specialisation across all species in a partition.
    ///
    /// Skips species with undefined d' (zero interactions). Returns `NaN` if
    /// no species have a defined d'.
    pub fn mean_d_prime(&self, partition: Partition) -> f64 {
        let values: Vec<f64> = self
            .d_prime(partition, None)
            .into_iter()
            .filter_map(|(_, v)| v)
            .collect();
        if values.is_empty() {
            return f64::NAN;
        }
        values.iter().sum::<f64>() / values.len() as f64
    }

    /// Generate a null matrix by randomly shuffling all elements (r00 null model).
    ///
    /// Preserves matrix shape and labels; randomises placement of all values.
    pub fn permute_null(&self) -> Self {
        let mut values: Vec<f64> = self.inner.iter().copied().collect();
        values.shuffle(&mut rand::thread_rng());
        let new_inner = Array2::from_shape_vec(self.inner.dim(), values).unwrap();
        InteractionMatrix {
            inner: new_inner,
            rownames: self.rownames.clone(),
            colnames: self.colnames.clone(),
        }
    }

    /// Generate one null matrix under `model` with the same row and column names.
    pub fn null(&self, model: NullModel, trades: usize, rng: &mut impl Rng) -> Result<Self, NullModelError> {
        Ok(InteractionMatrix {
            inner: null_matrix(&self.inner, model, trades, rng)?,
            rownames: self.rownames.clone(),
            colnames: self.colnames.clone(),
        })
    }

    /// Null-model significance test for any network-level statistic.
    ///
    /// Generates `n` null matrices in parallel under `model` (each from its own RNG seeded
    /// from `seed` and its index, so results are reproducible for a given seed), computes
    /// `stat` on each, and compares the observed value with the null distribution.
    /// `trades` is used only by the curveball model.
    pub fn null_test<F>(
        &self,
        n: usize,
        model: NullModel,
        trades: usize,
        seed: Option<u64>,
        stat: F,
    ) -> Result<PermutationTestResult, NullModelError>
    where
        F: Fn(&InteractionMatrix) -> f64 + Sync,
    {
        self.null_test_seeded(n, model, trades, seed, |m, _| stat(m))
    }

    /// As [`InteractionMatrix::null_test`], for statistics that are themselves stochastic
    /// (e.g. modularity). `stat` receives a seed: `seed` for the observed matrix, and a value
    /// drawn from each null matrix's generator for the nulls.
    pub fn null_test_seeded<F>(
        &self,
        n: usize,
        model: NullModel,
        trades: usize,
        seed: Option<u64>,
        stat: F,
    ) -> Result<PermutationTestResult, NullModelError>
    where
        F: Fn(&InteractionMatrix, u64) -> f64 + Sync,
    {
        let base: u64 = seed.unwrap_or_else(|| rand::thread_rng().random());
        let observed = stat(self, base);
        let scores: Result<Vec<f64>, NullModelError> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut rng = StdRng::seed_from_u64(base.wrapping_add(i as u64 + 1));
                let null = self.null(model, trades, &mut rng)?;
                let stat_seed: u64 = rng.random();
                Ok(stat(&null, stat_seed))
            })
            .collect();
        let scores: Vec<f64> = scores?.into_iter().filter(|v| !v.is_nan()).collect();
        Ok(permutation_stats(model, observed, &scores))
    }

    /// Null-model test of species-level d' for every species in `partition`.
    pub fn dprime_null_test(
        &self,
        partition: Partition,
        n: usize,
        model: NullModel,
        trades: usize,
        seed: Option<u64>,
    ) -> Result<Vec<DPrimeNullResult>, NullModelError> {
        let observed = self.d_prime(partition, None);
        let base: u64 = seed.unwrap_or_else(|| rand::thread_rng().random());
        let nulls: Result<Vec<Vec<(String, Option<f64>)>>, NullModelError> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut rng = StdRng::seed_from_u64(base.wrapping_add(i as u64 + 1));
                Ok(self.null(model, trades, &mut rng)?.d_prime(partition, None))
            })
            .collect();
        let nulls = nulls?;
        Ok(observed
            .into_iter()
            .enumerate()
            .map(|(k, (species, obs))| {
                let mut vals: Vec<f64> = nulls.iter().filter_map(|v| v[k].1).filter(|x| !x.is_nan()).collect();
                vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let m = vals.len() as f64;
                let mean_null = vals.iter().sum::<f64>() / m;
                let p_value = match obs {
                    Some(o) => (vals.iter().filter(|&&x| x >= o).count() as f64 + 1.0) / (m + 1.0),
                    None => f64::NAN,
                };
                DPrimeNullResult {
                    species,
                    observed: obs,
                    mean_null,
                    lower_null: quantile7(&vals, 0.025),
                    upper_null: quantile7(&vals, 0.975),
                    p_value,
                }
            })
            .collect())
    }

    /// Permutation test for binary NODF under the r00 null model.
    /// Kept for backwards compatibility; prefer [`InteractionMatrix::null_test`] with
    /// [`NullModel::Curveball`].
    pub fn nodf_permutation_test(&self, n: usize) -> PermutationTestResult {
        self.null_test(n, NullModel::R00, 0, None, |m| m.nodf(true, false, false).nodf)
            .expect("r00 cannot fail")
    }

    /// Permutation test for H2' under the r00 null model.
    /// Kept for backwards compatibility; prefer [`InteractionMatrix::null_test`] with
    /// [`NullModel::Patefield`].
    pub fn h2_permutation_test(&self, n: usize) -> PermutationTestResult {
        self.null_test(n, NullModel::R00, 0, None, |m| m.h2_prime())
            .expect("r00 cannot fail")
    }

    /// Compute Barber's matrix (modularity-related).
    ///
    /// Compute Barber's modularity matrix for this interaction matrix.
    pub fn barbers_matrix(&self) -> Array2<f64> {
        modularity::barbers_matrix(&self.inner)
    }
}

fn permutation_stats(null_model: NullModel, observed: f64, null_scores: &[f64]) -> PermutationTestResult {
    let n = null_scores.len() as f64;
    let mean_null = null_scores.iter().sum::<f64>() / n;
    let variance = null_scores.iter().map(|x| (x - mean_null).powi(2)).sum::<f64>() / (n - 1.0);
    let sd_null = variance.sqrt();
    let k = null_scores.iter().filter(|&&x| x >= observed).count() as f64;
    PermutationTestResult {
        null_model,
        observed,
        mean_null,
        sd_null,
        z: (observed - mean_null) / sd_null,
        p_value: (k + 1.0) / (n + 1.0),
        n_permutations: null_scores.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

        fn precision_f64(x: f64, decimals: u32) -> f64 {
            if x == 0. || decimals == 0 {
                0.
            } else {
                let shift = decimals as i32 - x.abs().log10().ceil() as i32;
                let shift_factor = 10_f64.powi(shift);

                (x * shift_factor).round() / shift_factor
            }
        }

        #[test]
        fn test_empty_matrix() {
            let data = Array2::<f64>::zeros((0, 0));
            let matrix = InteractionMatrix {
                inner: data,
                rownames: vec![],
                colnames: vec![],
            };

            let nodf_score = matrix.nodf(true, false, false);
            let ns = nodf_score.nodf;
            eprintln!("NODF: {}", ns);
            assert_eq!(ns, 0.0);
        }

        #[test]
        fn test_all_zero_matrix() {
            let data = Array2::<f64>::zeros((3, 3));
            let matrix = InteractionMatrix {
                inner: data,
                rownames: vec!["A".into(), "B".into(), "C".into()],
                colnames: vec!["X".into(), "Y".into(), "Z".into()],
            };

            let nodf_score = matrix.nodf(true, false, false);
            let ns = nodf_score.nodf;
            assert_eq!(ns, 0.0);
        }

        #[test]
        fn test_perfect_nested_matrix() {
            // This is a perfectly nested matrix
            let data = array![
                [1.0, 1.0, 1.0, 1.0], // Richest row
                [1.0, 1.0, 1.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0] // Poorest row
            ];

            let matrix = InteractionMatrix {
                inner: data,
                rownames: vec!["A".into(), "B".into(), "C".into(), "D".into()],
                colnames: vec!["W".into(), "X".into(), "Y".into(), "Z".into()],
            };

            let nodf_score = matrix.nodf(true, false, false);
            let ns = nodf_score.nodf;
            assert!(
                (ns - 100.0).abs() < 1e-6,
                "Expected 100, got {}",
                nodf_score.nodf
            );
        }

        #[test]
        fn test_no_nestedness_matrix() {
            // No nestedness: rows are disjoint
            let data = array![
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ];

            let matrix = InteractionMatrix {
                inner: data,
                rownames: vec!["A".into(), "B".into(), "C".into(), "D".into()],
                colnames: vec!["W".into(), "X".into(), "Y".into(), "Z".into()],
            };

            let nodf_score = matrix.nodf(true, false, false);
            let ns = nodf_score.nodf;
            assert_eq!(ns, 0.0);
        }

        #[test]
        fn test_unsorted_matrix_auto_sort() {
            // Unsorted matrix that needs sorting to show nestedness
            let data = array![
                [1.0, 1.0, 1.0, 1.0], // Should be first after sorting
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 0.0], // Should be second after sorting
            ];

            let matrix = InteractionMatrix {
                inner: data,
                rownames: vec!["D".into(), "B".into(), "C".into(), "A".into()],
                colnames: vec!["W".into(), "X".into(), "Y".into(), "Z".into()],
            };

            let nodf_score = matrix.nodf(true, false, false);
            assert!(
                nodf_score.nodf > 0.0 && nodf_score.nodf <= 100.0,
                "Expected NODF > 0, got {}",
                nodf_score.nodf
            );
        }

        #[test]
        fn test_unsorted_matrix_without_sorting() {
            let data = array![
                [1.0, 1.0, 1.0, 1.0], // Should be first after sorting
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 0.0], // Should be second after sorting
            ];

            let mut matrix = InteractionMatrix {
                inner: data.clone(),
                rownames: vec!["D".into(), "B".into(), "C".into(), "A".into()],
                colnames: vec!["W".into(), "X".into(), "Y".into(), "Z".into()],
            };

            let nodf_no_sort = matrix.nodf(false, false, false);
            let nodf_with_sort = {
                matrix.inner = data.clone(); // Reset matrix
                matrix.nodf(true, false, false)
            };

            // NODF with sort should be >= NODF without sort
            assert!(nodf_with_sort.nodf >= nodf_no_sort.nodf);
        }

        // a test from a subset of the safariland data (R bipartite package)
        //       [,1] [,2] [,3] [,4] [,5]
        // [1,]    1    0    1    0    0
        // [2,]    0    1    0    0    1
        // [3,]    0    0    0    0    0
        // [4,]    0    1    0    0    1
        // [5,]    0    0    1    0    1

        #[test]
        fn test_nodf_safariland() {
            let data = array![
                [1.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0, 1.0]
            ];

            let matrix = InteractionMatrix {
                inner: data,
                rownames: vec!["1".into(), "2".into(), "3".into(), "4".into(), "5".into()],
                colnames: vec!["1".into(), "2".into(), "3".into(), "4".into(), "5".into()],
            };

            let nodf_score = matrix.nodf(true, false, false);

            assert!(
                nodf_score.nodf == 12.5,
                "Expected NODF 12.5, got {}",
                nodf_score.nodf
            );
        }

        #[test]
        fn test_dprime_safariland() {
            let data = array![
                [673.0, 0.0, 110.0, 0.0, 0.0],
                [0.0, 154.0, 0.0, 0.0, 5.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 67.0, 0.0, 0.0, 5.0],
                [0.0, 0.0, 6.0, 0.0, 4.0]
            ];

            let matrix = InteractionMatrix {
                inner: data,
                rownames: vec![
                    "AC".into(),
                    "AA".into(),
                    "SP".into(),
                    "BD".into(),
                    "RE".into(),
                ],
                colnames: vec![
                    "PA".into(),
                    "BD".into(),
                    "RM".into(),
                    "TA".into(),
                    "SO".into(),
                ],
            };

            let dprimes = matrix.d_prime(Partition::Parasites, None);

            let expected = vec![
                Some(0.9721944),
                Some(0.7947769),
                None,
                Some(0.5547131),
                Some(0.5060748),
            ];

            for ((spp, dp), expected) in dprimes.iter().zip(expected) {
                assert!(
                    precision_f64(dp.unwrap_or(0.0), 2)
                        == precision_f64(expected.unwrap_or(0.0), 2),
                    "For {:?} got {:?}: expected {:?}",
                    spp,
                    dp,
                    expected
                );
            }
        }

        #[test]
        fn test_h2_integer_matches_bipartite() {
            // References from bipartite::H2fun(m, H2_integer = TRUE), deterministic in R.
            let m1 = InteractionMatrix {
                inner: array![[0.0, 4.0], [1.0, 5.0], [3.0, 0.0]],
                rownames: vec!["r1".into(), "r2".into(), "r3".into()],
                colnames: vec!["c1".into(), "c2".into()],
            };
            assert_eq!(precision_f64(m1.h2_prime(), 6), 0.661146);

            // Dominant cell: exercises the local refinement of H2_max.
            let m2 = InteractionMatrix {
                inner: array![
                    [23.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 3.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0, 0.0]
                ],
                rownames: vec!["r1".into(), "r2".into(), "r3".into(), "r4".into()],
                colnames: vec!["c1".into(), "c2".into(), "c3".into(), "c4".into()],
            };
            assert_eq!(precision_f64(m2.h2_prime(), 6), 0.927897);
        }

        #[test]
        fn test_h2_continuous_matches_bipartite() {
            // Reference: bipartite::H2fun(m, H2_integer = FALSE) = 0.525638281578
            let data = array![[1.5, 0.0, 2.25], [0.0, 3.1, 0.4], [0.2, 1.7, 1.1]];
            let matrix = InteractionMatrix {
                inner: data,
                rownames: vec!["a".into(), "b".into(), "c".into()],
                colnames: vec!["x".into(), "y".into(), "z".into()],
            };
            assert_eq!(precision_f64(matrix.h2_prime(), 4), 0.5256);
        }

        #[test]
        fn test_h2() {
            let data = array![
                [673.0, 0.0, 110.0, 0.0, 0.0],
                [0.0, 154.0, 0.0, 0.0, 5.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 67.0, 0.0, 0.0, 5.0],
                [0.0, 0.0, 6.0, 0.0, 4.0]
            ];

            let matrix = InteractionMatrix {
                inner: data,
                rownames: vec![
                    "AC".into(),
                    "AA".into(),
                    "SP".into(),
                    "BD".into(),
                    "RE".into(),
                ],
                colnames: vec![
                    "PA".into(),
                    "BD".into(),
                    "RM".into(),
                    "TA".into(),
                    "SO".into(),
                ],
            };

            let h2p = matrix.h2_prime();

            eprintln!("H2': {}", h2p);
            assert!(precision_f64(h2p, 2) == precision_f64(0.9804165, 2));
        }

        #[test]
        fn test_nodf_single_row_no_panic() {
            // A 1-row matrix must return 0.0 without panicking.
            // Before the || fix the early-return used &&, so a 1×N matrix
            // would reach `for i in 0..(1usize - 1)` safely (that's 0..0),
            // but a 0×N matrix would underflow. The || guard handles both.
            let data = array![[1.0, 0.0, 1.0]];
            let matrix = InteractionMatrix {
                inner: data,
                rownames: vec!["A".into()],
                colnames: vec!["X".into(), "Y".into(), "Z".into()],
            };
            let result = matrix.nodf(true, false, false);
            assert_eq!(result.nodf, 0.0);
        }

        #[test]
        fn test_nodf_single_col_no_panic() {
            let data = array![[1.0], [0.0], [1.0]];
            let matrix = InteractionMatrix {
                inner: data,
                rownames: vec!["A".into(), "B".into(), "C".into()],
                colnames: vec!["X".into()],
            };
            let result = matrix.nodf(true, false, false);
            assert_eq!(result.nodf, 0.0);
        }
}
