//! A bipartite graph can be converted to an interaction
//! matrix, which is a binary matrix representing all the
//! possible combinations of hosts/parasites (or sites/species).

use crate::bipartite::BipartiteGraph;
use crate::modularity;
use crate::modularity::PlotData;
use crate::sort::*;
use crate::LpaWbPlus;
use crate::MARGIN_LR;
use calm_io::*;
use itertools::Itertools;
use ndarray::{Array2, ArrayBase, Axis, Dim, OwnedRepr};
use std::collections::BTreeMap;
use std::fmt;
use thiserror::Error;

/// Error returned when constructing Barber's matrix fails.
///
/// Usually caused by a matrix shape mismatch when reshaping rows/columns.
#[derive(Error, Debug)]
pub enum BarbersMatrixError {
    #[error("Could not coerce Barbers matrix into a 2 x 1 matrix.")]
    Error(#[from] ndarray::ShapeError),
}

/// A 2D interaction matrix using floating-point weights.
///
/// Internally represented by `ndarray::Array2<f64>`.
pub type Matrix = Array2<f64>;

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
}

impl InteractionMatrix {
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

        let perc_ints = bin_mat.sum() / no_poss_ints as f64;

        InteractionMatrixStats {
            weighted,
            no_rows,
            no_cols,
            no_poss_ints,
            perc_ints,
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
        let (parasites, hosts) = graph.get_parasite_host_from_graph();

        // Early return if graph is empty
        if parasites.is_empty() || hosts.is_empty() {
            return InteractionMatrix::new(0, 0);
        }

        let mut int_max = InteractionMatrix::new(parasites.len(), hosts.len());

        for (i, (n1, _)) in parasites.iter().enumerate() {
            for (j, (n2, _)) in hosts.iter().enumerate() {
                if let Some(e) = graph.0.find_edge(*n1, *n2) {
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
                    let is_assoc = self.inner[[parasite, host]] == 1.0;
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
    pub fn nodf(&mut self, order: bool, weighted: bool, wbinary: bool) -> NestedNODFResult {
        let nr = self.inner.nrows();
        let nc = self.inner.ncols();

        // return early here to avoid error
        if nr < 2 && nc < 2 {
            return NestedNODFResult {
                comm: self.inner.clone(),
                fill: 0.0,
                n_rows: 0.0,
                n_cols: 0.0,
                nodf: 0.0,
            };
        }

        if order {
            self.sort_by_decreasing_fill(weighted);
        }

        let bin_comm = self.inner.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
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
            let first_row = self.inner.row(i);
            for j in (i + 1)..nr {
                if rfill[i] <= rfill[j] || rfill[i] == 0 || rfill[j] == 0 {
                    continue;
                }

                valid_row_pairs += 1;
                let second_row = self.inner.row(j);

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
            let first_col = self.inner.column(i);
            for j in (i + 1)..nc {
                if cfill[i] <= cfill[j] || cfill[i] == 0 || cfill[j] == 0 {
                    continue;
                }

                valid_col_pairs += 1;
                let second_col = self.inner.column(j);

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
            comm: self.inner.clone(),
            fill: total_fill,
            n_rows,
            n_cols,
            nodf,
        }
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

    /// Compute Barber's matrix (modularity-related).
    ///
    /// # Returns
    /// * `Ok` with Barber's matrix (2D array).
    /// * `Err` if the shape is incorrect for calculation.
    pub fn barbers_matrix(
        &self,
    ) -> Result<ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>, BarbersMatrixError> {
        // compute row sums and turn into 2d array with 1 column
        let row_sums = &self
            .row_sums()
            .into_shape_with_order((self.rownames.len(), 1))?;
        // col sums remain as a 1d array.
        let col_sums = &self.col_sums();

        Ok(&self.inner - ((row_sums * col_sums) / self.sum_matrix()))
    }
}

#[cfg(test)]
mod tests {
    // bring everything in from above
    use super::*;

    #[cfg(test)]
    mod tests {
        use super::*;
        use ndarray::array;

        #[test]
        fn test_empty_matrix() {
            let data = Array2::<f64>::zeros((0, 0));
            let mut matrix = InteractionMatrix {
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
            let mut matrix = InteractionMatrix {
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

            let mut matrix = InteractionMatrix {
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

            let mut matrix = InteractionMatrix {
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

            let mut matrix = InteractionMatrix {
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

            let mut matrix = InteractionMatrix {
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
    }
}
