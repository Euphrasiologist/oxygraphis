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
use rand::seq::SliceRandom;
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
    /// Observed value of the metric.
    pub observed: f64,
    /// Mean of the null distribution.
    pub mean_null: f64,
    /// Standard deviation of the null distribution.
    pub sd_null: f64,
    /// P-value: proportion of null values >= observed.
    pub p_value: f64,
    /// Number of valid (non-NaN) permutations used.
    pub n_permutations: usize,
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
        let (parasites, hosts) = graph.get_parasite_host_from_graph();

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
    /// This implementation replicates the integer-aware behavior of the R `bipartite::H2fun` function
    /// with `H2_integer = TRUE`. It assumes the matrix contains only non-negative integer entries.
    ///
    /// The method measures network-level specialization based on deviation from maximum entropy
    /// (uncorrected Shannon entropy of interaction frequencies) and compares this with a maximum
    /// entropy matrix (subject to integer constraints) and a minimum entropy configuration derived
    /// by greedily filling the matrix while maintaining row and column marginal totals.
    ///
    /// Returned value is:
    /// ```
    /// H2' = (H2_max - H2_uncorr) / (H2_max - H2_min)
    /// ```
    /// where:
    /// - `H2_uncorr` is the observed entropy of the interaction matrix
    /// - `H2_max` is the entropy of an integer-approximated expected matrix under independence
    /// - `H2_min` is the entropy of a maximally specialized (minimum entropy) matrix
    ///
    /// Panics if the matrix contains non-integer values.
    ///
    /// # Returns
    /// `f64` — the H2' value, in the range [0, 1], where 1 indicates maximum specialization.
    ///
    /// # Example
    /// ```rust
    /// let data = array![
    ///     [1.0, 0.0, 1.0],
    ///     [0.0, 2.0, 0.0],
    ///     [0.0, 1.0, 1.0]
    /// ];
    /// let matrix = InteractionMatrix::from(data);
    /// let h2p = matrix.h2_prime();
    /// assert!(h2p >= 0.0 && h2p <= 1.0);
    /// ```
    pub fn h2_prime(&self) -> f64 {
        let matrix = &self.inner;

        if matrix.iter().any(|&v| v.fract() != 0.0) {
            panic!("Matrix contains non-integer values. Set H2_integer = FALSE to bypass.");
        }

        let total: f64 = matrix.sum();
        let row_sums = matrix.sum_axis(Axis(1));
        let col_sums = matrix.sum_axis(Axis(0));

        // H2uncorr = entropy of original matrix
        let h2_uncorr = entropy(matrix);

        let expected = row_sums.clone()
    .insert_axis(Axis(1)) // shape (rows, 1)
    * col_sums.clone().insert_axis(Axis(0)) // shape (1, cols)
    / total;

        // Build integer-aware expected matrix
        let mut newweb = expected.mapv(f64::floor);
        let mut difexp = &expected - &newweb;
        let mut webfull = Array2::<bool>::from_elem(matrix.raw_dim(), false);

        while newweb.sum() < total {
            // Mark filled rows and columns
            for (i, sum) in newweb.sum_axis(Axis(1)).iter().enumerate() {
                if (*sum - row_sums[i]).abs() < 1e-6 {
                    for j in 0..matrix.ncols() {
                        webfull[[i, j]] = true;
                    }
                }
            }
            for (j, sum) in newweb.sum_axis(Axis(0)).iter().enumerate() {
                if (*sum - col_sums[j]).abs() < 1e-6 {
                    for i in 0..matrix.nrows() {
                        webfull[[i, j]] = true;
                    }
                }
            }

            let mut best_val = f64::MIN;
            let mut best_pos = None;
            for ((i, j), &val) in newweb.indexed_iter() {
                if !webfull[[i, j]] {
                    if val == difexp[[i, j]].floor() {
                        let diff = difexp[[i, j]];
                        if diff > best_val {
                            best_val = diff;
                            best_pos = Some((i, j));
                        }
                    }
                }
            }

            if let Some((i, j)) = best_pos {
                newweb[[i, j]] += 1.0;
                difexp[[i, j]] = expected[[i, j]] - newweb[[i, j]];
            } else {
                break;
            }
        }

        let mut h2_max = entropy(&newweb);

        // Local refinement
        if expected.iter().cloned().fold(f64::MIN, f64::max) > (1.0 / std::f64::consts::E) * total {
            for _ in 0..500 {
                let mut newmx = newweb.clone();
                let difexp = &expected - &newmx;

                let min_val = difexp.iter().cloned().fold(f64::INFINITY, f64::min);
                let mut best = (0, 0);
                for ((i, j), &val) in difexp.indexed_iter() {
                    if val == min_val
                        && newmx[[i, j]] == newmx.iter().cloned().fold(f64::MIN, f64::max)
                    {
                        best = (i, j);
                        break;
                    }
                }

                newmx[[best.0, best.1]] -= 1.0;

                let row_dif = difexp.row(best.0);
                let col_dif = difexp.column(best.1);

                let mr = row_dif.iter().cloned().fold(f64::MIN, f64::max);
                let mc = col_dif.iter().cloned().fold(f64::MIN, f64::max);

                if mr >= mc {
                    let scnd = row_dif
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                        .unwrap()
                        .0;
                    newmx[[best.0, scnd]] += 1.0;

                    let thrd = difexp
                        .column(scnd)
                        .iter()
                        .enumerate()
                        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                        .unwrap()
                        .0;
                    newmx[[thrd, scnd]] -= 1.0;
                    newmx[[thrd, best.1]] += 1.0;
                } else {
                    let scnd = col_dif
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                        .unwrap()
                        .0;
                    newmx[[scnd, best.1]] += 1.0;

                    let thrd = difexp
                        .row(scnd)
                        .iter()
                        .enumerate()
                        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                        .unwrap()
                        .0;
                    newmx[[scnd, thrd]] -= 1.0;
                    newmx[[best.0, thrd]] += 1.0;
                }

                newweb = newmx;
            }
        }

        let h2_max_improved = entropy(&newweb);
        if h2_max_improved > h2_max {
            h2_max = h2_max_improved;
        }

        // H2_min construction
        let mut newweb_min = Array2::<f64>::zeros(matrix.raw_dim());
        let mut rs_remaining = row_sums.to_vec();
        let mut cs_remaining = col_sums.to_vec();

        while rs_remaining.iter().sum::<f64>().round() != 0.0 {
            let (i, &rmax) = rs_remaining
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap();
            let (j, &cmax) = cs_remaining
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap();
            let minval = rmax.min(cmax);
            newweb_min[[i, j]] = minval;
            rs_remaining[i] -= minval;
            cs_remaining[j] -= minval;
        }

        let pnew = newweb_min.mapv(|x| x / newweb_min.sum());
        let h2_min = entropy(&pnew);

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

    /// Permutation significance test for NODF using the r00 null model.
    ///
    /// Shuffles matrix elements `n` times and computes NODF on each null matrix.
    /// P-value is the proportion of null NODF values ≥ the observed NODF.
    pub fn nodf_permutation_test(&self, n: usize) -> PermutationTestResult {
        let observed = self.nodf(true, false, false).nodf;

        let null_scores: Vec<f64> = (0..n)
            .into_par_iter()
            .map(|_| self.permute_null().nodf(true, false, false).nodf)
            .filter(|v| !v.is_nan())
            .collect();

        permutation_stats(observed, &null_scores)
    }

    /// Permutation significance test for H2' using the r00 null model.
    ///
    /// P-value is the proportion of null H2' values ≥ the observed H2'.
    pub fn h2_permutation_test(&self, n: usize) -> PermutationTestResult {
        let observed = self.h2_prime();

        let null_scores: Vec<f64> = (0..n)
            .into_par_iter()
            .map(|_| self.permute_null().h2_prime())
            .filter(|v| !v.is_nan())
            .collect();

        permutation_stats(observed, &null_scores)
    }

    /// Compute Barber's matrix (modularity-related).
    ///
    /// Compute Barber's modularity matrix for this interaction matrix.
    pub fn barbers_matrix(&self) -> Array2<f64> {
        modularity::barbers_matrix(&self.inner)
    }
}

fn permutation_stats(observed: f64, null_scores: &[f64]) -> PermutationTestResult {
    let n = null_scores.len() as f64;
    let mean_null = null_scores.iter().sum::<f64>() / n;
    let variance = null_scores
        .iter()
        .map(|x| (x - mean_null).powi(2))
        .sum::<f64>()
        / n;
    let sd_null = variance.sqrt();
    let p_value = null_scores.iter().filter(|&&x| x >= observed).count() as f64 / n;
    PermutationTestResult {
        observed,
        mean_null,
        sd_null,
        p_value,
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
