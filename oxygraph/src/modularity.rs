//! Compute the modularity of an (optionally weighted) interaction matrix.
//!
//! Original methods from:
//! https://github.com/sjbeckett/weighted-modularity-LPAwbPLUS
//!
//! It's pretty much a direct translation, as I wanted to ensure correctness
//! over rustiness.

use crate::int_matrix::{BarbersMatrixError, Matrix};
use crate::{sort::*, InteractionMatrix};
use ndarray::{Array, Array1, Array2, Axis};
use rand::seq::{IndexedRandom, IteratorRandom};
use rand::thread_rng;
use std::collections::HashSet;
use thiserror::Error;

/// An error type for LPAwb+.
#[derive(Error, Debug)]
pub enum LpaWbPlusError {
    #[error("Could not calculate Barbers Matrix.")]
    BarbersMatrix(#[from] BarbersMatrixError),
}

/// An error type for DIRTLpawb+.
#[derive(Error, Debug)]
pub enum DirtLpaWbError {
    Error(#[from] LpaWbPlusError),
}

impl std::fmt::Display for DirtLpaWbError {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        let dis = self;
        write!(fmt, "{}", dis)
    }
}

/// A struct just to hold the data from the output of the modularity
/// computation.
#[derive(Debug)]
pub struct LpaWbPlus {
    pub row_labels: Vec<Option<usize>>,
    pub column_labels: Vec<Option<usize>>,
    pub modularity: f64,
}

/// To plot the modules on an interaction plot, these three
/// pieces of data must be known. The rows, columns, and
/// the vector of modules.
#[derive(Debug, Clone)]
pub struct PlotData {
    pub rows: Array1<usize>,
    pub cols: Array1<usize>,
    pub modules: Vec<usize>,
}

impl LpaWbPlus {
    // TODO: a function to return the modules with the species in them
    pub fn modules(&self, int_mat: &mut InteractionMatrix) -> Vec<(Vec<String>, Vec<String>)> {
        // get the permutation order of the rows
        let array_from_row: Array1<usize> = Array::from(
            self.row_labels
                .iter()
                .map(|e| e.unwrap())
                .collect::<Vec<usize>>(),
        );
        let array_from_row_permutation =
            array_from_row.sort_axis_by(Axis(0), |i, j| array_from_row[i] < array_from_row[j]);

        // and the columns
        let array_from_col: Array1<usize> = Array::from(
            self.column_labels
                .iter()
                .map(|e| e.unwrap())
                .collect::<Vec<usize>>(),
        );
        let array_from_col_permutation =
            array_from_col.sort_axis_by(Axis(0), |i, j| array_from_col[i] < array_from_col[j]);

        // sort the rows/cols
        let rows = array_from_row.permute_axis(Axis(0), &array_from_row_permutation);
        let cols = array_from_col.permute_axis(Axis(0), &array_from_col_permutation);

        // find the number of modules
        let (mut uniq_rows, _) = rows.clone().into_raw_vec_and_offset();
        uniq_rows.sort();
        uniq_rows.dedup();

        // sort the original interaction matrix
        int_mat.inner = int_mat
            .inner
            .clone()
            .permute_axis(Axis(0), &array_from_row_permutation);
        int_mat.inner = int_mat
            .inner
            .clone()
            .permute_axis(Axis(1), &array_from_col_permutation);

        fn sort_strings_by_indices(strings: Vec<String>, indices: Vec<usize>) -> Vec<String> {
            // Pair the indices with the strings
            let mut paired: Vec<(usize, String)> =
                indices.into_iter().zip(strings.into_iter()).collect();

            // Sort the pairs by the indices
            paired.sort_by_key(|(index, _)| *index);

            // Extract the sorted strings
            paired.into_iter().map(|(_, string)| string).collect()
        }

        // and reorder the rows and cols of the interaction matrix
        int_mat.rownames =
            sort_strings_by_indices(int_mat.rownames.clone(), array_from_row_permutation.indices);

        int_mat.colnames =
            sort_strings_by_indices(int_mat.colnames.clone(), array_from_col_permutation.indices);

        // now get the modules. The logic for this actually resides in int_matrix.rs
        // TODO: fix up the redundancy in this code.

        let mut per_module: Vec<(Vec<String>, Vec<String>)> = Vec::new();
        // keep track of cumulative column & row sizes
        let mut cumulative_col_size = 0;
        let mut cumulative_row_size = 0;

        for module in 0..uniq_rows.len() {
            // get this row size and the previous row size information
            let row_size = rows.iter().filter(|e| **e == uniq_rows[module]).count();
            let prev_row_size = rows
                .iter()
                .filter(|e| **e == *uniq_rows.get(module - 1).unwrap_or(&module))
                .count();
            // and the same for the columns
            let col_size = cols.iter().filter(|e| **e == uniq_rows[module]).count();
            let prev_col_size = cols
                .iter()
                .filter(|e| **e == *uniq_rows.get(module - 1).unwrap_or(&module))
                .count();

            let hosts =
                int_mat.rownames[cumulative_row_size..cumulative_row_size + row_size].to_vec();
            let parasites =
                int_mat.colnames[cumulative_col_size..cumulative_col_size + col_size].to_vec();

            per_module.push((hosts, parasites));

            // as a by-product of the unwrap_or() on the .get() function above,
            // skip the first iteration in the cumulative sums.
            if module > 0 {
                cumulative_col_size += prev_col_size;
                cumulative_row_size += prev_row_size;
            }
        }

        per_module
    }
    /// Generate a plot.
    ///
    /// `int_mat` is the original interaction matrix used to
    /// generate the `LpaWbPlus` object
    pub fn plot(&mut self, mut int_mat: InteractionMatrix) -> Vec<(Vec<String>, Vec<String>)> {
        if int_mat.inner.nrows() > int_mat.inner.ncols() {
            // swap the row labels and col labels
            let rn = self.row_labels.clone();
            let cn = self.column_labels.clone();

            self.row_labels = cn;
            self.column_labels = rn;
        }

        let array_from_row: Array1<usize> = Array::from(
            self.row_labels
                .iter()
                .map(|e| e.unwrap())
                .collect::<Vec<usize>>(),
        );

        let array_from_row_permutation =
            array_from_row.sort_axis_by(Axis(0), |i, j| array_from_row[i] < array_from_row[j]);

        // and the columns
        let array_from_col: Array1<usize> = Array::from(
            self.column_labels
                .iter()
                .map(|e| e.unwrap())
                .collect::<Vec<usize>>(),
        );

        let array_from_col_permutation =
            array_from_col.sort_axis_by(Axis(0), |i, j| array_from_col[i] < array_from_col[j]);

        // sort the rows/cols
        let rows = array_from_row.permute_axis(Axis(0), &array_from_row_permutation);
        let cols = array_from_col.permute_axis(Axis(0), &array_from_col_permutation);

        // find the number of modules
        let (mut uniq_rows, _) = rows.clone().into_raw_vec_and_offset();
        uniq_rows.sort();
        uniq_rows.dedup();

        // sort the original interaction matrix
        int_mat.inner = int_mat
            .inner
            .permute_axis(Axis(0), &array_from_row_permutation);
        int_mat.inner = int_mat
            .inner
            .permute_axis(Axis(1), &array_from_col_permutation);

        // given the sort_strings_by_indices function below
        fn sort_strings_by_indices(strings: Vec<String>, indices: Vec<usize>) -> Vec<String> {
            // Pair the indices with the strings
            let mut paired: Vec<(usize, String)> =
                indices.into_iter().zip(strings.into_iter()).collect();
            // Sort the pairs by the indices
            paired.sort_by_key(|(index, _)| *index);
            // Extract the sorted strings
            paired.into_iter().map(|(_, string)| string).collect()
        }

        // and sort the rownames and colnames from the original interaction matrix
        int_mat.rownames =
            sort_strings_by_indices(int_mat.rownames.clone(), array_from_row_permutation.indices);
        int_mat.colnames =
            sort_strings_by_indices(int_mat.colnames.clone(), array_from_col_permutation.indices);

        let plot_data = PlotData {
            rows,
            cols,
            modules: uniq_rows,
        };

        int_mat.plot(1000, Some(plot_data.clone()));

        self.modules(&mut int_mat)
    }
}

/// Calculates the Barber's matrix for modularity calculation in a bipartite network.
///
/// # Parameters
/// - `matrix`: A reference to a `Matrix` (2D array of `f64`), representing the adjacency matrix of a bipartite graph.
///
/// # Returns
/// - A `Matrix` with the modularity adjustments applied, as specified by the Barber's matrix formula.
///
/// This function operates by subtracting the outer product of row and column sums, divided by
/// the total sum of `matrix`, from `matrix` itself. The result is returned as a new `Matrix`.
pub fn barbers_matrix(matrix: &Matrix) -> Matrix {
    // Calculate the sum of all elements in the matrix, which is used as the denominator.
    let total_sum = matrix.sum();

    // Compute the row sums as a 1D array and expand it to a column vector (nx1) for matrix operations.
    let row_sums = matrix.sum_axis(Axis(1)).insert_axis(Axis(1));

    // Compute the column sums as a 1D array and expand it to a row vector (1xm) for matrix operations.
    let col_sums = matrix.sum_axis(Axis(0)).insert_axis(Axis(0));

    // Compute the outer product of row_sums and col_sums, divided by total_sum.
    let outer_product = &row_sums.dot(&col_sums) / total_sum;

    // Calculate the Barber's matrix by subtracting the outer product from the original matrix.
    matrix - &outer_product
}

/// The DIRTLPAwb+ algorithm.
///
/// It is a wrapper of the LPAwb+ algorithm, exploring
/// more parameter space. Therefore for large graphs, will
/// take a lot longer to run.
pub fn dirt_lpa_wb_plus(
    matrix: &InteractionMatrix,
    mini: usize,
    reps: usize,
) -> Result<LpaWbPlus, DirtLpaWbError> {
    // initial modularity
    let mut a = lpa_wb_plus(matrix.clone(), None)?;
    // number of modules from this initial guess.
    let mut modules = a.row_labels.clone();
    modules.sort();
    modules.dedup();
    let module_no = modules.len();

    // now optimise over a small parameter space.
    if module_no - mini > 0 {
        for aa in mini..module_no {
            for _ in 0..reps {
                let b = lpa_wb_plus(matrix.clone(), Some(aa))?;
                let mut inner_modules = b.row_labels.clone();
                inner_modules.sort();
                inner_modules.dedup();

                if b.modularity > a.modularity {
                    a = b;
                }
            }
        }
    }
    Ok(a)
}

/// Label propagation algorithm for weighted bipartite networks that finds modularity.
///
/// The LPAwb+ algorithm.
///
/// Translated from the R code here with permission from the author:
/// Stephen Beckett ( https://github.com/sjbeckett/weighted-modularity-LPAwbPLUS )
/// Main function for the Label Propagation Algorithm with Weighted Bipartite Networks.
///
/// # Parameters
/// - `matrix`: The adjacency matrix representing the bipartite network.
/// - `initial_module_guess`: Optional initial guess for the number of modules; if `None`, each red label gets a unique initial label.
///
/// # Returns
/// - Tuple containing row labels, column labels, and the modularity score.
pub fn lpa_wb_plus(
    matrix: InteractionMatrix,
    initial_module_guess: Option<usize>,
) -> Result<LpaWbPlus, LpaWbPlusError> {
    let mut matrix = matrix.inner;
    // Determine if matrix should be transposed (red labels represent rows)
    let mut flipped = false;
    if matrix.nrows() > matrix.ncols() {
        matrix = matrix.t().to_owned();
        flipped = true;
    }

    let mat_sum = matrix.sum();
    let col_marginals = matrix.sum_axis(Axis(0));
    let row_marginals = matrix.sum_axis(Axis(1));
    let b_matrix = barbers_matrix(&matrix); // Generate Barber's matrix from adjacency matrix

    // Initialize labels
    let mut blue_labels: Vec<Option<usize>> = vec![None; matrix.ncols()];

    // Initialize red labels based on `initial_module_guess`
    let mut red_labels: Vec<Option<usize>> = if let Some(initial_guess) = initial_module_guess {
        let mut rng = thread_rng();
        (0..matrix.nrows())
            .map(|_| Some((1..=initial_guess + 1).choose(&mut rng).unwrap()))
            .collect()
    } else {
        (0..matrix.nrows()).map(Some).collect()
    };

    // Run Stage 1: Locally update labels to maximize modularity `Qb`
    let (new_red_labels, new_blue_labels, qb_now) = stage_one_lpa_wbdash(
        &row_marginals,
        &col_marginals,
        &matrix,
        &b_matrix,
        mat_sum,
        red_labels.clone(),
        blue_labels.clone(),
    );

    red_labels = new_red_labels.to_vec();
    blue_labels = new_blue_labels.to_vec();

    // Run Stage 2: Connect divisions to improve `Qb`, then re-run Stage 1 until `Qb` stabilizes
    let (final_red_labels, final_blue_labels, final_qb_now) = stage_two_lpa_wbdash(
        &row_marginals,
        &col_marginals,
        &matrix,
        &b_matrix,
        mat_sum,
        red_labels.clone(),
        blue_labels.clone(),
        qb_now,
    );

    // If the matrix was transposed, swap red and blue labels to correct orientation
    if flipped {
        std::mem::swap(&mut red_labels, &mut blue_labels);
    }

    // Convert labels to Array1 for returning in the expected format
    Ok(LpaWbPlus {
        row_labels: final_red_labels.to_vec(),
        column_labels: final_blue_labels.to_vec(),
        modularity: final_qb_now,
    })
}

/// Calculates the trace of a matrix, which is the sum of its diagonal elements.
///
/// # Parameters
/// - `matrix`: A reference to a `Matrix` (2D array of `f64`).
///
/// # Returns
/// - A `f64` value representing the trace of the matrix.
///
/// The trace is calculated by summing the elements along the main diagonal.
pub fn trace(matrix: &Matrix) -> f64 {
    // Sum of diagonal elements
    matrix.diag().sum()
}

/// Calculates the weighted modularity for a bipartite network based on Barber's matrix and labels.
/// This function uses equation 8 for modularity calculation.
///
/// # Parameters
/// - `b_matrix`: A reference to the `Matrix` (2D array of `f64`) representing Barber's matrix.
/// - `mat_sum`: The total sum of elements in `b_matrix`, as an `f64` value.
/// - `red_labels`: A slice of `Option<usize>` representing labels for the red nodes.
/// - `blue_labels`: A slice of `Option<usize>` representing labels for the blue nodes.
///
/// # Returns
/// - A `f64` value representing the calculated weighted modularity.
///
/// The function iterates over each red and blue label pair, calculates the Kronecker delta
/// based on label equality, and accumulates the weighted modularity based on `b_matrix` values.
pub fn weighted_modularity(
    b_matrix: &Matrix,
    mat_sum: f64,
    red_labels: &[Option<usize>],
    blue_labels: &[Option<usize>],
) -> f64 {
    let mut holdsum = 0.0;

    // Iterate over each label in `red_labels` and `blue_labels` to calculate the modularity
    for (rr, red_label) in red_labels.iter().enumerate() {
        for (cc, blue_label) in blue_labels.iter().enumerate() {
            // Calculate Kronecker delta: 1.0 if labels are equal, otherwise 0.0
            let kronecker_delta = if red_label == blue_label { 1.0 } else { 0.0 };

            // Update holdsum with the product of the matrix entry and the Kronecker delta
            holdsum += b_matrix[(rr, cc)] * kronecker_delta;
        }
    }

    // Divide accumulated sum by the total matrix sum to get the weighted modularity
    holdsum / mat_sum
}

/// Calculates the second form of weighted modularity for a bipartite network, based on equation 9.
///
/// # Parameters
/// - `b_matrix`: A reference to the `Matrix` (2D array of `f64`) representing Barber's matrix.
/// - `mat_sum`: The total sum of elements in `b_matrix`, as an `f64`.
/// - `red_labels`: A slice of `Option<usize>` representing labels for the red nodes.
/// - `blue_labels`: A slice of `Option<usize>` representing labels for the blue nodes.
///
/// # Returns
/// - A `f64` value representing the calculated weighted modularity.
///
/// The function constructs two indicator matrices (`labelmat1` and `labelmat2`) that map unique labels
/// to original labels in `red_labels` and `blue_labels`. Then it computes the modularity as the trace
/// of the product `LABELMAT1 * BMatrix * LABELMAT2`, divided by the total matrix sum.
pub fn weighted_modularity2(
    b_matrix: &Matrix,
    mat_sum: f64,
    red_labels: &[Option<usize>],
    blue_labels: &[Option<usize>],
) -> f64 {
    // Get unique values from red and blue labels
    let uni_red: Vec<Option<usize>> = red_labels
        .iter()
        .copied()
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    let uni_blue: Vec<Option<usize>> = blue_labels
        .iter()
        .copied()
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    let l_red = uni_red.len();
    let l_blue = uni_blue.len();

    // Initialize label matrices with zeros
    let mut labelmat1 = Array2::<f64>::zeros((l_red, red_labels.len()));
    let mut labelmat2 = Array2::<f64>::zeros((blue_labels.len(), l_blue));

    // Fill labelmat1 to map each unique red label to its position in red_labels
    for (col, &label) in red_labels.iter().enumerate() {
        if let Some(pos) = uni_red.iter().position(|&x| x == label) {
            labelmat1[(pos, col)] = 1.0;
        }
    }

    // Fill labelmat2 to map each unique blue label to its position in blue_labels
    for (row, &label) in blue_labels.iter().enumerate() {
        if let Some(pos) = uni_blue.iter().position(|&x| x == label) {
            labelmat2[(row, pos)] = 1.0;
        }
    }

    // Compute the modularity matrix product: LABELMAT1 * BMatrix * LABELMAT2
    let product = labelmat1.dot(b_matrix).dot(&labelmat2);

    // Calculate and return the trace of the product, divided by mat_sum
    trace(&product) / mat_sum
}

/// Stage One of the Label Propagation Algorithm with Weighted Bipartite Networks that Finds Modularity.
///
/// # Parameters
/// - `row_marginals`: The row marginals for each red node.
/// - `col_marginals`: The column marginals for each blue node.
/// - `matrix`: The adjacency matrix of the bipartite network.
/// - `b_matrix`: Barber's matrix derived from the adjacency matrix.
/// - `mat_sum`: The sum of all elements in the adjacency matrix.
/// - `red_labels`: Initial labels for the red nodes, can be updated within the function.
/// - `blue_labels`: Initial labels for the blue nodes, can be updated within the function.
///
/// # Returns
/// - Tuple containing updated red labels, blue labels, and the current modularity score.
pub fn stage_one_lpa_wbdash(
    row_marginals: &Array1<f64>,
    col_marginals: &Array1<f64>,
    matrix: &Matrix,
    b_matrix: &Matrix,
    mat_sum: f64,
    red_labels: Vec<Option<usize>>,
    blue_labels: Vec<Option<usize>>,
) -> (Vec<Option<usize>>, Vec<Option<usize>>, f64) {
    // Define lengths of blue and red label vectors
    let blue_label_length = blue_labels.len();
    let red_label_length = red_labels.len();

    // Initialize total degree containers for red and blue labels
    let mut total_red_degrees =
        vec![None; red_labels.iter().copied().flatten().max().unwrap_or(0) + 1];
    let mut total_blue_degrees = vec![None; blue_label_length.max(red_label_length)];

    // Fill up total degrees for red labels
    for (aa, &label) in red_labels.iter().enumerate() {
        if let Some(label) = label {
            total_red_degrees[label] = match total_red_degrees[label] {
                Some(degree) => Some(degree + row_marginals[aa]),
                None => Some(row_marginals[aa]),
            };
        }
    }

    // Fill up total degrees for blue labels if they are initially labeled
    if blue_labels.iter().any(|label| label.is_some()) {
        for (bb, &label) in blue_labels.iter().enumerate() {
            if let Some(label) = label {
                total_blue_degrees[label] = match total_blue_degrees[label] {
                    Some(degree) => Some(degree + col_marginals[bb]),
                    None => Some(col_marginals[bb]),
                };
            }
        }
    } else {
        // If blue labels are all initially unassigned, set their total degrees to 0
        total_blue_degrees.fill(Some(0.0));
    }

    // Call the local maximisation function to improve modularity
    let (new_red_labels, new_blue_labels, qb_now) = local_maximisation(
        row_marginals,
        col_marginals,
        matrix,
        b_matrix,
        mat_sum,
        &mut red_labels.clone(),
        &mut blue_labels.clone(),
        &mut total_red_degrees
            .clone()
            .into_iter()
            .map(|x| x.unwrap_or(0.0))
            .collect::<Vec<_>>(),
        &mut total_blue_degrees
            .clone()
            .into_iter()
            .map(|x| x.unwrap_or(0.0))
            .collect::<Vec<_>>(),
    );

    (new_red_labels, new_blue_labels, qb_now)
}

/// Stage Two of the Label Propagation Algorithm with Weighted Bipartite Networks that Finds Modularity.
///
/// # Parameters
/// - `row_marginals`: The row marginals for each red node.
/// - `col_marginals`: The column marginals for each blue node.
/// - `matrix`: The adjacency matrix of the bipartite network.
/// - `b_matrix`: Barber's matrix derived from the adjacency matrix.
/// - `mat_sum`: The sum of all elements in the adjacency matrix.
/// - `red_labels`: Initial mutable labels for the red nodes.
/// - `blue_labels`: Initial mutable labels for the blue nodes.
/// - `qb_now`: Initial modularity score.
///
/// # Returns
/// - Tuple containing updated red labels, blue labels, and the current modularity score.
pub fn stage_two_lpa_wbdash(
    row_marginals: &Array1<f64>,
    col_marginals: &Array1<f64>,
    matrix: &Matrix,
    b_matrix: &Matrix,
    mat_sum: f64,
    mut red_labels: Vec<Option<usize>>,
    mut blue_labels: Vec<Option<usize>>,
    mut qb_now: f64,
) -> (Array1<Option<usize>>, Array1<Option<usize>>, f64) {
    // Find initial divisions
    let mut divisions_found: HashSet<Option<usize>> = red_labels
        .iter()
        .chain(blue_labels.iter())
        .cloned()
        .collect();

    let mut iterate_flag = true;

    while iterate_flag {
        let mut combined_divisions_this_time = 0;
        let num_div = divisions_found.len();

        if num_div > 1 {
            // Convert divisions_found into a vector for indexing
            let divisions_found_vec: Vec<Option<usize>> = divisions_found.iter().cloned().collect();

            for div1_idx in 0..num_div - 1 {
                let mod1 = divisions_found_vec[div1_idx];

                for div2_idx in div1_idx + 1..num_div {
                    let mod2 = divisions_found_vec[div2_idx];

                    // Create new red and blue labels for testing
                    let mut check_red = red_labels.clone();
                    let mut check_blue = blue_labels.clone();

                    for red in check_red.iter_mut() {
                        if *red == mod1 {
                            *red = mod2;
                        }
                    }
                    for blue in check_blue.iter_mut() {
                        if *blue == mod1 {
                            *blue = mod2;
                        }
                    }

                    // Calculate modularity for the test configuration
                    let qq = weighted_modularity2(b_matrix, mat_sum, &check_red, &check_blue);

                    if qq > qb_now {
                        // Check for better modularity score by further division adjustments
                        let mut found_better = false;

                        for &division in &divisions_found_vec {
                            // Test moving all instances of `division` to `mod1` or `mod2`
                            let mut check_red2 = red_labels.clone();
                            let mut check_blue2 = blue_labels.clone();

                            for red in check_red2.iter_mut() {
                                if *red == division {
                                    *red = mod1;
                                }
                            }
                            for blue in check_blue2.iter_mut() {
                                if *blue == division {
                                    *blue = mod1;
                                }
                            }
                            if weighted_modularity2(b_matrix, mat_sum, &check_red2, &check_blue2)
                                > qq
                            {
                                found_better = true;
                            }

                            check_red2 = red_labels.clone();
                            check_blue2 = blue_labels.clone();
                            for red in check_red2.iter_mut() {
                                if *red == division {
                                    *red = mod2;
                                }
                            }
                            for blue in check_blue2.iter_mut() {
                                if *blue == division {
                                    *blue = mod2;
                                }
                            }
                            if weighted_modularity2(b_matrix, mat_sum, &check_red2, &check_blue2)
                                > qq
                            {
                                found_better = true;
                            }
                        }

                        if !found_better {
                            // Commit to the division merge as no better option was found
                            red_labels = check_red.clone();
                            blue_labels = check_blue.clone();
                            combined_divisions_this_time += 1;
                        }
                    }
                }
            }

            // If no divisions were combined, end the iteration
            if combined_divisions_this_time == 0 {
                iterate_flag = false;
            }
        } else {
            iterate_flag = false;
        }

        // Re-run StageOne_LPAwbdash to locally maximize modularity
        let (new_red_labels, new_blue_labels, new_qb_now) = stage_one_lpa_wbdash(
            row_marginals,
            col_marginals,
            matrix,
            b_matrix,
            mat_sum,
            red_labels.clone(),
            blue_labels.clone(),
        );

        red_labels = new_red_labels.to_vec();
        blue_labels = new_blue_labels.to_vec();
        qb_now = new_qb_now;

        // Update divisions_found based on the current red and blue labels
        divisions_found = red_labels
            .iter()
            .chain(blue_labels.iter())
            .cloned()
            .collect();
    }

    (Array1::from(red_labels), Array1::from(blue_labels), qb_now)
}

/// Performs local maximization for modularity in a bipartite network.
///
/// # Parameters
/// - `row_marginals`: The row marginals as an array.
/// - `col_marginals`: The column marginals as an array.
/// - `matrix`: The adjacency matrix of the bipartite network.
/// - `b_matrix`: Barber's matrix derived from the adjacency matrix.
/// - `mat_sum`: The total sum of elements in the adjacency matrix.
/// - `red_labels`: Mutable reference to the red node labels.
/// - `blue_labels`: Mutable reference to the blue node labels.
/// - `total_red_degrees`: Mutable reference to the total degrees of the red nodes.
/// - `total_blue_degrees`: Mutable reference to the total degrees of the blue nodes.
///
/// # Returns
/// - Tuple containing updated red labels, blue labels, and the current modularity score.
pub fn local_maximisation(
    row_marginals: &Array1<f64>,
    col_marginals: &Array1<f64>,
    matrix: &Matrix,
    b_matrix: &Matrix,
    mat_sum: f64,
    red_labels: &mut Vec<Option<usize>>,
    blue_labels: &mut Vec<Option<usize>>,
    total_red_degrees: &mut Vec<f64>,
    total_blue_degrees: &mut Vec<f64>,
) -> (Vec<Option<usize>>, Vec<Option<usize>>, f64) {
    // Initial modularity score for current partition
    let mut qb_after = weighted_modularity2(b_matrix, mat_sum, red_labels, blue_labels);

    if qb_after.is_nan() {
        qb_after = -999.9;
    }

    let mut iterate_flag = true;
    while iterate_flag {
        // Save current state for potential rollback
        let qb_before = qb_after;
        let old_red_labels = red_labels.clone();
        let old_blue_labels = blue_labels.clone();
        let old_total_red_degrees = total_red_degrees.clone();
        let old_total_blue_degrees = total_blue_degrees.clone();

        // Update blue node labels based on red node information
        let blue_label_choices: Vec<_> = red_labels
            .iter()
            .copied()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        for (bb, blue_label) in blue_labels.iter_mut().enumerate() {
            if let Some(bl_label) = *blue_label {
                total_blue_degrees[bl_label] -= col_marginals[bb];
            }

            let mut change_blue_label_test = Vec::with_capacity(blue_label_choices.len());
            for &choice in &blue_label_choices {
                let score = matrix
                    .column(bb)
                    .iter()
                    .zip(red_labels.iter())
                    .map(|(&m, &r_label)| if r_label == choice { m } else { 0.0 })
                    .sum::<f64>()
                    - col_marginals[bb] * total_red_degrees[choice.unwrap()] / mat_sum;
                change_blue_label_test.push(score);
            }

            let max_score = change_blue_label_test
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let max_indices: Vec<_> = change_blue_label_test
                .iter()
                .enumerate()
                .filter(|&(_, &score)| score == max_score)
                .map(|(i, _)| i)
                .collect();

            if let Some(&new_label_index) = max_indices.choose(&mut rand::thread_rng()) {
                *blue_label = blue_label_choices[new_label_index];
                let bl_label = blue_label.unwrap();
                if bl_label >= total_blue_degrees.len() {
                    total_blue_degrees.push(0.0);
                }
                total_blue_degrees[bl_label] += col_marginals[bb];
            }
        }

        // Update red node labels based on blue node information
        let red_label_choices: Vec<_> = blue_labels
            .iter()
            .copied()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        for (aa, red_label) in red_labels.iter_mut().enumerate() {
            if let Some(rd_label) = *red_label {
                total_red_degrees[rd_label] -= row_marginals[aa];
            }

            let mut change_red_label_test = Vec::with_capacity(red_label_choices.len());
            for &choice in &red_label_choices {
                let score = matrix
                    .row(aa)
                    .iter()
                    .zip(blue_labels.iter())
                    .map(|(&m, &b_label)| if b_label == choice { m } else { 0.0 })
                    .sum::<f64>()
                    - row_marginals[aa] * total_blue_degrees[choice.unwrap()] / mat_sum;
                change_red_label_test.push(score);
            }

            let max_score = change_red_label_test
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let max_indices: Vec<_> = change_red_label_test
                .iter()
                .enumerate()
                .filter(|&(_, &score)| score == max_score)
                .map(|(i, _)| i)
                .collect();

            if let Some(&new_label_index) = max_indices.choose(&mut rand::thread_rng()) {
                *red_label = red_label_choices[new_label_index];
                let rd_label = red_label.unwrap();
                if rd_label >= total_red_degrees.len() {
                    total_red_degrees.push(0.0);
                }
                total_red_degrees[rd_label] += row_marginals[aa];
            }
        }

        // Calculate the new modularity score
        qb_after = weighted_modularity(b_matrix, mat_sum, red_labels, blue_labels);

        // Check if modularity has improved, otherwise rollback to the previous state
        if qb_after <= qb_before {
            *red_labels = old_red_labels;
            *blue_labels = old_blue_labels;
            *total_red_degrees = old_total_red_degrees;
            *total_blue_degrees = old_total_blue_degrees;
            iterate_flag = false;
        }
    }

    (red_labels.clone(), blue_labels.clone(), qb_after)
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::InteractionMatrix;

    #[test]
    fn test_dirtlpawbplus() {
        // 20 x 20 matrix
        let mut int_mat = InteractionMatrix::new(5, 5);
        int_mat.rownames = (0..5).map(|e| format!("{}r", e)).collect();
        int_mat.colnames = (0..5).map(|e| format!("{}c", e)).collect();

        int_mat.inner = Array::from_shape_vec(
            (5, 5),
            vec![
                2.0, 3.0, 3.0, 3.0, 2.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 3.0, 1.0, 0.0, 2.0, 1.0,
                2.0, 3.0, 2.0, 2.0, 0.0, 0.0, 2.0, 2.0, 2.0,
            ],
        )
        .unwrap();

        let LpaWbPlus {
            modularity,
            row_labels,
            column_labels,
        } = dirt_lpa_wb_plus(&int_mat, 50000, 50000).unwrap();

        eprintln!("{:?}", row_labels);
        eprintln!("{:?}", column_labels);

        // this fails most of the time, sometimes it's what the author of the R code got.
        // I think this is because of the random nature of the algorithm, in the red label
        // initialisation.

        assert_eq!((modularity * 100.0).round(), 13.0)
    }
}
