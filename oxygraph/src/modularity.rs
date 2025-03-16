//! Compute the modularity of an (optionally weighted) interaction matrix.
//!
//! Original methods from:
//! https://github.com/sjbeckett/weighted-modularity-LPAwbPLUS
//!
//! It's pretty much a direct translation, as I wanted to ensure correctness
//! over rustiness.

use crate::{sort::*, InteractionMatrix};
use std::collections::{BTreeMap, HashSet};

use ndarray::{Array, Array1, Array2, Axis};
use rand::{seq::IndexedRandom, Rng};
use rayon::prelude::*;

/// A struct just to hold the data from the output of the modularity
/// computation.
#[derive(Debug)]
pub struct LpaWbPlus {
    pub row_labels: Vec<Option<u32>>,
    pub column_labels: Vec<Option<u32>>,
    pub modularity: f64,
}

/// To plot the modules on an interaction plot, these three
/// pieces of data must be known. The rows, columns, and
/// the vector of modules.
#[derive(Debug, Clone)]
pub struct PlotData {
    pub rows: Array1<u32>,
    pub cols: Array1<u32>,
    pub modules: Vec<u32>,
}

impl LpaWbPlus {
    /// Generate a plot.
    ///
    /// `int_mat` is the original interaction matrix used to
    /// generate the `LpaWbPlus` object
    pub fn plot(
        &mut self,
        mut int_mat: InteractionMatrix,
    ) -> Option<BTreeMap<usize, Vec<(String, String)>>> {
        let array_from_row: Array1<u32> = Array::from(
            self.row_labels
                .iter()
                .map(|e| e.unwrap())
                .collect::<Vec<u32>>(),
        );

        let array_from_row_permutation =
            array_from_row.sort_axis_by(Axis(0), |i, j| array_from_row[i] < array_from_row[j]);

        // and the columns
        let array_from_col: Array1<u32> = Array::from(
            self.column_labels
                .iter()
                .map(|e| e.unwrap())
                .collect::<Vec<u32>>(),
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

        int_mat.plot(1000, Some(plot_data.clone()))
    }
}

fn division(red_labels: &[Option<u32>], blue_labels: &[Option<u32>]) -> HashSet<u32> {
    let red_set: HashSet<u32> = red_labels.iter().filter_map(|&x| x).collect();
    let blue_set: HashSet<u32> = blue_labels.iter().filter_map(|&x| x).collect();

    red_set.intersection(&blue_set).cloned().collect()
}

fn barbers_matrix(matrix: &Array2<f64>) -> Array2<f64> {
    let row_sums = matrix.sum_axis(Axis(1)).insert_axis(Axis(1)); // Column vector
    let col_sums = matrix.sum_axis(Axis(0)).insert_axis(Axis(0)); // Row vector
    let total_sum = matrix.sum();

    matrix - (&row_sums.dot(&col_sums) / total_sum)
}

fn weighted_modularity(
    matrix: &Array2<f64>,
    mat_sum: f64,
    red_labels: &[Option<u32>],
    blue_labels: &[Option<u32>],
) -> f64 {
    let mut hold_sum = 0.0;

    for (rr, &red) in red_labels.iter().enumerate() {
        for (cc, &blue) in blue_labels.iter().enumerate() {
            let kronecker_delta = if red == blue { 1.0 } else { 0.0 };
            hold_sum += matrix[(rr, cc)] * kronecker_delta;
        }
    }

    hold_sum / mat_sum
}

fn trace(matrix: &Array2<f64>) -> f64 {
    matrix.diag().sum()
}

fn dedup_preserve_order(vec: Vec<Option<u32>>) -> Vec<Option<u32>> {
    let mut seen = HashSet::new();
    vec.into_iter()
        .filter(|item| seen.insert(*item)) // Insert returns true if the item was not already present
        .collect()
}

fn weighted_modularity2(
    matrix: &Array2<f64>,
    mat_sum: f64,
    red_labels: &[Option<u32>],
    blue_labels: &[Option<u32>],
) -> f64 {
    let uni_red: Vec<Option<u32>> = dedup_preserve_order(red_labels.to_vec());
    let uni_blue: Vec<Option<u32>> = dedup_preserve_order(blue_labels.to_vec());
    let l_red = uni_red.len();
    let l_blue = uni_blue.len();

    let mut label_mat1 = Array2::<f64>::zeros((l_red, red_labels.len()));
    let mut label_mat2 = Array2::<f64>::zeros((blue_labels.len(), l_blue));

    for (i, &label) in red_labels.iter().enumerate() {
        if let Some(_) = label {
            if let Some(pos) = uni_red.iter().position(|&x| x == label) {
                label_mat1[(pos, i)] = 1.0;
            }
        }
    }

    for (i, &label) in blue_labels.iter().enumerate() {
        if let Some(_) = label {
            if let Some(pos) = uni_blue.iter().position(|&x| x == label) {
                label_mat2[(i, pos)] = 1.0;
            }
        }
    }

    let intermediate = label_mat1.dot(matrix).dot(&label_mat2);

    trace(&intermediate) / mat_sum
}

fn dedup_preserve_order2<T: Eq + std::hash::Hash + Copy>(vec: &[Option<T>]) -> Vec<T> {
    let mut seen = std::collections::HashSet::new();
    let mut result = Vec::new();
    for &item in vec.iter() {
        if let Some(val) = item {
            if seen.insert(val) {
                result.push(val);
            }
        }
    }
    result
}

fn local_maximisation(
    matrix: &Array2<f64>,
    row_marginals: &Array1<f64>,
    col_marginals: &Array1<f64>,
    b_matrix: &Array2<f64>,
    mat_sum: f64,
    red_labels: &mut Vec<Option<u32>>,
    blue_labels: &mut Vec<Option<u32>>,
    total_red_degrees: &mut Vec<Option<f64>>,
    total_blue_degrees: &mut Vec<Option<f64>>,
) -> (Vec<Option<u32>>, Vec<Option<u32>>, f64) {
    let mut qb_after = weighted_modularity2(b_matrix, mat_sum, red_labels, blue_labels);
    if qb_after.is_nan() {
        qb_after = -999.0;
    }

    let mut iterate_flag = true;
    while iterate_flag {
        let qb_before = qb_after;
        let old_red_labels = red_labels.clone();
        let old_blue_labels = blue_labels.clone();
        let old_trd = total_red_degrees.clone();
        let old_tbd = total_blue_degrees.clone();

        // Update blue labels
        let blue_label_choices: Vec<_> = dedup_preserve_order2(red_labels);

        for (bb, blue_label) in blue_labels.iter_mut().enumerate() {
            if let Some(blue_label_val) = blue_label {
                if let Some(val) = total_blue_degrees[*blue_label_val as usize] {
                    total_blue_degrees[*blue_label_val as usize] = Some(val - col_marginals[bb]);
                }
            }

            let change_blue_label_test: Vec<Option<f64>> = blue_label_choices
                .iter()
                .map(|&choice| {
                    let sum_val: f64 = red_labels
                        .iter()
                        .zip(matrix.column(bb).iter())
                        .filter(|(r, _)| r.map_or(false, |x| x == choice))
                        .map(|(_, &val)| val)
                        .sum();
                    let degree = total_red_degrees[choice as usize].unwrap_or(0.0);
                    Some(sum_val - col_marginals[bb] * degree / mat_sum)
                })
                .collect();

            let max_val =
                change_blue_label_test
                    .iter()
                    .filter_map(|&e| e)
                    .fold(
                        f64::NEG_INFINITY,
                        |a, b| if b.is_nan() { a } else { a.max(b) },
                    );

            let best_labels: Vec<_> = blue_label_choices
                .iter()
                .enumerate()
                .filter(|(i, _)| change_blue_label_test[*i] == Some(max_val))
                .map(|(_, &val)| val)
                .collect();
            *blue_label = Some(*best_labels.choose(&mut rand::rng()).unwrap());

            // print blue labels during iteration, getting around immutable borrow

            // copy this logic to the rest of the function
            if let Some(blue_label_val) = blue_label {
                if *blue_label_val as usize >= total_blue_degrees.len() {
                    total_blue_degrees.resize_with(*blue_label_val as usize + 1, || None);
                    total_blue_degrees[*blue_label_val as usize] = Some(col_marginals[bb]);
                } else {
                    total_blue_degrees[*blue_label_val as usize] = Some(
                        // FIXME: again, how to handle the unwrap here
                        total_blue_degrees[*blue_label_val as usize].unwrap_or(0.0)
                            + col_marginals[bb],
                    );
                }
            }
        }

        // Update red labels
        let red_label_choices: Vec<_> = dedup_preserve_order2(blue_labels);

        for (aa, red_label) in red_labels.iter_mut().enumerate() {
            if let Some(red_label_val) = red_label {
                if let Some(val) = total_red_degrees[*red_label_val as usize] {
                    total_red_degrees[*red_label_val as usize] = Some(val - row_marginals[aa]);
                }
            }

            let change_red_label_test: Vec<Option<f64>> = red_label_choices
                .iter()
                .map(|&choice| {
                    let sum_val: f64 = blue_labels
                        .iter()
                        .zip(matrix.row(aa).iter())
                        .filter(|(r, _)| r.map_or(false, |x| x == choice))
                        .map(|(_, &val)| val)
                        .sum();
                    // FIXME: handle unwrap
                    let degree = total_blue_degrees[choice as usize].unwrap_or(0.0);
                    Some(sum_val - row_marginals[aa] * degree / mat_sum)
                })
                .collect();

            let max_val =
                change_red_label_test
                    .iter()
                    .filter_map(|&e| e)
                    .fold(
                        f64::NEG_INFINITY,
                        |a, b| if b.is_nan() { a } else { a.max(b) },
                    );

            let best_labels: Vec<_> = red_label_choices
                .iter()
                .enumerate()
                .filter(|(i, _)| change_red_label_test[*i] == Some(max_val))
                .map(|(_, &val)| val)
                .collect();
            *red_label = Some(*best_labels.choose(&mut rand::rng()).unwrap());

            if let Some(red_label_val) = red_label {
                if *red_label_val as usize >= total_red_degrees.len() {
                    total_red_degrees.resize_with(*red_label_val as usize + 1, || None);
                    total_red_degrees[*red_label_val as usize] = Some(0.0);
                } else {
                    total_red_degrees[*red_label_val as usize] = Some(
                        // FIXME: again, how to handle the unwrap here
                        total_red_degrees[*red_label_val as usize].unwrap_or(0.0)
                            + row_marginals[aa],
                    );
                }
            }
        }

        qb_after = weighted_modularity(b_matrix, mat_sum, red_labels, blue_labels);

        if qb_after <= qb_before {
            *red_labels = old_red_labels;
            *blue_labels = old_blue_labels;
            *total_red_degrees = old_trd;
            *total_blue_degrees = old_tbd;
            iterate_flag = false;
        }
    }

    (red_labels.clone(), blue_labels.clone(), qb_after)
}

fn stage_one_lpa_wb_dash(
    row_marginals: &Array1<f64>,
    col_marginals: &Array1<f64>,
    matrix: &Array2<f64>,
    b_matrix: &Array2<f64>,
    mat_sum: f64,
    red_labels: &mut Vec<Option<u32>>,
    blue_labels: &mut Vec<Option<u32>>,
) -> (Vec<Option<u32>>, Vec<Option<u32>>, f64) {
    let blue_label_length = blue_labels.len();
    let red_label_length = red_labels.len();

    let max_red_label = red_labels.iter().filter_map(|&x| x).max().unwrap_or(0);
    let max_label_len = std::cmp::max(blue_label_length, red_label_length);

    // Initialise total degrees with None
    let mut total_red_degrees = vec![None; (max_red_label + 1) as usize];
    let mut total_blue_degrees = vec![None; max_label_len];

    // Fill up total red degrees
    for (aa, &label_option) in red_labels.iter().enumerate() {
        if let Some(label) = label_option {
            let label_idx = label as usize;
            if label_idx >= total_red_degrees.len() {
                total_red_degrees.resize_with(label_idx + 1, || None);
            }

            total_red_degrees[label_idx] =
                Some(total_red_degrees[label_idx].unwrap_or(0.0) + row_marginals[aa]);
        }
    }

    // Fill up total blue degrees if any blue label isn't None
    if blue_labels.iter().any(|&label| label.is_some()) {
        for (bb, &label_option) in blue_labels.iter().enumerate() {
            if let Some(label) = label_option {
                let label_idx = label as usize;
                if label_idx >= total_blue_degrees.len() {
                    total_blue_degrees.resize_with(label_idx + 1, || None);
                }

                total_blue_degrees[label_idx] =
                    Some(total_blue_degrees[label_idx].unwrap_or(0.0) + col_marginals[bb]);
            }
        }
    } else {
        // all blue labels are None, so set zero degrees
        total_blue_degrees = vec![Some(0.0); max_label_len];
    }

    let (updated_red_labels, updated_blue_labels, qb_now) = local_maximisation(
        matrix,
        row_marginals,
        col_marginals,
        b_matrix,
        mat_sum,
        red_labels,
        blue_labels,
        &mut total_red_degrees,
        &mut total_blue_degrees,
    );

    (updated_red_labels, updated_blue_labels, qb_now)
}

fn stage_two_lpa_wb_dash(
    row_marginals: &Array1<f64>,
    col_marginals: &Array1<f64>,
    matrix: &Array2<f64>,
    b_matrix: &Array2<f64>,
    mat_sum: f64,
    mut red_labels: Vec<Option<u32>>,
    mut blue_labels: Vec<Option<u32>>,
    mut qb_now: f64,
) -> (Vec<Option<u32>>, Vec<Option<u32>>, f64) {
    let mut divisions_found: Vec<u32> = division(&red_labels, &blue_labels).into_iter().collect();
    let mut num_div = divisions_found.len();
    let mut iterate_flag = true;

    while iterate_flag {
        let mut combined_divisions_this_time = 0;

        if num_div > 1 {
            for div1_idx in 0..(num_div - 1) {
                let mod1 = divisions_found[div1_idx];

                for div2_idx in (div1_idx + 1)..num_div {
                    let mod2 = divisions_found[div2_idx];

                    let mut check_red = red_labels.clone();
                    let mut check_blue = blue_labels.clone();

                    // Merge mod1 into mod2 for both red and blue labels
                    for label in check_red.iter_mut() {
                        if label == &Some(mod1) {
                            *label = Some(mod2);
                        }
                    }

                    for label in check_blue.iter_mut() {
                        if label == &Some(mod1) {
                            *label = Some(mod2);
                        }
                    }

                    let qq = weighted_modularity2(b_matrix, mat_sum, &check_red, &check_blue);

                    if qq > qb_now {
                        let mut found_better = false;

                        for &div in &divisions_found {
                            // Try replacing divisions with mod1
                            let mut check_red2 = red_labels.clone();
                            let mut check_blue2 = blue_labels.clone();

                            for label in check_red2.iter_mut() {
                                if label == &Some(div) {
                                    *label = Some(mod1);
                                }
                            }

                            for label in check_blue2.iter_mut() {
                                if label == &Some(div) {
                                    *label = Some(mod1);
                                }
                            }

                            if weighted_modularity2(b_matrix, mat_sum, &check_red2, &check_blue2)
                                > qq
                            {
                                found_better = true;
                            }

                            // Try replacing divisions with mod2
                            let mut check_red2 = red_labels.clone();
                            let mut check_blue2 = blue_labels.clone();

                            for label in check_red2.iter_mut() {
                                if label == &Some(div) {
                                    *label = Some(mod2);
                                }
                            }

                            for label in check_blue2.iter_mut() {
                                if label == &Some(div) {
                                    *label = Some(mod2);
                                }
                            }

                            if weighted_modularity2(b_matrix, mat_sum, &check_red2, &check_blue2)
                                > qq
                            {
                                found_better = true;
                            }
                        }

                        if !found_better {
                            red_labels = check_red;
                            blue_labels = check_blue;
                            combined_divisions_this_time += 1;
                        }
                    }
                }
            }

            if combined_divisions_this_time == 0 {
                iterate_flag = false;
            }
        } else {
            iterate_flag = false;
        }

        let (new_red_labels, new_blue_labels, new_qb_now) = stage_one_lpa_wb_dash(
            row_marginals,
            col_marginals,
            matrix,
            b_matrix,
            mat_sum,
            &mut red_labels,
            &mut blue_labels,
        );

        red_labels = new_red_labels;
        blue_labels = new_blue_labels;
        qb_now = new_qb_now;

        divisions_found = division(&red_labels, &blue_labels).into_iter().collect();
        num_div = divisions_found.len();
    }

    (red_labels, blue_labels, qb_now)
}

pub fn lpa_wb_plus(input_matrix: &Array2<f64>, initial_module_guess: Option<u32>) -> LpaWbPlus {
    let mut matrix = input_matrix.clone();
    let mut flipped = false;

    // Flip matrix if rows > cols
    if matrix.nrows() > matrix.ncols() {
        matrix = matrix.t().to_owned();
        flipped = true;
    }

    let mat_sum = matrix.sum();
    let col_marginals = matrix.sum_axis(Axis(0));
    let row_marginals = matrix.sum_axis(Axis(1));
    let b_matrix = barbers_matrix(&matrix);

    // Initialise labels
    let mut blue_labels: Vec<Option<u32>> = vec![None; matrix.ncols()];

    let mut red_labels: Vec<Option<u32>> = if initial_module_guess.is_none() {
        (0..matrix.nrows()).map(|x| Some(x as u32)).collect()
    } else {
        // Sample randomly from 1 to (initial_module_guess + 1)
        let mut rng = rand::rng();
        let max_label = initial_module_guess.unwrap() + 1;
        (0..matrix.nrows())
            .map(|_| Some(rng.random_range(1..=max_label)))
            .collect()
    };

    // Run Phase 1
    let (red_labels, blue_labels, qb_now) = stage_one_lpa_wb_dash(
        &row_marginals,
        &col_marginals,
        &matrix,
        &b_matrix,
        mat_sum,
        &mut red_labels,
        &mut blue_labels,
    );

    // Run Phase 2
    let (mut red_labels, mut blue_labels, qb_now) = stage_two_lpa_wb_dash(
        &row_marginals,
        &col_marginals,
        &matrix,
        &b_matrix,
        mat_sum,
        red_labels,
        blue_labels,
        qb_now,
    );

    // Swap labels back if we flipped the matrix
    if flipped {
        std::mem::swap(&mut red_labels, &mut blue_labels);
    }

    LpaWbPlus {
        row_labels: red_labels,
        column_labels: blue_labels,
        modularity: qb_now,
    }
}

pub fn dirt_lpa_wb_plus(matrix: &Array2<f64>, mini: u32, reps: u32) -> LpaWbPlus {
    let LpaWbPlus {
        row_labels,
        column_labels,
        modularity,
    } = lpa_wb_plus(matrix, None);
    let mut best_row_labels = row_labels.clone();
    let mut best_column_labels = column_labels.clone();
    let mut best_modularity = modularity.clone();

    let divisions_found: HashSet<u32> = division(&row_labels, &column_labels);
    let mods = divisions_found.len() as u32;

    if mods > mini {
        for aa in mini..=mods {
            // Parallelize over reps
            let results: Vec<LpaWbPlus> = (0..reps)
                .into_par_iter()
                .map(|_| lpa_wb_plus(matrix, Some(aa)))
                .collect();

            // Find the best result from parallel reps
            for LpaWbPlus {
                row_labels,
                column_labels,
                modularity,
            } in results
            {
                if modularity > best_modularity {
                    best_row_labels = row_labels;
                    best_column_labels = column_labels;
                    best_modularity = modularity;
                }
            }
        }
    }

    LpaWbPlus {
        row_labels: best_row_labels,
        column_labels: best_column_labels,
        modularity: best_modularity,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_division() {
        let red_labels = vec![
            Some(5),
            Some(6),
            Some(2),
            Some(6),
            Some(1),
            Some(3),
            Some(1),
            Some(6),
            Some(4),
        ];
        let blue_labels = vec![
            Some(5),
            Some(6),
            Some(5),
            Some(1),
            Some(6),
            Some(6),
            Some(1),
            Some(6),
            Some(6),
            Some(5),
            Some(6),
            Some(3),
            Some(3),
            Some(1),
            Some(2),
            Some(3),
            Some(4),
            Some(6),
            Some(3),
            Some(6),
            Some(6),
            Some(6),
            Some(6),
            Some(6),
            Some(6),
            Some(6),
            Some(5),
        ];

        let expected: HashSet<u32> = vec![1, 2, 3, 4, 5, 6].into_iter().collect();
        let result = division(&red_labels, &blue_labels);

        assert_eq!(result, expected);
    }

    use ndarray::array;

    #[test]
    fn test_barbers_matrix() {
        let matrix = array![[1.0, 2.0], [3.0, 4.0]];
        let result = barbers_matrix(&matrix);
        let expected = array![[-0.2, 0.2], [0.2, -0.2]];

        let x = precision_f64(result[[0, 0]], 2);
        let y = precision_f64(expected[[0, 0]], 2);

        assert_eq!(x, y);
    }

    #[test]
    fn test_weighted_modularity() {
        let o_matrix = array![[1.0, 2.0], [3.0, 4.0]];
        let matrix = o_matrix.clone();
        let bmatrix = barbers_matrix(&matrix);
        // sum the matrix
        let mat_sum = o_matrix.sum();
        let red_labels = vec![Some(1), Some(2)];
        let blue_labels = vec![Some(1), Some(2)];

        let result = weighted_modularity(&bmatrix, mat_sum, &red_labels, &blue_labels);
        let expected = -0.04;

        assert_eq!(precision_f64(result, 2), expected);
    }

    #[test]
    fn test_trace() {
        let matrix = array![[1.0, 2.0], [3.0, 4.0]];
        let result = trace(&matrix);
        let expected = 5.0;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_weighted_modularity2() {
        let o_matrix = array![[1.0, 2.0], [3.0, 4.0]];
        let matrix = o_matrix.clone();
        let bmatrix = barbers_matrix(&matrix);
        // sum the matrix
        let mat_sum = o_matrix.sum();
        let red_labels = vec![Some(1), Some(2)];
        let blue_labels = vec![Some(1), Some(2)];
        let result = weighted_modularity2(&bmatrix, mat_sum, &red_labels, &blue_labels);

        let expected = -0.04;
        assert_eq!(precision_f64(result, 2), expected);
    }

    #[test]
    fn test_local_maximisation() {
        let matrix = array![
            [673.0, 0.0, 110.0, 0.0, 0.0],
            [0.0, 154.0, 0.0, 0.0, 5.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 67.0, 0.0, 0.0, 5.0]
        ];

        let b_matrix = barbers_matrix(&matrix);

        // row_marginals:  783 159 0 72
        let row_marginals = array![783.0, 159.0, 0.0, 72.0];
        // col_marginals:  673 221 110 0 10
        let col_marginals = array![673.0, 221.0, 110.0, 0.0, 10.0];
        // matsum:  1014
        let mat_sum = 1014.0;
        // redlabels:
        let mut red_labels: Vec<Option<u32>> = vec![Some(4), Some(5), Some(4), Some(3)];
        // bluelabels:  NA NA NA NA NA
        let mut blue_labels: Vec<Option<u32>> = vec![None, None, None, None, None];
        // totalreddegrees:  783 NA 159 NA 72
        let mut total_red_degrees = vec![None, None, None, Some(72.0), Some(783.0), Some(159.0)];
        // totalbluedegrees:  0 0 0 0 0
        let mut total_blue_degrees = vec![Some(0.0), Some(0.0), Some(0.), Some(0.), Some(0.)];

        let result = local_maximisation(
            &matrix,
            &row_marginals,
            &col_marginals,
            &b_matrix,
            mat_sum,
            &mut red_labels,
            &mut blue_labels,
            &mut total_red_degrees,
            &mut total_blue_degrees,
        );

        // multiple expectations here
        eprintln!("red: {:?}", result.0);
        eprintln!("blue: {:?}", result.1);
        let expected_qb_now = 0.3518259;
        assert_eq!(
            precision_f64(result.2, 2),
            precision_f64(expected_qb_now, 2)
        );

        // this is sometimes true, due to randomness
        assert_eq!(red_labels, vec![Some(4), Some(5), Some(4), Some(5)]);
        assert_eq!(
            blue_labels,
            vec![Some(4), Some(5), Some(4), Some(4), Some(5)]
        );
    }

    #[test]
    fn test_stage_one_lpa_wb_dash() {
        let matrix = array![
            [673.0, 0.0, 110.0, 0.0, 0.0],
            [0.0, 154.0, 0.0, 0.0, 5.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 67.0, 0.0, 0.0, 5.0]
        ];

        let b_matrix = barbers_matrix(&matrix);

        let row_marginals = array![783.0, 159.0, 0.0, 72.0];
        let col_marginals = array![673.0, 221.0, 110.0, 0.0, 10.0];
        let mat_sum = 1014.0;

        let mut red_labels: Vec<Option<u32>> = vec![Some(4), Some(5), Some(4), Some(3)];
        let mut blue_labels: Vec<Option<u32>> = vec![None, None, None, None, None];

        let (result_red_labels, result_blue_labels, qb_now) = stage_one_lpa_wb_dash(
            &row_marginals,
            &col_marginals,
            &matrix,
            &b_matrix,
            mat_sum,
            &mut red_labels,
            &mut blue_labels,
        );

        // Expected output should be based on your LOCALMAXIMISATION logic
        eprintln!("result_red_labels: {:?}", result_red_labels);
        eprintln!("result_blue_labels: {:?}", result_blue_labels);
        eprintln!("qb_now: {:?}", qb_now);

        // Example assertions (these may need adjusting based on your outputs)
        let expected_qb_now = 0.3518259;
        assert_eq!(precision_f64(qb_now, 2), precision_f64(expected_qb_now, 2));
    }

    #[test]
    fn test_stage_two_lpa_wb_dash() {
        let matrix = array![
            [673.0, 0.0, 110.0, 0.0, 0.0],
            [0.0, 154.0, 0.0, 0.0, 5.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 67.0, 0.0, 0.0, 5.0]
        ];

        let b_matrix = barbers_matrix(&matrix);

        let row_marginals = array![783.0, 159.0, 0.0, 72.0];
        let col_marginals = array![673.0, 221.0, 110.0, 0.0, 10.0];
        let mat_sum = 1014.0;

        let mut red_labels: Vec<Option<u32>> = vec![Some(4), Some(5), Some(4), Some(3)];
        let mut blue_labels: Vec<Option<u32>> = vec![None, None, None, None, None];

        // Run stage one first to get initial labels and qb_now
        let (red_labels_stage1, blue_labels_stage1, qb_now_stage1) = stage_one_lpa_wb_dash(
            &row_marginals,
            &col_marginals,
            &matrix,
            &b_matrix,
            mat_sum,
            &mut red_labels,
            &mut blue_labels,
        );

        // Now run stage two with output from stage one
        let (result_red_labels, result_blue_labels, qb_now_stage2) = stage_two_lpa_wb_dash(
            &row_marginals,
            &col_marginals,
            &matrix,
            &b_matrix,
            mat_sum,
            red_labels_stage1,
            blue_labels_stage1,
            qb_now_stage1,
        );

        eprintln!("result_red_labels: {:?}", result_red_labels);
        eprintln!("result_blue_labels: {:?}", result_blue_labels);
        eprintln!("qb_now_stage2: {:?}", qb_now_stage2);

        // Example assertions, values depend on the data and labels
        let expected_qb_now = 0.3518259; // Adjust this as per your R output
        assert_eq!(
            precision_f64(qb_now_stage2, 2),
            precision_f64(expected_qb_now, 2)
        );
    }

    #[test]
    fn test_lpa_wb_plus() {
        let matrix = array![
            [673.0, 0.0, 110.0, 0.0, 0.0],
            [0.0, 154.0, 0.0, 0.0, 5.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 67.0, 0.0, 0.0, 5.0]
        ];

        let LpaWbPlus {
            row_labels,
            column_labels,
            modularity,
        } = lpa_wb_plus(&matrix, None);

        eprintln!("result_red_labels: {:?}", row_labels);
        eprintln!("result_blue_labels: {:?}", column_labels);
        eprintln!("qb_now: {:?}", modularity);

        // Adjust expected_qb_now according to your R outputs!
        let expected_qb_now = 0.3518259;
        assert_eq!(
            precision_f64(modularity, 2),
            precision_f64(expected_qb_now, 2)
        );

        // You can add assertions for red_labels and blue_labels if stable
    }
    #[test]
    fn test_dirt_lpa_wb_plus() {
        let matrix = array![
            [673.0, 0.0, 110.0, 0.0, 0.0],
            [0.0, 154.0, 0.0, 0.0, 5.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 67.0, 0.0, 0.0, 5.0]
        ];

        let LpaWbPlus {
            row_labels,
            column_labels,
            modularity,
        } = dirt_lpa_wb_plus(&matrix, 4, 10);

        eprintln!("result_red_labels: {:?}", row_labels);
        eprintln!("result_blue_labels: {:?}", column_labels);
        eprintln!("qb_now: {:?}", modularity);

        // You can set an expected value based on R output if deterministic
        let expected_qb_now = 0.3518259;
        assert_eq!(
            precision_f64(modularity, 2),
            precision_f64(expected_qb_now, 2)
        );
    }
}
