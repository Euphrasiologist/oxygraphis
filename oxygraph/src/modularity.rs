//! Compute the modularity of an (optionally weighted) interaction matrix.
//!
//! Original methods from:
//! https://github.com/sjbeckett/weighted-modularity-LPAwbPLUS
//!
//! It's pretty much a direct translation, as I wanted to ensure correctness
//! over rustiness.

use crate::{sort::*, InteractionMatrix};
use core::f64::NAN;
use ndarray::{Array, Array1, Array2, ArrayBase, Axis, Dim, OwnedRepr, ViewRepr};
use rand::seq::SliceRandom;
use std::collections::HashSet;

// Will need an error type for this module in the future.
// TODO:
// Implement DIRT_LPA_wb_plus

/// A struct just to hold the data from the output of the modularity
/// computation.
pub struct LpaWbPlus {
    pub row_labels: Vec<usize>,
    pub column_labels: Vec<usize>,
    pub modularity: f64,
}

/// To plot the modules on an interaction plot, these three
/// pieces of data must be known. The rows, columns, and
/// the vector of modules.
pub struct PlotData<'a> {
    pub rows: ArrayBase<ViewRepr<&'a usize>, Dim<[usize; 1]>>,
    pub cols: ArrayBase<ViewRepr<&'a usize>, Dim<[usize; 1]>>,
    pub modules: Vec<usize>,
}

impl LpaWbPlus {
    /// Generate a plot.
    ///
    /// `int_mat` is the original interaction matrix used to
    /// generate the `LpaWbPlus` object
    pub fn plot(&self, mut int_mat: InteractionMatrix) {
        // get the permutation order of the rows
        let array_from_row: Array1<usize> = Array::from(self.row_labels.clone());
        let array_from_row_permutation =
            array_from_row.sort_axis_by(Axis(0), |i, j| array_from_row[i] < array_from_row[j]);

        // and the columns
        let array_from_col: Array1<usize> = Array::from(self.column_labels.clone());
        let array_from_col_permutation =
            array_from_col.sort_axis_by(Axis(0), |i, j| array_from_col[i] < array_from_col[j]);

        // sort the rows/cols
        let rows = array_from_row
            .clone()
            .permute_axis(Axis(0), &array_from_row_permutation);
        let cols = array_from_col
            .clone()
            .permute_axis(Axis(0), &array_from_col_permutation);

        // find the number of modules
        let mut uniq_rows = rows.clone().into_raw_vec();
        uniq_rows.sort();
        uniq_rows.dedup();

        // sort the original interaction matrix

        int_mat.inner = int_mat
            .inner
            .permute_axis(Axis(0), &array_from_row_permutation);
        int_mat.inner = int_mat
            .inner
            .permute_axis(Axis(1), &array_from_col_permutation);

        // view so we don't have to clone the thing
        let rows_view = rows.view();
        let cols_view = cols.view();

        int_mat.plot(
            1000,
            Some(PlotData {
                rows: rows_view,
                cols: cols_view,
                modules: uniq_rows,
            }),
        );
    }
}

/// Label propagation algorithm for weighted bipartite networks that finds modularity.
/// Contains the LPAwb+ and the DIRTLPAwb+ algorithms
///
/// Translated from the R code here with permission from the author:
/// Stephen Beckett ( https://github.com/sjbeckett/weighted-modularity-LPAwbPLUS )
///
/// TODO: No initial module guess to start with.
pub fn lba_wb_plus(mut matrix: InteractionMatrix) -> LpaWbPlus {
    // Make sure the smallest matrix dimension represent the red labels by making
    // them the rows (if matrix is transposed here, will be transposed back at the end)
    let row_len = matrix.rownames.len();
    let col_len = matrix.colnames.len();
    // not sure if I need this yet
    let mut flipped = 0;

    if row_len > col_len {
        matrix = matrix.transpose();
        flipped = 1;
    }

    let mat_sum = matrix.sum_matrix();
    let col_marginals = matrix.col_sums();
    let row_marginals = matrix.row_sums();
    let b_matrix = matrix.barbers_matrix();

    // initialise labels
    // columns are hosts, rows are parasites
    let mut blue_labels: Array1<f64> = Array1::zeros(col_len);
    blue_labels.fill(NAN);
    let red_labels: Array1<f64> = Array::linspace(0.0, row_len as f64 - 1.0, row_len);

    // Run phase 1
    let out_list = stage_one_lpa_wbdash(
        row_marginals.clone(),
        col_marginals.clone(),
        matrix.inner.clone(),
        b_matrix.clone(),
        mat_sum,
        red_labels,
        blue_labels,
        0, // hack the first iteration.
    );

    let mut red_labels = out_list.0;
    let mut blue_labels = out_list.1;
    let qb_now = out_list.2;

    // Run phase 2
    let out_list2 = stage_two_lpa_wbdash(
        row_marginals,
        col_marginals,
        matrix.inner,
        &b_matrix,
        mat_sum,
        &mut red_labels,
        &mut blue_labels,
        qb_now,
    );

    let row_labels: Vec<usize> = out_list2
        .0
        .into_raw_vec()
        .iter()
        .map(|e| *e as usize)
        .collect();
    let column_labels: Vec<usize> = out_list2
        .1
        .into_raw_vec()
        .iter()
        .map(|e| *e as usize)
        .collect();
    let modularity = out_list2.2;

    if flipped == 1 {
        return LpaWbPlus {
            row_labels: column_labels,
            column_labels: row_labels,
            modularity,
        };
    }

    LpaWbPlus {
        row_labels,
        column_labels,
        modularity,
    }
}

/// Returns the sum of the diagonal of a 2D Array<f64>.
fn trace(matrix: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>) -> f64 {
    matrix.diag().sum()
}

/// Weighted modularity computation.
fn weighted_modularity(
    b_matrix: &ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
    mat_sum: f64,
    red_labels: &Array1<f64>,
    blue_labels: &Array1<f64>,
) -> f64 {
    let mut hold_sum = 0f64;

    for rr in 0..red_labels.len() {
        for cc in 0..blue_labels.len() {
            let kroneckerdelta = red_labels[rr] == blue_labels[cc];
            if kroneckerdelta {
                hold_sum = hold_sum + (b_matrix[[rr, cc]] * 1.0);
            } else {
                hold_sum = hold_sum + (b_matrix[[rr, cc]] * 0.0);
            }
        }
    }

    hold_sum / mat_sum
}

/// Second weighted modularity calculation.
fn weighted_modularity_2(
    b_matrix: &ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
    mat_sum: f64,
    red_labels: Array1<f64>,
    blue_labels: Array1<f64>,
    _iteration: i32, // for debugging
) -> f64 {
    // create the unique red elements
    let mut uniq_red = red_labels.clone().into_raw_vec();
    uniq_red.sort_by(|a, b| a.partial_cmp(b).unwrap());
    uniq_red.dedup();
    let uniq_red_len = uniq_red.len();

    // create the unique blue elements
    let mut uniq_blue = blue_labels.clone().into_raw_vec();
    uniq_blue.sort_by(|a, b| {
        let cmp = a.partial_cmp(b);
        match cmp {
            Some(c) => c,
            // treat NANs as less.
            None => std::cmp::Ordering::Greater,
        }
    });
    uniq_blue.dedup();
    let number_nans_uniq_blue = uniq_blue.iter().filter(|e| e.is_nan()).count();
    let mut uniq_blue_len = uniq_blue.len();
    if number_nans_uniq_blue == uniq_blue_len {
        uniq_blue = vec![NAN];
        uniq_blue_len = 1;
    }

    // initiate the labelled matrices
    let mut label_mat_1: Array2<f64> = Array2::zeros((uniq_red_len, red_labels.len()));
    let mut label_mat_2: Array2<f64> = Array2::zeros((blue_labels.len(), uniq_blue_len));

    // populate the labelled matrices
    for aa in 0..red_labels.len() - 1 {
        let aa_index = uniq_red.iter().position(|&x| x == red_labels[aa]).unwrap();
        label_mat_1[[aa_index, aa]] = 1.0;
    }

    for aa in 0..blue_labels.len() {
        let aa_index = uniq_blue
            .iter()
            .position(|&x| x == blue_labels[aa])
            .unwrap_or(0);
        label_mat_2[[aa, aa_index]] = 1.0;
    }

    let inner_matrix = (label_mat_1.dot(&b_matrix.dot(&label_mat_2))) / mat_sum;

    trace(inner_matrix)
}

/// Stage one of the label propagation algorithm.
fn stage_one_lpa_wbdash(
    row_marginals: ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>>,
    col_marginals: ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>>,
    matrix: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
    b_matrix: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
    mat_sum: f64,
    mut red_labels: Array1<f64>,
    mut blue_labels: Array1<f64>,
    iteration_counter: i32,
) -> (
    ndarray::ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>>,
    ndarray::ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>>,
    f64,
) {
    // label lengths
    let blue_label_length = blue_labels.len();
    let red_label_length = red_labels.len();

    // red and blue 1D degree arrays.
    let mut total_red_degrees: Array1<f64> = Array1::zeros(
        *red_labels
            .iter()
            // or is it b.partial_cmp(a)..?
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap() as usize
            + 1,
    );
    total_red_degrees.fill(NAN);

    // annoying bit of logic here. R is much more concise in this regard.
    let mut total_blue_degrees: Array1<f64>;
    if blue_label_length > red_label_length {
        total_blue_degrees = Array1::zeros(blue_label_length);
        total_blue_degrees.fill(NAN);
    } else {
        total_blue_degrees = Array1::zeros(red_label_length);
        total_blue_degrees.fill(NAN);
    }

    // now fill up these containers according to current labels
    // red!
    for aa in 0..red_label_length {
        let red_deg_index = total_red_degrees[red_labels[aa] as usize];

        if red_deg_index.is_nan() {
            total_red_degrees[red_labels[aa] as usize] = row_marginals[aa];
        } else {
            total_red_degrees[red_labels[aa] as usize] =
                total_red_degrees[red_labels[aa] as usize] + row_marginals[aa];
        }
    }

    // blue!
    let no_blue_nans = blue_labels.iter().filter(|&e| e.is_nan()).count();

    if no_blue_nans != blue_label_length {
        for bb in 0..blue_label_length {
            if total_blue_degrees[blue_labels[bb] as usize].is_nan() {
                total_blue_degrees[blue_labels[bb] as usize] = col_marginals[bb];
            } else {
                total_blue_degrees[blue_labels[bb] as usize] =
                    total_blue_degrees[blue_labels[bb] as usize] + col_marginals[bb];
            }
        }
    } else {
        total_blue_degrees.fill(0.0);
    }

    // locally maximise modularity!
    let out_list = local_maximisation(
        row_marginals,
        col_marginals,
        matrix,
        &b_matrix,
        mat_sum,
        &mut red_labels,
        &mut blue_labels,
        total_red_degrees,
        total_blue_degrees,
        iteration_counter,
    );

    let red_labels = out_list.0;
    let blue_labels = out_list.1;
    let qb_now = out_list.2;

    (red_labels, blue_labels, qb_now)
}

/// Hacky intersection based on casting f64 to usizes.
fn division(red_labels: &mut Array1<f64>, blue_labels: &mut Array1<f64>) -> Vec<usize> {
    let red_label_set: HashSet<_> = red_labels.iter().map(|e| *e as usize).collect();
    let blue_label_set: HashSet<_> = blue_labels.iter().map(|e| *e as usize).collect();

    red_label_set
        .intersection(&blue_label_set)
        .map(|e| *e)
        .collect()
}

/// Stage two of the LPAwbplus algorithm.
fn stage_two_lpa_wbdash(
    row_marginals: ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>>,
    col_marginals: ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>>,
    matrix: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
    b_matrix: &ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
    mat_sum: f64,
    red_labels: &mut Array1<f64>,
    blue_labels: &mut Array1<f64>,
    mut qb_now: f64,
) -> (
    ndarray::ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>>,
    ndarray::ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>>,
    f64,
) {
    let mut divisions_found = division(red_labels, blue_labels);
    let mut num_div = divisions_found.len();

    let mut iterate_flag = true;
    let mut iteration_counter = 1;

    while iterate_flag {
        let mut combined_divisions_this_time = 0;

        if num_div > 1 {
            //
            for div1check in 0..num_div - 1 {
                let mod_1 = divisions_found[div1check];
                for div2check in div1check + 1..num_div {
                    // red
                    let mut check_red = red_labels.clone();
                    let check_red_index = red_labels.iter().find(|e| **e == mod_1 as f64).unwrap();
                    check_red[*check_red_index as usize] = divisions_found[div2check] as f64;
                    // blue
                    let mut check_blue = blue_labels.clone();
                    let check_blue_index =
                        blue_labels.iter().find(|e| **e == mod_1 as f64).unwrap();
                    check_blue[*check_blue_index as usize] = divisions_found[div2check] as f64;

                    // calc modularity
                    let qq = weighted_modularity_2(
                        b_matrix,
                        mat_sum,
                        check_red.clone(),
                        check_blue.clone(),
                        0,
                    );

                    if qq > qb_now {
                        let mut found_better = false;
                        for aa in 0..num_div {
                            // red
                            let mut check_red_2 = red_labels.clone();
                            let check_red_2_index = red_labels
                                .iter()
                                .find(|e| **e == divisions_found[aa] as f64)
                                .unwrap();
                            check_red_2[*check_red_2_index as usize] = mod_1 as f64;
                            // blue
                            let mut check_blue_2 = blue_labels.clone();
                            let check_blue_2_index = blue_labels
                                .iter()
                                .find(|e| **e == divisions_found[aa] as f64)
                                .unwrap();
                            check_blue_2[*check_blue_2_index as usize] = mod_1 as f64;

                            let first_qq = weighted_modularity_2(
                                b_matrix,
                                mat_sum,
                                check_red_2,
                                check_blue_2,
                                0,
                            );

                            if first_qq > qq {
                                found_better = true;
                            }

                            // red
                            let mut check_red_2 = red_labels.clone();
                            let check_red_2_index = red_labels
                                .iter()
                                .find(|e| **e == divisions_found[aa] as f64)
                                .unwrap();
                            check_red_2[*check_red_2_index as usize] =
                                divisions_found[div2check] as f64;
                            // blue
                            let mut check_blue_2 = blue_labels.clone();
                            let check_blue_2_index = blue_labels
                                .iter()
                                .find(|e| **e == divisions_found[aa] as f64)
                                .unwrap();
                            check_blue_2[*check_blue_2_index as usize] =
                                divisions_found[div2check] as f64;

                            let second_qq = weighted_modularity_2(
                                b_matrix,
                                mat_sum,
                                check_red_2,
                                check_blue_2,
                                0,
                            );

                            if second_qq > qq {
                                found_better = true;
                            }
                        }
                        if !found_better {
                            *red_labels = check_red;
                            *blue_labels = check_blue;
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

        // oof lots of cloning.
        let out_list = stage_one_lpa_wbdash(
            row_marginals.clone(),
            col_marginals.clone(),
            matrix.clone(),
            b_matrix.clone(),
            mat_sum,
            red_labels.clone(),
            blue_labels.clone(),
            iteration_counter,
        );

        *red_labels = out_list.0;
        *blue_labels = out_list.1;
        qb_now = out_list.2;
        divisions_found = division(red_labels, blue_labels);
        num_div = divisions_found.len();

        iteration_counter += 1;
    }

    (red_labels.clone(), blue_labels.clone(), qb_now)
}

/// Locally maximise inside `stage_one_lpa_wbdash`.
fn local_maximisation(
    row_marginals: ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>>,
    col_marginals: ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>>,
    matrix: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
    b_matrix: &ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
    mat_sum: f64,
    red_labels: &mut Array1<f64>,
    blue_labels: &mut Array1<f64>,
    mut total_red_degrees: ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>>,
    mut total_blue_degrees: ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>>,
    outer_iteration_counter: i32,
) -> (
    ndarray::ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>>,
    ndarray::ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>>,
    f64,
) {
    // find the score for the current partition
    let mut qb_after = weighted_modularity_2(
        &b_matrix,
        mat_sum,
        red_labels.to_owned(),
        blue_labels.to_owned(),
        outer_iteration_counter,
    );

    // not sure why we do this.
    if qb_after.is_nan() {
        qb_after = -999.0;
    }

    // turn this to false once we have optimised.
    let mut iterate_flag = true;
    let mut _iteration_counter = 1;

    while iterate_flag {
        // Save old information
        let qb_before = qb_after;

        let old_red_labels = red_labels.clone();
        let old_blue_labels = blue_labels.clone();
        let old_trd = total_red_degrees.clone();
        let old_tbd = total_blue_degrees.clone();

        // update blue nodes using red node information.
        // this is a mental unique() implementation by the way.
        // clone the original
        let blue_label_choices_arr = red_labels.clone();
        // turn to vec
        let mut blue_label_choices_vec = blue_label_choices_arr.into_raw_vec();
        // sort
        blue_label_choices_vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // dedup
        blue_label_choices_vec.dedup();
        let number_nans_blue_label_choices_vec =
            blue_label_choices_vec.iter().filter(|e| e.is_nan()).count();
        if number_nans_blue_label_choices_vec == blue_label_choices_vec.len() {
            blue_label_choices_vec = vec![NAN];
        }
        // turn back to array
        let blue_label_choices = Array::from(blue_label_choices_vec);

        for bb in 0..blue_labels.len() {
            if blue_labels[bb].is_nan() == false {
                total_blue_degrees[blue_labels[bb] as usize] =
                    total_blue_degrees[blue_labels[bb] as usize] - col_marginals[bb];
            }

            let mut change_blue_label_test: Array1<f64> = Array1::zeros(blue_label_choices.len());
            change_blue_label_test.fill(NAN);

            for ww in 0..blue_label_choices.len() {
                // so iterate over the red labels
                let red_label_test: Array1<f64> = red_labels
                    .iter()
                    .map(|e| (*e == blue_label_choices[ww]) as usize as f64)
                    .collect();
                change_blue_label_test[ww] = (red_label_test * matrix.column(bb)).sum()
                    - col_marginals[bb] * total_red_degrees[blue_label_choices[ww] as usize]
                        / mat_sum;
            }

            // assign new label based on maximisation of above condition
            // get the maximum of change_blue_label_test
            // not sure what happens if it's entirely NAN's
            // I guess we just return NAN.
            let max_change_blue_label_test = *change_blue_label_test
                .iter()
                .max_by(|a, b| {
                    let cmp = a.partial_cmp(b);
                    match cmp {
                        Some(c) => c,
                        // treat NANs as less.
                        None => std::cmp::Ordering::Greater,
                    }
                })
                .unwrap();

            // make the labels
            let mut labels = Vec::new();
            for (index, el) in change_blue_label_test.iter().enumerate() {
                if *el == max_change_blue_label_test {
                    labels.push(index);
                }
            }
            // generate the new label index
            let new_label_index = *labels.choose(&mut rand::thread_rng()).unwrap();
            blue_labels[bb] = blue_label_choices[new_label_index];

            if blue_labels[bb] > total_blue_degrees.len() as f64 {
                total_blue_degrees[blue_labels[bb] as usize] = 0.0;
            }

            // Update total marginals on new labelling
            total_blue_degrees[blue_labels[bb] as usize] =
                total_blue_degrees[blue_labels[bb] as usize] + col_marginals[bb]
        }

        // now we do the same for the red labels
        let red_label_choices_arr = blue_labels.clone();
        // turn to vec
        let mut red_label_choices_vec = red_label_choices_arr.into_raw_vec();
        // sort
        red_label_choices_vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // dedup
        red_label_choices_vec.dedup();
        // turn back to array
        let red_label_choices = Array::from(red_label_choices_vec);

        for aa in 0..red_labels.len() {
            total_red_degrees[red_labels[aa] as usize] =
                total_red_degrees[red_labels[aa] as usize] - row_marginals[aa];

            let mut change_red_label_test: Array1<f64> = Array1::zeros(red_label_choices.len());
            change_red_label_test.fill(NAN);

            for ww in 0..red_label_choices.len() {
                // so iterate over the red labels
                let blue_label_test: Array1<f64> = blue_labels
                    .iter()
                    .map(|e| (*e == red_label_choices[ww]) as usize as f64)
                    .collect();
                change_red_label_test[ww] = (blue_label_test * matrix.row(aa)).sum()
                    - row_marginals[aa] * total_blue_degrees[red_label_choices[ww] as usize]
                        / mat_sum;
            }

            // assign new label based on maximisation of above condition
            // get the maximum of change_blue_label_test
            // not sure what happens if it's entirely NAN's
            // I guess we just return NAN.
            let max_change_red_label_test = *change_red_label_test
                .iter()
                .max_by(|a, b| {
                    let cmp = a.partial_cmp(b);
                    match cmp {
                        Some(c) => c,
                        // treat NANs as less.
                        None => std::cmp::Ordering::Greater,
                    }
                })
                .unwrap();

            // make the labels
            let mut labels = Vec::new();
            for (index, el) in change_red_label_test.iter().enumerate() {
                if *el == max_change_red_label_test {
                    labels.push(index);
                }
            }
            // generate the new label index
            let new_label_index = *labels.choose(&mut rand::thread_rng()).unwrap_or(&0);
            red_labels[aa] = red_label_choices[new_label_index];

            if red_labels[aa] > total_red_degrees.len() as f64 {
                total_red_degrees[red_labels[aa] as usize] = 0.0;
            }

            // Update total marginals on new labelling
            total_red_degrees[red_labels[aa] as usize] =
                total_red_degrees[red_labels[aa] as usize] + row_marginals[aa]
        }

        qb_after = weighted_modularity(&b_matrix, mat_sum, &red_labels, &blue_labels);

        if qb_after <= qb_before {
            *red_labels = old_red_labels;
            *blue_labels = old_blue_labels;
            total_red_degrees = old_trd;
            total_blue_degrees = old_tbd;
            iterate_flag = false;
        }

        _iteration_counter += 1;
    }

    let qb_now = qb_after;

    (red_labels.clone(), blue_labels.clone(), qb_now)
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::InteractionMatrix;
    use ndarray::arr2;

    #[test]
    fn test_modularity() {
        let mut int_mat = InteractionMatrix::new(3, 3);
        int_mat.rownames = vec!["1r".into(), "2r".into(), "3r".into()];
        int_mat.colnames = vec!["1c".into(), "2c".into(), "3c".into()];

        int_mat.inner = arr2(&[[1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0]]);

        let LpaWbPlus { modularity, .. } = lba_wb_plus(int_mat);

        // kind of hacky way to test, but couldn't be bothered
        // to add another crate to test float equivalence.
        assert_eq!((modularity * 100.0).floor(), 25.0)
    }
}
