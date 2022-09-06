//! Compute the modularity of an (optionally weighted) interaction matrix.
//!
//! Original methods from:
//! https://github.com/sjbeckett/weighted-modularity-LPAwbPLUS

use crate::InteractionMatrix;
use thiserror::Error;

/// Error type for modularity.
#[derive(Error, Debug)]
pub enum ModularityError {
    #[error("Could not transpose matrix: {0}")]
    TransposeError(String),
}

/// Label propagation algorithm for weighted bipartite networks that finds modularity.
/// Contains the LPAwb+ and the DIRTLPAwb+ algorithms
///
/// Translated from the R code here:
/// Stephen Beckett ( https://github.com/sjbeckett/weighted-modularity-LPAwbPLUS )
pub fn lba_wb_plus(mut matrix: InteractionMatrix) -> Result<(), ModularityError> {
    // Make sure the smallest matrix dimension represent the red labels by making
    // them the rows (if matrix is transposed here, will be transposed back at the end)
    let row_len = matrix.rownames.len();
    let col_len = matrix.colnames.len();
    // not sure if I need this yet
    let mut flipped = 0;

    if row_len > col_len {
        matrix = matrix
            .transpose();
    }

    // save matrix

    println!("{:?}", matrix.row_sums());

    Ok(())
}
