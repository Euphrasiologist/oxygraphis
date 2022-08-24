//! `oxygraph` is a crate to aid in the analysis
//! of ecological graphs. It's very much in development.

/// A module to analyse and visualise bipartite
/// graphs from tab delimited input data.
pub mod bipartite;
pub use bipartite::BipartiteGraph;
/// A module to create, analyse, and visualise
/// interaction matrices.
pub mod int_matrix;
pub use int_matrix::InteractionMatrix;

/// The margins for the graphs
/// used in this crate.
const MARGIN_LR: f64 = 20.0;
