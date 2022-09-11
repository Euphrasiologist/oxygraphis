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

/// The derived graphs of a bipartite graph.
/// There are two derived graphs, one for each stratum,
/// and the edges in the graph correspond to how frequent
/// shared connections are between two nodes in that stratum.
pub mod derived;
pub use derived::DerivedGraphs;

/// Modularity calculations are in their own module, but they
/// are built on top of an interaction matrix.
pub mod modularity;
pub use modularity::LpaWbPlus;

/// Sorting algorithms on arrays
pub mod sort;

/// The margins for the all the graph plots
/// used in this crate.
const MARGIN_LR: f64 = 20.0;

/// Scale a number between zero and 1, given a min/max.
pub fn scale_fit(x: f64, min_x: f64, max_x: f64) -> f64 {
    ((1.0 - 0.1) * ((x - min_x) / (max_x - min_x))) + 0.1
}
