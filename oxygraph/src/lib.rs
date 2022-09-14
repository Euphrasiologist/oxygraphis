//! `oxygraph` is a crate to aid in the analysis
//! of ecological bipartite graphs. Still in development 
//! and might support other ecological graph types in the
//! future.
//! 
//! Creation of `BipartiteGraph` structs can only be done through
//! an input TSV currently. Otherwise you will have to create a 
//! `petgraph` graph e.g.:
//! 
//! ```rust
//! use petgraph::Graph;
//! use oxygraph::bipartite::BipartiteGraph;
//! 
//! let mut graph: Graph<String, f64> = Graph::new();
//! 
//! // add nodes/edges etc...
//! 
//! // now wrap in `BipartiteGraph` and access
//! // all the methods associated.
//! let bpgraph = BipartiteGraph(graph);
//! 
//! ```

/// Create, analyse, and visualise bipartite
/// graphs from tab delimited input data.
pub mod bipartite;
pub use bipartite::BipartiteGraph;
/// Create, analyse, and visualise
/// interaction matrices generated from a 
/// `BipartiteGraph`.
pub mod int_matrix;
pub use int_matrix::InteractionMatrix;

/// The derived graphs of a bipartite graph.
/// 
/// There are two derived graphs, one for each stratum,
/// and the edges in the graph correspond to how frequent
/// shared connections are between two nodes in that stratum.
pub mod derived;
pub use derived::DerivedGraphs;
pub use derived::DerivedGraphStats;

/// Modularity calculations are in their own module, but they
/// are built on top of an interaction matrix. Included are 
/// two algorithms from Beckett 2016.
pub mod modularity;
pub use modularity::LpaWbPlus;

/// Sorting algorithms on arrays.
pub mod sort;

/// The margins for the all the graph plots
/// used in this crate.
const MARGIN_LR: f64 = 20.0;

/// Scale a number between zero and 1, given a min/max.
pub fn scale_fit(x: f64, min_x: f64, max_x: f64) -> f64 {
    ((1.0 - 0.1) * ((x - min_x) / (max_x - min_x))) + 0.1
}
