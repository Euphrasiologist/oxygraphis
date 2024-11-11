//! `oxygraph` is a crate to aid in the analysis
//! of ecological bipartite graphs. Still in development
//! and *might* support other ecological graph types in the
//! future.
//!
//! Creation of `BipartiteGraph` structs can be done through
//! an input TSV:
//!
//! ```rust
//! use oxygraph;
//! use oxygraph::bipartite::Strata;
//! use std::path::PathBuf;
//!
//! let path = PathBuf::from("./path/to/bipartite.tsv");
//!
//! // load into a BipartiteGraph structure
//! // a quick peek at the source codes shows you
//! // it's a thin wrapper over `petgraph`.
//! let bpgraph = oxygraph::BipartiteGraph::from_dsv(&path, b'\t').unwrap();
//!
//! // check it's bipartite
//! let check = match bpgraph.is_bipartite() {
//!     Strata::Yes(_) => "it's bipartite!",
//!     Strata::No => "it's not bipartite :(",
//! };
//!
//!
//! // create an interaction matrix
//! let int_mat = oxygraph::InteractionMatrix::from_bipartite(bpgraph);
//! // and look at some basic stats
//! let stats = int_mat.stats();
//!
//! ```
//!
//! Or by creating a `petgraph` graph e.g.:
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
//!
//! More information and tutorials to follow.

/// Create, analyse, and visualise bipartite
/// graphs from tab delimited input data.
pub mod bipartite;
pub use bipartite::BipartiteGraph;
pub use bipartite::BipartiteStats;
/// Create, analyse, and visualise
/// interaction matrices generated from a
/// `BipartiteGraph`.
pub mod int_matrix;
pub use int_matrix::InteractionMatrix;
pub use int_matrix::InteractionMatrixStats;

/// The derived graphs of a bipartite graph.
///
/// There are two derived graphs, one for each stratum,
/// and the edges in the graph correspond to how frequent
/// shared connections are between two nodes in that stratum.
pub mod derived;
pub use derived::DerivedGraphStats;
pub use derived::DerivedGraphs;

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
