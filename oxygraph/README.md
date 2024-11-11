# `oxygraph`

`oxygraph` is a Rust library to analyse bipartite graphs, and implements several algorithms along with visualisations.

There is functionality to:
- Visualise bipartite graphs and their derivatives, as well as interaction matrices.
- Compute basic statistics on bipartite graphs and interaction matrices.
- Create random bipartite graphs through the Erdös-Rényi process.
- Algorithms for computing nestedness, and modularity.

The bipartite graphs are a thin wrapper over `petgraph` graphs, and the interaction matrices are two dimensional `ndarray`s.

As the wrappers are thin, implementation of new metrics/algorithms should be straightforward.

An example which illustrates initiation of the graph from a TSV:

```rust
// main bipartite graph struct
use oxygraph::BipartiteGraph;
// Which strata there are in a bipartite graph
use oxygraph::bipartite::Strata;
// Interaction matrix struct
use oxygraph::InteractionMatrix;

// read in some data
// in the format:
// from    to    weight
// 0       1     1.0 
// etc ...
let bpgraph = BipartiteGraph::from_dsv("path/to/tsv", b'\t').unwrap();
// is the graph bipartite?
let strata = bpgraph.is_bipartite();

match stata {
    Strata::Yes(map) => println!("{:?}", map),
    // tell the user which nodes are the offenders.
    Strata::No => {
        panic!("Uh oh, your graph isn't bipartite!");
    }
}

// basic stats
println!("{:?}", bpgraph.stats());

// calculate NODF
let mut im = InteractionMatrix::from_bipartite(bpgraph);
println!("{}", im.nodf().unwrap());

// make a random bipartite graph
let rand_graph = BipartiteGraph::random(80, 100, 250).unwrap();
let mut im_rand = InteractionMatrix::from_bipartite(rand_graph);
// and calculate modularity
let modularity = rand_graph.lpa_wb_plus(None);
println!("{:?}", modularity);

```
