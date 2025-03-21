//! Derived graphs represent projections of bipartite networks onto each stratum.
//!
//! In a bipartite graph, there are two strata (e.g., parasites and hosts). A derived graph
//! projects the bipartite relationships onto one stratum, such that nodes are connected by
//! how many shared partners they have in the opposing stratum.
//!
//! For example, two parasite species that both interact with the same host species will have
//! an edge between them in the parasite derived graph, weighted by the number of shared hosts.
//!
//! This module provides `DerivedGraph` for a single stratum and `DerivedGraphs` for both strata,
//! including statistics and simple circular graph visualizations.

use crate::bipartite::SpeciesNode;
use crate::BipartiteGraph;
use crate::{scale_fit, MARGIN_LR};
use calm_io::*;
use itertools::Itertools;
use petgraph::{
    graph::{NodeIndex, UnGraph},
    visit::{EdgeRef, IntoNodeReferences},
    Direction::{self, Incoming, Outgoing},
};
use std::collections::{HashMap, HashSet};

/// Species is a `String` type alias for clarity.
pub type Species = String;

/// Total number of connections a species has in a derived graph.
pub type Connections = usize;

/// Number of shared connections between two species in a derived graph.
pub type Shared = usize;

/// A derived graph representing the relationships among species in a single stratum.
///
/// Nodes represent species within one stratum (e.g., parasites or hosts).
/// Edges represent the number of shared connections they have to the opposing stratum.
/// - Node weights: `(Species, Connections)`
/// - Edge weights: `Shared`
#[derive(Debug)]
pub struct DerivedGraph(pub UnGraph<(Species, Connections), Shared>);

impl DerivedGraph {
    /// Compute an overlap measure for the derived graph.
    ///
    /// This function is a placeholder and will calculate overlap metrics (e.g., Jaccard similarity)
    /// for species in the same stratum. Not yet implemented.
    pub fn overlap_measure() {
        todo!()
    }

    /// Plot the derived graph as a circular layout in SVG format.
    ///
    /// # Arguments
    /// * `diameter` - The diameter of the circular layout.
    /// * `remove` - A threshold value. Edges with fewer shared partners than this value are ignored in the plot.
    ///
    /// # Details
    /// Nodes are placed evenly on a circle.
    /// Edges are drawn between nodes, scaled by the number of shared partners.
    ///
    /// Modified from a reference here:
    /// <https://observablehq.com/@euphrasiologist/hybridisation-in-the-genus-saxifraga>
    pub fn plot(&self, diameter: f64, remove: f64) {
        let graph = &self.0;
        // this will store the positions of the nodes in cartesian space.
        let mut pos = HashMap::new();

        // need to know the number of nodes.
        let node_number = graph.node_count();
        let angle = (2.0 * std::f64::consts::PI) / node_number as f64;
        let mut nodes = String::new();

        // now iterate over the nodes in the graph.
        // bit complex destructuring here:
        // i = index of iteration
        // node = NodeIndex, spp = species string name, connections = how many connections that spp has.
        for (i, (node, (spp, connections))) in graph.node_references().enumerate() {
            let angle_i = angle * i as f64;
            let x = diameter * angle_i.cos();
            let y = diameter * angle_i.sin();
            let title_string = format!("{}: {}", spp, connections);

            // add to map
            pos.insert(node, (x, y));

            // make the nodes as circles.
            nodes += &format!(
                "<circle cx=\"{x}\" cy=\"{y}\" r=\"10\" fill=\"white\" stroke=\"black\" stroke-width=\"3\"><title>{title_string}</title></circle>\n"
            );
        }

        // now let's make the edges
        let mut edges = String::new();
        let mut connections_vec = Vec::new();
        for edge in graph.edge_references() {
            connections_vec.push(*edge.weight());
        }

        // remove these unwraps.
        let con_min = *connections_vec.iter().min().unwrap() as f64;
        let con_max = *connections_vec.iter().max().unwrap() as f64;

        for edge in graph.edge_references() {
            let from = edge.source();
            let to = edge.target();
            // use in a minute
            let no_connections = *edge.weight();

            // needs to be mutable if logic hashed out below is used.
            let mut scaled_connections =
                scale_fit(no_connections as f64, con_min + 1.0, con_max) * 6.0;

            // as in, we don't care if they share one host.
            // but this could be an input parameter.
            if no_connections < remove as usize {
                scaled_connections = 0.0;
            }

            let (x1, y1) = pos.get(&from).unwrap();
            // to allow for self parasitism (or association I should say)!
            let (x2, y2) = pos.get(&to).unwrap_or_else(|| pos.get(&from).unwrap());
            // and a title string
            let edge_title_string = format!(
                "{} connections between {} and {}",
                no_connections,
                graph.node_weight(from).unwrap().0,
                graph.node_weight(to).unwrap().0
            );

            edges += &format!(
                "<line x1=\"{x1}\" y1=\"{y1}\" x2=\"{x2}\" y2=\"{y2}\" stroke=\"black\" stroke-width=\"{scaled_connections}\"><title>{edge_title_string}</title></line>\n"
            );
        }

        let viewbox_param1 = (-(diameter) / 1.0) - MARGIN_LR;
        let viewbox_param2 = (diameter * 2.0) + (MARGIN_LR * 2.0);

        let svg = format!(
            r#"<svg version="1.1"
    viewBox="{viewbox_param1},{viewbox_param1},{viewbox_param2},{viewbox_param2}" width="{diameter}" height="{diameter}"
    xmlns="http://www.w3.org/2000/svg">
    <g>
    {edges}
    </g>
    <g>
    {nodes}
    </g>
</svg>
        "#
        );
        let _ = stdoutln!("{}", svg);
    }
}

/// Holds two derived graphs: one for parasites and one for hosts.
///
/// Each derived graph represents the projected relationships within a single stratum.
pub struct DerivedGraphs {
    /// A derived graph of the parasites.
    pub parasites: DerivedGraph,
    /// A derived graph of the hosts.
    pub hosts: DerivedGraph,
}

/// Basic statistics for a `DerivedGraphs` object.
///
/// Provides node and edge counts for both strata, including filtered edge counts.
pub struct DerivedGraphStats {
    /// The number of parasite nodes in the graph.
    pub parasite_nodes: usize,
    /// The number of parasite edges in the graph.
    pub parasite_edges: usize,
    /// The number of parasite edges where the weight is > 1.
    pub parasite_edges_filtered: usize,
    /// The number of host nodes.
    pub host_nodes: usize,
    /// The number of host edges.
    pub host_edges: usize,
    /// The number of host edges where the weight > 1.
    pub host_edges_filtered: usize,
}

impl DerivedGraphs {
    /// Generate basic statistics on the derived graphs.
    ///
    /// Includes node counts, edge counts, and counts of edges with weights greater than 1.
    pub fn stats(&self) -> DerivedGraphStats {
        let parasites = &self.parasites.0;
        let hosts = &self.hosts.0;

        let parasite_nodes = parasites.node_count();
        let host_nodes = hosts.node_count();
        let parasite_edges = parasites.edge_count();
        let host_edges = hosts.edge_count();

        DerivedGraphStats {
            parasite_nodes,
            parasite_edges,
            parasite_edges_filtered: parasites
                .edge_references()
                .filter(|e| *e.weight() > 1)
                .count(),
            host_nodes,
            host_edges,
            host_edges_filtered: hosts.edge_references().filter(|e| *e.weight() > 1).count(),
        }
    }

    /// Generate `DerivedGraphs` from a `BipartiteGraph`.
    ///
    /// Projects the bipartite graph into two unipartite (derived) graphs:
    /// - One for parasites, where edges represent shared hosts.
    /// - One for hosts, where edges represent shared parasites.
    ///
    /// # Assumptions
    /// - Parasites have only outgoing edges in the bipartite graph.
    /// - Hosts have only incoming edges in the bipartite graph.
    pub fn from_bipartite(bpgraph: BipartiteGraph) -> Self {
        // extract nodes for each stratum
        // find the set of their connections
        // which will be outgoing for parasites
        // and incoming for hosts
        let (parasites, hosts) = bpgraph.get_parasite_host_from_graph();

        // closure so we can capture bpgraph.
        let make_derived_graph =
            |input: Vec<(NodeIndex, &SpeciesNode)>, dir: Direction| -> DerivedGraph {
                // make the parasite derived graph
                let mut ungraph: DerivedGraph = DerivedGraph(UnGraph::new_undirected());
                // store the species name + node index
                let mut node_index_map = HashMap::new();
                // store the node index + the connections it has (HashSet)
                let mut node_conn_map = HashMap::new();

                for (node, spp) in input {
                    // node ^ is the original node index from the original
                    // graph, we ignore it after calculating the set of neighbours.
                    // make the set of connecting nodes
                    let out: HashSet<NodeIndex> = bpgraph.0.neighbors_directed(node, dir).collect();
                    // add the parasite node to its own graph
                    let node_index = ungraph.0.add_node((spp.to_species(), 0));
                    // add these to a map
                    node_conn_map.insert(node_index, out);
                    // and insert the species + node into map for later addition.
                    node_index_map.insert(node_index, spp.clone());
                }

                // now we iterate over the combinations of the k,v in parasite_node_conn_map
                for node_pair in node_conn_map.iter().combinations(2) {
                    // get out each of the pairs separately
                    let n1 = node_pair[0];
                    let n2 = node_pair[1];
                    // calculate the lengths of the sets
                    // these will be the node weights eventually
                    let n1_len = n1.1.len();
                    let n2_len = n2.1.len();

                    // update the n1 weight
                    let n1_spp = match node_index_map.get(n1.0) {
                        Some(n) => n,
                        None => {
                            let _ = stdoutln!("NodeIndex: {:?} not found", n1.0);
                            continue;
                        }
                    };
                    *ungraph.0.node_weight_mut(*n1.0).unwrap() = (n1_spp.to_species(), n1_len);
                    // update the n2 weight
                    let n2_spp = node_index_map.get(n2.0).unwrap();
                    *ungraph.0.node_weight_mut(*n2.0).unwrap() = (n2_spp.to_species(), n2_len);

                    // now add the edges in our parasite graph
                    let overlap: HashSet<_> = n1.1.intersection(n2.1).collect();
                    // only add an edge if the overlap.len() > 0 ?
                    if !overlap.is_empty() {
                        ungraph.0.add_edge(*n1.0, *n2.0, overlap.len());
                    }
                }
                ungraph
            };

        // we need to switch up the direction,
        // as in our crate, parasites are only supposed to have outgoing edges
        // and hosts only incoming edges.
        let parasite_ungraph = make_derived_graph(parasites, Outgoing);
        let host_ungraph = make_derived_graph(hosts, Incoming);

        Self {
            parasites: parasite_ungraph,
            hosts: host_ungraph,
        }
    }
}
#[cfg(test)]
mod tests {
    use crate::bipartite::{Fitness, Partition};

    use super::*;
    use petgraph::graph::Graph;

    /// Helper function: build a simple bipartite graph:
    ///
    /// Parasites: A, B  
    /// Hosts:     X, Y  
    /// Edges:  
    /// - A -> X  
    /// - A -> Y  
    /// - B -> Y
    ///
    /// This will create overlaps in hosts (both A and B share host Y).
    fn make_simple_bipartite() -> BipartiteGraph {
        let mut graph = Graph::<SpeciesNode, Fitness>::new();

        // Parasites
        let a = graph.add_node(SpeciesNode::new("A".into(), Partition::Parasites));
        let b = graph.add_node(SpeciesNode::new("B".into(), Partition::Parasites));

        // Hosts
        let x = graph.add_node(SpeciesNode::new("X".into(), Partition::Hosts));
        let y = graph.add_node(SpeciesNode::new("Y".into(), Partition::Hosts));

        // Edges
        graph.add_edge(a, x, 1.0);
        graph.add_edge(a, y, 1.0);
        graph.add_edge(b, y, 1.0);

        BipartiteGraph(graph)
    }

    #[test]
    fn test_from_bipartite_creates_correct_derived_graphs() {
        let bp_graph = make_simple_bipartite();

        let derived_graphs = DerivedGraphs::from_bipartite(bp_graph);

        // Parasite graph:
        // A and B share host Y, so they should have an edge.
        let parasite_graph = &derived_graphs.parasites.0;

        assert_eq!(
            parasite_graph.node_count(),
            2,
            "Parasite graph should have 2 nodes"
        );
        assert_eq!(
            parasite_graph.edge_count(),
            1,
            "Parasite graph should have 1 edge"
        );

        // Edge weight should be 1 because they only share Y.
        let edge = parasite_graph.edge_references().next().unwrap();
        assert_eq!(*edge.weight(), 1, "Edge weight should be 1 (shared host Y)");

        // Host graph:
        // Hosts X and Y share parasite A, so they should have an edge.
        let host_graph = &derived_graphs.hosts.0;

        assert_eq!(host_graph.node_count(), 2, "Host graph should have 2 nodes");
        assert_eq!(host_graph.edge_count(), 1, "Host graph should have 1 edge");

        // Edge weight should be 1 because they only share parasite A.
        let edge = host_graph.edge_references().next().unwrap();
        assert_eq!(
            *edge.weight(),
            1,
            "Edge weight should be 1 (shared parasite A)"
        );
    }

    #[test]
    fn test_derived_graphs_stats_are_correct() {
        let bp_graph = make_simple_bipartite();

        let derived_graphs = DerivedGraphs::from_bipartite(bp_graph);
        let stats = derived_graphs.stats();

        assert_eq!(stats.parasite_nodes, 2);
        assert_eq!(stats.parasite_edges, 1);
        assert_eq!(stats.parasite_edges_filtered, 0); // edges are 1, not >1

        assert_eq!(stats.host_nodes, 2);
        assert_eq!(stats.host_edges, 1);
        assert_eq!(stats.host_edges_filtered, 0); // edges are 1, not >1
    }

    #[test]
    fn test_plot_does_not_panic() {
        let bp_graph = make_simple_bipartite();
        let derived_graphs = DerivedGraphs::from_bipartite(bp_graph);

        // Test that plotting doesn't panic
        derived_graphs.parasites.plot(500.0, 0.0);
        derived_graphs.hosts.plot(500.0, 0.0);
    }

    #[test]
    fn test_empty_bipartite_graph() {
        let empty_graph = BipartiteGraph(Graph::new());

        let derived_graphs = DerivedGraphs::from_bipartite(empty_graph);
        let stats = derived_graphs.stats();

        assert_eq!(stats.parasite_nodes, 0);
        assert_eq!(stats.parasite_edges, 0);
        assert_eq!(stats.parasite_edges_filtered, 0);

        assert_eq!(stats.host_nodes, 0);
        assert_eq!(stats.host_edges, 0);
        assert_eq!(stats.host_edges_filtered, 0);
    }
}
