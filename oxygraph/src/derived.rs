//! The derived graphs of a bipartite graph are
//! the relationships among each of the two strata
//! in the bipartite graph. Each node in a single
//! stratum will share edges weighted by how many connections
//! they share.

use crate::BipartiteGraph;
use crate::{scale_fit, MARGIN_LR};
use itertools::Itertools;
use petgraph::{
    graph::{NodeIndex, UnGraph},
    visit::{EdgeRef, IntoNodeReferences},
    Direction::{self, Incoming, Outgoing},
};
use std::collections::{HashMap, HashSet};

/// Species is a String
pub type Species = String;
/// Total connection number
pub type Connections = usize;
/// Size is currently a usize.
pub type Shared = usize;

/// A derived graph of either hosts or parasites.
/// It's made of the three types above.
#[derive(Debug)]
pub struct DerivedGraph(pub UnGraph<(Species, Connections), Shared>);

impl DerivedGraph {
    /// The plots of [`DerivedGraph`] are going to be circular
    /// graphs with some sugar.
    ///
    /// Modified from a reference here:
    /// https://observablehq.com/@euphrasiologist/hybridisation-in-the-genus-saxifraga
    pub fn plot(&self, diameter: f64) {
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

        for (i, edge) in graph.edge_references().enumerate() {
            let from = edge.source();
            let to = edge.target();
            // use in a minute
            let no_connections = *edge.weight();

            let mut scaled_connections =
                scale_fit(no_connections as f64, con_min + 1.0, con_max) * 6.0;

            if scaled_connections < 2.0 {
                scaled_connections = 0.0;
            }

            let (x1, y1) = pos.get(&from).unwrap();
            // to allow for self parasitism (or association I should say)!
            let (x2, y2) = pos.get(&to).unwrap_or(pos.get(&from).unwrap());

            edges += &format!(
                "<line x1=\"{x1}\" y1=\"{y1}\" x2=\"{x2}\" y2=\"{y2}\" stroke=\"black\" stroke-width=\"{scaled_connections}\"/>\n"
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
        println!("{}", svg);
    }
}
/// DerivedGraph will be a wrapper over `petgraph`'s
/// `UnGraph`.
///
/// There are two, one for the upper stratum (parasites),
/// and one for the lower stratum (hosts).
pub struct DerivedGraphs {
    pub parasites: DerivedGraph,
    pub hosts: DerivedGraph,
}

impl DerivedGraphs {
    pub fn from_bipartite(bpgraph: BipartiteGraph) -> Self {
        // extract nodes for each stratum
        // find the set of their connections
        // which will be outgoing for parasites
        // and incoming for hosts
        let (parasites, hosts) = bpgraph.get_parasite_host_from_graph();

        // closure so we can capture bpgraph.
        let make_derived_graph =
            |input: Vec<(NodeIndex, &String)>, dir: Direction| -> DerivedGraph {
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
                    let node_index = ungraph.0.add_node((spp.clone(), 0));
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
                            println!("NodeIndex: {:?} not found", n1.0);
                            continue;
                        }
                    };
                    *ungraph.0.node_weight_mut(*n1.0).unwrap() = (n1_spp.clone(), n1_len);
                    // update the n2 weight
                    let n2_spp = node_index_map.get(n2.0).unwrap();
                    *ungraph.0.node_weight_mut(*n2.0).unwrap() = (n2_spp.clone(), n2_len);

                    // now add the edges in our parasite graph
                    let overlap: HashSet<_> = n1.1.intersection(n2.1).collect();
                    ungraph.0.add_edge(*n1.0, *n2.0, overlap.len());
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
