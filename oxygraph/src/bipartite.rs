use csv::ReaderBuilder;
use petgraph::dot::{Config, Dot};
use petgraph::graph::NodeIndex;
use petgraph::visit::{EdgeRef, IntoNodeReferences, NodeRef};
use petgraph::Direction::{Incoming, Outgoing};
use petgraph::Graph;
use serde_derive::Deserialize;
use std::collections::HashMap;
use std::path::PathBuf;
use thiserror::Error;

use crate::{scale_fit, MARGIN_LR};

/// Error type for reading in a DSV.
#[derive(Error, Debug)]
pub enum ReadDSVError {
    #[error("Problem reading from path.")]
    FromPath { source: csv::Error },
    #[error("Problem with StringRecord: {source}")]
    StringRecordParseError { source: csv::Error },
}

/// A row in the DSV should only be these three columns currently.
#[derive(Debug, Deserialize, PartialEq)]
pub struct Row {
    /// The actor species (parasite/parasitoid/higher trophic level)
    pub from: String,
    /// The recipient species (host/lower trophic level)
    pub to: String,
    /// Weights between actor/recipient can only be floats
    /// at the moment. But could be more complex in the future.
    /// These are added to the *nodes* (just for me).
    pub weight: f64,
}

/// Species is a String
pub type Species = String;
/// Fitness is currently an f64.
pub type Fitness = f64;
/// A directed graph with two levels, parasite, and host.
/// Could be also used for plants and pollinators. Or other
/// such things.
pub struct BipartiteGraph(pub Graph<Species, Fitness>);

/// TODO:
/// Would be nice to have some methods to add
/// nodes, edges, and the weights to a graph
/// without having to go through `from_dsv`.
///
/// This means we would also be able to do things
/// like BipartiteGraph::create_random(<no_nodes>, <no_edges>)...
impl BipartiteGraph {
    /// Function to read into this graph struct from a DSV.
    ///
    /// Input must hav three columns in the order:
    /// - from
    /// - to
    /// - weight
    ///
    /// Using lower case names. Any delimiter can be used.
    pub fn from_dsv(input: &PathBuf, delimiter: u8) -> Result<Self, ReadDSVError> {
        let mut rdr = ReaderBuilder::new()
            .delimiter(delimiter)
            .from_path(input)
            .map_err(|s| ReadDSVError::FromPath { source: s })?;

        let mut edges = Vec::new();

        for result in rdr.deserialize() {
            let record: Row =
                result.map_err(|s| ReadDSVError::StringRecordParseError { source: s })?;
            edges.push(record);
        }
        Ok(Self::create_graph_from_dsv(edges))
    }

    /// Create a graph from a DSV, given the specific input criteria.
    fn create_graph_from_dsv(input: Vec<Row>) -> Self {
        // create a unique vector of nodes
        let froms: Vec<&String> = input.iter().map(|e| &e.from).collect();
        let tos: Vec<&String> = input.iter().map(|e| &e.to).collect();

        // collect into nodes, sort, dedup
        let mut nodes: Vec<&String> = froms.into_iter().chain(tos.into_iter()).collect();
        nodes.sort();
        nodes.dedup();

        // create the graph
        // it has name of the node (spp) and f64 node weights, and no edge weights.
        let mut graph: Graph<String, f64> = petgraph::Graph::new();
        // we also need to make a lookup of the nodes and their indices
        let mut node_index_map = HashMap::new();

        // add the nodes, and make the map
        for node in nodes {
            // let's not worry about allocation for the moment.
            let node = node.clone();

            let node_index = graph.add_node(node.clone());

            node_index_map.insert(node, node_index);
        }

        // add the edges
        for Row { from, to, weight } in input {
            let from_node_index = node_index_map.get(&from).unwrap();
            let to_node_index = node_index_map.get(&to).unwrap();

            graph.add_edge(*from_node_index, *to_node_index, weight);
        }
        // return the graph
        BipartiteGraph(graph)
    }

    pub fn get_parasite_host_from_graph(
        &self,
    ) -> (Vec<(NodeIndex, &String)>, Vec<(NodeIndex, &String)>) {
        let graph = &self.0;
        // store the parasites and hosts in their own vecs.
        let mut hosts = Vec::new();
        let mut parasites = Vec::new();

        // so we iterate over the nodes
        for (node, w) in graph.node_references() {
            // for this node, does it have any outgoing edges?
            let out: Vec<NodeIndex> = graph.neighbors_directed(node.id(), Outgoing).collect();
            // if it has outgoing edges
            let is_parasite = out.len() > 0;

            if is_parasite {
                parasites.push((node.id(), w));
            } else {
                hosts.push((node.id(), w));
            }
        }
        (parasites, hosts)
    }

    /// Make an SVG Plot of a bipartite graph.
    pub fn plot(&self, width: i32, height: i32) {
        let graph = &self.0;

        // some consts and fns
        // scale the nodes across the bipartite layers
        const NODE_SCALE: f64 = 4.0;

        let (parasites, hosts) = &self.get_parasite_host_from_graph();

        // make the circles.
        // want them 1/3 of the way down the SVG
        let mut parasite_nodes = String::new();
        let mut parasite_pos = HashMap::new();

        let parasite_spacing = (width as f64 - (MARGIN_LR * 2.0)) / parasites.len() as f64;

        // mut i, is because we want indexing to start at 1
        // for SVG formatting reasons. We -1 from i for x, as
        // we want to multiply by zero for the first node.
        for (mut i, (node, spp_name)) in parasites.iter().enumerate() {
            i += 1;
            // store the x and y coords of parasites
            let x = ((i - 1) as f64 * parasite_spacing) + (parasite_spacing / 2.0) + MARGIN_LR;
            let y = height as f64 / NODE_SCALE;
            // for edge drawing later
            parasite_pos.insert(*node, (x, y));

            parasite_nodes += &format!(
                "<circle cx=\"{x}\" cy=\"{y}\" r=\"6\" fill=\"green\"><title>{spp_name}</title></circle>\n{}",
                if i >= 1 { "\t" } else { "" }
            );
        }

        // host nodes here
        // scaling the nodes by how many incoming connections they have.
        let mut incoming_nodes_vec = Vec::new();
        for (node, _) in hosts.iter() {
            let out: Vec<NodeIndex> = graph.neighbors_directed(*node, Outgoing).collect();

            if out.len() > 0 {
                continue;
            } else {
                let r_vec: Vec<NodeIndex> = graph.neighbors_directed(*node, Incoming).collect();
                incoming_nodes_vec.push(r_vec.len());
            }
        }

        let mut host_nodes = String::new();
        let mut host_pos = HashMap::new();

        let host_spacing = (width as f64 - (MARGIN_LR * 2.0)) / hosts.len() as f64;

        for (mut i, (node, spp_name)) in hosts.iter().enumerate() {
            i += 1;
            // store the x and y coords of parasites
            let x = ((i - 1) as f64 * host_spacing) + (host_spacing / 2.0) + MARGIN_LR;
            let y = (height as f64 / NODE_SCALE) * 3.0;
            let r_vec: Vec<NodeIndex> = graph.neighbors_directed(*node, Incoming).collect();
            let r = r_vec.len();
            // for edge drawing later
            host_pos.insert(*node, (x, y));

            host_nodes += &format!(
                "<circle cx=\"{x}\" cy=\"{y}\" r=\"{}\" fill=\"red\"><title>{spp_name}</title></circle>\n{}",
                scale_fit(
                    r as f64,
                    *incoming_nodes_vec.iter().min().unwrap() as f64,
                    *incoming_nodes_vec.iter().max().unwrap() as f64
                ) * 5.0,
                if i >= 1 { "\t" } else { "" }
            );
        }

        let mut edge_links = String::new();
        // in order to scale the thickness of the lines ('fitness')
        // I'll need to find the min/max of the edge weights

        let mut fitness_vec = Vec::new();
        for edge in graph.edge_references() {
            fitness_vec.push(*edge.weight());
        }

        // remove these unwraps.
        let fit_min = fitness_vec
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let fit_max = fitness_vec
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();

        // now draw the edges
        for (mut i, edge) in graph.edge_references().enumerate() {
            i += 1;
            let from = edge.source();
            let to = edge.target();
            let fitness = *edge.weight();

            let (x1, y1) = parasite_pos.get(&from).unwrap();
            // to allow for self parasitism (or association I should say)!
            let (x2, y2) = host_pos
                .get(&to)
                .unwrap_or(parasite_pos.get(&from).unwrap());

            edge_links += &format!(
                "<line x1=\"{x1}\" y1=\"{y1}\" x2=\"{x2}\" y2=\"{y2}\" stroke=\"black\" stroke-width=\"{}\"/>\n{}",
                scale_fit(fitness, *fit_min, *fit_max),
                if i >= 1 { "\t" } else { "" }
            );
        }

        let svg = format!(
            r#"<svg version="1.1"
    width="{width}" height="{height}"
    xmlns="http://www.w3.org/2000/svg">
    {edge_links}
    {parasite_nodes}
    {host_nodes}
</svg>
        "#
        );

        println!("{}", svg);
    }

    /// Make a dot representation of the graph
    /// (It's not very good...)
    pub fn print_dot(&self) {
        println!("{}", Dot::with_config(&self.0, &[Config::GraphContentOnly]));
    }
}
