//! A bipartite graph is one where there are nodes
//! of two strata (or colours). Edges can form *between*
//! these two strata, but not *within*.
//!
//! In `oxygraph`, the bipartite graph is implemented
//! as a wrapper over a `petgraph::Graph`, and there are
//! no enforcements to maintain bipartite-ness. This is down
//! to the user. A bipartite graph can be checked, once created,
//! with the `is_bipartite` method.

use calm_io::*;
use csv::ReaderBuilder;
use ordered_float::OrderedFloat;
use petgraph::{
    graph::NodeIndex,
    visit::{EdgeRef, IntoNodeReferences},
    Direction::{self, Incoming, Outgoing},
    Graph,
};
use rand::{self, seq::IndexedRandom};
use serde_derive::Deserialize;
use std::collections::{BTreeMap, HashMap};
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
    #[error("Problem with graph creation")]
    GraphCreationError { source: DSVError },
}

/// Error type for generating a random graph.
#[derive(Error, Debug)]
pub enum RandomError {
    #[error("More edges than is possible for a bipartite graph ({0}).")]
    MaxEdges(usize),
    #[error("Number of nodes for a graph must be non-zero.")]
    NoNodes,
}

/// Errors that may occur while reading a delimited-separated values (DSV) file.
#[derive(Error, Debug)]
pub enum DSVError {
    #[error("No weight.")]
    Weight,
    #[error("No source node.")]
    Source,
    #[error("No target node.")]
    Target,
    #[error("Node appears in both parasites and hosts - i.e. not a bipartite graph.")]
    NotBipartite,
}

/// A row in the DSV should only be these three columns currently.
///
/// * `from` - Actor species (parasite/parasitoid/higher trophic level).
/// * `to` - Recipient species (host/lower trophic level).
/// * `weight` - The weight of the edge between the two species.
#[derive(Debug, Deserialize, PartialEq)]
pub struct Row {
    /// The actor species (parasite/parasitoid/higher trophic level).
    pub from: String,
    /// The recipient species (host/lower trophic level).
    pub to: String,
    /// Weights between actor/recipient can only be floats
    /// at the moment. But could be more complex in the future.
    /// These are added to the *nodes*.
    pub weight: f64,
}

/// Represents the two partitions in a bipartite graph:
/// - `Hosts` (lower level, typically recipients)
/// - `Parasites` (higher level, typically actors)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Partition {
    Hosts,
    Parasites,
}

impl std::fmt::Display for Partition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Partition::Hosts => write!(f, "Hosts"),
            Partition::Parasites => write!(f, "Parasites"),
        }
    }
}

/// Represents a species node in the bipartite graph.
///
/// # Fields
/// * `name` - The species name.
/// * `partition` - Whether it's a host or parasite.
#[derive(Debug, Clone)]
pub struct SpeciesNode {
    pub name: String,
    pub partition: Partition,
}

impl SpeciesNode {
    /// Create a new node
    pub fn new(name: String, partition: Partition) -> Self {
        Self { name, partition }
    }

    /// Convert a `SpeciesNode` to a species name.
    pub fn to_species(&self) -> String {
        self.name.clone()
    }
}

impl std::fmt::Display for SpeciesNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // ignore partition for now
        write!(f, "{}", self.name)
    }
}

/// Fitness is currently an `f64`.
pub type Fitness = f64;

/// Wrapper for a petgraph `Graph`, representing a bipartite network.
///
/// Nodes are `SpeciesNode`, and edges are weighted with `Fitness`.
#[derive(Debug, Clone)]
pub struct BipartiteGraph(pub Graph<SpeciesNode, Fitness>);

/// Output from bipartiteness checking.
/// * `Yes` contains a color map for nodes.
/// * `No` means the graph is not bipartite.
#[derive(Debug)]
pub enum Strata {
    /// If there are strata present, return these
    /// as a map of node indices &
    Yes(HashMap<NodeIndex, bool>),
    /// There weren't any strata. This isn't a
    /// bipartite graph! And what are the offending nodes?
    No,
}

/// A structure to hold the output of the
/// bipartite graph statistics.
#[derive(Debug)]
pub struct BipartiteStats {
    /// The number of parasites in the graph.
    pub no_parasites: usize,
    /// The number of hosts in the graph.
    pub no_hosts: usize,
    /// The number of edges in the graph.
    pub no_edges: usize,
}

impl BipartiteGraph {
    /// Checks whether the graph is bipartite.
    ///
    /// This uses a two-coloring algorithm to verify that no two adjacent nodes
    /// share the same partition. If the graph is bipartite, returns `Strata::Yes`
    /// with a map of node indices and their assigned color (bool).
    /// If not bipartite, returns `Strata::No`.
    ///
    /// # Returns
    /// * `Strata::Yes(map)` if the graph is bipartite.
    /// * `Strata::No` if the graph is not bipartite.
    pub fn is_bipartite(&self) -> Strata {
        // create a map to store the colours
        let mut colour_map: HashMap<NodeIndex, bool> = HashMap::new();
        // iterate over all the nodes
        // ignoring the weights.
        for (node, _) in self.0.node_references() {
            // does the map contain the node?
            let contains_node = colour_map.contains_key(&node);
            // now get the neighbours of this node.

            let no_neighbours = self.0.neighbors_undirected(node).count();

            if contains_node || no_neighbours == 0 {
                continue;
            }

            // make a queue
            let mut queue = vec![node];
            colour_map.insert(node, true);

            while !queue.is_empty() {
                let v = queue.pop().unwrap();
                let c = !colour_map.get(&v).unwrap();
                let inner_neighbours: Vec<NodeIndex> = self.0.neighbors_undirected(v).collect();
                for w in &inner_neighbours {
                    let contains_node_inner = colour_map.contains_key(w);
                    if contains_node_inner {
                        if colour_map.get(w).unwrap() == colour_map.get(&v).unwrap() {
                            return Strata::No;
                        }
                    } else {
                        colour_map.insert(*w, c);
                        queue.push(*w);
                    }
                }
            }
        }

        Strata::Yes(colour_map)
    }

    /// Generates a random bipartite graph with a given number of parasites,
    /// hosts, and edges.
    ///
    /// # Arguments
    /// * `parasite_no` - Number of parasite nodes.
    /// * `host_no` - Number of host nodes.
    /// * `edge_no` - Number of edges between parasites and hosts.
    ///
    /// # Returns
    /// * `Ok(BipartiteGraph)` if successful.
    /// * `Err(RandomError)` if node counts are zero or edges exceed the maximum possible.
    pub fn random(parasite_no: usize, host_no: usize, edge_no: usize) -> Result<Self, RandomError> {
        // must be greater than no nodes.
        if parasite_no == 0 || host_no == 0 {
            return Err(RandomError::NoNodes);
        }

        let max_edges = parasite_no * host_no;
        if edge_no > max_edges {
            return Err(RandomError::MaxEdges(max_edges));
        }
        // so we make a new bipartite graph with the number of nodes =
        // parasite_no + host_no
        // then get random node from parasites
        // and random node from hosts
        // then add while edge count is less than edge_no
        let mut graph: Graph<SpeciesNode, Fitness> = Graph::new();

        let mut p_node_indices = Vec::new();
        // add the parasite node indices to the graph
        for _ in 0..parasite_no {
            let spnode = SpeciesNode::new("".into(), Partition::Parasites);
            let nidx = graph.add_node(spnode);
            p_node_indices.push(nidx);
        }

        let mut h_node_indices = Vec::new();
        // add the host node indices to the graph
        for _ in 0..host_no {
            let spnode = SpeciesNode::new("".into(), Partition::Hosts);
            let nidx = graph.add_node(spnode);
            h_node_indices.push(nidx);
        }

        let mut edge_count = 0;

        while edge_count < edge_no {
            // guarantee these slices are non-empty pls.
            let p = *p_node_indices.choose(&mut rand::thread_rng()).unwrap();
            let h = *h_node_indices.choose(&mut rand::thread_rng()).unwrap();

            // check if this edge already exists.
            if graph.contains_edge(p, h) {
                continue;
            }
            graph.add_edge(p, h, 0.0);
            edge_count += 1;
        }

        Ok(BipartiteGraph(graph))
    }

    /// Returns simple statistics on the bipartite graph: counts of parasites,
    /// hosts, and edges.
    ///
    /// # Returns
    /// `BipartiteStats` struct with parasite count, host count, and edge count.
    pub fn stats(&self) -> BipartiteStats {
        let (parasites, hosts) = &self.get_parasite_host_from_graph();

        let no_parasites = parasites.len();
        let no_hosts = hosts.len();

        let no_edges = &self.0.edge_count();

        BipartiteStats {
            no_parasites,
            no_hosts,
            no_edges: *no_edges,
        }
    }

    /// Reads a bipartite graph from a delimited file.
    ///
    /// The file should contain three columns: `from`, `to`, and `weight`.
    /// Any delimiter can be specified (e.g., comma, tab).
    ///
    /// # Arguments
    /// * `input` - Path to the DSV file.
    /// * `delimiter` - Delimiter as a byte, e.g. `b'\t'` for TSV.
    ///
    /// # Returns
    /// `Ok(BipartiteGraph)` on success, or `ReadDSVError` on failure.
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
        Self::create_graph_from_dsv(edges)
            .map_err(|e| ReadDSVError::GraphCreationError { source: e })
    }

    /// Internal function to create a graph from DSV rows.
    ///
    /// The DSV must represent a valid bipartite network: "from" nodes
    /// are treated as parasites, and "to" nodes as hosts. Nodes cannot
    /// belong to both strata.
    ///
    /// # Returns
    /// `Ok(BipartiteGraph)` if valid, otherwise `DSVError`.
    fn create_graph_from_dsv(input: Vec<Row>) -> Result<BipartiteGraph, DSVError> {
        use std::collections::{HashMap, HashSet};

        // Step 1: Collect unique node names and their partitions
        let mut parasite_nodes: HashSet<&String> = HashSet::new();
        let mut host_nodes: HashSet<&String> = HashSet::new();

        for row in &input {
            parasite_nodes.insert(&row.from);
            host_nodes.insert(&row.to);
        }

        // Step 2: Build list of all nodes
        let mut all_nodes: HashSet<&String> = parasite_nodes.union(&host_nodes).copied().collect();

        // Step 3: Create the graph
        let mut graph: Graph<SpeciesNode, Fitness> = Graph::new();
        let mut node_index_map: HashMap<String, NodeIndex> = HashMap::new();

        // Step 4: Add nodes with SpeciesNode (name + partition)
        for node_name in all_nodes.drain() {
            let partition = if parasite_nodes.contains(node_name) && host_nodes.contains(node_name)
            {
                // we can't have a node in both partitions
                return Err(DSVError::NotBipartite);
            } else if parasite_nodes.contains(node_name) {
                Partition::Parasites
            } else {
                Partition::Hosts
            };

            let species_node = SpeciesNode {
                name: node_name.clone(),
                partition,
            };

            let node_index = graph.add_node(species_node);
            node_index_map.insert(node_name.clone(), node_index);
        }

        // Step 5: Add edges
        for Row { from, to, weight } in input {
            let from_node_index = node_index_map.get(&from).unwrap();
            let to_node_index = node_index_map.get(&to).unwrap();

            graph.add_edge(*from_node_index, *to_node_index, weight);
        }

        // Return wrapped graph
        Ok(BipartiteGraph(graph))
    }

    /// Extracts parasite and host nodes from the bipartite graph.
    ///
    /// # Returns
    /// A tuple with two vectors:
    /// * Parasites: Vec of `(NodeIndex, &SpeciesNode)`
    /// * Hosts: Vec of `(NodeIndex, &SpeciesNode)`
    pub fn get_parasite_host_from_graph(
        &self,
    ) -> (
        Vec<(NodeIndex, &SpeciesNode)>,
        Vec<(NodeIndex, &SpeciesNode)>,
    ) {
        let graph = &self.0;

        let hosts: Vec<_> = graph
            .node_references()
            .filter(|(_, node)| node.partition == Partition::Hosts)
            .collect();

        let parasites: Vec<_> = graph
            .node_references()
            .filter(|(_, node)| node.partition == Partition::Parasites)
            .collect();

        (parasites, hosts)
    }

    /// Returns the degree (number of connections) of a single node.
    ///
    /// # Arguments
    /// * `node` - NodeIndex of the node.
    ///
    /// # Returns
    /// Degree (usize).
    pub fn node_degree(&self, node: NodeIndex) -> usize {
        let graph = &self.0;
        graph.edges_directed(node, Direction::Incoming).count()
            + graph.edges_directed(node, Direction::Outgoing).count()
    }

    /// Returns a list of degrees for nodes in the graph.
    ///
    /// # Arguments
    /// * `partition` - Optional filter by `Partition::Hosts` or `Partition::Parasites`.
    ///
    /// # Returns
    /// Vector of tuples: `(node name, partition, degree)`.
    pub fn degrees(&self, partition: Option<Partition>) -> Vec<(String, Partition, usize)> {
        let graph = &self.0;

        graph
            .node_references()
            .filter(|(_, node)| partition.map_or(true, |p| node.partition == p))
            .map(|(node_index, node)| {
                (
                    node.name.clone(),
                    node.partition,
                    self.node_degree(node_index),
                )
            })
            .collect()
    }

    /// Returns the weighted degree (also known as strength) of a single node.
    ///
    /// # Arguments
    /// * `node` - NodeIndex of the node.
    ///
    /// # Returns
    /// Weighted degree (sum of edge weights).
    pub fn weighted_node_degree(&self, node: NodeIndex) -> f64 {
        let graph = &self.0;

        graph.edges(node).map(|e| *e.weight()).sum()
    }

    /// Returns weighted degrees (strengths) for nodes in the graph.
    ///
    /// # Arguments
    /// * `partition` - Optional filter by `Partition::Hosts` or `Partition::Parasites`.
    ///
    /// # Returns
    /// Vector of tuples: `(node name, partition, strength)`.
    pub fn weighted_degrees(&self, partition: Option<Partition>) -> Vec<(String, Partition, f64)> {
        let graph = &self.0;

        graph
            .node_references()
            .filter(|(_, node)| partition.map_or(true, |p| node.partition == p))
            .map(|(node_index, node)| {
                (
                    node.name.clone(),
                    node.partition,
                    self.weighted_node_degree(node_index),
                )
            })
            .collect()
    }

    /// Computes the degree distribution of the graph (unweighted or weighted).
    ///
    /// Uses Freedman-Diaconis rule to bin the degrees into a histogram.
    ///
    /// # Arguments
    /// * `partition` - Optional filter by `Partition::Hosts` or `Partition::Parasites`.
    /// * `weighted` - Whether to compute weighted degrees.
    ///
    /// # Returns
    /// A tuple:
    /// * Bin width (f64).
    /// * BTreeMap where keys are bin lower bounds and values are counts.
    pub fn degree_distribution(
        &self,
        partition: Option<Partition>,
        weighted: bool,
    ) -> (f64, BTreeMap<OrderedFloat<f64>, usize>) {
        // 1. Get degrees or weighted degrees
        let degrees: Vec<f64> = if weighted {
            self.weighted_degrees(partition)
                .into_iter()
                .map(|(_, _, d)| d)
                .collect()
        } else {
            self.degrees(partition)
                .into_iter()
                .map(|(_, _, d)| d as f64)
                .collect()
        };

        if degrees.is_empty() {
            return (0.0, BTreeMap::new());
        }

        // 2. Sort degrees for binning
        let mut sorted_degrees = degrees.clone();
        sorted_degrees.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // 3. Calculate IQR (Interquartile Range)
        fn calculate_iqr(sorted_degrees: &[f64]) -> f64 {
            let len = sorted_degrees.len();
            let q1_index = len / 4;
            let q3_index = 3 * len / 4;
            let q1 = sorted_degrees[q1_index];
            let q3 = sorted_degrees[q3_index];
            q3 - q1
        }

        let iqr = calculate_iqr(&sorted_degrees);
        let bin_width = (2.0 * iqr / (degrees.len() as f64).cbrt()).max(1.0).round();

        // 4. Build histogram
        let min_degree = *sorted_degrees.first().unwrap();
        let mut histogram = BTreeMap::new();

        for &degree in &degrees {
            // Calculate bin, offset from min_degree
            let bin = ((degree - min_degree) / bin_width).floor() * bin_width + min_degree;
            histogram
                .entry(OrderedFloat(bin))
                .and_modify(|count| *count += 1)
                .or_insert(1);
        }

        (bin_width, histogram)
    }

    /// Computes the bivariate degree distribution of edges.
    ///
    /// For each edge in the graph, returns a tuple containing the degree
    /// of the source node and the degree of the target node.
    ///
    /// # Returns
    /// Vector of `(degree_source, degree_target)` pairs.
    pub fn bivariate_degree_distribution(&self) -> Vec<(usize, usize)> {
        let graph = &self.0;

        graph
            .edge_references()
            .map(|e| (self.node_degree(e.source()), self.node_degree(e.target())))
            .collect()
    }

    /// Plots the bipartite graph as an SVG image.
    ///
    /// Parasites and hosts are drawn as circles at the top and bottom
    /// of the SVG, respectively. Edges are drawn as lines between them.
    ///
    /// # Arguments
    /// * `width` - Width of the SVG canvas.
    /// * `height` - Height of the SVG canvas.
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
            if graph.neighbors_directed(*node, Outgoing).count() > 0 {
                continue;
            } else {
                incoming_nodes_vec.push(graph.neighbors_directed(*node, Incoming).count());
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
            let r = graph.neighbors_directed(*node, Incoming).count();
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
                .unwrap_or_else(|| parasite_pos.get(&from).unwrap());

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

        let _ = stdoutln!("{}", svg);
    }

    /// Plots the bipartite graph as a proportional diagram (Sankey-like).
    ///
    /// Nodes are drawn as rectangles sized according to the number of
    /// connections. Edges are drawn as polygons representing connections
    /// between hosts and parasites.
    ///
    /// # Arguments
    /// * `width` - Width of the SVG canvas.
    /// * `height` - Height of the SVG canvas.
    pub fn plot_prop(&self, width: i32, height: i32) {
        let graph = &self.0;

        let canvas_width = width as f64 - (2.0 * MARGIN_LR);
        let upper_stratum_height = height as f64 / 4.0;
        let lower_stratum_height = height as f64 / 4.0 * 3.0;

        let rect_height = 20.0;
        let inner_margin = 3.0;

        // calculate total number of parasite connections
        let (parasites, hosts) = &self.get_parasite_host_from_graph();
        // TODO: it's possible the hosts and parasites
        // can be reordered here to increase visual appeal

        let _ = stderrln!("Number of parasites: {}", parasites.len());
        let _ = stderrln!("Number of hosts: {}", hosts.len());

        let mut parasite_connections = vec![];
        for (node, s) in parasites.iter() {
            parasite_connections.push((
                graph.neighbors_directed(*node, Incoming).count() as f64
                    + graph.neighbors_directed(*node, Outgoing).count() as f64,
                node,
                s,
            ));
        }
        let total_parasite_connections =
            parasite_connections.iter().map(|(e, _, _)| e).sum::<f64>();

        // now iterate over the parasites and draw the polygons
        let mut cumsum_node_connections = 0.0;
        let mut parasite_polygons = String::new();
        let mut parasite_pos = HashMap::new();

        let (y1_u, y2_u) = (upper_stratum_height, upper_stratum_height + rect_height);
        for (i, (node_connections, node, parasite)) in parasite_connections.iter().enumerate() {
            // special case for first node
            let (x1, x2) = if i == 0 {
                cumsum_node_connections += node_connections;
                let pos = (
                    MARGIN_LR,
                    ((cumsum_node_connections / total_parasite_connections) * canvas_width)
                        - inner_margin,
                );

                // for drawing edges
                parasite_pos.insert(**node, (pos.0, pos.0, pos.1, y1_u, y2_u));

                pos
            } else {
                let curr_conns = cumsum_node_connections;
                cumsum_node_connections += node_connections;
                let pos = (
                    ((curr_conns / total_parasite_connections) * canvas_width) + inner_margin,
                    ((cumsum_node_connections / total_parasite_connections) * canvas_width)
                        - inner_margin,
                );
                // for drawing edges
                parasite_pos.insert(**node, (pos.0, pos.0, pos.1, y1_u, y2_u));

                pos
            };

            // now make the polygon element
            let polygon = format!(
                r#"<polygon points="{x2},{y1_u} {x1},{y1_u} {x1},{y2_u} {x2},{y2_u}" fill="green" stroke="black" stroke-width="1" ><title>{parasite}</title></polygon>\n{end}"#,
                x1 = x1,
                y1_u = y1_u,
                x2 = x2,
                y2_u = y2_u,
                parasite = parasite,
                end = if i >= 1 { "\t" } else { "" }
            );

            parasite_polygons += &polygon;
        }

        // hosts
        let mut host_connections = vec![];
        for (node, s) in hosts.iter() {
            host_connections.push((
                graph.neighbors_directed(*node, Incoming).count() as f64
                    + graph.neighbors_directed(*node, Outgoing).count() as f64,
                node,
                s,
            ));
        }

        let total_host_connections = host_connections.iter().map(|(e, _, _)| e).sum::<f64>();

        // now we iterate over the hosts and draw the polygons
        let mut cumsum_node_connections = 0.0;
        let mut host_polygons = String::new();
        let mut host_pos = HashMap::new();

        let (y1_l, y2_l) = (lower_stratum_height, lower_stratum_height + rect_height);
        for (i, (node_connections, node, host)) in host_connections.iter().enumerate() {
            // special case for first node
            let (x1, x2) = if i == 0 {
                cumsum_node_connections += node_connections;
                let pos = (
                    MARGIN_LR,
                    ((cumsum_node_connections / total_host_connections) * canvas_width)
                        - inner_margin,
                );

                host_pos.insert(**node, (pos.0, pos.0, pos.1, y1_l, y2_l));

                pos
            } else {
                let curr_conns = cumsum_node_connections;
                cumsum_node_connections += node_connections;
                let pos = (
                    ((curr_conns / total_host_connections) * canvas_width) + inner_margin,
                    ((cumsum_node_connections / total_host_connections) * canvas_width)
                        - inner_margin,
                );

                host_pos.insert(**node, (pos.0, pos.0, pos.1, y1_l, y2_l));
                pos
            };

            // now make the polygon element
            let polygon = format!(
                r#"<polygon points="{x2},{y1_l} {x1},{y1_l} {x1},{y2_l} {x2},{y2_l}" fill="red" stroke="black" stroke-width="1"><title>{host}</title></polygon>\n{end}"#,
                x1 = x1,
                y1_l = y1_l,
                x2 = x2,
                y2_l = y2_l,
                host = host,
                end = if i >= 1 { "\t" } else { "" }
            );

            host_polygons += &polygon;
        }

        // and now the edges using polygons again
        let mut edge_polygons = String::new();
        for (mut i, edge) in graph.edge_references().enumerate() {
            i += 1;
            let from = edge.source();
            let to = edge.target();
            let _fitness = *edge.weight();

            // we need to mutate the x coords of the polygons
            let (x1_update_p, x1_p, x2_p, _y1_p, y2_p) = parasite_pos.get_mut(&from).unwrap();
            // FIXME: this unwrap panics; only when the graph is not bipartite
            // I haven't been bothered yet to figure it out properly.
            let (x1_update_h, x1_h, x2_h, y1_h, _y2_h) = host_pos.get_mut(&to).unwrap();

            // get total number of connections for the parasite
            let total_parasite_connections = graph.neighbors_directed(from, Outgoing).count()
                as f64
                + graph.neighbors_directed(from, Incoming).count() as f64;

            // get the number of connections for the parasite from the current host
            let p_to_h = graph
                .neighbors_directed(from, Outgoing)
                .filter(|e| e == &to)
                .count();

            // now scale the x coordinates of the polygons by the number of connections
            let parasite_pos_width = *x2_p - *x1_p;
            let current_parasite_width =
                parasite_pos_width * (p_to_h as f64 / total_parasite_connections);

            // goes from x1_update to x1_update + current_parasite_width
            let x1_update_p_clone = x1_update_p.clone();
            let (p_poly_1, p_poly_2) = (
                x1_update_p_clone,
                x1_update_p_clone + current_parasite_width,
            );

            // the hosts

            // get total number of connections for the parasite
            let total_host_connections = graph.neighbors_directed(to, Outgoing).count() as f64
                + graph.neighbors_directed(to, Incoming).count() as f64;

            let h_from_p = graph
                .neighbors_directed(to, Incoming)
                .filter(|e| e == &from)
                .count();

            // now scale the x coordinates of the polygons by the number of connections for the host
            let host_pos_width = *x2_h - *x1_h;
            let current_host_width = host_pos_width * (h_from_p as f64 / total_host_connections);

            // goes from x1_update to x1_update + current_parasite_width
            let x1_update_h_clone = x1_update_h.clone();
            let (h_poly_1, h_poly_2) = (x1_update_h_clone, x1_update_h_clone + current_host_width);

            // now update the x's
            *x1_update_p += current_parasite_width;
            *x1_update_h += current_host_width;

            // and create the polygon edges
            let edge_polygon = format!(
                r#"<polygon points="{p_poly_1},{y2_p} {p_poly_2},{y2_p} {h_poly_2},{y1_h} {h_poly_1},{y1_h}" fill-opacity="50%" />\n{}"#,
                if i >= 1 { "\t" } else { "" }
            );
            edge_polygons += &edge_polygon;
        }

        let svg = format!(
            r#"<svg version="1.1"
    width="{width}" height="{height}"
    xmlns="http://www.w3.org/2000/svg">
    {parasite_polygons}
    {host_polygons}
    {edge_polygons}
</svg>
        "#
        );

        let _ = stdoutln!("{}", svg);
    }

    /// Exports the bipartite graph to a tab-separated values (TSV) string.
    ///
    /// Output includes columns: `from`, `to`, `weight`.
    ///
    /// # Returns
    /// * `Ok(String)` with TSV contents.
    /// * `Err(DSVError)` if data is missing or invalid.
    pub fn to_tsv(&self) -> Result<String, DSVError> {
        let mut tsv = String::new();
        tsv += "from\tto\tweight\n";

        let g = &self.0;
        for e in g.edge_references() {
            let w = g.edge_weight(e.id()).ok_or(DSVError::Weight)?;

            let s = g.node_weight(e.source()).ok_or(DSVError::Source)?;
            let t = g.node_weight(e.target()).ok_or(DSVError::Target)?;

            tsv += &format!("{s}\t{t}\t{w}\n");
        }

        Ok(tsv)
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    // the test graph looks like this
    //
    //         a    b
    //         | \ /
    //         | /\
    //         |/  \
    //         c    d

    fn make_graph() -> BipartiteGraph {
        let mut graph: Graph<SpeciesNode, Fitness> = Graph::new();
        let a = graph.add_node(SpeciesNode::new("a".into(), Partition::Parasites));
        let b = graph.add_node(SpeciesNode::new("b".into(), Partition::Parasites));
        let c = graph.add_node(SpeciesNode::new("c".into(), Partition::Hosts));
        let d = graph.add_node(SpeciesNode::new("d".into(), Partition::Hosts));

        graph.add_edge(a, c, 1.0);
        graph.add_edge(a, d, 1.0);
        graph.add_edge(b, c, 1.0);

        BipartiteGraph(graph)
    }

    #[test]
    fn test_bipartite() {
        let g = make_graph();
        let bp = g.is_bipartite();

        match bp {
            Strata::Yes(_) => (),
            Strata::No => panic!(),
        }
    }

    #[test]
    fn test_tsv() {
        let g = make_graph();

        let tsv = g.to_tsv().unwrap();

        //  from    to      weight
        //  a       c       1
        //  a       d       1
        //  b       c       1

        assert_eq!(
            "from\tto\tweight\na\tc\t1\na\td\t1\nb\tc\t1\n".to_string(),
            tsv
        )
    }

    #[test]
    fn test_bivariate_degree_dist() {
        let g = make_graph();

        let bdd = g.bivariate_degree_distribution();

        assert_eq!(vec![(2, 2), (2, 1), (1, 2)], bdd)
    }

    // binary degree tests
    #[test]
    fn test_degrees_all() {
        let g = make_graph();
        let degrees = g.degrees(None);

        // Expected degrees:
        // a -> 2 (Parasites)
        // b -> 1 (Parasites)
        // c -> 2 (Hosts)
        // d -> 1 (Hosts)

        assert!(degrees.contains(&("a".to_string(), Partition::Parasites, 2)));
        assert!(degrees.contains(&("b".to_string(), Partition::Parasites, 1)));
        assert!(degrees.contains(&("c".to_string(), Partition::Hosts, 2)));
        assert!(degrees.contains(&("d".to_string(), Partition::Hosts, 1)));
    }

    #[test]
    fn test_degrees_partition_hosts() {
        let g = make_graph();
        let degrees = g.degrees(Some(Partition::Hosts));

        assert_eq!(degrees.len(), 2);
        assert!(degrees.contains(&("c".to_string(), Partition::Hosts, 2)));
        assert!(degrees.contains(&("d".to_string(), Partition::Hosts, 1)));
    }

    // weighted degrees
    #[test]
    fn test_weighted_degrees_partition_parasites() {
        let g = make_graph();
        let weighted_degrees = g.weighted_degrees(Some(Partition::Parasites));

        // a -> 2.0, b -> 1.0
        assert!(weighted_degrees.contains(&("a".to_string(), Partition::Parasites, 2.0)));
        assert!(weighted_degrees.contains(&("b".to_string(), Partition::Parasites, 1.0)));
    }

    // degree distribution
    #[test]
    fn test_degree_distribution_unweighted_hosts() {
        let g = make_graph();
        let (bin_width, histogram) = g.degree_distribution(Some(Partition::Hosts), false);

        assert!(bin_width > 0.0);
        eprintln!("{:?}", histogram);
        // Hosts degrees: c -> 2, d -> 1
        // Everything lumped in the same bin for this small example
        assert_eq!(histogram.get(&OrderedFloat(1.0)).cloned(), Some(2));
        // Fails for this small dataset..
        // assert_eq!(histogram.get(&OrderedFloat(2.0)).cloned(), Some(1));
    }

    // check host/parasite partitioning
    #[test]
    fn test_get_parasite_host_from_graph() {
        let g = make_graph();
        let (parasites, hosts) = g.get_parasite_host_from_graph();

        assert_eq!(parasites.len(), 2);
        assert!(parasites.iter().any(|(_, n)| n.name == "a"));
        assert!(parasites.iter().any(|(_, n)| n.name == "b"));

        assert_eq!(hosts.len(), 2);
        assert!(hosts.iter().any(|(_, n)| n.name == "c"));
        assert!(hosts.iter().any(|(_, n)| n.name == "d"));
    }

    // random graph generation
    #[test]
    fn test_random_graph_valid() {
        let g = BipartiteGraph::random(5, 5, 10).unwrap();
        let stats = g.stats();

        assert_eq!(stats.no_parasites, 5);
        assert_eq!(stats.no_hosts, 5);
        assert_eq!(stats.no_edges, 10);
    }

    #[test]
    #[should_panic(expected = "MaxEdges")]
    fn test_random_graph_too_many_edges() {
        let _ = BipartiteGraph::random(2, 2, 5).unwrap();
    }

    #[test]
    #[should_panic(expected = "NoNodes")]
    fn test_random_graph_zero_nodes() {
        let _ = BipartiteGraph::random(0, 2, 1).unwrap();
    }

    #[test]
    fn test_is_bipartite_simple() {
        let g = make_graph();

        let bp = g.is_bipartite();

        match bp {
            Strata::Yes(partition_map) => {
                assert_eq!(partition_map.len(), 4);
            }
            Strata::No => panic!("Graph should be bipartite but was not detected as such."),
        }
    }

    #[test]
    fn test_is_bipartite_triangle_not_bipartite() {
        let mut graph: Graph<SpeciesNode, Fitness> = Graph::new();

        let a = graph.add_node(SpeciesNode::new("a".into(), Partition::Parasites));
        let b = graph.add_node(SpeciesNode::new("b".into(), Partition::Parasites));
        let c = graph.add_node(SpeciesNode::new("c".into(), Partition::Parasites));

        graph.add_edge(a, b, 1.0);
        graph.add_edge(b, c, 1.0);
        graph.add_edge(c, a, 1.0);

        let g = BipartiteGraph(graph);

        let bp = g.is_bipartite();

        match bp {
            Strata::No => (),
            Strata::Yes(_) => panic!("Graph should NOT be bipartite but was detected as such."),
        }
    }

    #[test]
    fn test_is_bipartite_disconnected_nodes() {
        let mut graph: Graph<SpeciesNode, Fitness> = Graph::new();

        let _a = graph.add_node(SpeciesNode::new("a".into(), Partition::Parasites));
        let _b = graph.add_node(SpeciesNode::new("b".into(), Partition::Parasites));
        let _c = graph.add_node(SpeciesNode::new("c".into(), Partition::Hosts));

        let g = BipartiteGraph(graph);

        let bp = g.is_bipartite();

        match bp {
            Strata::Yes(partition_map) => {
                assert_eq!(partition_map.len(), 0);
            }
            Strata::No => panic!("Disconnected graph should be bipartite."),
        }
    }

    #[test]
    fn test_is_bipartite_single_node() {
        let mut graph: Graph<SpeciesNode, Fitness> = Graph::new();

        let _a = graph.add_node(SpeciesNode::new("a".into(), Partition::Parasites));

        let g = BipartiteGraph(graph);

        let bp = g.is_bipartite();

        match bp {
            Strata::Yes(partition_map) => {
                assert_eq!(partition_map.len(), 0);
            }
            Strata::No => panic!("Single node graph should be bipartite."),
        }
    }
}
