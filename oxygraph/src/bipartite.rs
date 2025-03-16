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
use petgraph::{
    graph::NodeIndex,
    visit::{EdgeRef, IntoNodeReferences, NodeRef},
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
}

/// Error type for generating a random graph.
#[derive(Error, Debug)]
pub enum RandomError {
    #[error("More edges than is possible for a bipartite graph ({0}).")]
    MaxEdges(usize),
    #[error("Number of nodes for a graph must be non-zero.")]
    NoNodes,
}

/// Error type for generating a TSV.
#[derive(Error, Debug)]
pub enum TSVError {
    #[error("No weight.")]
    Weight,
    #[error("No source node.")]
    Source,
    #[error("No target node.")]
    Target,
}

/// A row in the DSV should only be these three columns currently.
#[derive(Debug, Deserialize, PartialEq)]
pub struct Row {
    /// The actor species (parasite/parasitoid/higher trophic level).
    pub from: String,
    /// The recipient species (host/lower trophic level).
    pub to: String,
    /// Weights between actor/recipient can only be floats
    /// at the moment. But could be more complex in the future.
    /// These are added to the *nodes* (just for me).
    pub weight: f64,
}

/// Species is a `String`.
pub type Species = String;
/// Fitness is currently an `f64`.
pub type Fitness = f64;
/// A directed graph with two levels, parasite, and host.
/// Could be also used for plants and pollinators. Or other
/// such things.
#[derive(Debug, Clone)]
pub struct BipartiteGraph(pub Graph<Species, Fitness>);

/// This enum might replace `get_parasite_host_from_graph`.
/// As it should display the same information.
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
    /// Check that a data set passed to `oxygraph` is actually
    /// bipartite in nature.
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
    /// Generate a set of random bipartite graphs with specified
    /// numbers of nodes in each stratum, and edges between the strata.
    ///
    /// See https://networkx.org/documentation/networkx-1.10/_modules/networkx/algorithms/bipartite/generators.html#gnmk_random_graph
    ///
    /// Guard against more nodes than possible?
    pub fn random(parasite_no: usize, host_no: usize, edge_no: usize) -> Result<Self, RandomError> {
        let max_edges = parasite_no * host_no;
        if edge_no > max_edges {
            return Err(RandomError::MaxEdges(max_edges));
        }
        // so we make a new bipartite graph with the number of nodes =
        // parasite_no + host_no
        // then get random node from parasites
        // and random node from hosts
        // then add while edge count is less than edge_no
        let mut graph: Graph<Species, Fitness> = Graph::new();
        // must be greater than no nodes.
        if parasite_no == 0 || host_no == 0 {
            return Err(RandomError::NoNodes);
        }

        let mut p_node_indices = Vec::new();
        // add the parasite node indices to the graph
        for _ in 0..parasite_no {
            let nidx = graph.add_node("".into());
            p_node_indices.push(nidx);
        }

        let mut h_node_indices = Vec::new();
        // add the host node indices to the graph
        for _ in 0..host_no {
            let nidx = graph.add_node("".into());
            h_node_indices.push(nidx);
        }

        let mut edge_count = 0;

        while edge_count < edge_no {
            // guarantee these slices are non-empty pls.
            let p = *p_node_indices.choose(&mut rand::rng()).unwrap();
            let h = *h_node_indices.choose(&mut rand::rng()).unwrap();

            // check if this edge already exists.
            if graph.contains_edge(p, h) {
                continue;
            }
            graph.add_edge(p, h, 0.0);
            edge_count += 1;
        }

        Ok(BipartiteGraph(graph))
    }

    /// Print some stats when the default subcommand is called.
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
        let mut nodes: Vec<&String> = input
            .iter()
            .map(|e| &e.from)
            .chain(input.iter().map(|e| &e.to))
            .collect();
        // sort, dedup
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

    /// Extract the nodes from each stratum of a bipartite graph.
    /// FIXME: change return type here.
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
            let is_parasite = graph.neighbors_directed(node.id(), Outgoing).count() > 0;

            if is_parasite {
                parasites.push((node.id(), w));
            } else {
                hosts.push((node.id(), w));
            }
        }
        (parasites, hosts)
    }

    /// Degree of each node in the graph. Simply calculate the degree for each node in the
    /// graph. Optionally split by stratum?
    pub fn degrees(&self) -> Vec<(String, usize)> {
        // I imagine there will be tabular data output?
        let graph = &self.0;

        let mut dist = Vec::new();
        for (node, spp) in graph.node_references() {
            dist.push((
                spp.clone(),
                graph.edges_directed(node, Direction::Incoming).count()
                    + graph.edges_directed(node, Direction::Outgoing).count(),
            ))
        }
        dist
    }

    pub fn degree_distribution(&self) -> (usize, BTreeMap<usize, usize>) {
        let degrees = self
            .degrees()
            .iter()
            .map(|(_, d)| *d)
            .collect::<Vec<usize>>();

        fn calculate_iqr(sorted_degrees: &[usize]) -> f64 {
            let q1_index = sorted_degrees.len() / 4;
            let q3_index = 3 * sorted_degrees.len() / 4;
            let q1 = sorted_degrees[q1_index] as f64;
            let q3 = sorted_degrees[q3_index] as f64;
            q3 - q1
        }

        if degrees.is_empty() {
            return (0, BTreeMap::new());
        }

        // Sort degrees for IQR calculation
        let mut sorted_degrees = degrees.clone();
        sorted_degrees.sort_unstable();

        // Calculate the bin width using Freedman-Diaconis rule
        let iqr = calculate_iqr(&sorted_degrees);
        let bin_width = (2.0 * iqr / (degrees.len() as f64).cbrt()).max(1.0).round() as usize;

        // Determine the minimum and maximum degrees
        let min_degree = *sorted_degrees.first().unwrap();

        // Initialize an ordered histogram using BTreeMap
        let mut histogram = BTreeMap::new();
        for &degree in &degrees {
            let bin = ((degree - min_degree) / bin_width) * bin_width + min_degree;
            *histogram.entry(bin).or_insert(0) += 1;
        }

        (bin_width, histogram)
    }

    /// Bivariate degree distributions. Enumerate all adjacent nodes
    /// and calculate the degree for each.
    ///
    /// Iterate over the nodes, derive neighbours for each node (Incoming+Outgoing).
    /// Append these to a list in a sorted order
    /// sort this final list, and dedup.
    /// Now for each node pair, calculate the degree for each.
    pub fn bivariate_degree_distribution(&self) -> Vec<(usize, usize)> {
        let graph = &self.0;

        let edge_list: Vec<(NodeIndex, NodeIndex)> = graph
            .edge_references()
            .map(|e| (e.source().id(), e.target().id()))
            .collect();

        let mut biv_dist = Vec::new();
        for (node1, node2) in edge_list {
            biv_dist.push((
                graph.edges_directed(node1, Direction::Incoming).count()
                    + graph.edges_directed(node1, Direction::Outgoing).count(),
                graph.edges_directed(node2, Direction::Incoming).count()
                    + graph.edges_directed(node2, Direction::Outgoing).count(),
            ))
        }
        biv_dist
    }

    /// Plot the bipartite graph using SVG. The parasites and hosts
    /// are plotted as circles, and the edges are plotted as lines.
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

    /// Turn `BipartiteGraph` into a TSV.
    pub fn to_tsv(&self) -> Result<String, TSVError> {
        let mut tsv = String::new();
        tsv += "from\tto\tweight\n";

        let g = &self.0;
        for e in g.edge_references() {
            let w = g.edge_weight(e.id()).ok_or(TSVError::Weight)?;

            let s = g.node_weight(e.source()).ok_or(TSVError::Source)?;
            let t = g.node_weight(e.target()).ok_or(TSVError::Target)?;

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
        let mut graph: Graph<Species, Fitness> = Graph::new();
        let a = graph.add_node("a".into());
        let b = graph.add_node("b".into());
        let c = graph.add_node("c".into());
        let d = graph.add_node("d".into());

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
}
