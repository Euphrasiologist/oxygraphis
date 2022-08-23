// a binary to create a graph from
// TSV input
// and then we'd like to compute
// various metrics on these graphs

use clap::{arg, value_parser, Command};
use oxygraph::{BipartiteGraph, InteractionMatrix};
use std::error::Error;
use std::path::PathBuf;

fn cli() -> Command<'static> {
    Command::new("oxygraphis")
        .bin_name("oxygraphis")
        .subcommand_required(false)
        .arg_required_else_help(true)
        .arg(
            arg!(-i --input <FILE>)
                // File always required
                .required(true)
                // and we expect it to be a PathBuf
                .value_parser(value_parser!(PathBuf)),
        )
}

fn main() -> Result<(), Box<dyn Error>> {
    let matches = cli().get_matches();

    let input = matches.get_one::<PathBuf>("input").expect("required");

    let bpgraph = BipartiteGraph::from_dsv(input, b'\t')?;

    // bpgraph.plot(420, 200);
    let mut im = InteractionMatrix::from_bipartite(bpgraph);
    im.sort();
    im.plot(600, 100);

    Ok(())
}
