use clap::{arg, value_parser, Command};
use oxygraph::{BipartiteGraph, DerivedGraphs, InteractionMatrix};
use std::error::Error;
use std::path::PathBuf;

// next thing to do is pause and clean up the functionality we already have.
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

    let dg = DerivedGraphs::from_bipartite(bpgraph);

    // plot
    dg.hosts.plot(500.0);

    // bpgraph.plot(420, 200);
    // let mut im = InteractionMatrix::from_bipartite(bpgraph);
    // im.sort();
    // let nodf = im.nodf()?;
    // eprintln!("NODF for input = {}", nodf);
    // im.plot(600, 600);

    Ok(())
}
