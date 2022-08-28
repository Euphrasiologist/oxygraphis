use clap::{arg, value_parser, ArgMatches, Command};
use oxygraph::{BipartiteGraph, DerivedGraphs, InteractionMatrix};
use std::error::Error;
use std::path::PathBuf;

/// Create the CLI in clap.
///
/// Better to have subcommands for each of derived + interaction matrix.
pub fn cli() -> Command<'static> {
    Command::new("oxygraphis")
        .bin_name("oxygraphis")
        .arg_required_else_help(true)
        .subcommand(
            Command::new("bipartite")
                .about("Generate and analyse bipartite graphs.")
                .arg_required_else_help(true)
                // generic parameters
                .arg(
                    arg!(<INPUT_DSV> "An input DSV with three headers only: from, to, and weight.")
                        // File always required
                        .required(true)
                        // and we expect it to be a PathBuf
                        .value_parser(value_parser!(PathBuf)),
                )
                .arg(
                    arg!([DELIMITER] "Specify the delimiter of the DSV; we assume tabs.")
                        .required(false),
                )
                .arg(
                    arg!(-p --plotbp "Render an SVG bipartite graph plot.")
                        .action(clap::ArgAction::SetTrue)
                )
                .subcommand(
                    Command::new("interaction-matrix")
                        .about("Coerce a bipartite graph into an interaction matrix.")
                        .arg(
                            arg!(-p --plotim "Render an SVG interaction matrix plot.")
                                .action(clap::ArgAction::SetTrue)
                        )
                        .arg(
                            arg!(-n --nodf "Compute the NODF number of a *sorted* interaction matrix.")
                                .action(clap::ArgAction::SetTrue)
                        ),
                )
                .subcommand(Command::new("derived-graphs")
                    .about("Coerce a bipartite graph into two derived graphs.")
                    .arg(
                        arg!(-p --plotdg "Render an SVG derived graph of a stratum.")
                            .action(clap::ArgAction::SetTrue)
                    )
                    .arg(
                        arg!(-s --stratum [STRATUM] "The stratum to display.")
                        .default_value("host")
                        .possible_values(["host", "parasite"])
                    )
            )
        )
}

/// Process all of the matches from the CLI.
pub fn process_matches(matches: &ArgMatches) -> Result<(), Box<dyn Error>> {
    match matches.subcommand() {
        // all current functionality under the bipartite subcommand
        Some(("bipartite", sub_matches)) => {
            // parse all of the command line args here.
            // globals
            let input = sub_matches
                .get_one::<PathBuf>("INPUT_DSV")
                .expect("required");
            let delimiter = match sub_matches.get_one::<String>("DELIMITER") {
                Some(d) => d.bytes().next().unwrap_or(b'\t'),
                None => b'\t',
            };
            // did user want a bipartite plot?
            let bipartite_plot = *sub_matches
                .get_one::<bool>("plotbp")
                .expect("defaulted by clap.");

            // everything requires the bipartite graph
            // and must currently go through a DSV.
            let bpgraph = BipartiteGraph::from_dsv(input, delimiter)?;

            match sub_matches.subcommand() {
                // user just called bipartite
                None => {
                    // pass args from above
                    if bipartite_plot {
                        // make the plot dims CLI args.
                        // but 600 x 400 for now.
                        bpgraph.plot(600, 400);
                    } else {
                        // default subcommand output
                        // probably pass this to another function later.
                        let (no_parasites, no_hosts, no_edges) = bpgraph.stats();
                        println!("#_parasite_nodes\t#_host_nodes\t#_total_edges");
                        println!("{}\t{}\t{}", no_parasites, no_hosts, no_edges);
                    }
                }
                // user called interaction-matrix
                Some(("interaction-matrix", im_matches)) => {
                    // generate the matrix
                    let mut im_mat = InteractionMatrix::from_bipartite(bpgraph);

                    let im_plot = *im_matches
                        .get_one::<bool>("plotim")
                        .expect("defaulted by clap.");
                    let nodf = *im_matches
                        .get_one::<bool>("nodf")
                        .expect("defaulted by clap.");
                    if im_plot {
                        // change these, especially height might need to
                        // be auto generated
                        im_mat.plot(600, 400);
                    } else if nodf {
                        // sort and make nodf.
                        im_mat.sort();
                        let nodf = im_mat.nodf()?;
                        println!("NODF\n{}", nodf);
                    } else {
                        // default subcommand output
                        let (no_rows, no_cols) = im_mat.stats();
                        println!("#_rows\t#_cols");
                        println!("{}\t{}", no_rows, no_cols);
                    }
                }
                // user called derived-graphs
                Some(("derived-graphs", dg_matches)) => {
                    let dgs = DerivedGraphs::from_bipartite(bpgraph);

                    let dg_plot = *dg_matches
                        .get_one::<bool>("plotdg")
                        .expect("defaulted by clap.");
                    let stratum = &*dg_matches
                        .get_one::<String>("stratum")
                        .expect("defaulted by clap.");

                    if dg_plot {
                        match stratum.as_str() {
                            "host" => dgs.hosts.plot(600.0),
                            "parasite" => dgs.parasites.plot(600.0),
                            _ => unreachable!("Should never reach here."),
                        }
                    } else {
                        let (p_nodes, p_edges, p_edge_fil, h_nodes, h_edges, h_edge_fil) =
                            dgs.stats();

                        println!("p_nodes\tp_edges\tp_edge_fil\th_nodes\th_edges\th_edge_fil");
                        println!(
                            "{}\t{}\t{}\t{}\t{}\t{}",
                            p_nodes, p_edges, p_edge_fil, h_nodes, h_edges, h_edge_fil
                        );
                    }
                }
                _ => unreachable!("Should never reach here."),
            }
        }
        _ => unreachable!("Should never reach here."),
    }

    Ok(())
}
