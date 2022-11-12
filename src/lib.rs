use anyhow::Result;
use clap::{arg, crate_version, value_parser, ArgMatches, Command};
use oxygraph::{
    BipartiteGraph, BipartiteStats, DerivedGraphStats, DerivedGraphs, InteractionMatrix,
    InteractionMatrixStats, LpaWbPlus,
};
use rayon::prelude::*;
use std::path::PathBuf;

/// Create the CLI in clap.
///
/// Better to have subcommands for each of derived + interaction matrix.
pub fn cli() -> Command {
    Command::new("oxygraphis")
        .bin_name("oxygraphis")
        .arg_required_else_help(true)
        .version(crate_version!())
        .author("Max Brown <euphrasiamax@gmail.com>")
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
                .arg(
                    arg!(-d --degreedistribution "Return the degree distribution of a bipartite graph.")
                        .action(clap::ArgAction::SetTrue)
                )
                .arg(
                    arg!(-b --bivariatedistribution  "Return the bivariate degree distribution of a bipartite graph.")
                        .action(clap::ArgAction::SetTrue)
                )
                .subcommand(
                    Command::new("interaction-matrix")
                        .about("Coerce a bipartite graph into an interaction matrix.")
                        .arg(
                            arg!(--print "Print the inner matrix as a TSV. Mainly for debugging.")
                                .action(clap::ArgAction::SetTrue)
                        )
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
                            .value_parser(["host", "parasite"])
                    )
                    .arg(
                        arg!(-r --remove [REMOVE] "Edges with fewer than this number of connections are removed from the graph.")
                            .default_value("2.0")
                            .value_parser(value_parser!(f64))
                    )
                )
                .subcommand(Command::new("modularity")
                    .about("Derive the modularity of a bipartite graph.")
                    .arg(
                        arg!(-l --lpawbplus "Compute the modularity of a bipartite network using LPAwb+ algorithm.")
                            .action(clap::ArgAction::SetTrue)
                            .conflicts_with("dirtlpawbplus")
                    )
                    .arg(
                        arg!(-d --dirtlpawbplus "Compute the modularity of a bipartite network using DIRTLPAwb+ algorithm.")
                            .action(clap::ArgAction::SetTrue)
                            .conflicts_with("lpawbplus")
                    )
                    .arg(
                        arg!(-p --plotmod "Plot the interaction matrix of a bipartite network, sorted to maximise modularity.")
                            .action(clap::ArgAction::SetTrue)
                    )
                )
            )
            .subcommand(Command::new("simulate")
                .about("Simulate a number of graphs, and return calculations over the samples.")
                .arg(
                    arg!(--parasitenumber <PARASITENUMBER> "Number of parasite nodes in the graph.")
                        .required(true)
                        .value_parser(value_parser!(usize))
                )
                .arg(
                    arg!(--hostnumber <HOSTNUMBER> "Number of host nodes in the graph.")
                        .required(true)
                        .value_parser(value_parser!(usize))
                )
                .arg(
                    arg!(-e --edgecount <EDGECOUNT> "Number of edges in the graph.")
                        .required(true)
                        .value_parser(value_parser!(usize))
                )
                .arg(
                    arg!(-n --nsims [NSIMS] "Number of random samples to make.")
                        .value_parser(value_parser!(i32))
                        .default_value("1000")
                )
                .arg(
                    arg!(-c --calculation [CALCULATION] "The calculation to make.")
                        .default_value("nodf")
                        .value_parser(["nodf", "lpawbplus", "dirtlpawbplus", "degree-distribution", "bivariate-distribution"])
                )
                .arg(
                    arg!(--plot "Plot the simulated bipartite network.")
                        .action(clap::ArgAction::SetTrue)
                )
            )
}

/// Process all of the matches from the CLI.
pub fn process_matches(matches: &ArgMatches) -> Result<()> {
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
            let degee_distribution = *sub_matches
                .get_one::<bool>("degreedistribution")
                .expect("defaulted by clap.");
            let bivariate_distribution = *sub_matches
                .get_one::<bool>("bivariatedistribution")
                .expect("defaulted by clap.");

            // everything requires the bipartite graph
            // and must currently go through a DSV.
            let bpgraph = BipartiteGraph::from_dsv(input, delimiter)?;

            match bpgraph.is_bipartite() {
                // don't care here
                oxygraph::bipartite::Strata::Yes(_) => (),
                // tell the user which nodes are the offenders.
                oxygraph::bipartite::Strata::No => {
                    eprintln!("Warning: The input graph is not (fully) bipartite.");
                }
            }

            match sub_matches.subcommand() {
                // user just called bipartite
                None => {
                    // pass args from above
                    if bipartite_plot {
                        // make the plot dims CLI args.
                        // but 600 x 400 for now.
                        bpgraph.plot(1600, 700);
                    } else if degee_distribution {
                        let dist = bpgraph.degree_distribution();
                        println!("spp\tvalue");
                        for (s, v) in dist {
                            println!("{}\t{}", s, v);
                        }
                    } else if bivariate_distribution {
                        let biv_dist = bpgraph.bivariate_degree_distribution();
                        println!("node1\tnode2");
                        for (n1, n2) in biv_dist {
                            println!("{}\t{}", n1, n2);
                        }
                    } else {
                        // default subcommand output
                        // probably pass this to another function later.
                        let BipartiteStats {
                            no_parasites,
                            no_hosts,
                            no_edges,
                        } = bpgraph.stats();
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
                    let print = *im_matches
                        .get_one::<bool>("print")
                        .expect("defaulted by clap");

                    if im_plot {
                        // change these, especially height might need to
                        // be auto generated
                        im_mat.sort();
                        im_mat.plot(1600, None);
                    } else if nodf {
                        // sort and make nodf.
                        im_mat.sort();
                        let nodf = im_mat.nodf()?;
                        println!("NODF\n{}", nodf);
                    } else if print {
                        println!("{}", im_mat);
                    } else {
                        // default subcommand output
                        let InteractionMatrixStats {
                            no_rows,
                            no_cols,
                            no_poss_ints,
                            perc_ints,
                        } = im_mat.stats();
                        println!("#_rows\t#_cols\t#_poss_ints\tperc_ints");
                        println!(
                            "{}\t{}\t{}\t{}",
                            no_rows,
                            no_cols,
                            no_poss_ints,
                            perc_ints * 100.0
                        );
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
                    let remove = *dg_matches
                        .get_one::<f64>("remove")
                        .expect("defaulted by clap.");

                    if dg_plot {
                        match stratum.as_str() {
                            "host" => dgs.hosts.plot(600.0, remove),
                            "parasite" => dgs.parasites.plot(600.0, remove),
                            _ => unreachable!("Should never reach here."),
                        }
                    } else {
                        let DerivedGraphStats {
                            parasite_nodes,
                            parasite_edges,
                            parasite_edges_filtered,
                            host_nodes,
                            host_edges,
                            host_edges_filtered,
                        } = dgs.stats();

                        println!("p_nodes\tp_edges\tp_edge_fil\th_nodes\th_edges\th_edge_fil");
                        println!(
                            "{}\t{}\t{}\t{}\t{}\t{}",
                            parasite_nodes,
                            parasite_edges,
                            parasite_edges_filtered,
                            host_nodes,
                            host_edges,
                            host_edges_filtered
                        );
                    }
                }
                Some(("modularity", mod_matches)) => {
                    // this will probably be used later. Currently not!
                    let _lpawbplus = *mod_matches
                        .get_one::<bool>("lpawbplus")
                        .expect("defaulted by clap.");
                    let dirtlpawbplus = *mod_matches
                        .get_one::<bool>("dirtlpawbplus")
                        .expect("defaulted by clap.");
                    let plot = *mod_matches
                        .get_one::<bool>("plotmod")
                        .expect("defaulted by clap.");

                    // create the interaction matrix
                    let int_mat = InteractionMatrix::from_bipartite(bpgraph);

                    if plot {
                        let kind: &str;
                        let modularity_obj = if dirtlpawbplus {
                            kind = "DIRTLPAwb+";
                            oxygraph::modularity::dirt_lpa_wb_plus(int_mat.clone(), 2, 2)
                        } else {
                            kind = "LPAwb+";
                            oxygraph::modularity::lpa_wb_plus(int_mat.clone(), None)
                        };
                        let modularity = modularity_obj.modularity;
                        modularity_obj.plot(int_mat);
                        eprintln!("{} modularity: {}", kind, modularity);
                    } else if dirtlpawbplus {
                        // probably let user input reps in future.
                        let LpaWbPlus { modularity, .. } =
                            oxygraph::modularity::dirt_lpa_wb_plus(int_mat, 2, 2);
                        println!("DIRTLPAwb+\n{}", modularity);
                    } else {
                        let LpaWbPlus { modularity, .. } =
                            oxygraph::modularity::lpa_wb_plus(int_mat, None);
                        println!("LPAwb+\n{}", modularity);
                    }
                }
                _ => unreachable!("Should never reach here."),
            }
        }
        Some(("simulate", sm_matches)) => {
            let parasite_number = *sm_matches
                .get_one::<usize>("parasitenumber")
                .expect("defaulted by clap?");
            let host_number = *sm_matches
                .get_one::<usize>("hostnumber")
                .expect("defaulted by clap?");
            let edge_count = *sm_matches
                .get_one::<usize>("edgecount")
                .expect("defaulted by clap?");
            let n_sims = *sm_matches
                .get_one::<i32>("nsims")
                .expect("defaulted by clap?");
            let plot = *sm_matches
                .get_one::<bool>("plot")
                .expect("defaulted by clap?");

            let calculation = &*sm_matches
                .get_one::<String>("calculation")
                .expect("defaulted by clap.");

            if plot {
                let rand_graph = BipartiteGraph::random(parasite_number, host_number, edge_count)?;

                rand_graph.plot(1000, 400);

                // return early here.
                return Ok(());
            }

            (0..n_sims).into_par_iter().for_each(|_| {
                {
                    let rand_graph =
                        BipartiteGraph::random(parasite_number, host_number, edge_count).unwrap();

                    match calculation.as_str() {
                        "nodf" => {
                            let mut im_mat = InteractionMatrix::from_bipartite(rand_graph);
                            im_mat.sort();
                            let nodf = im_mat.nodf().unwrap();
                            if !nodf.is_nan() {
                                println!("{}", nodf);
                            }
                            // sim_vec.push(nodf);
                        }
                        "lpawbplus" => {
                            let im_mat = InteractionMatrix::from_bipartite(rand_graph);
                            let LpaWbPlus { modularity, .. } =
                                oxygraph::modularity::lpa_wb_plus(im_mat, None);
                            // sim_vec.push(modularity);
                            println!("{}", modularity);
                        }
                        "dirtlpawbplus" => {
                            let im_mat = InteractionMatrix::from_bipartite(rand_graph);
                            let LpaWbPlus { modularity, .. } =
                                oxygraph::modularity::dirt_lpa_wb_plus(im_mat, 2, 2);
                            // sim_vec.push(modularity);
                            println!("{}", modularity);
                        }
                        // not sure how to implement these two yet, or how useful they will be.
                        "degree-distribution" => {
                            unimplemented!()
                        }
                        "bivariate-distribution" => {
                            unimplemented!()
                        }
                        _ => unreachable!("clap should make sure we never reach here."),
                    }
                }
            });
        }
        _ => unreachable!("Should never reach here."),
    }

    Ok(())
}
