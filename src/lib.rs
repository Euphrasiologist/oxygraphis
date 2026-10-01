use anyhow::{bail, Error, Result};
use calm_io::*;
use clap::{arg, crate_version, value_parser, ArgMatches, Command};
use oxygraph::{
    bipartite, BipartiteGraph, BipartiteStats, DerivedGraphStats, DerivedGraphs, InteractionMatrix,
    InteractionMatrixStats, LpaWbPlus, NullModel, PermutationTestResult,
};
use rayon::prelude::*;
use std::{io::Write, path::PathBuf};

/// Create the CLI in clap.
///
/// Better to have subcommands for each of derived + interaction matrix.
pub fn cli() -> Command {
    Command::new("oxygraphis")
        .bin_name("oxygraphis")
        .arg_required_else_help(true)
        .version(crate_version!())
        .author("Max Brown <max.carter-brown@aru.ac.uk>")
        .about("Analyse bipartite ecological networks. Compute nestedness, specialisation, modularity, and derived graphs from a delimited edge list.")
        .subcommand(
            Command::new("bipartite")
                .about("Load and analyse a bipartite graph from a delimited edge list.")
                .long_about(
                    "Load a bipartite graph from a tab-separated (or custom-delimited) edge list \
                    with columns `from`, `to`, and `weight`. The graph must be strictly bipartite: \
                    all edges go from one stratum (e.g. parasites) to the other (e.g. hosts). \
                    Use subcommands to analyse the graph as an interaction matrix, derived graphs, \
                    or to compute modularity. Flags on this command return graph-level summaries \
                    (degrees, degree distributions, plots) without requiring a subcommand."
                )
                .arg_required_else_help(true)
                .arg(
                    arg!(<INPUT_DSV> "Path to a delimited file with columns: from, to, weight. \
                        Rows represent edges. The `from` column is treated as the parasite/row stratum \
                        and `to` as the host/column stratum. Weights must be numeric (integer or float).")
                        .required(true)
                        .value_parser(value_parser!(PathBuf)),
                )
                .arg(
                    arg!(-d --delimiter [DELIMITER] "Column delimiter character. Defaults to tab (\\t). \
                        Pass a single character, e.g. -d ',' for CSV.")
                        .required(false),
                )
                .arg(
                    arg!(-p --plotbp "Render an SVG bipartite graph and print to stdout. \
                        Nodes are scaled uniformly. Pipe to a file: oxygraphis bipartite input.tsv --plotbp > out.svg")
                        .conflicts_with("plotbp2")
                        .action(clap::ArgAction::SetTrue)
                )
                .arg(
                    arg!(-q --plotbp2 "Render an SVG bipartite graph with node size proportional to degree. \
                        Useful for visualising hub species. Pipe to a file.")
                        .conflicts_with("plotbp")
                        .action(clap::ArgAction::SetTrue)
                )
                .arg(
                    arg!(--degrees "Print the degree of every node (number of interaction partners). \
                        Output columns: spp, stratum, value.")
                        .action(clap::ArgAction::SetTrue)
                )
                .arg(
                    arg!(-e --degreedistribution "Print the degree distribution for each stratum as a frequency table. \
                        Output columns: stratum, degree, count.")
                        .action(clap::ArgAction::SetTrue)
                )
                .arg(
                    arg!(-b --bivariatedistribution "Print the bivariate degree distribution: the joint frequency of \
                        (parasite degree, host degree) across all edges. Useful for detecting degree-degree correlations.")
                        .action(clap::ArgAction::SetTrue)
                )
                .subcommand(
                    Command::new("interaction-matrix")
                        .about("Coerce the bipartite graph into an interaction matrix and compute network metrics.")
                        .long_about(
                            "Builds an n×m interaction matrix (parasites × hosts) from the edge list, \
                            then computes one or more metrics. With no metric flags, prints summary \
                            statistics: whether weights are present, matrix dimensions, percentage fill, \
                            and link density. Metric flags can be combined freely."
                        )
                        .arg(
                            arg!(--print "Print the raw interaction matrix as a TSV to stdout. \
                                Rows are parasites, columns are hosts. Useful for debugging or passing \
                                to downstream tools.")
                                .action(clap::ArgAction::SetTrue)
                        )
                        .arg(
                            arg!(-p --plotim "Render an SVG heatmap of the interaction matrix and print to stdout. \
                                Cell colour intensity is proportional to edge weight.")
                                .action(clap::ArgAction::SetTrue)
                        )
                        .arg(
                            arg!(-n --nodf "Compute NODF (Nestedness metric based on Overlap and Decreasing Fill). \
                                The matrix is sorted by decreasing marginal totals before calculation. \
                                Outputs the NODF score (0–100, higher = more nested). \
                                Use --weighted or --wbinary for weighted variants. \
                                Use --permutations N to test significance against a null model.")
                                .action(clap::ArgAction::SetTrue)
                        )
                        .arg(
                            arg!(-w --weighted "Compute weighted NODF instead of binary NODF. \
                                Interactions are compared by weight rather than presence/absence. \
                                Requires --nodf.")
                                .action(clap::ArgAction::SetTrue)
                                .requires("nodf")
                        )
                        .arg(
                            arg!(--wbinary "Compute weighted-binary NODF: applies a binary filter \
                                to the weighted matrix before computing nestedness. Requires --nodf.")
                                .action(clap::ArgAction::SetTrue)
                                .requires("nodf")
                        )
                        .arg(
                            arg!(-P --permutations [PERMUTATIONS] "Run a null-model significance test with N null \
                                matrices (see --null). Applies to --nodf, --h2 and --dprime. Outputs the observed \
                                value, null mean, null SD, standardised effect size z, one-tailed P = (k + 1)/(N + 1), \
                                N and the null model. For --dprime, outputs per-species null means and 95% intervals.")
                                .value_parser(value_parser!(usize))
                        )
                        .arg(
                            arg!(--null [NULL] "Null model for --permutations. r00: shuffle all cells \
                                (fixed fill only). patefield: random integer matrices with the observed \
                                row and column totals (as R's r2dtable); use for H2', d' and weighted metrics. \
                                curveball: random binary matrices with the observed row and column degrees \
                                (Strona et al. 2014); use for NODF. [default: r00]")
                                .value_parser(["r00", "patefield", "curveball"])
                                .default_value("r00")
                        )
                        .arg(
                            arg!(--seed [SEED] "Random seed for the null matrices, for reproducible results. \
                                Null matrix i uses seed + i.")
                                .value_parser(value_parser!(u64))
                        )
                        .arg(
                            arg!(--trades [TRADES] "Curveball trades per null matrix. Each null matrix starts \
                                from the observed matrix. [default: max(1000, 50 x number of rows)]")
                                .value_parser(value_parser!(usize))
                        )
                        .arg(
                            arg!(-d --dprime <PARTITION> "Compute d' (d-prime) for each species in the chosen stratum. \
                                d' measures how much a species deviates from using partners in proportion \
                                to their overall availability (marginal frequencies). \
                                0 = complete generalist (uses partners proportional to abundance); \
                                1 = complete specialist (uses only a subset regardless of availability). \
                                Also prints the mean d' across the stratum. \
                                Choose 'parasites' for row species or 'hosts' for column species.")
                                .value_parser(clap::builder::PossibleValuesParser::new(["parasites", "hosts"]))
                        )
                        .arg(
                            arg!(--h2 "Compute H2' (H2-prime), a network-level specialisation index. \
                                H2' measures how much the whole network deviates from random partner use, \
                                scaled between the most generalised (H2'=0) and most specialised (H2'=1) \
                                network possible given the observed marginal totals. \
                                Works with both integer and continuous weights. \
                                Use --permutations N to test significance against a null model.")
                                .action(clap::ArgAction::SetTrue)
                        )
                )
                .subcommand(Command::new("derived-graphs")
                    .about("Project the bipartite graph into unipartite derived graphs for each stratum.")
                    .long_about(
                        "Constructs two unipartite derived graphs: one connecting parasites that \
                        share hosts, and one connecting hosts that share parasites. Edge weights \
                        are the number of shared partners. Without flags, prints summary statistics \
                        (node and edge counts before and after filtering). Use --overlap to get \
                        pairwise Jaccard similarity between parasite species based on shared hosts, \
                        or --plotdg to visualise one stratum's derived graph."
                    )
                    .arg(
                        arg!(-p --plotdg "Render an SVG of the derived graph for the chosen stratum and print to stdout. \
                            Requires --stratum. Node size is proportional to degree in the derived graph.")
                            .action(clap::ArgAction::SetTrue)
                            .requires("stratum")
                    )
                    .arg(
                        arg!(-s --stratum [STRATUM] "Which stratum's derived graph to plot or summarise. \
                            'host': connect hosts that share parasites. \
                            'parasite': connect parasites that share hosts. [default: host]")
                            .num_args(1)
                            .default_value("host")
                            .value_parser(["host", "parasite"])
                    )
                    .arg(
                        arg!(-r --remove [REMOVE] "Remove edges from the derived graph with weight below this threshold \
                            before plotting or summarising. Higher values retain only strongly overlapping pairs. \
                            [default: 2.0]")
                            .default_value("2.0")
                            .value_parser(value_parser!(f64))
                    )
                    .arg(
                        arg!(-d --diameter [DIAMETER] "Width and height of the SVG plot in pixels (plot is square). \
                            [default: 600.0]")
                            .default_value("600.0")
                            .value_parser(value_parser!(f64))
                    )
                    .arg(
                        arg!(-v --overlap "Print pairwise Jaccard host-overlap between all parasite species. \
                            Jaccard = shared_hosts / union_hosts. Output columns: sp1, sp2, jaccard. \
                            Sorted by descending Jaccard (most similar pairs first).")
                            .action(clap::ArgAction::SetTrue)
                    )
                )
                .subcommand(Command::new("modularity")
                    .about("Compute the modularity of the bipartite network.")
                    .long_about(
                        "Detects modules (groups of parasites and hosts that interact more with each \
                        other than with the rest of the network) using label-propagation algorithms \
                        optimised for bipartite graphs. Modularity Q ranges from 0 (no modular \
                        structure) to 1 (perfectly modular). Use --plotmod to also write module \
                        assignments and the sorted interaction matrix to files."
                    )
                    .arg(
                        arg!(-l --lpawbplus "Compute modularity using LPAwb+, the standard weighted \
                            bipartite label-propagation algorithm (Beckett 2016). Faster but less \
                            thorough than DIRTLPAwb+. Mutually exclusive with --dirtlpawbplus.")
                            .action(clap::ArgAction::SetTrue)
                            .conflicts_with("dirtlpawbplus")
                    )
                    .arg(
                        arg!(-d --dirtlpawbplus "Compute modularity using DIRTLPAwb+ (Beckett 2016), \
                            which reruns LPAwb+ from multiple starting conditions to escape local optima. \
                            Recommended for publication-quality results. Slower than --lpawbplus. \
                            Mutually exclusive with --lpawbplus.")
                            .action(clap::ArgAction::SetTrue)
                            .conflicts_with("lpawbplus")
                    )
                    .arg(
                        arg!(-P --permutations [PERMUTATIONS] "Test Q against N null matrices (see --null), \
                            running the chosen algorithm on each. Outputs the observed Q, null mean, null SD, \
                            z, one-tailed P = (k + 1)/(N + 1), N and the null model.")
                            .value_parser(value_parser!(usize))
                    )
                        .arg(
                        arg!(--null [NULL] "Null model for --permutations. r00: shuffle all cells \
                            (fixed fill only). patefield: random integer matrices with the observed \
                            row and column totals (as R's r2dtable); use for H2', d' and weighted metrics. \
                            curveball: random binary matrices with the observed row and column degrees \
                            (Strona et al. 2014); use for NODF. [default: patefield]")
                            .value_parser(["r00", "patefield", "curveball"])
                            .default_value("patefield")
                    )
                    .arg(
                        arg!(--seed [SEED] "Random seed for the null matrices, for reproducible results. \
                            Null matrix i uses seed + i.")
                            .value_parser(value_parser!(u64))
                    )
                    .arg(
                        arg!(--trades [TRADES] "Curveball trades per null matrix. Each null matrix starts \
                            from the observed matrix. [default: max(1000, 50 x number of rows)]")
                            .value_parser(value_parser!(usize))
                    )
                    .arg(
                        arg!(--mini [MINI] "DIRTLPAwb+ only: minimum number of modules from which to restart \
                            label propagation (Beckett 2016). [default: 4]")
                            .value_parser(value_parser!(u32))
                            .default_value("4")
                    )
                    .arg(
                        arg!(--reps [REPS] "DIRTLPAwb+ only: number of LPAwb+ restarts per module number, \
                            run in parallel (Beckett 2016). [default: 10]")
                            .value_parser(value_parser!(u32))
                            .default_value("10")
                    )
                    .arg(
                        arg!(-p --plotmod "In addition to the Q value, write two files to the output directory: \
                            (1) an SVG interaction matrix sorted by module membership, and \
                            (2) a TSV listing each (module, parasite, host) assignment. \
                            Requires --lpawbplus or --dirtlpawbplus.")
                            .action(clap::ArgAction::SetTrue)
                    )
                    .arg(
                        arg!(-o --output [OUTPUT] "Directory to write module output files when --plotmod is used. \
                            Files are named '<algorithm>_interaction_matrix.tsv' and '<algorithm>_modules.tsv'. \
                            [default: current directory]")
                            .value_parser(value_parser!(PathBuf))
                            .default_value(".")
                    )
                )
            )
            .subcommand(Command::new("simulate")
                .about("Simulate random Erdős–Rényi bipartite graphs and compute metrics over the ensemble.")
                .long_about(
                    "Generates N random bipartite graphs with the specified number of parasite nodes, \
                    host nodes, and edges, then runs a chosen calculation on each. Useful for building \
                    null distributions and understanding expected metric values under random wiring."
                )
                .arg(
                    arg!(--parasitenumber <PARASITENUMBER> "Number of parasite (row) nodes in each simulated graph.")
                        .required(true)
                        .value_parser(value_parser!(usize))
                )
                .arg(
                    arg!(--hostnumber <HOSTNUMBER> "Number of host (column) nodes in each simulated graph.")
                        .required(true)
                        .value_parser(value_parser!(usize))
                )
                .arg(
                    arg!(-e --edgecount <EDGECOUNT> "Number of edges to place in each simulated graph. \
                        Must be ≤ parasitenumber × hostnumber.")
                        .required(true)
                        .value_parser(value_parser!(usize))
                )
                .arg(
                    arg!(-n --nsims [NSIMS] "Number of random graphs to generate. \
                        The chosen calculation is run on each. [default: 1000]")
                        .value_parser(value_parser!(i32))
                        .default_value("1000")
                )
                .arg(
                    arg!(-c --calculation [CALCULATION] "Metric to compute on each simulated graph. \
                        nodf: binary NODF nestedness score. \
                        lpawbplus: modularity Q via LPAwb+. \
                        dirtlpawbplus: modularity Q via DIRTLPAwb+. \
                        degree-distribution: degree distribution for each stratum. \
                        bivariate-distribution: joint (parasite degree, host degree) distribution. \
                        [default: nodf]")
                        .default_value("nodf")
                        .value_parser(["nodf", "lpawbplus", "dirtlpawbplus", "degree-distribution", "bivariate-distribution"])
                )
                .arg(
                    arg!(--mini [MINI] "DIRTLPAwb+ only: minimum number of modules from which to restart \
                        label propagation (Beckett 2016). [default: 4]")
                        .value_parser(value_parser!(u32))
                        .default_value("4")
                    )
                .arg(
                    arg!(--reps [REPS] "DIRTLPAwb+ only: number of LPAwb+ restarts per module number, \
                        run in parallel (Beckett 2016). [default: 10]")
                        .value_parser(value_parser!(u32))
                        .default_value("10")
                    )
                .arg(
                    arg!(--plot "Render an SVG of the first simulated bipartite graph and print to stdout.")
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
            let delimiter = match sub_matches.get_one::<String>("delimiter") {
                Some(d) => d.bytes().next().unwrap_or(b'\t'),
                None => b'\t',
            };
            // did user want a bipartite plot?
            let bipartite_plot = *sub_matches
                .get_one::<bool>("plotbp")
                .expect("defaulted by clap.");
            let bipartite_plot_2 = *sub_matches
                .get_one::<bool>("plotbp2")
                .expect("defaulted by clap.");
            let degrees = *sub_matches
                .get_one::<bool>("degrees")
                .expect("defaulted by clap.");
            let degreedistribution = *sub_matches
                .get_one::<bool>("degreedistribution")
                .expect("defaulted by clap.");
            let bivariate_distribution = *sub_matches
                .get_one::<bool>("bivariatedistribution")
                .expect("defaulted by clap.");

            // everything requires the bipartite graph
            // and must currently go through a DSV.
            // input and delimiter
            let bpgraph = BipartiteGraph::from_dsv(input, delimiter)?;

            match bpgraph.is_bipartite() {
                // don't care here
                bipartite::Strata::Yes(_) => (),
                // tell the user which nodes are the offenders.
                bipartite::Strata::No => {
                    return Err(Error::msg(
                        "Graph is not bipartite. Check the input file for errors.",
                    ))
                }
            }

            match sub_matches.subcommand() {
                // user just called bipartite
                None => {
                    // pass args from above
                    if bipartite_plot {
                        // make the plot dims CLI args.
                        // but 600 x 400 for now.
                        stdoutln!("{}", bpgraph.plot(1600, 700))?;
                    } else if bipartite_plot_2 {
                        stdoutln!("{}", bpgraph.plot_prop(1800, 700))?;
                    } else if degrees {
                        let degs = bpgraph.degrees(None);
                        stdoutln!("spp\tstratum\tvalue")?;
                        for (s, p, v) in degs {
                            stdoutln!("{}\t{}\t{}", s, p, v)?;
                        }
                    } else if degreedistribution {
                        let (bin_size, deg_dist) = bpgraph.degree_distribution(None, false);
                        // print the distribution
                        stdoutln!("degree\tcount")?;

                        for (deg, count) in deg_dist {
                            let bin_end = deg + bin_size - 1.0;
                            stdoutln!("{}-{}\t{}", deg, bin_end, count)?;
                        }
                    } else if bivariate_distribution {
                        let biv_dist = bpgraph.bivariate_degree_distribution();
                        stdoutln!("node1\tnode2")?;
                        for (n1, n2) in biv_dist {
                            stdoutln!("{}\t{}", n1, n2)?;
                        }
                    } else {
                        // default subcommand output
                        // probably pass this to another function later.
                        let BipartiteStats {
                            no_parasites,
                            no_hosts,
                            no_edges,
                        } = bpgraph.stats();
                        stdoutln!("#_parasite_nodes\t#_host_nodes\t#_total_edges")?;
                        stdoutln!("{}\t{}\t{}", no_parasites, no_hosts, no_edges)?;
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
                    let weighted = *im_matches
                        .get_one::<bool>("weighted")
                        .expect("defaulted by clap.");
                    let wbinary = *im_matches
                        .get_one::<bool>("wbinary")
                        .expect("defaulted by clap.");
                    let h2 = *im_matches.get_one::<bool>("h2").expect("defaulted by clap");
                    let d_prime = im_matches.get_one::<String>("dprime").cloned();
                    let print = *im_matches
                        .get_one::<bool>("print")
                        .expect("defaulted by clap.");
                    let permutations = im_matches.get_one::<usize>("permutations").copied();

                    if im_plot {
                        // change these, especially height might need to
                        // be auto generated
                        im_mat.sort();
                        im_mat.plot(1600, None);
                    } else if let Some(dp) = d_prime {
                        let partition = if dp == "hosts" {
                            bipartite::Partition::Hosts
                        } else {
                            bipartite::Partition::Parasites
                        };
                        if let Some(n) = permutations {
                            let (model, seed, trades) = null_settings(im_matches, im_mat.inner.nrows())?;
                            let res = im_mat
                                .dprime_null_test(partition, n, model, trades, seed)
                                .map_err(|e| Error::msg(format!("{}", e)))?;
                            stdoutln!("species\tdprime\tmean_null\tlower_null\tupper_null\tp_value\tn_perms\tnull_model")?;
                            for r in res {
                                let obs = r.observed.map(|e| e.to_string()).unwrap_or("None".into());
                                stdoutln!(
                                    "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                                    r.species, obs, r.mean_null, r.lower_null, r.upper_null, r.p_value, n, model
                                )?;
                            }
                            return Ok(());
                        }
                        let dprimes = im_mat.d_prime(partition, None);
                        for (spp, d_prime) in &dprimes {
                            let d_prime_fmt =
                                d_prime.map(|e| e.to_string()).unwrap_or("None".into());
                            stdoutln!("{}\t{}", spp, d_prime_fmt)?;
                        }
                        let mean = im_mat.mean_d_prime(partition);
                        stdoutln!("mean_d'\t{}", mean)?;
                    } else if h2 {
                        let h2 = im_mat.h2_prime();
                        stdoutln!("H2'\t{}", h2)?;
                        if let Some(n) = permutations {
                            let (model, seed, trades) = null_settings(im_matches, im_mat.inner.nrows())?;
                            let r = im_mat
                                .null_test(n, model, trades, seed, |m| m.h2_prime())
                                .map_err(|e| Error::msg(format!("{}", e)))?;
                            print_null_test(&r)?;
                        }
                    } else if nodf {
                        let nodf_result = im_mat.nodf(true, weighted, wbinary);
                        stdoutln!("NODF\t{}", nodf_result.nodf)?;
                        if let Some(n) = permutations {
                            let (model, seed, trades) = null_settings(im_matches, im_mat.inner.nrows())?;
                            let r = im_mat
                                .null_test(n, model, trades, seed, |m| m.nodf(true, weighted, wbinary).nodf)
                                .map_err(|e| Error::msg(format!("{}", e)))?;
                            print_null_test(&r)?;
                        }
                    } else if print {
                        stdoutln!("{}", im_mat)?;
                    } else {
                        // default subcommand output
                        let InteractionMatrixStats {
                            weighted,
                            no_rows,
                            no_cols,
                            no_poss_ints,
                            perc_ints,
                            link_density,
                        } = im_mat.stats();
                        stdoutln!("weighted\t#_rows\t#_cols\t#_poss_ints\tperc_ints\tlink_density")?;
                        stdoutln!(
                            "{}\t{}\t{}\t{}\t{}\t{}",
                            weighted,
                            no_rows,
                            no_cols,
                            no_poss_ints,
                            perc_ints * 100.0,
                            link_density
                        )?;
                    }
                }
                // user called derived-graphs
                Some(("derived-graphs", dg_matches)) => {
                    let dgs = DerivedGraphs::from_bipartite(bpgraph);

                    let dg_plot = *dg_matches
                        .get_one::<bool>("plotdg")
                        .expect("defaulted by clap.");
                    let stratum = dg_matches
                        .get_one::<String>("stratum")
                        .expect("defaulted by clap.");
                    let remove = *dg_matches
                        .get_one::<f64>("remove")
                        .expect("defaulted by clap.");
                    let diameter = *dg_matches
                        .get_one::<f64>("diameter")
                        .expect("defaulted by clap.");

                    let overlap = *dg_matches
                        .get_one::<bool>("overlap")
                        .expect("defaulted by clap.");

                    if dg_plot {
                        let svg = match stratum.as_str() {
                            "host" => dgs.hosts.plot(diameter, remove),
                            "parasite" => dgs.parasites.plot(diameter, remove),
                            _ => unreachable!("Should never reach here."),
                        };
                        stdoutln!("{}", svg)?;
                    } else if overlap {
                        stdoutln!("sp1\tsp2\tjaccard")?;
                        let mut pairs = dgs.parasites.overlap_measure();
                        pairs.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
                        for (sp1, sp2, j) in pairs {
                            stdoutln!("{}\t{}\t{}", sp1, sp2, j)?;
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

                        stdoutln!("p_nodes\tp_edges\tp_edge_fil\th_nodes\th_edges\th_edge_fil")?;
                        stdoutln!(
                            "{}\t{}\t{}\t{}\t{}\t{}",
                            parasite_nodes,
                            parasite_edges,
                            parasite_edges_filtered,
                            host_nodes,
                            host_edges,
                            host_edges_filtered
                        )?;
                    }
                }
                Some(("modularity", mod_matches)) => {
                    let lpawbplus = *mod_matches
                        .get_one::<bool>("lpawbplus")
                        .expect("defaulted by clap.");
                    let dirtlpawbplus = *mod_matches
                        .get_one::<bool>("dirtlpawbplus")
                        .expect("defaulted by clap.");
                    let plot = *mod_matches
                        .get_one::<bool>("plotmod")
                        .expect("defaulted by clap.");
                    let dir = mod_matches
                        .get_one::<PathBuf>("output")
                        .expect("defaulted by clap.");

                    let mod_seed = mod_matches.get_one::<u64>("seed").copied();
                    let mini = *mod_matches.get_one::<u32>("mini").expect("defaulted by clap.");
                    let reps = *mod_matches.get_one::<u32>("reps").expect("defaulted by clap.");

                    // create the interaction matrix
                    let int_mat = InteractionMatrix::from_bipartite(bpgraph);

                    if plot {
                        let kind: &str;
                        let mut modularity_obj = if dirtlpawbplus {
                            kind = "DIRTLPAwb+";
                            match mod_seed {
                                Some(sd) => int_mat.dirt_lpa_wb_plus_seeded(mini, reps, sd),
                                None => int_mat.clone().dirt_lpa_wb_plus(mini, reps),
                            }
                        } else {
                            kind = "LPAwb+";
                            match mod_seed {
                                Some(sd) => int_mat.lpa_wb_plus_seeded(None, sd),
                                None => int_mat.clone().lpa_wb_plus(None),
                            }
                        };
                        let modularity = modularity_obj.modularity;
                        let (int_mat, modules) = modularity_obj.plot(int_mat);

                        // create the file path for the interaction matrix
                        let im_path = dir.join(format!("{}_interaction_matrix.tsv", kind));

                        // write the interaction matrix to a TSV file
                        int_mat.write_tsv(im_path, kind)?;

                        // now write the modules to a TSV file
                        let modules_path = dir.join(format!("{}_modules.tsv", kind));
                        let mut modules_file = std::fs::File::create(modules_path)?;

                        let module_header = format!("# {} modularity: {}\n", kind, modularity);
                        modules_file.write_all(module_header.as_bytes())?;

                        let module_headers = format!("module\tparasite\thost\n");
                        modules_file.write_all(module_headers.as_bytes())?;
                        for (module, s) in modules.unwrap() {
                            for (host, parasite) in s.iter() {
                                let line = format!("{}\t{}\t{}\n", module, host, parasite);
                                modules_file.write_all(line.as_bytes())?;
                            }
                        }
                        modules_file.flush()?;
                    } else if let Some(n) = mod_matches.get_one::<usize>("permutations").copied() {
                        if !(dirtlpawbplus || lpawbplus) {
                            return Err(Error::msg("Please specify --lpawbplus or --dirtlpawbplus."));
                        }
                        let (model, seed, trades) = null_settings(mod_matches, int_mat.inner.nrows())?;
                        let r = int_mat
                            .null_test_seeded(n, model, trades, seed, |m, sd| {
                                if dirtlpawbplus {
                                    m.dirt_lpa_wb_plus_seeded(mini, reps, sd).modularity
                                } else {
                                    m.lpa_wb_plus_seeded(None, sd).modularity
                                }
                            })
                            .map_err(|e| Error::msg(format!("{}", e)))?;
                        stdoutln!("{}", if dirtlpawbplus { "DIRTLPAwb+" } else { "LPAwb+" })?;
                        print_null_test(&r)?;
                    } else if dirtlpawbplus {
                        let LpaWbPlus { modularity, .. } = match mod_seed {
                            Some(sd) => int_mat.dirt_lpa_wb_plus_seeded(mini, reps, sd),
                            None => int_mat.dirt_lpa_wb_plus(mini, reps),
                        };
                        stdoutln!("DIRTLPAwb+\n{}", modularity)?;
                    } else if lpawbplus {
                        let LpaWbPlus { modularity, .. } = match mod_seed {
                            Some(sd) => int_mat.lpa_wb_plus_seeded(None, sd),
                            None => int_mat.lpa_wb_plus(None),
                        };
                        stdoutln!("LPAwb+\n{}", modularity)?;
                    } else {
                        return Err(Error::msg("Please specify --lpawbplus or --dirtlpawbplus."));
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

            let calculation = sm_matches
                .get_one::<String>("calculation")
                .expect("defaulted by clap.");
            let mini = *sm_matches.get_one::<u32>("mini").expect("defaulted by clap.");
            let reps = *sm_matches.get_one::<u32>("reps").expect("defaulted by clap.");

            if plot {
                let rand_graph = BipartiteGraph::random(parasite_number, host_number, edge_count)?;

                stdoutln!("{}", rand_graph.plot(1000, 400))?;

                // return early here.
                return Ok(());
            }

            (0..n_sims).into_par_iter().try_for_each(|_| {
                {
                    let rand_graph =
                        BipartiteGraph::random(parasite_number, host_number, edge_count).unwrap();

                    match calculation.as_str() {
                        "nodf" => {
                            let im_mat = InteractionMatrix::from_bipartite(rand_graph);
                            let nodf = im_mat.nodf(true, false, false);
                            if !nodf.nodf.is_nan() {
                                stdoutln!("{}", nodf.nodf)?;
                            }
                            Ok::<(), Error>(())
                        }
                        "lpawbplus" => {
                            let im_mat = InteractionMatrix::from_bipartite(rand_graph);
                            let LpaWbPlus { modularity, .. } = im_mat.lpa_wb_plus(None);
                            stdoutln!("{}", modularity)?;
                            Ok::<(), Error>(())
                        }
                        "dirtlpawbplus" => {
                            let im_mat = InteractionMatrix::from_bipartite(rand_graph);
                            let LpaWbPlus { modularity, .. } = im_mat.dirt_lpa_wb_plus(mini, reps);
                            stdoutln!("{}", modularity)?;
                            Ok::<(), Error>(())
                        }
                        // not sure how to implement these two yet, or how useful they will be.
                        "degree-distribution" => {
                            bail!("Degree distribution simulations are not yet implemented.")
                        }
                        "bivariate-distribution" => {
                            bail!("Bivariate distributions not yet implemented.")
                        }
                        _ => unreachable!("clap should make sure we never reach here."),
                    }
                }
            })?;
        }
        _ => unreachable!("Should never reach here."),
    }

    Ok(())
}

/// Parse the shared null-model arguments.
fn null_settings(m: &ArgMatches, nrows: usize) -> Result<(NullModel, Option<u64>, usize)> {
    let model: NullModel = m
        .get_one::<String>("null")
        .expect("defaulted by clap.")
        .parse()
        .map_err(|e| Error::msg(format!("{}", e)))?;
    let seed = m.get_one::<u64>("seed").copied();
    let trades = m
        .get_one::<usize>("trades")
        .copied()
        .unwrap_or_else(|| oxygraph::null::default_trades(nrows));
    Ok((model, seed, trades))
}

/// Print a null-model test result as a two-line TSV.
fn print_null_test(r: &PermutationTestResult) -> Result<()> {
    stdoutln!("obs\tmean_null\tsd_null\tz\tp_value\tn_perms\tnull_model")?;
    stdoutln!(
        "{}\t{}\t{}\t{}\t{}\t{}\t{}",
        r.observed, r.mean_null, r.sd_null, r.z, r.p_value, r.n_permutations, r.null_model
    )?;
    Ok(())
}
