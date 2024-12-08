# `oxygraphis`

## Introduction

`oxygraphis` is a lightweight small crate, and command line tool to interact with bipartite ecological graphs.

<img src="./euphrasia_hp.svg">

## CLI details

### Install

Currently you will need to clone this repository and build from source. Never fear, just download the rust toolchain. Then:

```bash
# install from crates.io (oxygraphis 0.1.62)
cargo install oxygraphis
# or from this repo
git clone https://github.com/Euphrasiologist/oxygraphis
cd oxygraphis
# install to path.
cargo install --path=.
```

### Interface

Bipartite graphs are the graph of interest. Analyse these graphs directly, or simulate them.

```bash
Usage: oxygraphis [COMMAND]

Commands:
  bipartite  Generate and analyse bipartite graphs.
  simulate   Simulate a number of graphs, and return calculations over the samples.
  help       Print this message or the help of the given subcommand(s)

Options:
  -h, --help     Print help
  -V, --version  Print version
```

#### Bipartite graphs

The `bipartite` subcommand.

```bash
Generate and analyse bipartite graphs.

Usage: oxygraphis bipartite [OPTIONS] <INPUT_DSV> [COMMAND]

Commands:
  interaction-matrix  Coerce a bipartite graph into an interaction matrix.
  derived-graphs      Coerce a bipartite graph into two derived graphs.
  modularity          Derive the modularity of a bipartite graph.
  help                Print this message or the help of the given subcommand(s)

Arguments:
  <INPUT_DSV>  An input DSV with three headers only: from, to, and weight.

Options:
  -d, --delimiter [<DELIMITER>]  Specify the delimiter of the DSV; we assume tabs.
  -p, --plotbp                   Render an SVG bipartite graph plot.
  -q, --plotbp2                  Render an SVG bipartite graph plot with proportional node size.
  -d, --degrees                  Return the degrees of a bipartite graph.
  -e, --degreedistribution       Return the degree distribution of a bipartite graph.
  -b, --bivariatedistribution    Return the bivariate degree distribution of a bipartite graph.
  -h, --help                     Print help
```

The input must be a delimited file with three columns only:

```txt
from    to    weight
Sp1    Sp2    1.0
Sp2    Sp3    1.0
Sp1    Sp3    1.0
```

And I guess should be bipartite in structure (i.e. edges only from stratum 1 -> stratum 2). A warning is issued if it isn't, but does not halt the program.

#### Derived graphs

Derived graphs are graphs which show the relationships between species in a stratum.

```bash
Coerce a bipartite graph into two derived graphs.

Usage: oxygraphis bipartite <INPUT_DSV> derived-graphs [OPTIONS]

Options:
  -p, --plotdg                 Render an SVG derived graph of a stratum.
  -s, --stratum <STRATUM>      The stratum to display. [default: host] [possible values: host, parasite]
  -r, --remove [<REMOVE>]      Edges with fewer than this number of connections are removed from the graph. [default: 2.0]
  -d, --diameter [<DIAMETER>]  The diameter (width and height; plot is square) of the plot. [default: 600.0]
  -h, --help                   Print help
```

#### Interaction matrix

These form the core of some interesting bipartite analyses. Essentially an `n x m` matrix of all possible species-species interactions in the network.

```bash
Coerce a bipartite graph into an interaction matrix.

Usage: oxygraphis bipartite <INPUT_DSV> interaction-matrix [OPTIONS]

Options:
      --print   Print the inner matrix as a TSV. Mainly for debugging.
  -p, --plotim  Render an SVG interaction matrix plot.
  -n, --nodf    Compute the NODF number of a *sorted* interaction matrix.
  -h, --help    Print help
```

#### Modularity 

An algorithm operating on the interaction matrix made from a bipartite graph. It attempts to find modules of species-host interactions in a matrix.

```bash
Derive the modularity of a bipartite graph.

Usage: oxygraphis bipartite <INPUT_DSV> modularity [OPTIONS]

Options:
  -l, --lpawbplus      Compute the modularity of a bipartite network using LPAwb+ algorithm.
  -d, --dirtlpawbplus  Compute the modularity of a bipartite network using DIRTLPAwb+ algorithm.
  -m, --modules        Compute the modules of a bipartite graph.
  -p, --plotmod        Plot the interaction matrix of a bipartite network, sorted to maximise modularity.
  -h, --help           Print help
```

#### Simulations

A subcommand to simulate a number of random graphs (Erdös-Rényi) and execute a calculation on each.

```bash
Simulate a number of graphs, and return calculations over the samples.

Usage: oxygraphis simulate [OPTIONS] --parasitenumber <PARASITENUMBER> --hostnumber <HOSTNUMBER> --edgecount <EDGECOUNT>

Options:
      --parasitenumber <PARASITENUMBER>
          Number of parasite nodes in the graph.
      --hostnumber <HOSTNUMBER>
          Number of host nodes in the graph.
  -e, --edgecount <EDGECOUNT>
          Number of edges in the graph.
  -n, --nsims [<NSIMS>]
          Number of random samples to make. [default: 1000]
  -c, --calculation [<CALCULATION>]
          The calculation to make. [default: nodf] [possible values: nodf, lpawbplus, dirtlpawbplus, degree-distribution, bivariate-distribution]
      --plot
          Plot the simulated bipartite network.
  -h, --help
          Print help
```

## Oxygraphis..?

*Oxygraphis* is one of only 5-6 genera in the flowering plants which have *graph* included fully in the name. It's in the Ranunculaceae.
