use anyhow::Result;
use oxygraphis::{cli, process_matches};

fn main() -> Result<()> {
    std::env::set_var("RUST_BACKTRACE", "full");
    let matches = cli().get_matches();

    process_matches(&matches)?;

    Ok(())
}
