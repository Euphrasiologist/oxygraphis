use anyhow::Result;
use oxygraphis::{cli, process_matches};

fn main() -> Result<()> {
    let matches = cli().get_matches();

    process_matches(&matches)?;

    Ok(())
}
