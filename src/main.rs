use oxygraphis::{cli, process_matches};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let matches = cli().get_matches();

    process_matches(&matches)?;

    Ok(())
}
