/// CPT/SFT/RL-specific processing
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io;
use std::io::BufRead;
use std::path::PathBuf;

/// Canonical schema message
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Message {
    content: String,
}

/// Canonical schema record
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Record {
    // Messages
    messages: Vec<Message>,
}

/// Get the correct task
pub fn get_task(path: &PathBuf) -> Box<dyn Task> {
    let extension = path
        .extension()
        .expect("Cannot find a valid extension")
        .to_str()
        .expect("failed to convert extension to string");

    match extension {
        "parquet" => Box::new(CPTTask),
        _ => Box::new(SFTTask),
    }
}

/// SFT/CPT-specific operations
pub trait Task {
    // Load the text column
    fn load_text(&self, path: &PathBuf, key: &str) -> DataFrame;
    // Load the whole table
    fn load(&self, path: &PathBuf) -> DataFrame;
    // Save the whole table
    fn save(&self, df: &mut DataFrame, path: &PathBuf);
}

/// CPT task
pub struct CPTTask;
impl Task for CPTTask {
    /// We have a parquet table with a unique text column
    fn load_text(&self, path: &PathBuf, key: &str) -> DataFrame {
        let plpath = PlPath::from_str(path.to_str().unwrap());
        LazyFrame::scan_parquet(plpath, Default::default())
            .expect(&format!("Failed to scan parquet file {}", path.display()))
            .select([col(key)])
            .collect()
            .expect(&format!("Failed to read parquet file {}", path.display()))
    }

    /// Load the whole parquet table
    fn load(&self, path: &PathBuf) -> DataFrame {
        let plpath = PlPath::from_str(path.to_str().expect("failed to create path"));
        LazyFrame::scan_parquet(plpath, Default::default())
            .unwrap()
            .collect()
            .expect("failed to read dataframe")
    }

    /// Save the parquet table
    fn save(&self, df: &mut DataFrame, path: &PathBuf) {
        let mut file = fs::File::create(path).expect("failed to create file.");
        ParquetWriter::new(&mut file)
            .finish(df)
            .expect("failed to write parquet");
    }
}

/// JSONL table with messages
pub struct SFTTask;
impl SFTTask {
    // Concatenate messages in the conversation
    fn concat_messages(&self, record: Record) -> String {
        record
            .messages
            .into_iter()
            .map(|message| message.content)
            .collect::<Vec<_>>()
            .join("\n")
    }
}
impl Task for SFTTask {
    /// Load the SFT text column
    fn load_text(&self, path: &PathBuf, key: &str) -> DataFrame {
        let file = io::BufReader::new(fs::File::open(path).unwrap());
        let text: Vec<String> = file
            .lines()
            .map(|line| serde_json::from_str::<Record>(&line.unwrap()).unwrap())
            .map(move |record| self.concat_messages(record))
            .collect();

        df!(key => text).unwrap()
    }

    /// Load the whole JSONL table
    fn load(&self, path: &PathBuf) -> DataFrame {
        let plpath = PlPath::from_str(path.to_str().expect("failed to create path"));
        LazyJsonLineReader::new(plpath)
            .finish()
            .unwrap()
            .collect()
            .unwrap()
    }

    /// Save the whole JSONL table
    fn save(&self, df: &mut DataFrame, path: &PathBuf) {
        let mut file = fs::File::create(path).expect("failed to create file.");
        JsonWriter::new(&mut file)
            .with_json_format(JsonFormat::JsonLines)
            .finish(df)
            .unwrap();
    }
}
