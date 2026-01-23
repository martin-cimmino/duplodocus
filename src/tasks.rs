use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io;
use std::io::BufRead;
use std::path::PathBuf;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Message {
    content: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Record {
    messages: Vec<Message>,
}

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

pub trait Task {
    fn load_text(&self, path: &PathBuf, key: &str) -> DataFrame;
    fn load(&self, path: &PathBuf) -> DataFrame;
    fn save(&self, df: &mut DataFrame, path: &PathBuf);
}

pub struct CPTTask;
impl Task for CPTTask {
    fn load_text(&self, path: &PathBuf, key: &str) -> DataFrame {
        let plpath = PlPath::from_str(path.to_str().unwrap());
        LazyFrame::scan_parquet(plpath, Default::default())
            .unwrap()
            .select([col(key)])
            .collect()
            .unwrap()
    }

    fn load(&self, path: &PathBuf) -> DataFrame {
        let plpath = PlPath::from_str(path.to_str().expect("failed to create path"));
        LazyFrame::scan_parquet(plpath, Default::default())
            .unwrap()
            .collect()
            .expect("failed to read dataframe")
    }

    fn save(&self, df: &mut DataFrame, path: &PathBuf) {
        let mut file = fs::File::create(path).expect("failed to create file.");
        ParquetWriter::new(&mut file)
            .finish(df)
            .expect("failed to write parquet");
    }
}

pub struct SFTTask;
impl SFTTask {
    fn concat_messages(&self, record: Record) -> String {
        record
            .messages
            .into_iter()
            .map(|message| message.content)
            .collect::<Vec<_>>()
            .join(" ")
    }
}
impl Task for SFTTask {
    fn load_text(&self, path: &PathBuf, key: &str) -> DataFrame {
        let file = io::BufReader::new(fs::File::open(path).unwrap());
        let text: Vec<String> = file
            .lines()
            .map(|line| serde_json::from_str::<Record>(&line.unwrap()).unwrap())
            .map(move |record| self.concat_messages(record))
            .collect();

        df!(key => text).unwrap()
    }

    fn load(&self, path: &PathBuf) -> DataFrame {
        let plpath = PlPath::from_str(path.to_str().expect("failed to create path"));
        LazyJsonLineReader::new(plpath)
            .finish()
            .unwrap()
            .collect()
            .unwrap()
    }

    fn save(&self, df: &mut DataFrame, path: &PathBuf) {
        let mut file = fs::File::create(path).expect("failed to create file.");
        JsonWriter::new(&mut file)
            .with_json_format(JsonFormat::JsonLines)
            .finish(df)
            .unwrap();
    }
}
