use std::fs;
use serde::Serialize;
use serde::Deserialize;
use std::path::PathBuf;
use anyhow::{Result, Error};


const DEFAULT_MAX_LINES_PER_PATH: usize = 1_000_000_000;
const DEFAULT_TOKENIZER: &str = "bytes"; //.to_string();
const DEFAULT_TEXT_KEY: &str = "text";
const DEFAULT_RANDOM_SEED: u64 = 42;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SAConfig {
    pub tokenizer: String,
    /// Defaults to using bytes
    pub max_lines_per_path: usize,
    /// Defaults to 1_000_000_000
    pub text_key: String,
    pub random_seed: u64,
}

impl Default for SAConfig {
    fn default() -> Self {
        Self {
            tokenizer: DEFAULT_TOKENIZER.to_string(),
            max_lines_per_path: DEFAULT_MAX_LINES_PER_PATH,
            text_key: DEFAULT_TEXT_KEY.to_string(),
            random_seed: DEFAULT_RANDOM_SEED,
        }
    }
}

pub struct SAConfigOverrides {
    pub tokenizer: Option<String>,
    pub max_lines_per_path: Option<usize>,
    pub text_key: Option<String>,
    pub random_seed: Option<u64>,
}

impl SAConfigOverrides {
    fn apply_to(self, config: &mut SAConfig) {
        if let Some(val) = self.tokenizer {
            config.tokenizer = String::from(val.clone());
        }

        if let Some(val) = self.max_lines_per_path {
            config.max_lines_per_path = val;
        }

        if let Some(val) = self.text_key {
            config.text_key = String::from(val);
        }

        if let Some(val) = self.random_seed {
            config.random_seed = val;
        }
    }
}

impl SAConfig {
    pub fn load_with_overrides(
        config_path: Option<PathBuf>,
        overrides: SAConfigOverrides,
    ) -> Result<Self, Error> {
        let mut config = Self::default();
        if let Some(path) = config_path {
            let config_content = fs::read_to_string(path)?;
            let file_config: SAConfig = serde_yaml::from_str(&config_content)?;
            config = file_config;
        }

        overrides.apply_to(&mut config);
        Ok(config)
    }
}
