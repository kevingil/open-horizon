//! File-system adapter registry.
//!
//! Layout under `root/`: `{adapter_id}/manifest.json` plus whatever
//! weight files the trainer produced. Every list call rereads manifests
//! so adapters dropped in by external tools show up immediately.

use std::path::{Path, PathBuf};

use horizon_core::models::AdapterRecord;

#[derive(Debug, Clone)]
pub struct LocalAdapterRegistry {
    root: PathBuf,
}

impl LocalAdapterRegistry {
    pub fn new(root: impl Into<PathBuf>) -> std::io::Result<Self> {
        let root: PathBuf = root.into();
        std::fs::create_dir_all(&root)?;
        let root = root.canonicalize()?;
        Ok(Self { root })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn list_adapters(&self) -> Vec<AdapterRecord> {
        let mut records = Vec::new();
        let Ok(entries) = std::fs::read_dir(&self.root) else {
            return records;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            let manifest = path.join("manifest.json");
            let Ok(text) = std::fs::read_to_string(&manifest) else {
                continue;
            };
            if let Ok(record) = serde_json::from_str::<AdapterRecord>(&text) {
                records.push(record);
            }
        }
        records.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        records
    }

    pub fn get(&self, adapter_id: &str) -> Option<AdapterRecord> {
        let text = std::fs::read_to_string(self.path_for(adapter_id).join("manifest.json")).ok()?;
        serde_json::from_str(&text).ok()
    }

    /// Persist the manifest, normalising `path` to the registry directory.
    pub fn register(&self, record: &AdapterRecord) -> std::io::Result<AdapterRecord> {
        let dir = self.path_for(&record.id);
        std::fs::create_dir_all(&dir)?;
        let canonical = AdapterRecord {
            path: dir.to_string_lossy().to_string(),
            ..record.clone()
        };
        let text = serde_json::to_string_pretty(&canonical).expect("adapter serializes");
        std::fs::write(dir.join("manifest.json"), text)?;
        Ok(canonical)
    }

    pub fn path_for(&self, adapter_id: &str) -> PathBuf {
        self.root.join(adapter_id)
    }

    pub fn children_of(&self, adapter_id: Option<&str>) -> Vec<AdapterRecord> {
        self.list_adapters()
            .into_iter()
            .filter(|r| r.parent_id.as_deref() == adapter_id)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_list_children() {
        let dir = tempfile::tempdir().unwrap();
        let reg = LocalAdapterRegistry::new(dir.path()).unwrap();
        let parent = AdapterRecord {
            id: "adapter-p".into(),
            parent_id: None,
            base_model: "stub:base".into(),
            training_run_id: None,
            eval_score: None,
            path: String::new(),
            tags: vec![],
            metadata: Default::default(),
            created_at: horizon_core::utc_now(),
        };
        let parent = reg.register(&parent).unwrap();
        assert!(parent.path.ends_with("adapter-p"));
        let child = AdapterRecord {
            id: "adapter-c".into(),
            parent_id: Some("adapter-p".into()),
            ..parent.clone()
        };
        reg.register(&child).unwrap();
        assert_eq!(reg.list_adapters().len(), 2);
        assert_eq!(reg.children_of(Some("adapter-p")).len(), 1);
        assert_eq!(
            reg.get("adapter-c").unwrap().parent_id.as_deref(),
            Some("adapter-p")
        );
        assert!(reg.get("missing").is_none());
    }
}
