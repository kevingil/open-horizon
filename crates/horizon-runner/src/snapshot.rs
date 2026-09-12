//! Cheap per-rollout repo snapshots: copy only git-tracked files so
//! node_modules, .venv, and artifacts never get duplicated. Falls back to
//! a filtered copy when the source is not a git checkout.

use std::path::{Path, PathBuf};

const ALWAYS_IGNORE: &[&str] = &[
    "__pycache__",
    ".git",
    ".venv",
    "node_modules",
    "dist",
    "build",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "artifacts",
    "target",
];

pub async fn snapshot_repo(source: &Path, destination: &Path) -> std::io::Result<PathBuf> {
    let source = source.canonicalize()?;
    tokio::fs::create_dir_all(destination).await?;
    let destination = destination.canonicalize()?;
    let files = git_tracked_files(&source).await;
    let src = source.clone();
    let dst = destination.clone();
    tokio::task::spawn_blocking(move || -> std::io::Result<()> {
        match files {
            Some(files) => {
                for rel in files {
                    let from = src.join(&rel);
                    if !from.is_file() {
                        continue;
                    }
                    let to = dst.join(&rel);
                    if let Some(parent) = to.parent() {
                        std::fs::create_dir_all(parent)?;
                    }
                    std::fs::copy(&from, &to)?;
                }
                Ok(())
            }
            None => copy_tree_filtered(&src, &dst),
        }
    })
    .await
    .map_err(|e| std::io::Error::other(e.to_string()))??;
    Ok(destination)
}

async fn git_tracked_files(source: &Path) -> Option<Vec<String>> {
    if !source.join(".git").exists() {
        return None;
    }
    let out = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        tokio::process::Command::new("git")
            .arg("-C")
            .arg(source)
            .arg("ls-files")
            .kill_on_drop(true)
            .output(),
    )
    .await
    .ok()?
    .ok()?;
    if !out.status.success() {
        return None;
    }
    Some(
        String::from_utf8_lossy(&out.stdout)
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(str::to_string)
            .collect(),
    )
}

fn copy_tree_filtered(source: &Path, destination: &Path) -> std::io::Result<()> {
    for entry in std::fs::read_dir(source)? {
        let entry = entry?;
        let name = entry.file_name();
        if ALWAYS_IGNORE.iter().any(|ig| name == *ig) {
            continue;
        }
        let from = entry.path();
        let to = destination.join(&name);
        if from.is_dir() {
            std::fs::create_dir_all(&to)?;
            copy_tree_filtered(&from, &to)?;
        } else if from.is_file() {
            std::fs::copy(&from, &to)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn non_git_source_uses_filtered_copy() {
        let src = tempfile::tempdir().unwrap();
        std::fs::write(src.path().join("a.txt"), "a").unwrap();
        std::fs::create_dir_all(src.path().join("node_modules/x")).unwrap();
        std::fs::write(src.path().join("node_modules/x/big.js"), "junk").unwrap();
        std::fs::create_dir_all(src.path().join("src")).unwrap();
        std::fs::write(src.path().join("src/b.txt"), "b").unwrap();
        let dst = tempfile::tempdir().unwrap();
        snapshot_repo(src.path(), dst.path()).await.unwrap();
        assert!(dst.path().join("a.txt").is_file());
        assert!(dst.path().join("src/b.txt").is_file());
        assert!(!dst.path().join("node_modules").exists());
    }
}
