//! Keeping what a trial left under `results/` (RFD reference, "Retained results").
//!
//! The evaluation root is temporary; `results/` keeps, per trial, what lets a result be
//! checked by hand. The agent could write much of it, so a link is never followed: it
//! could name any file on the machine, which would then land in `results/`. A link, a
//! special file, or a tree that cannot be listed is left out and named in
//! [`SKIPPED_FILE`]. A name that is not UTF-8 stops its whole tree from being listed, so
//! the agent can keep its own submission out this way; that is accepted, and noted.
//!
//! Everything is copied into a temporary sibling of the destination, which takes the
//! destination's name only once complete, so a failed retention leaves nothing behind
//! that could be taken for a result. The destination must be new.

#[cfg(test)]
mod tests;

use std::fs::{self, File};
use std::io::{self, BufWriter, Read as _, Seek as _, SeekFrom, Write as _};
use std::path::{Path, PathBuf};

use tempfile::TempDir;
use thiserror::Error;

use super::{Done, RESULT_FILE};
use crate::run::Configuration;
use crate::tree::{self, EntryKind, TreeEntry};

/// What was left out, one line each, when anything was.
pub(crate) const SKIPPED_FILE: &str = "retention-skipped.txt";

#[derive(Debug, Error)]
pub(crate) enum RetainError {
    #[error("`{}` already exists", .0.display())]
    Exists(PathBuf),
    /// Something the trial reached always leaves, such as its result.
    #[error("`{}` is missing or not a regular file", .0.display())]
    Required(PathBuf),
    #[error("cannot create `{}`", .path.display())]
    Create {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("cannot keep `{}`", .path.display())]
    Copy {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
}

/// Copies into a staging directory, noting what it leaves out.
struct Keeper {
    staging: TempDir,
    skipped: Vec<String>,
}

impl Done {
    /// Copies what the RFD keeps of this trial into `destination`, which must not exist.
    ///
    /// # Errors
    ///
    /// When `destination` exists, the result or a launched client's transcript or
    /// standard error is missing, or something kept cannot be written; then nothing is
    /// left at `destination`. A part the trial never reached is simply absent.
    pub(crate) fn retain(&self, destination: &Path) -> Result<(), RetainError> {
        let mut keeper = Keeper::new(destination)?;
        self.keep(&mut keeper)?;
        keeper.publish(destination)
    }

    fn keep(&self, keeper: &mut Keeper) -> Result<(), RetainError> {
        let root = &self.root;
        keeper.required(&root.join(RESULT_FILE), RESULT_FILE)?;
        keeper.file(&root.join("srt.json"), "srt.json")?;
        keeper.tree(&root.join("project"), "submission", |path| {
            path != "target" && !path.starts_with("target/")
        })?;
        keeper.tree(&root.join("grading/logs"), "grading", |_| true)?;
        if let Some(session) = &self.session {
            // A launched client always has both: the backend creates them as it starts.
            keeper.required(&session.transcript, "transcript.jsonl")?;
            keeper.required(&session.stderr, "stderr.log")?;
            if let Some(response) = &session.response {
                keeper.write("response.txt", response.as_bytes())?;
            }
        }
        if self.configuration == Configuration::Symposium {
            let evidence = root.join("evidence");
            keeper.file(&evidence.join("hook-trace.jsonl"), "hook-trace.jsonl")?;
            keeper.tree(&evidence, "evidence", |_| true)?;
            keeper.joined_logs(&root.join("home/.symposium/logs"), "symposium.log")?;
        }
        Ok(())
    }
}

impl Keeper {
    /// A staging directory beside `destination`, which must not exist.
    fn new(destination: &Path) -> Result<Self, RetainError> {
        if !matches!(kind(destination), Ok(None)) {
            return Err(RetainError::Exists(destination.to_owned()));
        }
        let parent = destination.parent().unwrap_or(Path::new("."));
        fs::create_dir_all(parent).map_err(|source| create(parent, source))?;
        let staging = tempfile::Builder::new()
            .prefix(".retaining-")
            .tempdir_in(parent)
            .map_err(|source| create(parent, source))?;
        Ok(Self {
            staging,
            skipped: Vec::new(),
        })
    }

    fn staging_dir(&self) -> &Path {
        self.staging.path()
    }

    /// Copies the regular file at `source` as `name`, which the trial must have left.
    fn required(&self, source: &Path, name: &str) -> Result<(), RetainError> {
        match kind(source) {
            Ok(Some(EntryKind::File)) => copy(source, &self.staging_dir().join(name)),
            _ => Err(RetainError::Required(source.to_owned())),
        }
    }

    /// What is at `source`, kept as `name`; `None`, noting why, when it cannot be told.
    fn inspect(&mut self, source: &Path, name: &str) -> Option<EntryKind> {
        match kind(source) {
            Ok(found) => found,
            Err(error) => {
                self.skip(name, &format!("cannot inspect: {error}"));
                None
            }
        }
    }

    /// Copies the regular file at `source` as `name`. A missing one is not kept.
    fn file(&mut self, source: &Path, name: &str) -> Result<(), RetainError> {
        match self.inspect(source, name) {
            None => Ok(()),
            Some(EntryKind::File) => copy(source, &self.staging_dir().join(name)),
            Some(other) => {
                self.skip(name, &other.to_string());
                Ok(())
            }
        }
    }

    /// Copies the directory at `source` as `name`, with the entries `keep` accepts.
    fn tree(
        &mut self,
        source: &Path,
        name: &str,
        keep: impl Fn(&str) -> bool,
    ) -> Result<(), RetainError> {
        let Some(entries) = self.listing(source, name) else {
            return Ok(());
        };
        let destination = self.staging_dir().join(name);
        fs::create_dir(&destination).map_err(|source| create(&destination, source))?;
        for entry in entries.into_iter().filter(|entry| keep(&entry.path)) {
            let to = destination.join(&entry.path);
            match entry.kind {
                EntryKind::Dir => fs::create_dir(&to).map_err(|source| create(&to, source))?,
                EntryKind::File => copy(&source.join(&entry.path), &to)?,
                EntryKind::Symlink | EntryKind::Other => {
                    self.skip(&format!("{name}/{}", entry.path), &entry.kind.to_string());
                }
            }
        }
        Ok(())
    }

    /// Joins the regular files in `source`, in name order, each under a header, as
    /// `name`. Symposium writes one log per process, named by the time it started.
    fn joined_logs(&mut self, source: &Path, name: &str) -> Result<(), RetainError> {
        let Some(entries) = self.listing(source, name) else {
            return Ok(());
        };
        let joined = self.staging_dir().join(name);
        let file = File::create_new(&joined).map_err(|source| create(&joined, source))?;
        let mut out = BufWriter::new(file);
        for entry in entries {
            if entry.kind != EntryKind::File || entry.path.contains('/') {
                self.skip(&format!("{name}: {}", entry.path), &entry.kind.to_string());
                continue;
            }
            let path = source.join(&entry.path);
            append_log(&mut out, &entry.path, &path)
                .map_err(|source| RetainError::Copy { path, source })?;
        }
        out.flush().map_err(|source| RetainError::Copy {
            path: joined,
            source,
        })
    }

    /// The entries of the real directory at `source`, kept as `name`; `None` when there
    /// is none, or, noting why, when it cannot be listed.
    fn listing(&mut self, source: &Path, name: &str) -> Option<Vec<TreeEntry>> {
        match self.inspect(source, name)? {
            EntryKind::Dir => {}
            other => {
                self.skip(name, &other.to_string());
                return None;
            }
        }
        tree::walk(source)
            .inspect_err(|error| self.skip(name, &error.to_string()))
            .ok()
    }

    fn write(&self, name: &str, bytes: &[u8]) -> Result<(), RetainError> {
        let path = self.staging_dir().join(name);
        fs::write(&path, bytes).map_err(|source| RetainError::Copy { path, source })
    }

    fn skip(&mut self, what: &str, why: &str) {
        self.skipped.push(format!("{what}: {why}\n"));
    }

    /// Names what was left out, then gives the staging directory `destination`'s name.
    /// Until then a failure removes it.
    fn publish(self, destination: &Path) -> Result<(), RetainError> {
        if !self.skipped.is_empty() {
            self.write(SKIPPED_FILE, self.skipped.concat().as_bytes())?;
        }
        fs::rename(self.staging_dir(), destination)
            .map_err(|source| create(destination, source))?;
        // Renamed away, so there is nothing left for the staging directory to remove.
        let _ = self.staging.keep();
        Ok(())
    }
}

/// Appends the log at `path`, named `name`, under its header, ending it with a newline
/// so the next header starts a line of its own.
fn append_log(out: &mut impl io::Write, name: &str, path: &Path) -> io::Result<()> {
    writeln!(out, "== {name} ==")?;
    let mut log = File::open(path)?;
    if io::copy(&mut log, out)? > 0 && !ends_with_newline(&mut log)? {
        writeln!(out)?;
    }
    Ok(())
}

fn ends_with_newline(file: &mut File) -> io::Result<bool> {
    file.seek(SeekFrom::End(-1))?;
    let mut last = [0];
    file.read_exact(&mut last)?;
    Ok(last == *b"\n")
}

/// What is at `path`, a link not followed; `Ok(None)` when nothing is.
fn kind(path: &Path) -> io::Result<Option<EntryKind>> {
    match fs::symlink_metadata(path) {
        Ok(metadata) => Ok(Some(EntryKind::of(metadata.file_type()))),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error),
    }
}

fn copy(source: &Path, destination: &Path) -> Result<(), RetainError> {
    fs::copy(source, destination)
        .map(drop)
        .map_err(|error| RetainError::Copy {
            path: source.to_owned(),
            source: error,
        })
}

fn create(path: &Path, source: io::Error) -> RetainError {
    RetainError::Create {
        path: path.to_owned(),
        source,
    }
}
