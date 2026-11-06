use std::fs;
use std::path::{Path, PathBuf};

use tempfile::TempDir;

use super::*;
use crate::run::trial::SessionOutput;

/// A trial root as a Symposium trial that finished leaves it, under `temp`.
fn finished_trial(temp: &Path) -> PathBuf {
    let root = temp.join("trial-1-with-symposium");
    for (path, contents) in [
        (RESULT_FILE, "{}"),
        ("srt.json", "{}"),
        ("project/src/lib.rs", "pub fn double() {}"),
        ("project/target/debug/build", "output"),
        ("grading/logs/build.stdout.log", "built"),
        ("grading/work/scratch", "not kept"),
        ("session/claude.stdout.log", "transcript"),
        ("session/claude.stderr.log", "errors"),
        ("evidence/hook-trace.jsonl", "trace"),
        ("evidence/hook-records.jsonl", "records"),
        // No final newline, which the joined log adds before the next header.
        ("home/.symposium/logs/symposium-1.log", "first"),
        ("home/.symposium/logs/symposium-2.log", "second\n"),
        ("home/.claude/settings.json", "not kept"),
    ] {
        let path = root.join(path);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, contents).unwrap();
    }
    root
}

fn done(root: &Path, configuration: Configuration, launched: bool) -> Done {
    Done {
        root: root.to_owned(),
        configuration,
        session: launched.then(|| SessionOutput {
            transcript: root.join("session/claude.stdout.log"),
            stderr: root.join("session/claude.stderr.log"),
            response: Some("done".to_owned()),
        }),
    }
}

fn read(path: &Path) -> String {
    fs::read_to_string(path).unwrap()
}

#[test]
fn a_symposium_trial_keeps_what_the_rfd_lists() {
    let temp = TempDir::new().unwrap();
    let root = finished_trial(temp.path());
    let kept = temp.path().join("results/trial-1-with-symposium");

    done(&root, Configuration::Symposium, true)
        .retain(&kept)
        .unwrap();

    for (path, contents) in [
        (RESULT_FILE, "{}"),
        ("srt.json", "{}"),
        ("submission/src/lib.rs", "pub fn double() {}"),
        ("grading/build.stdout.log", "built"),
        ("transcript.jsonl", "transcript"),
        ("stderr.log", "errors"),
        ("response.txt", "done"),
        ("hook-trace.jsonl", "trace"),
        ("evidence/hook-records.jsonl", "records"),
    ] {
        assert_eq!(read(&kept.join(path)), contents, "{path}");
    }
    assert_eq!(
        read(&kept.join("symposium.log")),
        "== symposium-1.log ==\nfirst\n== symposium-2.log ==\nsecond\n"
    );
    for absent in ["submission/target", "grading/scratch", SKIPPED_FILE] {
        assert!(!kept.join(absent).exists(), "{absent}");
    }
}

#[test]
fn a_baseline_trial_keeps_no_symposium_evidence() {
    let temp = TempDir::new().unwrap();
    let root = finished_trial(temp.path());
    let kept = temp.path().join("kept");

    done(&root, Configuration::Baseline, true)
        .retain(&kept)
        .unwrap();

    for absent in ["hook-trace.jsonl", "evidence", "symposium.log"] {
        assert!(!kept.join(absent).exists(), "{absent}");
    }
    assert!(kept.join("transcript.jsonl").is_file());
}

#[test]
fn a_trial_that_never_launched_keeps_no_session() {
    let temp = TempDir::new().unwrap();
    let root = temp.path().join("trial");
    fs::create_dir(&root).unwrap();
    fs::write(root.join(RESULT_FILE), "{}").unwrap();
    let kept = temp.path().join("kept");

    done(&root, Configuration::Symposium, false)
        .retain(&kept)
        .unwrap();

    let names: Vec<_> = fs::read_dir(&kept)
        .unwrap()
        .map(|entry| entry.unwrap().file_name())
        .collect();
    assert_eq!(names, [RESULT_FILE]);
}

#[test]
fn results_are_never_overwritten() {
    let temp = TempDir::new().unwrap();
    let root = finished_trial(temp.path());
    let kept = temp.path().join("kept");
    fs::create_dir(&kept).unwrap();
    fs::write(kept.join(RESULT_FILE), "earlier").unwrap();

    let retained = done(&root, Configuration::Symposium, true).retain(&kept);

    assert!(
        matches!(retained, Err(RetainError::Exists(_))),
        "{retained:?}"
    );
    assert_eq!(read(&kept.join(RESULT_FILE)), "earlier");
}

/// Nothing is left beside where the results would have gone.
fn assert_nothing_kept(results: &Path) {
    let left: Vec<_> = fs::read_dir(results)
        .unwrap()
        .map(|entry| entry.unwrap().file_name())
        .collect();
    assert!(left.is_empty(), "{left:?}");
}

/// The result, and a launched client's transcript and standard error, are always left,
/// so one missing, or something else in its place, keeps nothing.
#[test]
fn a_trial_missing_what_it_must_leave_keeps_nothing() {
    let required = [
        RESULT_FILE,
        "session/claude.stdout.log",
        "session/claude.stderr.log",
    ];
    for file in required {
        for replaced in [false, true] {
            let temp = TempDir::new().unwrap();
            let root = finished_trial(temp.path());
            fs::remove_file(root.join(file)).unwrap();
            if replaced {
                fs::create_dir(root.join(file)).unwrap();
            }
            let results = temp.path().join("results");

            let retained = done(&root, Configuration::Symposium, true).retain(&results.join("1"));

            assert!(
                matches!(retained, Err(RetainError::Required(_))),
                "{file}, replaced: {replaced}: {retained:?}"
            );
            assert_nothing_kept(&results);
        }
    }
}

/// A copy that fails partway leaves no partial results to be taken for complete ones.
#[cfg(unix)]
#[test]
fn a_failed_retention_leaves_nothing_behind() {
    use std::os::unix::fs::PermissionsExt;

    let temp = TempDir::new().unwrap();
    let root = finished_trial(temp.path());
    let unreadable = root.join("grading/logs/build.stdout.log");
    fs::set_permissions(&unreadable, fs::Permissions::from_mode(0o000)).unwrap();
    let results = temp.path().join("results");

    let retained = done(&root, Configuration::Symposium, true).retain(&results.join("1"));

    assert!(
        matches!(retained, Err(RetainError::Copy { .. })),
        "{retained:?}"
    );
    assert_nothing_kept(&results);
}

/// The agent can leave links in its project and its evidence. None is followed, so what
/// they name never reaches `results/`, and each is named in the skipped list.
#[cfg(unix)]
#[test]
fn a_link_the_agent_left_is_never_followed() {
    use std::os::unix::fs::symlink;

    let temp = TempDir::new().unwrap();
    let root = finished_trial(temp.path());
    let secret = temp.path().join("secret");
    fs::write(&secret, "private").unwrap();
    symlink(&secret, root.join("project/src/key")).unwrap();
    fs::remove_file(root.join("evidence/hook-trace.jsonl")).unwrap();
    symlink(&secret, root.join("evidence/hook-trace.jsonl")).unwrap();
    symlink(&secret, root.join("home/.symposium/logs/symposium-3.log")).unwrap();
    let kept = temp.path().join("kept");

    done(&root, Configuration::Symposium, true)
        .retain(&kept)
        .unwrap();

    for entry in crate::tree::walk(&kept).unwrap() {
        let path = kept.join(&entry.path);
        if entry.kind == EntryKind::File {
            assert!(!read(&path).contains("private"), "{}", entry.path);
        }
        assert_ne!(entry.kind, EntryKind::Symlink, "{}", entry.path);
    }
    assert_eq!(
        read(&kept.join(SKIPPED_FILE)),
        "submission/src/key: symlink\n\
         hook-trace.jsonl: symlink\n\
         evidence/hook-trace.jsonl: symlink\n\
         symposium.log: symposium-3.log: symlink\n"
    );
}
