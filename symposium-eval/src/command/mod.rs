//! `cargo xtask agent-eval <task>`: one batch of a task on one client and model (RFD,
//! "What happens when you run an eval").
//!
//! A regression task asks whether Symposium delivered, so it runs one Symposium trial. A
//! capability task asks whether Symposium helps, so it runs one round: a baseline trial,
//! then a Symposium trial. More trials, alternating order, and `benchmark.json` come with
//! `--trials`.
//!
//! Everything a trial could fail on is checked before the first one starts: the
//! credential, `srt` and its stack, the client's version, and the `cargo-agents` under
//! evaluation. Each trial's result is then kept under
//! `results/<task>/<client>/<batch>/`. An outcome, even a failed one, is not an error.

mod batch_name;
mod claude;
mod credential;
pub(crate) mod host;
mod report;
mod symposium;

use std::ffi::{OsStr, OsString};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, bail};
use chrono::Utc;

use crate::client::claude::version::Version;
use crate::client::claude::{Claude, version};
use crate::evals::Evals;
use crate::result::{Arm, Deadlines, IsolationMode, Provenance, SymposiumBuild};
use crate::run::stage::{Audience, StagedTool, Tool};
use crate::run::trial::Driver;
use crate::run::{ApiKey, Batch, Inputs, Machine, SRT_VERSION};
use crate::task::{TaskKind, TaskName, ToolName};

/// The only host a Claude Code trial reaches.
const CLAUDE_API_HOST: &str = "api.anthropic.com";

/// The deadlines every trial runs with.
const DEADLINES: Deadlines = Deadlines {
    setup: Duration::from_secs(120),
    agent: Duration::from_secs(600),
    grading: Duration::from_secs(600),
};

/// What the developer asked for.
#[derive(Debug)]
pub(crate) struct Request {
    pub(crate) task: TaskName,
    pub(crate) model: String,
    pub(crate) symposium_bin: Option<PathBuf>,
    pub(crate) eval_root: Option<PathBuf>,
}

/// Where the runner reads tasks and keeps results.
#[derive(Debug)]
pub(crate) struct Checkout {
    pub(crate) root: PathBuf,
    pub(crate) evals: Evals,
    pub(crate) results: PathBuf,
}

impl Checkout {
    /// The checkout this crate is built in: its `evals/`, and `results/` beside it.
    pub(crate) fn in_repository() -> Self {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("symposium-eval lives inside the repository")
            .to_owned();
        Self {
            evals: Evals::in_repository(),
            results: root.join("results"),
            root,
        }
    }
}

/// A batch ready to run trials: everything checked and staged.
struct Ready {
    batch: Batch,
    inputs: Inputs,
    machine: Machine,
    srt: crate::run::srt::Srt,
    api_key: ApiKey,
    provenance: Provenance,
    claude: version::Version,
    build: SymposiumBuild,
}

/// What preparation found on the developer's machine before creating a batch.
struct PreparedHost {
    api_key: ApiKey,
    path: OsString,
    srt: crate::run::srt::Srt,
    machine: Machine,
}

/// The tools copied into a batch, together with the source revision of Symposium when
/// one can be named truthfully.
struct StagedTools {
    claude: StagedTool,
    cargo_agents: StagedTool,
    symposium_revision: Option<String>,
}

/// Runs `request` from `checkout`, with `var` reading the runner's environment.
///
/// # Errors
///
/// When the machine, the credential, or a tool is not fit to run trials, or a trial has
/// no result. A trial that failed is an outcome, not an error.
pub(crate) fn run(
    request: &Request,
    checkout: &Checkout,
    var: impl Fn(&str) -> Option<OsString>,
) -> anyhow::Result<()> {
    if !cfg!(target_os = "linux") {
        bail!("trials run under srt, which this runner supports on Linux only");
    }
    let ready = prepare(request, checkout, &var)?;
    let batch = ready
        .batch
        .root()
        .file_name()
        .context("a batch has a name")?;
    let results = checkout
        .results
        .join(request.task.as_str())
        .join("claude")
        .join(batch);
    let failed = run_trials(&ready, &request.model, &results);
    println!("results in {}", results.display());
    if failed > 0 {
        bail!("{failed} trial(s) have no result; see above");
    }
    Ok(())
}

/// Checks the machine and credential, then creates and stages the batch.
fn prepare(
    request: &Request,
    checkout: &Checkout,
    var: &impl Fn(&str) -> Option<OsString>,
) -> anyhow::Result<Ready> {
    let host = inspect_host(checkout, var)?;
    let mut batch = create_batch(request)?;
    let scratch = scratch(&batch, "preflight")?;
    host.srt.preflight(&scratch)?;

    let staged = stage_tools(&mut batch, request, checkout, &host.path)?;
    let build = symposium::recorded(&staged.cargo_agents, staged.symposium_revision, &scratch)?;
    let claude = version::preflight(&batch, &staged.claude, &scratch)?;
    let inputs = batch.snapshot(&checkout.evals, &request.task)?;

    Ok(Ready {
        batch,
        inputs,
        machine: host.machine,
        srt: host.srt,
        api_key: host.api_key,
        provenance: provenance(claude)?,
        claude,
        build,
    })
}

/// Checks the host state needed by every later preparation phase.
fn inspect_host(
    checkout: &Checkout,
    var: &impl Fn(&str) -> Option<OsString>,
) -> anyhow::Result<PreparedHost> {
    let api_key = credential::claude(var)?;
    let path = var("PATH").context("PATH is not set")?;
    let srt = host::srt(&path)?;
    let machine = Machine::new(Vec::new(), &checkout.root, &host::sysroot()?)?;
    Ok(PreparedHost {
        api_key,
        path,
        srt,
        machine,
    })
}

/// Creates the fresh batch in which preparation stages its inputs.
fn create_batch(request: &Request) -> anyhow::Result<Batch> {
    Ok(Batch::create(
        &eval_root(request)?,
        &batch_name::new_batch_name(Utc::now()),
    )?)
}

/// Discovers and stages the tools the trials will run.
fn stage_tools(
    batch: &mut Batch,
    request: &Request,
    checkout: &Checkout,
    path: &OsStr,
) -> anyhow::Result<StagedTools> {
    let under_evaluation =
        symposium::under_evaluation(&checkout.root, request.symposium_bin.as_deref())?;
    let claude_tools = claude::claude_tools(path)?;
    if let Some(node) = &claude_tools.node {
        stage(batch, "node", node, Audience::Common)?;
    }
    let claude = stage(batch, "claude", &claude_tools.claude, Audience::Common)?;
    let cargo_agents = stage(
        batch,
        "cargo-agents",
        &Tool::Executable(under_evaluation.binary),
        Audience::Symposium,
    )?;
    Ok(StagedTools {
        claude,
        cargo_agents,
        symposium_revision: under_evaluation.revision,
    })
}

/// Metadata shared by every result in the prepared batch.
fn provenance(claude: Version) -> anyhow::Result<Provenance> {
    Ok(Provenance {
        client_version: Some(claude.to_string()),
        os: host::os(),
        toolchain: Some(host::toolchain()?),
        isolation: IsolationMode::Enforced,
        srt_version: Some(SRT_VERSION.to_owned()),
        deadlines: DEADLINES,
    })
}

/// Runs each trial the task's kind asks for, keeps its results under `results`, and
/// prints its outcome. Returns how many trials have no result.
fn run_trials(ready: &Ready, model: &str, results: &Path) -> usize {
    let driver = Driver {
        batch: &ready.batch,
        inputs: &ready.inputs,
        machine: &ready.machine,
        sandbox: &ready.srt,
        api_host: CLAUDE_API_HOST,
        api_key: &ready.api_key,
        provenance: &ready.provenance,
    };
    let mut failed = 0;
    for (number, arm) in (1..).zip(arms(ready.inputs.task().kind(), &ready.build)) {
        let runner = Claude::enforced(model.to_owned(), ready.claude);
        let kept = driver
            .run(number, arm, &runner)
            .map_err(anyhow::Error::from)
            .and_then(|done| {
                let destination =
                    results.join(done.root.file_name().context("a trial has a name")?);
                done.retain(&destination)?;
                Ok(destination)
            });
        match kept {
            Ok(destination) => report::trial(&destination),
            Err(error) => {
                failed += 1;
                report::no_result(number, &error);
            }
        }
    }
    failed
}

/// The configurations a task of `kind` runs, in order.
fn arms(kind: TaskKind, build: &SymposiumBuild) -> Vec<Arm> {
    match kind {
        TaskKind::Regression => vec![Arm::Symposium(build.clone())],
        TaskKind::Capability => vec![Arm::Baseline, Arm::Symposium(build.clone())],
    }
}

/// The evaluation root asked for, or one under the system's temporary directory.
fn eval_root(request: &Request) -> anyhow::Result<PathBuf> {
    let root = match &request.eval_root {
        Some(root) => std::path::absolute(root)
            .with_context(|| format!("cannot resolve `{}`", root.display()))?,
        None => std::env::temp_dir().join("symposium-eval"),
    };
    Ok(root)
}

/// A directory of the batch's own for checks before any trial.
fn scratch(batch: &Batch, name: &str) -> anyhow::Result<PathBuf> {
    let dir = batch.root().join(name);
    fs::create_dir(&dir).with_context(|| format!("cannot create `{}`", dir.display()))?;
    Ok(dir)
}

fn stage(
    batch: &mut Batch,
    name: &str,
    tool: &Tool,
    audience: Audience,
) -> anyhow::Result<StagedTool> {
    let name = ToolName::new(name).expect("a tool name the runner chose");
    batch
        .stage(name.clone(), tool, audience)
        .with_context(|| format!("cannot stage `{name}`"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn symposium_build() -> SymposiumBuild {
        SymposiumBuild {
            version: "cargo-agents 1.0.0".to_owned(),
            sha256: crate::digest::file(
                Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("Cargo.toml")
                    .as_path(),
            )
            .unwrap(),
            revision: None,
        }
    }

    #[test]
    fn a_regression_task_runs_with_symposium_only() {
        let arms = arms(TaskKind::Regression, &symposium_build());

        assert!(matches!(arms[..], [Arm::Symposium(_)]), "{arms:?}");
    }

    #[test]
    fn a_capability_task_runs_a_baseline_then_symposium() {
        let arms = arms(TaskKind::Capability, &symposium_build());

        assert!(
            matches!(arms[..], [Arm::Baseline, Arm::Symposium(_)]),
            "{arms:?}"
        );
    }
}
