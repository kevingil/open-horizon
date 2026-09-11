use std::path::PathBuf;

use clap::{Parser, Subcommand};
use tokio_util::sync::CancellationToken;
use utoipa::OpenApi;

use horizon_core::models::TrainingStatus;
use horizon_core::rewards;
use horizon_server::coordinator::rescore_run;
use horizon_server::jobs::JobRunner;
use horizon_server::training::TrainingRequest;
use horizon_server::{api, build_app, logging, sglang, Settings};

#[derive(Parser)]
#[command(name = "horizon", version, about = "Open Horizon control plane")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run the API server and job runner.
    Serve {
        #[arg(long)]
        host: Option<String>,
        #[arg(long)]
        port: Option<u16>,
    },
    /// Queue a training run and wait for it (or list adapters).
    Train {
        #[arg(long, value_delimiter = ',')]
        samples: Vec<String>,
        #[arg(long)]
        parent: Option<String>,
        #[arg(long)]
        steps: Option<i64>,
        #[arg(long)]
        list: bool,
    },
    /// Run the eval harness against an adapter.
    Eval {
        #[arg(long)]
        adapter: String,
    },
    /// Rescore a stored run against a rubric.
    Replay {
        #[arg(long)]
        run: Option<String>,
        #[arg(long)]
        rubric: Option<String>,
        #[arg(long)]
        dry_run: bool,
        #[arg(long)]
        list: bool,
    },
    /// Print the OpenAPI document.
    Openapi {
        #[arg(long)]
        out: Option<PathBuf>,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let settings = Settings::from_env();
    match cli.command {
        Command::Serve { host, port } => serve(settings, host, port).await,
        Command::Train {
            samples,
            parent,
            steps,
            list,
        } => train(settings, samples, parent, steps, list).await,
        Command::Eval { adapter } => eval(settings, adapter).await,
        Command::Replay {
            run,
            rubric,
            dry_run,
            list,
        } => replay(settings, run, rubric, dry_run, list).await,
        Command::Openapi { out } => {
            let doc = api::ApiDoc::openapi().to_pretty_json()?;
            match out {
                Some(path) => std::fs::write(path, doc)?,
                None => println!("{doc}"),
            }
            Ok(())
        }
    }
}

async fn serve(
    mut settings: Settings,
    host: Option<String>,
    port: Option<u16>,
) -> anyhow::Result<()> {
    if let Some(h) = host {
        settings.host = h;
    }
    if let Some(p) = port {
        settings.port = p;
    }
    let state = build_app(settings).await?;
    logging::init(
        &state.settings.log_level,
        state.settings.log_json,
        Some(state.bus.clone()),
    );
    let shutdown = CancellationToken::new();
    let runner = JobRunner::new(state.clone(), shutdown.clone()).spawn();
    if state.settings.sglang_autoload_lora {
        if let Some(url) = &state.settings.sglang_admin_url {
            sglang::spawn_autoreload(state.bus.clone(), sglang::SglangLoraReloader::new(url));
        }
    }
    let addr = format!("{}:{}", state.settings.host, state.settings.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!(addr, store = %state.settings.store_backend, env = %state.settings.env_backend, "horizon.serve");
    let app = api::router(state.clone());
    let server_shutdown = shutdown.clone();
    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            let _ = tokio::signal::ctrl_c().await;
            server_shutdown.cancel();
        })
        .await?;
    shutdown.cancel();
    let _ = runner.await;
    state.bridge.shutdown().await;
    Ok(())
}

async fn train(
    settings: Settings,
    samples: Vec<String>,
    parent: Option<String>,
    steps: Option<i64>,
    list: bool,
) -> anyhow::Result<()> {
    let state = build_app(settings).await?;
    logging::init(&state.settings.log_level, state.settings.log_json, None);
    if list {
        for a in state.adapters.list_adapters() {
            let score = a
                .eval_score
                .map(|s| format!("{s:.3}"))
                .unwrap_or_else(|| "  -  ".into());
            println!(
                "{}  parent={}  eval={score}  base={}",
                a.id,
                a.parent_id.as_deref().unwrap_or("-"),
                a.base_model
            );
        }
        return Ok(());
    }
    if samples.is_empty() {
        anyhow::bail!("--samples is required unless --list is passed");
    }
    let mut hyperparams = horizon_core::models::Hyperparams::new();
    if let Some(s) = steps {
        hyperparams.insert("steps".into(), horizon_core::models::HyperValue::Int(s));
    }
    let record = state.training.start(TrainingRequest {
        sample_run_ids: samples.clone(),
        parent_adapter_id: parent.clone(),
        hyperparams,
    })?;
    println!(
        "queued    {}  samples={}  parent={}",
        record.id,
        samples.len(),
        parent.as_deref().unwrap_or("-")
    );
    // Run the job inline: the CLI is its own worker.
    let shutdown = CancellationToken::new();
    let runner = JobRunner::new(state.clone(), shutdown.clone()).spawn();
    let latest = loop {
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        if let Some(r) = state.store.get_training_run(&record.id)? {
            if r.status.is_terminal() {
                break r;
            }
        }
    };
    shutdown.cancel();
    let _ = runner.await;
    state.bridge.shutdown().await;
    if latest.status == TrainingStatus::Completed {
        println!(
            "completed {}  adapter={}  steps={}",
            latest.id,
            latest.adapter_out.as_deref().unwrap_or("-"),
            latest.metrics.len()
        );
        Ok(())
    } else {
        eprintln!(
            "FAILED    {}  error={}",
            latest.id,
            latest.error.as_deref().unwrap_or("-")
        );
        std::process::exit(1)
    }
}

async fn eval(settings: Settings, adapter: String) -> anyhow::Result<()> {
    let state = build_app(settings).await?;
    logging::init(&state.settings.log_level, state.settings.log_json, None);
    let report = match state.eval.run(&adapter, None, None).await {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(2)
        }
    };
    state.bridge.shutdown().await;
    println!("adapter   {}", report.adapter_id);
    println!("task_set  {}", report.task_set);
    println!("mean      {:+.4}", report.mean_reward);
    for entry in report.per_task {
        println!("  - {:24} {:+.4}", entry.task_id, entry.terminal_reward);
    }
    Ok(())
}

async fn replay(
    settings: Settings,
    run: Option<String>,
    rubric: Option<String>,
    dry_run: bool,
    list: bool,
) -> anyhow::Result<()> {
    if list {
        for name in rewards::rubric_names() {
            let spec = rewards::get_rubric(&name).expect("registered");
            println!("{name:20}  {}", spec.signal_names().join(", "));
        }
        return Ok(());
    }
    let (Some(run), Some(rubric)) = (run, rubric) else {
        anyhow::bail!("--run and --rubric are required unless --list is passed");
    };
    let state = build_app(settings).await?;
    let result = match rescore_run(&state.store, &run, &rubric, !dry_run) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(2)
        }
    };
    println!("run    {}", result.run_id);
    println!(
        "rubric {} ({})",
        result.rubric,
        if dry_run { "dry-run" } else { "saved" }
    );
    let prov: String = result.previous.provenance.chars().take(60).collect();
    println!(
        "  was: {:+.4}  provenance={prov}",
        result.previous.terminal_reward
    );
    println!(
        "  now: {:+.4}  delta={:+.4}",
        result.new.terminal_reward,
        result.delta()
    );
    Ok(())
}
