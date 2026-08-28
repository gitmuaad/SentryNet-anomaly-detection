"""SentryNet review dashboard (Gradio).

    python app.py

Scope:

* CSV upload
* anomaly score
* binary decision
* score distribution

plus a few useful extras: Normal/Suspicious counts, the model name and its frozen threshold,
a downloadable scored CSV, and a PSI drift summary against the stored reference profile.

Deliberately **not** in scope: live packet capture, packet sniffing, network monitoring
agents, host-log parsing, free-text analysis, chatbot, LLM, RAG, agentic AI, SIEM replacement.

The app **loads already-trained artifacts** and never fits a model. There is no ``fit`` call
in this file. Uploaded rows are scored with exactly the feature engineering and preprocessing
that produced the reported metrics, because both come from the persisted bundle in
``artifacts/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import gradio as gr  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from sentrynet.config import load_config  # noqa: E402
from sentrynet.inference import (  # noqa: E402
    DECISION_SUSPICIOUS,
    KIND_BASELINE,
    KIND_IFOREST,
    KIND_OCSVM,
    build_pipeline,
)
from sentrynet.monitoring import psi_report_from_frame, psi_summary_table  # noqa: E402
from sentrynet.persistence import load_scoring_bundle  # noqa: E402

CFG = load_config()
PROTOCOL = CFG["splits"]["active_protocol"]
ARTIFACTS = CFG.path("artifacts_dir") / PROTOCOL
SCORED_CSV = CFG.path("outputs_dir") / "scored_upload.csv"

DETECTOR_FILES = {
    "Statistical Baseline": (KIND_BASELINE, "baseline.joblib"),
    "Isolation Forest": (KIND_IFOREST, "isolation_forest.joblib"),
    "One-Class SVM": (KIND_OCSVM, "one_class_svm.joblib"),
}
REQUIRED_COLUMNS = ["duration", "src_bytes", "dst_bytes", "packet_count", "protocol", "failed_logins"]


def _load_pipeline(display_name: str):
    kind, filename = DETECTOR_FILES[display_name]
    bundle = load_scoring_bundle(ARTIFACTS / filename)
    return build_pipeline(
        kind=bundle["kind"],
        model=bundle["model"],
        threshold=bundle["threshold"],
        preprocessor=bundle["preprocessor"],
        metadata={"params": bundle["params"], "seed": bundle["seed"]},
    )


def _load_psi_reference():
    path = ARTIFACTS / "psi_reference.joblib"
    if not path.exists():
        return None
    return load_scoring_bundle(path)["psi_reference"]


def score_csv(file_obj, detector_name: str, show_psi: bool):
    """Score an uploaded CSV. Loads artifacts; never trains."""
    empty_df = pd.DataFrame()
    if file_obj is None:
        return "Upload a CSV to begin.", empty_df, None, None, empty_df

    try:
        uploaded = pd.read_csv(file_obj.name if hasattr(file_obj, "name") else file_obj)
    except Exception as exc:  # noqa: BLE001 - surface any parse problem to the reviewer
        return f"### Could not read the CSV\n\n```\n{exc}\n```", empty_df, None, None, empty_df

    missing = [c for c in REQUIRED_COLUMNS if c not in uploaded.columns]
    if missing:
        return (
            "### Missing required columns\n\n"
            f"The uploaded file is missing: `{missing}`\n\n"
            f"SentryNet scores **structured network-flow summaries**. Required columns:\n\n"
            f"`{REQUIRED_COLUMNS}`",
            empty_df, None, None, empty_df,
        )

    try:
        pipeline = _load_pipeline(detector_name)
    except FileNotFoundError as exc:
        return f"### Artifacts not found\n\n```\n{exc}\n```", empty_df, None, None, empty_df

    scored = pipeline.score_frame(uploaded)
    n_rows = len(scored)
    n_suspicious = int((scored["decision"] == DECISION_SUSPICIOUS).sum())
    n_normal = n_rows - n_suspicious

    SCORED_CSV.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(SCORED_CSV, index=False)

    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.hist(scored["anomaly_score"], bins=60, color="#2a9d8f", edgecolor="white", linewidth=0.4)
    ax.axvline(
        pipeline.threshold, color="#e76f51", ls="--", lw=1.6,
        label=f"threshold = {pipeline.threshold:.6f}",
    )
    ax.set_xlabel("anomaly score (higher = more anomalous)")
    ax.set_ylabel("rows")
    ax.set_title(f"Score distribution — {detector_name}")
    ax.legend()
    fig.tight_layout()

    params = pipeline.metadata.get("params")
    summary = "\n".join([
        f"### {detector_name}",
        "",
        f"- **Rows scored:** {n_rows:,}",
        f"- **Suspicious:** {n_suspicious:,} ({100 * n_suspicious / n_rows:.2f}%)" if n_rows else "",
        f"- **Normal:** {n_normal:,} ({100 * n_normal / n_rows:.2f}%)" if n_rows else "",
        f"- **Operating threshold (frozen at training time):** `{pipeline.threshold:.6f}`",
        f"- **Model parameters:** `{params}`",
        "",
        "_Scores come from the saved preprocessor and model. Nothing was retrained._",
    ])

    psi_table = pd.DataFrame()
    if show_psi:
        reference = _load_psi_reference()
        if reference is not None:
            report = psi_report_from_frame(
                uploaded, reference, float(CFG["monitoring"]["psi_review_threshold"])
            )
            psi_table = psi_summary_table(report)
            summary += (
                f"\n\n**Drift (PSI vs. the Normal training baseline):** max PSI "
                f"`{report['max_psi']:.4f}` — {report['decision']}"
            )

    preview = scored.head(200)
    return summary, preview, fig, str(SCORED_CSV), psi_table


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="SentryNet — Network Flow Anomaly Review") as demo:
        gr.Markdown(
            "# SentryNet — Network Flow Anomaly Review\n"
            "Upload a CSV of **structured network-flow summaries** to get an anomaly score and "
            "a binary decision from a detector trained on confirmed-Normal flows only.\n\n"
            f"**Required columns:** `{REQUIRED_COLUMNS}`\n\n"
            "This is a **prototype review interface** for a synthetic teaching dataset. It does "
            "not capture packets, monitor a network, read host logs, or replace a SIEM."
        )
        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(label="Network-flow CSV", file_types=[".csv"])
                detector = gr.Dropdown(
                    choices=list(DETECTOR_FILES), value="Statistical Baseline", label="Detector"
                )
                psi_toggle = gr.Checkbox(
                    value=True, label="Show PSI drift summary vs. the training baseline"
                )
                run = gr.Button("Score CSV", variant="primary")
            with gr.Column(scale=2):
                summary_out = gr.Markdown("Upload a CSV to begin.")
        plot_out = gr.Plot(label="Score distribution")
        with gr.Row():
            table_out = gr.Dataframe(label="Scored rows (first 200)", wrap=True)
        with gr.Row():
            psi_out = gr.Dataframe(label="PSI drift summary")
            download_out = gr.File(label="Download scored CSV")

        run.click(
            fn=score_csv,
            inputs=[file_input, detector, psi_toggle],
            outputs=[summary_out, table_out, plot_out, download_out, psi_out],
        )
        gr.Markdown(
            "---\n"
            "**Limitations.** The dataset is synthetic and has no timestamps, IP addresses, or "
            "user identifiers. Reported performance is not claimed to transfer to production "
            "enterprise traffic. See the README's *Limitations* section."
        )
    return demo


if __name__ == "__main__":
    if not ARTIFACTS.exists():
        raise SystemExit(
            f"No artifacts found in {ARTIFACTS}.\n"
            "Run: python scripts/prepare_data.py && python scripts/train.py"
        )
    build_ui().launch(server_name="0.0.0.0", server_port=7860)
