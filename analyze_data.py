"""
=============================================================================
Offline Data Analysis & Visualization Module
=============================================================================
Reads recorded eye tracking and facial telemetry from CSV, computes summary
statistics, and generates synchronized multi-panel time-series plots saved to
the data/plots/ directory.
=============================================================================
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Path resolution
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CSV_PATH = DATA_DIR / "eye_tracking_data.csv"
PLOTS_DIR = DATA_DIR / "plots"


def analyze_and_plot(csv_file: Path = CSV_PATH, output_dir: Path = PLOTS_DIR) -> None:
    """
    Parses eye tracking CSV data and outputs visualization graphs and statistics.
    """
    if not csv_file.exists():
        print(f"[Analysis Error] File not found: {csv_file}")
        print("Please run `python main.py` first to collect driving telemetry data.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Analysis] Loading telemetry from: {csv_file}")
    df = pd.read_csv(csv_file)

    if df.empty or len(df) < 5:
        print("[Analysis Warning] Not enough data points recorded for meaningful analysis.")
        return

    # Convert numeric columns
    numeric_cols = [
        "elapsed_time", "left_ear", "right_ear", "average_ear",
        "eye_closed_duration", "blink_event", "blink_count", "blink_rate",
        "perclos", "mouth_ratio", "yawn_event", "yawn_count",
        "head_pitch", "head_yaw", "head_roll", "fps"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Clean NaN values
    df.fillna(0.0, inplace=True)

    time_sec = df["elapsed_time"]
    total_duration = time_sec.max() - time_sec.min()

    # ---------------------------------------------------------
    # Compute Summary Statistics
    # ---------------------------------------------------------
    total_blinks = int(df["blink_count"].max()) if "blink_count" in df.columns else 0
    total_yawns = int(df["yawn_count"].max()) if "yawn_count" in df.columns else 0
    avg_ear = float(df["average_ear"].mean())
    max_closed_dur = float(df["eye_closed_duration"].max())
    mean_blink_rate = float(df["blink_rate"].mean())
    final_perclos = float(df["perclos"].iloc[-1])

    print("\n" + "=" * 60)
    print("DRIVER MONITORING SESSION SUMMARY STATISTICS")
    print("=" * 60)
    print(f" Total Session Duration:      {total_duration:.1f} seconds ({total_duration/60.0:.2f} min)")
    print(f" Total Valid Data Points:      {len(df):,}")
    print(f" Average EAR:                 {avg_ear:.4f}")
    print(f" Total Blinks Detected:       {total_blinks}")
    print(f" Average Blink Rate:          {mean_blink_rate:.1f} blinks/min")
    print(f" Max Single Eye Closure:      {max_closed_dur:.2f} seconds")
    print(f" Final PERCLOS (60s window):  {final_perclos:.1f} %")
    print(f" Total Confirmed Yawns:       {total_yawns}")
    print("=" * 60 + "\n")

    # ---------------------------------------------------------
    # Generate Synchronized Multi-Panel Plots
    # ---------------------------------------------------------
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)
    fig.suptitle("Multimodal Driver Monitoring Telemetry (Phase 1: Vision Features)", fontsize=14, fontweight="bold")

    # 1. Eye Aspect Ratio (EAR) & Closure Threshold
    ax1 = axes[0]
    ax1.plot(time_sec, df["average_ear"], label="Average EAR", color="#1f77b4", lw=1.5)
    ax1.axhline(0.21, color="#d62728", linestyle="--", lw=1.2, label="EAR Closure Threshold (0.21)")
    ax1.set_ylabel("EAR", fontsize=10, fontweight="bold")
    ax1.set_ylim(0.0, 0.5)
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.set_title("1. Eye Aspect Ratio (EAR) vs Time", fontsize=11, loc="left")

    # 2. PERCLOS (%)
    ax2 = axes[1]
    ax2.plot(time_sec, df["perclos"], label="PERCLOS (%)", color="#ff7f0e", lw=1.8)
    ax2.axhline(15.0, color="#d62728", linestyle=":", lw=1.2, label="15% Caution Reference")
    ax2.set_ylabel("PERCLOS (%)", fontsize=10, fontweight="bold")
    ax2.set_ylim(0.0, max(30.0, df["perclos"].max() + 5.0))
    ax2.legend(loc="upper right", framealpha=0.9)
    ax2.set_title("2. Percentage of Eye Closure (PERCLOS) (60s Rolling Window)", fontsize=11, loc="left")

    # 3. Blink Rate & Blink Events
    ax3 = axes[2]
    ax3.plot(time_sec, df["blink_rate"], label="Blink Rate (blinks/min)", color="#2ca02c", lw=1.6)
    # Highlight discrete blink impulses
    blink_events = df[df["blink_event"] == 1]
    if not blink_events.empty:
        ax3.scatter(blink_events["elapsed_time"], [df["blink_rate"].mean()] * len(blink_events),
                    color="#2ca02c", marker="|", s=80, label=f"Blink Events (n={len(blink_events)})")
    ax3.set_ylabel("Blinks / Min", fontsize=10, fontweight="bold")
    ax3.legend(loc="upper right", framealpha=0.9)
    ax3.set_title("3. Physiological Blink Rate", fontsize=11, loc="left")

    # 4. Mouth Aspect Ratio (MAR) & Yawning Events
    ax4 = axes[3]
    ax4.plot(time_sec, df["mouth_ratio"], label="Mouth Ratio (MAR)", color="#9467bd", lw=1.5)
    ax4.axhline(0.60, color="#d62728", linestyle="--", lw=1.2, label="Yawn MAR Threshold (0.60)")
    yawn_events = df[df["yawn_event"] == 1]
    if not yawn_events.empty:
        ax4.scatter(yawn_events["elapsed_time"], yawn_events["mouth_ratio"],
                    color="#d62728", marker="o", s=60, label=f"Confirmed Yawns (n={len(yawn_events)})")
    ax4.set_ylabel("MAR", fontsize=10, fontweight="bold")
    ax4.set_ylim(0.0, 1.0)
    ax4.legend(loc="upper right", framealpha=0.9)
    ax4.set_title("4. Mouth Aspect Ratio (MAR) & Yawn Detection", fontsize=11, loc="left")

    # 5. Head Pose Angles (Pitch, Yaw, Roll)
    ax5 = axes[4]
    ax5.plot(time_sec, df["head_pitch"], label="Pitch (Nodding)", color="#8c564b", lw=1.2)
    ax5.plot(time_sec, df["head_yaw"], label="Yaw (Turn L/R)", color="#e377c2", lw=1.2)
    ax5.plot(time_sec, df["head_roll"], label="Roll (Tilt)", color="#7f7f7f", lw=1.2)
    ax5.axhline(0.0, color="#cccccc", linestyle="-", lw=0.8)
    ax5.set_ylabel("Degrees (°)", fontsize=10, fontweight="bold")
    ax5.set_xlabel("Elapsed Time (seconds)", fontsize=11, fontweight="bold")
    ax5.set_ylim(-45.0, 45.0)
    ax5.legend(loc="upper right", framealpha=0.9, ncol=3)
    ax5.set_title("5. 3D Head Orientation (solvePnP Euler Angles)", fontsize=11, loc="left")

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    plot_output_file = output_dir / "drowsiness_features_analysis.png"
    plt.savefig(plot_output_file, dpi=300)
    print(f"[Analysis] High-resolution figure saved to: {plot_output_file}")
    plt.close()


if __name__ == "__main__":
    csv_target = Path(sys.argv[1]) if len(sys.argv) > 1 else CSV_PATH
    analyze_and_plot(csv_target)
