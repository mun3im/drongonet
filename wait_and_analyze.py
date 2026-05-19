#!/usr/bin/env python3
"""
Wait for ablation study to complete and then run analysis
Monitors progress and provides periodic updates
"""

import time
import subprocess
import re
from pathlib import Path
from datetime import datetime, timedelta

LOG_DIR = Path("ablation_logs")
TOTAL_EXPERIMENTS = 75
CHECK_INTERVAL = 300  # Check every 5 minutes

def get_latest_master_log():
    """Find the latest master log file"""
    master_logs = sorted(LOG_DIR.glob("master_*.log"), key=lambda x: x.stat().st_mtime)
    return master_logs[-1] if master_logs else None

def parse_progress(log_file):
    """Parse progress from master log"""
    if not log_file or not log_file.exists():
        return None

    content = log_file.read_text()

    # Count successes and failures
    success_count = content.count("✓ SUCCESS")
    fail_count = content.count("✗ FAILED")
    total_done = success_count + fail_count

    # Find latest experiment info
    exp_match = re.findall(r'Experiment (\d+)/(\d+)', content)
    current_exp = int(exp_match[-1][0]) if exp_match else 0

    # Find elapsed and ETA
    eta_match = re.findall(r'ETA: (\d+):(\d+):(\d+)', content)
    if eta_match:
        h, m, s = map(int, eta_match[-1])
        eta_seconds = h * 3600 + m * 60 + s
    else:
        eta_seconds = 0

    return {
        'current': current_exp,
        'total': TOTAL_EXPERIMENTS,
        'success': success_count,
        'failed': fail_count,
        'done': total_done,
        'eta_seconds': eta_seconds
    }

def format_time(seconds):
    """Format seconds to readable time"""
    return str(timedelta(seconds=int(seconds)))

def print_progress_bar(progress, width=50):
    """Print a nice progress bar"""
    if progress is None:
        return

    pct = progress['done'] / progress['total']
    filled = int(width * pct)
    bar = '█' * filled + '░' * (width - filled)

    print(f"\n{'='*80}")
    print(f"ABLATION STUDY PROGRESS")
    print(f"{'='*80}")
    print(f"Progress: [{bar}] {progress['done']}/{progress['total']} ({pct*100:.1f}%)")
    print(f"Success: {progress['success']}, Failed: {progress['failed']}")
    if progress['eta_seconds'] > 0:
        print(f"Estimated time remaining: {format_time(progress['eta_seconds'])}")
    print(f"{'='*80}\n")

def monitor_until_complete():
    """Monitor progress until all experiments complete"""
    print(f"Starting monitor at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Checking progress every {CHECK_INTERVAL} seconds...")

    start_time = time.time()
    last_progress = None

    while True:
        log_file = get_latest_master_log()
        progress = parse_progress(log_file)

        if progress:
            # Print progress update
            print_progress_bar(progress)

            # Check if complete
            if progress['done'] >= TOTAL_EXPERIMENTS:
                print("✓ All experiments completed!")
                elapsed = time.time() - start_time
                print(f"Total monitoring time: {format_time(elapsed)}")
                break

            last_progress = progress
        else:
            print("⏳ Waiting for experiments to start...")

        # Wait before next check
        time.sleep(CHECK_INTERVAL)

    return progress

def run_analysis():
    """Run the results analysis script"""
    print(f"\n{'='*80}")
    print("RUNNING COMPREHENSIVE ANALYSIS")
    print(f"{'='*80}\n")

    try:
        result = subprocess.run(
            ['python3', 'analyze_ablation_results.py'],
            capture_output=True,
            text=True,
            timeout=300
        )

        print(result.stdout)
        if result.returncode != 0:
            print(f"⚠ Analysis had errors:\n{result.stderr}")
        else:
            print("\n✓ Analysis completed successfully!")

    except Exception as e:
        print(f"✗ Failed to run analysis: {e}")

def main():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║         ABLATION STUDY - AUTONOMOUS MONITOR & ANALYZER               ║
╚══════════════════════════════════════════════════════════════════════╝
""")

    # Monitor progress
    final_progress = monitor_until_complete()

    # Run analysis
    if final_progress and final_progress['success'] > 0:
        run_analysis()
    else:
        print("❌ No successful experiments to analyze")

    print(f"\n{'='*80}")
    print("AUTONOMOUS EXECUTION COMPLETE")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    main()
