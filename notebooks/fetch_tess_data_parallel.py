#!/usr/bin/env python3

from lightkurve import search_lightcurvefile
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import warnings
import time
import random

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def fetch_and_save_lightcurve(tic_id, output_dir="data", max_retries=5):
    """
    Fetch and save light curve for a single TIC ID with retry logic.
    Returns (tic_id, success, message)
    """
    Path(output_dir).mkdir(exist_ok=True)
    output_file = f"{output_dir}/{tic_id}.csv"

    if os.path.exists(output_file):
        return (tic_id, True, f"Already exists, skipped")

    for attempt in range(max_retries):
        try:
            # Add random delay to avoid rate limiting
            time.sleep(random.uniform(0.1, 0.5))

            # Search and download
            lc_file = search_lightcurvefile(f"TIC {tic_id}", mission="TESS").download(download_dir=None)

            # Try PDCSAP first, fall back to SAP
            try:
                lc = lc_file.PDCSAP_FLUX
            except:
                lc = lc_file.SAP_FLUX

            lc = lc.remove_nans()

            # Save data to file
            data = np.column_stack([
                np.asarray(lc.time.value),
                np.asarray(lc.flux.value),
                np.asarray(lc.flux_err.value)
            ])
            np.savetxt(output_file, data, delimiter=',',
                       header='time,flux,flux_err', comments='')

            return (tic_id, True, f"Saved {len(lc)} points")

        except Exception as e:
            error_msg = str(e)
            # Retry on 429 errors with exponential backoff
            if "429" in error_msg and attempt < max_retries - 1:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(wait_time)
                continue
            else:
                return (tic_id, False, f"Error: {error_msg}")

    return (tic_id, False, f"Error: Max retries exceeded")

def fetch_multiple_parallel(tic_ids, max_workers=10, output_dir="data"):
    """
    Fetch multiple TIC IDs in parallel.

    Args:
        tic_ids: List of TIC ID strings
        max_workers: Maximum number of parallel downloads (default: 10)
        output_dir: Directory to save CSV files
    """
    print(f"Fetching {len(tic_ids)} TIC IDs with {max_workers} parallel workers...")
    print("Please wait...")

    results = {"success": [], "failed": [], "details": []}

    # Open log file for real-time updates
    log_file = open("download_progress.txt", "w")
    log_file.write(f"=== Download Progress ===\n")
    log_file.write(f"Total TIC IDs: {len(tic_ids)}\n")
    log_file.write(f"Workers: {max_workers}\n\n")
    log_file.flush()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_tic = {
            executor.submit(fetch_and_save_lightcurve, tic_id, output_dir): tic_id
            for tic_id in tic_ids
        }

        # Process results as they complete
        completed = 0
        for future in as_completed(future_to_tic):
            try:
                tic_id, success, message = future.result()
                completed += 1

                if success:
                    results["success"].append(tic_id)
                    results["details"].append((tic_id, True, message))
                else:
                    results["failed"].append(tic_id)
                    results["details"].append((tic_id, False, message))

                # Write to log file in real-time
                status = "✓" if success else "✗"
                progress_line = f"[{completed}/{len(tic_ids)}] {status} TIC {tic_id}: {message}\n"
                log_file.write(progress_line)
                log_file.flush()

            except Exception as e:
                completed += 1
                results["failed"].append("unknown")
                results["details"].append(("unknown", False, str(e)))
                log_file.write(f"[{completed}/{len(tic_ids)}] ✗ TIC unknown: {str(e)}\n")
                log_file.flush()

    # Write summary
    log_file.write(f"\n=== Summary ===\n")
    log_file.write(f"Successful: {len(results['success'])}/{len(tic_ids)}\n")
    log_file.write(f"Failed: {len(results['failed'])}/{len(tic_ids)}\n")
    log_file.close()

    # Write results to file to avoid stdout issues
    with open("download_results.txt", "w") as f:
        f.write("=== Results ===\n")
        for tic_id, success, message in results["details"]:
            status = "✓" if success else "✗"
            f.write(f"{status} TIC {tic_id}: {message}\n")

        f.write(f"\n=== Summary ===\n")
        f.write(f"Successful: {len(results['success'])}/{len(tic_ids)}\n")
        f.write(f"Failed: {len(results['failed'])}/{len(tic_ids)}\n")

    # Also try to print (may fail if stdout is closed)
    try:
        with open("download_results.txt", "r") as f:
            print(f.read())
    except:
        pass

    return results

def main():
    import sys

    # Allow custom input file and output directory
    input_file = sys.argv[1] if len(sys.argv) > 1 else "ids.txt"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "data"

    # Read TIC IDs from file
    with open(input_file, "r") as f:
        all_tic_ids = [line.strip() for line in f if line.strip()]

    # Use all TIC IDs
    tic_ids = all_tic_ids

    print(f"Loaded {len(tic_ids)} TIC IDs from {input_file}")

    # Fetch with 8 parallel workers to avoid rate limiting
    results = fetch_multiple_parallel(tic_ids, max_workers=8, output_dir=output_dir)

if __name__ == "__main__":
    main()
