import csv
from pathlib import Path
from typing import Dict, Any

def append_run_to_csv(run_summary: Dict[str, Any], csv_filename: str = "total_runs.csv") -> None:
    """
    Append a single run summary entry to a CSV file.

    Parameters:
        run_summary (dict): The dictionary containing run summary data.
        csv_filename (str): The name/path of the CSV file (default: 'test_run_summary.csv').

    This function appends a new row to the CSV file using selected fields from the run summary.
    It will automatically create the file and write the header if it doesn't exist.
    """

    csv_path = Path(csv_filename)
    csv_columns = [
        "run_id",
        "no_of_mrs",
        "num_mrs_passed",
        "num_mrs_failed",
        "fault_detection_rate_pct",
        "total_api_ops",
        "server_500_count",
        "no_of_iterations",
        "api_coverage",
        "stop_reason"
    ]

    # Prepare a flattened row dict
    csv_row = {
        "run_id": run_summary.get("run_id"),
        "no_of_mrs": run_summary.get("no_of_mrs"),
        "num_mrs_passed": run_summary.get("num_mrs_passed"),
        "num_mrs_failed": run_summary.get("num_mrs_failed"),
        "fault_detection_rate_pct": run_summary.get("fault_detection_rate_pct"),
        "total_api_ops": run_summary.get("total_api_ops"),
        "server_500_count": run_summary.get("server_500_count"),
        "no_of_iterations": run_summary.get("no_of_iterations"),
        "api_coverage": run_summary.get("api_coverage"),
        "stop_reason": (
            run_summary.get("stop_reason", {}).get("reason")
            if isinstance(run_summary.get("stop_reason"), dict)
            else run_summary.get("stop_reason")
        )
    }

    file_exists = csv_path.exists()

    try:
        with csv_path.open("a", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            if not file_exists:
                writer.writeheader()
            writer.writerow(csv_row)
    except Exception as e:
        print(f"[ERROR] Failed to append run to CSV: {e}")
