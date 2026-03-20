#!/usr/bin/env python3
"""
Anthropic API Cost Tracker

Logs token usage and costs for each API call.
Can be integrated into OpenClaw to track spending.
"""

import json
import csv
from pathlib import Path
from datetime import datetime

# Pricing (as of 2026-03)
PRICING = {
    "anthropic/claude-haiku-4-5": {"input": 0.80, "output": 2.40},
    "anthropic/claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "anthropic/claude-opus-4-1": {"input": 15.00, "output": 75.00},
    "openai/gpt-5.4": {"input": 2.50, "output": 10.00},
    "openai/gpt-4.1": {"input": 0.03, "output": 0.06},
}

LOG_FILE = Path("cost_log.csv")
SUMMARY_FILE = Path("cost_summary.json")


def log_api_call(model: str, input_tokens: int, output_tokens: int, session_id: str = None):
    """Log an API call to CSV."""
    
    # Calculate cost
    rates = PRICING.get(model, {"input": 0, "output": 0})
    input_cost = input_tokens * rates["input"] / 1_000_000
    output_cost = output_tokens * rates["output"] / 1_000_000
    total_cost = input_cost + output_cost
    
    # Prepare row
    row = {
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost": f"${input_cost:.6f}",
        "output_cost": f"${output_cost:.6f}",
        "total_cost": f"${total_cost:.6f}",
        "session_id": session_id or "unknown"
    }
    
    # Append to CSV
    is_new = not LOG_FILE.exists()
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if is_new:
            writer.writeheader()
        writer.writerow(row)
    
    return total_cost


def get_summary() -> dict:
    """Calculate total costs by model."""
    if not LOG_FILE.exists():
        return {}
    
    summary = {}
    total_spend = 0.0
    
    with open(LOG_FILE) as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row["model"]
            cost = float(row["total_cost"].replace("$", ""))
            
            if model not in summary:
                summary[model] = {"calls": 0, "tokens": 0, "cost": 0.0}
            
            summary[model]["calls"] += 1
            summary[model]["tokens"] += int(row["input_tokens"]) + int(row["output_tokens"])
            summary[model]["cost"] += cost
            total_spend += cost
    
    summary["TOTAL"] = {"cost": total_spend, "calls": sum(s["calls"] for s in summary.values() if s != summary.get("TOTAL", {}))}
    
    return summary


def print_summary():
    """Print cost summary to console."""
    summary = get_summary()
    
    print("\n" + "=" * 80)
    print("💰 ANTHROPIC API COST SUMMARY")
    print("=" * 80)
    
    for model, stats in summary.items():
        if model == "TOTAL":
            print(f"\n{'TOTAL SPEND':<30} ${stats['cost']:.2f}")
        else:
            print(f"\n{model:<30}")
            print(f"  Calls: {stats['calls']}")
            print(f"  Tokens: {stats['tokens']:,}")
            print(f"  Cost: ${stats['cost']:.2f}")
    
    print("\n" + "=" * 80)


def export_summary_json():
    """Export summary to JSON."""
    summary = get_summary()
    with open(SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"✅ Summary exported to {SUMMARY_FILE}")


if __name__ == "__main__":
    # Example usage
    print("Cost Tracker Ready")
    print(f"Log file: {LOG_FILE}")
    print(f"Summary file: {SUMMARY_FILE}")
    
    # Test: log a fake call
    log_api_call("anthropic/claude-haiku-4-5", 1000, 500)
    print_summary()
