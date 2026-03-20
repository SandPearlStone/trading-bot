#!/usr/bin/env python3
"""
alert_on_grade.py — Alert when A-grade or B-grade setups are found

Usage:
    python3 alert_on_grade.py --grade A              # Alert on A-grade only
    python3 alert_on_grade.py --grade B --watch 30   # Check every 30 min for A or B
"""

import subprocess
import json
import time
from datetime import datetime
from pathlib import Path

ALERT_STATE = "alert_state.json"


def run_scanner():
    """Run find_trades.py and capture output."""
    try:
        result = subprocess.run(
            ["python3", "find_trades.py", "--min-grade", "B"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        return result.stdout
    except Exception as e:
        print(f"❌ Scanner error: {e}")
        return None


def parse_grades(output):
    """Extract grades from scanner output."""
    grades = {}
    for line in output.split("\n"):
        if "Grade:" in line and "/" in line:
            # Example: "Grade: A (81/100)"
            parts = line.split("|")
            if len(parts) >= 2:
                symbol = parts[0].split("/")[0].strip()
                grade_part = parts[1].strip()
                if "Grade:" in grade_part:
                    grade = grade_part.split("Grade:")[1].strip()[0]  # Get first char (A/B/C/F)
                    grades[symbol] = grade
    return grades


def load_state():
    """Load last known grades."""
    if Path(ALERT_STATE).exists():
        with open(ALERT_STATE) as f:
            return json.load(f)
    return {}


def save_state(grades):
    """Save current grades."""
    with open(ALERT_STATE, "w") as f:
        json.dump(grades, f)


def alert(symbol, grade, is_new=False):
    """Print alert."""
    if is_new:
        print(f"\n🚨 NEW {grade}-GRADE SETUP FOUND!")
    else:
        print(f"\n✨ {grade}-GRADE: {symbol}")
    print(f"   Time: {datetime.now().strftime('%H:%M:%S')}")
    print(f"   Action: Run find_trades.py for details")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Alert on high-grade setups")
    parser.add_argument("--grade", default="A", help="Alert on this grade or better (A or B)")
    parser.add_argument("--watch", type=int, default=0, help="Check every N minutes (0=once)")
    args = parser.parse_args()

    min_grade = args.grade
    alert_threshold = {"A": 1, "B": 2, "C": 3}[min_grade]

    last_grades = load_state()

    try:
        while True:
            print(f"\n🔍 Scanning for {min_grade}-grade setups...")
            output = run_scanner()

            if output:
                grades = parse_grades(output)

                # Check for new or upgraded grades
                for symbol, grade in grades.items():
                    grade_value = {"A": 1, "B": 2, "C": 3, "F": 4}.get(grade, 5)

                    # Alert if grade meets threshold
                    if grade_value <= alert_threshold:
                        old_grade = last_grades.get(symbol)
                        is_new = symbol not in last_grades

                        if is_new or old_grade != grade:
                            alert(symbol, grade, is_new=is_new)

                last_grades = grades
                save_state(grades)

            if args.watch > 0:
                print(f"\n⏳ Next scan in {args.watch} minutes...")
                time.sleep(args.watch * 60)
            else:
                break

    except KeyboardInterrupt:
        print("\n✅ Alert stopped")


if __name__ == "__main__":
    main()
