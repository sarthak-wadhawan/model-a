"""Verify target lookup mechanism - investigate 12:04 target."""

import sys
sys.path.insert(0, 'src')

import csv
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pytz


def main():
    print("=" * 80)
    print("TARGET LOOKUP VERIFICATION")
    print("=" * 80)
    
    cache_path = "artifacts/model_a/minute_features.csv"
    ny_tz = pytz.timezone("America/New_York")
    
    # Load Aug 25
    summaries = []
    with open(cache_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['date'] == '2026-08-25':
                summaries.append({
                    'timestamp': datetime.fromisoformat(row['timestamp']),
                    'midpoint': float(row['midpoint'])
                })
    
    print(f"\nAug 25 cache rows: {len(summaries)}")
    print(f"First: {summaries[0]['timestamp'].astimezone(ny_tz).strftime('%H:%M:%S')}")
    print(f"Last: {summaries[-1]['timestamp'].astimezone(ny_tz).strftime('%H:%M:%S')}")
    
    # Simulate target lookup for 11:59 -> 12:04
    print("\n" + "=" * 80)
    print("TARGET LOOKUP SIMULATION: 11:59 -> 12:04")
    print("=" * 80)
    
    # Find 11:59 summary
    decision_time = None
    for summary in summaries:
        et_time = summary['timestamp'].astimezone(ny_tz).strftime("%H:%M")
        if et_time == '11:59':
            decision_time = summary['timestamp']
            decision_midpoint = summary['midpoint']
            break
    
    if decision_time:
        target_time = decision_time + timedelta(minutes=5)
        
        print(f"\nDecision time: {decision_time.astimezone(ny_tz).strftime('%H:%M:%S.%f')[:-3]}")
        print(f"Decision midpoint: {decision_midpoint:.4f}")
        print(f"\nTarget time: {target_time.astimezone(ny_tz).strftime('%H:%M:%S.%f')[:-3]}")
        
        # Find target midpoint using current algorithm
        target_midpoint = None
        for summary in reversed(summaries):
            if summary['timestamp'] <= target_time:
                target_midpoint = summary['midpoint']
                target_summary_time = summary['timestamp']
                break
        
        if target_midpoint:
            print(f"\nTarget midpoint found: {target_midpoint:.4f}")
            print(f"Target summary time: {target_summary_time.astimezone(ny_tz).strftime('%H:%M:%S.%f')[:-3]}")
            print(f"\nPROBLEM:")
            print(f"  Target time is 12:04:00")
            print(f"  But returned midpoint is from {target_summary_time.astimezone(ny_tz).strftime('%H:%M:%S')}")
            print(f"  This is NOT the midpoint at 12:04:00")
            print(f"  This is the most recent summary <= 12:04")
            print(f"  Which is the 11:59 summary")
            print(f"  THIS IS INCORRECT - target should use midpoint AT 12:04, not BEFORE")
        else:
            print(f"\nTarget midpoint: NOT FOUND")
    
    print("\n" + "=" * 80)
    print("ADDITIONAL DEFECT IDENTIFIED")
    print("=" * 80)
    print("\nTARGET LOOKUP DEFECT:")
    print("  - Current algorithm finds most recent summary <= target_time")
    print("  - For target 12:04, it returns 11:59 summary (last <= 12:04)")
    print("  - This uses the WRONG midpoint (11:59 instead of 12:04)")
    print("  - Target should use midpoint AT exactly 12:04:00, not before")
    print("  - This creates TARGET LEAKAGE (using stale data)")
    
    print("\nROOT CAUSE:")
    print("  - Cache only contains 145 rows (09:35-11:59)")
    print("  - It does NOT contain target rows (12:00-12:04)")
    print("  - Target lookup incorrectly uses stale data instead of failing")


if __name__ == '__main__':
    main()
