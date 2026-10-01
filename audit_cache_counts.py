"""Audit corrected cache counts to resolve inconsistency."""

import sys
sys.path.insert(0, 'src')

import csv
from datetime import datetime
from pathlib import Path
import pytz


def main():
    print("=" * 80)
    print("CACHE COUNT AUDIT")
    print("=" * 80)
    
    cache_path = "artifacts/model_a/minute_features_fixed.csv"
    ny_tz = pytz.timezone("America/New_York")
    
    with open(cache_path, 'r') as f:
        reader = csv.DictReader(f)
        
        # Group by date
        dates_data = {}
        for row in reader:
            date = row['date']
            if date not in dates_data:
                dates_data[date] = []
            
            timestamp = datetime.fromisoformat(row['timestamp'])
            is_target_only = row.get('is_target_only', 'False') == 'True'
            
            dates_data[date].append({
                'timestamp': timestamp,
                'time': row['time'],
                'midpoint': float(row['midpoint']),
                'is_target_only': is_target_only
            })
    
    # Audit each date
    for date in sorted(dates_data.keys()):
        summaries = dates_data[date]
        summaries.sort(key=lambda x: x['timestamp'])
        
        input_summaries = [s for s in summaries if not s['is_target_only']]
        target_summaries = [s for s in summaries if s['is_target_only']]
        
        print(f"\n{date}:")
        print(f"  Total rows: {len(summaries)}")
        print(f"  Input-eligible: {len(input_summaries)}")
        print(f"  Target-only: {len(target_summaries)}")
        
        if summaries:
            print(f"  First timestamp: {summaries[0]['timestamp'].astimezone(ny_tz).strftime('%H:%M:%S')}")
            print(f"  Last timestamp: {summaries[-1]['timestamp'].astimezone(ny_tz).strftime('%H:%M:%S')}")
        
        if input_summaries:
            print(f"  First input: {input_summaries[0]['timestamp'].astimezone(ny_tz).strftime('%H:%M:%S')}")
            print(f"  Last input: {input_summaries[-1]['timestamp'].astimezone(ny_tz).strftime('%H:%M:%S')}")
        
        if target_summaries:
            print(f"  First target: {target_summaries[0]['timestamp'].astimezone(ny_tz).strftime('%H:%M:%S')}")
            print(f"  Last target: {target_summaries[-1]['timestamp'].astimezone(ny_tz).strftime('%H:%M:%S')}")
    
    # Detailed Aug 25 audit
    print("\n" + "=" * 80)
    print("AUG 25 DETAILED CLASSIFICATION")
    print("=" * 80)
    
    aug25 = dates_data.get('2026-08-25', [])
    aug25.sort(key=lambda x: x['timestamp'])
    
    print("\nTimestamps around key boundaries:")
    key_times = ['09:34', '09:35', '09:36', '11:59', '12:00', '12:01', '12:05']
    
    for summary in aug25:
        time_str = summary['timestamp'].astimezone(ny_tz).strftime("%H:%M")
        if time_str in key_times:
            marker = "INPUT" if not summary['is_target_only'] else "TARGET"
            print(f"  {time_str}: {summary['timestamp'].astimezone(ny_tz).strftime('%H:%M:%S')} - {marker}")
    
    # Show all input-eligible times
    print("\nAll input-eligible times:")
    input_times = [s['timestamp'].astimezone(ny_tz).strftime("%H:%M") for s in aug25 if not s['is_target_only']]
    print(f"  {' '.join(input_times)}")
    print(f"  Count: {len(input_times)}")
    
    # Show all target-only times
    print("\nAll target-only times:")
    target_times = [s['timestamp'].astimezone(ny_tz).strftime("%H:%M") for s in aug25 if s['is_target_only']]
    print(f"  {' '.join(target_times)}")
    print(f"  Count: {len(target_times)}")
    
    # Expected vs actual
    print("\n" + "=" * 80)
    print("EXPECTED vs ACTUAL")
    print("=" * 80)
    print(f"\nExpected input-eligible: 09:35-12:00 = 146 boundaries")
    print(f"Actual input-eligible: {len(input_summaries)}")
    print(f"\nExpected target-only: 12:01-12:05 = 5 boundaries")
    print(f"Actual target-only: {len(target_summaries)}")
    print(f"\nExpected total: 151")
    print(f"Actual total: {len(summaries)}")
    
    # Identify missing boundaries
    expected_input = [f"{h:02d}:{m:02d}" for h in range(9, 13) for m in range(35, 60) if not (h == 9 and m < 35)]
    expected_input.extend(["12:00"])
    
    actual_input = [s['timestamp'].astimezone(ny_tz).strftime("%H:%M") for s in aug25 if not s['is_target_only']]
    
    missing_input = set(expected_input) - set(actual_input)
    if missing_input:
        print(f"\nMISSING input boundaries: {sorted(missing_input)}")
    else:
        print(f"\nNo missing input boundaries")
    
    expected_target = ["12:01", "12:02", "12:03", "12:04", "12:05"]
    actual_target = [s['timestamp'].astimezone(ny_tz).strftime("%H:%M") for s in aug25 if s['is_target_only']]
    
    missing_target = set(expected_target) - set(actual_target)
    if missing_target:
        print(f"MISSING target boundaries: {sorted(missing_target)}")
    else:
        print(f"No missing target boundaries")


if __name__ == '__main__':
    main()