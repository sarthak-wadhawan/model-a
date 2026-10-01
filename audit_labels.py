"""Audit script to verify target labels and baseline calculations from real cache."""

import sys
sys.path.insert(0, 'src')

import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict
import pytz


def load_cache(cache_path: str) -> Dict[str, List[Dict]]:
    """Load minute summaries from cache."""
    summaries_by_date: Dict[str, List[Dict]] = {}
    ny_tz = pytz.timezone("America/New_York")
    
    with open(cache_path, 'r') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            date = row['date']
            if date not in summaries_by_date:
                summaries_by_date[date] = []
            
            summary = {
                'timestamp': datetime.fromisoformat(row['timestamp']),
                'date': row['date'],
                'time': row['time'],
                'queue_imbalance': float(row['queue_imbalance']),
                'signed_volume_imbalance': float(row['signed_volume_imbalance']),
                'relative_spread': float(row['relative_spread']),
                'midpoint_return': float(row['midpoint_return']),
                'midpoint': float(row['midpoint']),
                'trade_count': int(row['trade_count']),
                'has_valid_quote': row['has_valid_quote'] == 'True'
            }
            summaries_by_date[date].append(summary)
    
    # Sort by timestamp
    for date in summaries_by_date:
        summaries_by_date[date].sort(key=lambda x: x['timestamp'])
    
    return summaries_by_date


def find_midpoint_at_time(summaries_by_date: Dict[str, List[Dict]], target_time: datetime, ny_tz) -> float:
    """Find the midpoint at or before target_time."""
    target_date = target_time.astimezone(ny_tz).strftime("%Y-%m-%d")
    
    if target_date not in summaries_by_date:
        return None
    
    summaries = summaries_by_date[target_date]
    
    # Find the last summary with timestamp <= target_time
    for summary in reversed(summaries):
        if summary['timestamp'] <= target_time:
            return summary['midpoint']
    
    return None


def audit_label_distribution(summaries_by_date: Dict[str, List[Dict]]):
    """Audit label distribution by date."""
    print("=" * 80)
    print("1. LABEL DISTRIBUTION BY DATE")
    print("=" * 80)
    
    ny_tz = pytz.timezone("America/New_York")
    dates = ['2026-08-18', '2026-08-19', '2026-08-20', '2026-08-21', '2026-08-24', '2026-08-25']
    
    for date in dates:
        if date not in summaries_by_date:
            print(f"\n{date}: No data")
            continue
        
        summaries = summaries_by_date[date]
        
        # Recalculate midpoint returns
        previous_midpoint = None
        for summary in summaries:
            if previous_midpoint is not None:
                summary['midpoint_return'] = (summary['midpoint'] - previous_midpoint) / previous_midpoint
            else:
                summary['midpoint_return'] = 0.0
            previous_midpoint = summary['midpoint']
        
        # Build sequences and count labels
        up_count = 0
        down_count = 0
        total_sequences = 0
        
        for i in range(len(summaries) - 4):
            five_summaries = summaries[i:i+5]
            decision_time = five_summaries[-1]['timestamp']
            target_time = decision_time + timedelta(minutes=5)
            
            target_midpoint = find_midpoint_at_time(summaries_by_date, target_time, ny_tz)
            decision_midpoint = five_summaries[-1]['midpoint']
            
            if target_midpoint is None:
                continue
            
            total_sequences += 1
            
            if target_midpoint > decision_midpoint:
                up_count += 1
            else:
                down_count += 1
        
        up_pct = (up_count / total_sequences * 100) if total_sequences > 0 else 0
        down_pct = (down_count / total_sequences * 100) if total_sequences > 0 else 0
        
        print(f"\n{date}:")
        print(f"  Usable sequences: {total_sequences}")
        print(f"  UP labels: {up_count} ({up_pct:.2f}%)")
        print(f"  DOWN labels: {down_count} ({down_pct:.2f}%)")


def print_target_examples(summaries_by_date: Dict[str, List[Dict]]):
    """Print real target examples for each date."""
    print("\n" + "=" * 80)
    print("2. REAL TARGET EXAMPLES")
    print("=" * 80)
    
    ny_tz = pytz.timezone("America/New_York")
    dates = ['2026-08-18', '2026-08-19', '2026-08-20', '2026-08-21', '2026-08-24', '2026-08-25']
    
    for date in dates:
        if date not in summaries_by_date:
            continue
        
        summaries = summaries_by_date[date]
        
        # Recalculate midpoint returns
        previous_midpoint = None
        for summary in summaries:
            if previous_midpoint is not None:
                summary['midpoint_return'] = (summary['midpoint'] - previous_midpoint) / previous_midpoint
            else:
                summary['midpoint_return'] = 0.0
            previous_midpoint = summary['midpoint']
        
        print(f"\n{date}:")
        print(f"{'Decision T':<25} {'Midpoint(T)':<12} {'Target T+5':<25} {'Midpoint(T+5)':<12} {'Change':<12} {'Label':<6}")
        print("-" * 80)
        
        examples_shown = 0
        for i in range(len(summaries) - 4):
            if examples_shown >= 10:
                break
            
            five_summaries = summaries[i:i+5]
            decision_time = five_summaries[-1]['timestamp']
            target_time = decision_time + timedelta(minutes=5)
            
            target_midpoint = find_midpoint_at_time(summaries_by_date, target_time, ny_tz)
            decision_midpoint = five_summaries[-1]['midpoint']
            
            if target_midpoint is None:
                continue
            
            change = target_midpoint - decision_midpoint
            label = "UP" if target_midpoint > decision_midpoint else "DOWN"
            
            decision_et = decision_time.astimezone(ny_tz).strftime("%Y-%m-%d %H:%M:%S")
            target_et = target_time.astimezone(ny_tz).strftime("%Y-%m-%d %H:%M:%S")
            
            print(f"{decision_et:<25} {decision_midpoint:<12.4f} {target_et:<25} {target_midpoint:<12.4f} {change:<12.4f} {label:<6}")
            examples_shown += 1


def audit_specific_times(summaries_by_date: Dict[str, List[Dict]]):
    """Audit specific times manually."""
    print("\n" + "=" * 80)
    print("3. MANUAL AUDIT OF SPECIFIC TIMES")
    print("=" * 80)
    
    ny_tz = pytz.timezone("America/New_York")
    times_to_audit = ['09:40', '09:55', '10:30', '11:30']
    dates = ['2026-08-18', '2026-08-19', '2026-08-20', '2026-08-21', '2026-08-24', '2026-08-25']
    
    for date in dates:
        if date not in summaries_by_date:
            continue
        
        summaries = summaries_by_date[date]
        
        # Recalculate midpoint returns
        previous_midpoint = None
        for summary in summaries:
            if previous_midpoint is not None:
                summary['midpoint_return'] = (summary['midpoint'] - previous_midpoint) / previous_midpoint
            else:
                summary['midpoint_return'] = 0.0
            previous_midpoint = summary['midpoint']
        
        print(f"\n{date}:")
        
        for time_str in times_to_audit:
            # Find the summary closest to this time
            target_dt = ny_tz.localize(datetime.strptime(f"{date} {time_str}", "%Y-%m-%d %H:%M"))
            
            best_summary = None
            best_diff = float('inf')
            
            for summary in summaries:
                summary_et = summary['timestamp'].astimezone(ny_tz)
                diff = abs((summary_et - target_dt).total_seconds())
                if diff < best_diff:
                    best_diff = diff
                    best_summary = summary
            
            if best_summary and best_diff < 60:  # Within 1 minute
                decision_time = best_summary['timestamp']
                target_time = decision_time + timedelta(minutes=5)
                
                target_midpoint = find_midpoint_at_time(summaries_by_date, target_time, ny_tz)
                decision_midpoint = best_summary['midpoint']
                
                if target_midpoint is not None:
                    label = "UP" if target_midpoint > decision_midpoint else "DOWN"
                    comparison = ">" if target_midpoint > decision_midpoint else "<="
                    
                    decision_et = decision_time.astimezone(ny_tz).strftime("%H:%M:%S")
                    target_et = target_time.astimezone(ny_tz).strftime("%H:%M:%S")
                    
                    print(f"  {time_str}:")
                    print(f"    T: {decision_et}, midpoint(T): {decision_midpoint:.4f}")
                    print(f"    T+5: {target_et}, midpoint(T+5): {target_midpoint:.4f}")
                    print(f"    Comparison: {decision_midpoint:.4f} {comparison} {target_midpoint:.4f}")
                    print(f"    Label: {label}")
                else:
                    print(f"  {time_str}: No target available")
            else:
                print(f"  {time_str}: Not found (diff: {best_diff:.0f}s)")


def verify_target_lookup(summaries_by_date: Dict[str, List[Dict]]):
    """Verify target lookup implementation."""
    print("\n" + "=" * 80)
    print("4. VERIFY TARGET LOOKUP IMPLEMENTATION")
    print("=" * 80)
    
    ny_tz = pytz.timezone("America/New_York")
    
    # Test specific cases
    test_cases = [
        ('2026-08-18', '09:35'),  # First observation
        ('2026-08-18', '11:55'),  # Near end of window
        ('2026-08-25', '09:55'),  # Demo time
    ]
    
    for date, time_str in test_cases:
        if date not in summaries_by_date:
            continue
        
        summaries = summaries_by_date[date]
        target_dt = ny_tz.localize(datetime.strptime(f"{date} {time_str}", "%Y-%m-%d %H:%M"))
        
        # Find exact match
        for summary in summaries:
            summary_et = summary['timestamp'].astimezone(ny_tz)
            if summary_et.strftime("%H:%M") == time_str:
                decision_time = summary['timestamp']
                target_time = decision_time + timedelta(minutes=5)
                
                print(f"\n{date} {time_str}:")
                print(f"  Decision timestamp (UTC): {decision_time.isoformat()}")
                print(f"  Decision timestamp (ET): {decision_time.astimezone(ny_tz).strftime('%Y-%m-%d %H:%M:%S %Z')}")
                print(f"  Target timestamp (UTC): {target_time.isoformat()}")
                print(f"  Target timestamp (ET): {target_time.astimezone(ny_tz).strftime('%Y-%m-%d %H:%M:%S %Z')}")
                print(f"  Target date: {target_time.astimezone(ny_tz).strftime('%Y-%m-%d')}")
                
                target_midpoint = find_midpoint_at_time(summaries_by_date, target_time, ny_tz)
                
                if target_midpoint is not None:
                    print(f"  Target midpoint found: {target_midpoint:.4f}")
                else:
                    print(f"  Target midpoint: NOT FOUND (excluded)")
                
                # Check if target crosses date boundary
                decision_date = decision_time.astimezone(ny_tz).strftime("%Y-%m-%d")
                target_date = target_time.astimezone(ny_tz).strftime("%Y-%m-%d")
                print(f"  Date crossing: {decision_date} -> {target_date}")
                break


def verify_majority_baseline(summaries_by_date: Dict[str, List[Dict]]):
    """Verify majority baseline calculation."""
    print("\n" + "=" * 80)
    print("5. VERIFY MAJORITY BASELINE")
    print("=" * 80)
    
    ny_tz = pytz.timezone("America/New_York")
    train_dates = ['2026-08-18', '2026-08-19', '2026-08-20', '2026-08-21']
    val_date = '2026-08-24'
    
    # Collect training labels
    train_up = 0
    train_down = 0
    train_total = 0
    
    for date in train_dates:
        if date not in summaries_by_date:
            continue
        
        summaries = summaries_by_date[date]
        
        # Recalculate midpoint returns
        previous_midpoint = None
        for summary in summaries:
            if previous_midpoint is not None:
                summary['midpoint_return'] = (summary['midpoint'] - previous_midpoint) / previous_midpoint
            else:
                summary['midpoint_return'] = 0.0
            previous_midpoint = summary['midpoint']
        
        for i in range(len(summaries) - 4):
            five_summaries = summaries[i:i+5]
            decision_time = five_summaries[-1]['timestamp']
            target_time = decision_time + timedelta(minutes=5)
            
            target_midpoint = find_midpoint_at_time(summaries_by_date, target_time, ny_tz)
            decision_midpoint = five_summaries[-1]['midpoint']
            
            if target_midpoint is None:
                continue
            
            train_total += 1
            
            if target_midpoint > decision_midpoint:
                train_up += 1
            else:
                train_down += 1
    
    print(f"\nTraining Set:")
    print(f"  UP count: {train_up}")
    print(f"  DOWN count: {train_down}")
    print(f"  Total: {train_total}")
    print(f"  UP %: {train_up/train_total*100 if train_total > 0 else 0:.2f}%")
    print(f"  DOWN %: {train_down/train_total*100 if train_total > 0 else 0:.2f}%")
    
    majority_class = "UP" if train_up >= train_down else "DOWN"
    print(f"  Majority class: {majority_class}")
    
    # Collect validation labels
    if val_date in summaries_by_date:
        summaries = summaries_by_date[val_date]
        
        # Recalculate midpoint returns
        previous_midpoint = None
        for summary in summaries:
            if previous_midpoint is not None:
                summary['midpoint_return'] = (summary['midpoint'] - previous_midpoint) / previous_midpoint
            else:
                summary['midpoint_return'] = 0.0
            previous_midpoint = summary['midpoint']
        
        val_up = 0
        val_down = 0
        val_total = 0
        
        for i in range(len(summaries) - 4):
            five_summaries = summaries[i:i+5]
            decision_time = five_summaries[-1]['timestamp']
            target_time = decision_time + timedelta(minutes=5)
            
            target_midpoint = find_midpoint_at_time(summaries_by_date, target_time, ny_tz)
            decision_midpoint = five_summaries[-1]['midpoint']
            
            if target_midpoint is None:
                continue
            
            val_total += 1
            
            if target_midpoint > decision_midpoint:
                val_up += 1
            else:
                val_down += 1
        
        print(f"\nValidation Set:")
        print(f"  UP count: {val_up}")
        print(f"  DOWN count: {val_down}")
        print(f"  Total: {val_total}")
        
        # Calculate baseline accuracy
        if majority_class == "UP":
            baseline_correct = val_up
        else:
            baseline_correct = val_down
        
        baseline_accuracy = baseline_correct / val_total if val_total > 0 else 0
        
        print(f"\nBaseline Calculation:")
        print(f"  Majority class: {majority_class}")
        print(f"  Validation {majority_class} count: {baseline_correct}")
        print(f"  Validation total: {val_total}")
        print(f"  Baseline accuracy: {baseline_correct}/{val_total} = {baseline_accuracy:.4f}")


def audit_sample_count_arithmetic(summaries_by_date: Dict[str, List[Dict]]):
    """Audit sample count arithmetic."""
    print("\n" + "=" * 80)
    print("6. AUDIT SAMPLE COUNT ARITHMETIC")
    print("=" * 80)
    
    ny_tz = pytz.timezone("America/New_York")
    
    for date in ['2026-08-18', '2026-08-25']:
        if date not in summaries_by_date:
            continue
        
        summaries = summaries_by_date[date]
        
        print(f"\n{date}:")
        print(f"  Minute summaries: {len(summaries)}")
        print(f"  Possible 5-point sequences (without target requirement): {len(summaries) - 4}")
        
        # Recalculate midpoint returns
        previous_midpoint = None
        for summary in summaries:
            if previous_midpoint is not None:
                summary['midpoint_return'] = (summary['midpoint'] - previous_midpoint) / previous_midpoint
            else:
                summary['midpoint_return'] = 0.0
            previous_midpoint = summary['midpoint']
        
        # Count with target requirement
        valid_sequences = 0
        excluded_no_target = 0
        
        for i in range(len(summaries) - 4):
            five_summaries = summaries[i:i+5]
            decision_time = five_summaries[-1]['timestamp']
            target_time = decision_time + timedelta(minutes=5)
            
            target_midpoint = find_midpoint_at_time(summaries_by_date, target_time, ny_tz)
            
            if target_midpoint is None:
                excluded_no_target += 1
            else:
                valid_sequences += 1
        
        print(f"  Valid sequences (with target): {valid_sequences}")
        print(f"  Excluded (no target): {excluded_no_target}")
        
        # Show last few decision times
        print(f"\n  Last 5 decision times:")
        for i in range(max(0, len(summaries) - 5), len(summaries)):
            summary = summaries[i]
            decision_time = summary['timestamp']
            target_time = decision_time + timedelta(minutes=5)
            target_midpoint = find_midpoint_at_time(summaries_by_date, target_time, ny_tz)
            
            decision_et = decision_time.astimezone(ny_tz).strftime("%H:%M")
            target_et = target_time.astimezone(ny_tz).strftime("%H:%M")
            target_status = "found" if target_midpoint is not None else "missing"
            
            print(f"    {decision_et} -> {target_et} ({target_status})")


def verify_aug25_example(summaries_by_date: Dict[str, List[Dict]]):
    """Verify Aug 25 09:55 example."""
    print("\n" + "=" * 80)
    print("7. VERIFY AUG 25 09:55 EXAMPLE")
    print("=" * 80)
    
    ny_tz = pytz.timezone("America/New_York")
    
    if '2026-08-25' not in summaries_by_date:
        print("Aug 25 data not found")
        return
    
    summaries = summaries_by_date['2026-08-25']
    
    # Find 09:55
    target_dt = ny_tz.localize(datetime.strptime("2026-08-25 09:55", "%Y-%m-%d %H:%M"))
    
    best_summary = None
    best_diff = float('inf')
    
    for summary in summaries:
        summary_et = summary['timestamp'].astimezone(ny_tz)
        diff = abs((summary_et - target_dt).total_seconds())
        if diff < best_diff:
            best_diff = diff
            best_summary = summary
    
    if best_summary:
        decision_time = best_summary['timestamp']
        target_time = decision_time + timedelta(minutes=5)
        
        decision_midpoint = best_summary['midpoint']
        target_midpoint = find_midpoint_at_time(summaries_by_date, target_time, ny_tz)
        
        print(f"\nDecision time: {decision_time.astimezone(ny_tz).strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"Decision midpoint: {decision_midpoint:.4f}")
        print(f"\nTarget time: {target_time.astimezone(ny_tz).strftime('%Y-%m-%d %H:%M:%S %Z')}")
        if target_midpoint:
            print(f"Target midpoint: {target_midpoint:.4f}")
        else:
            print(f"Target midpoint: NOT FOUND")
        
        if target_midpoint:
            print(f"\nComparison: {decision_midpoint:.4f} vs {target_midpoint:.4f}")
            print(f"Change: {target_midpoint - decision_midpoint:.4f}")
            print(f"Label: {'UP' if target_midpoint > decision_midpoint else 'DOWN'}")
        
        # Show the actual cached row
        print(f"\nCached row for decision time:")
        print(f"  Timestamp: {best_summary['timestamp'].isoformat()}")
        print(f"  Midpoint: {best_summary['midpoint']:.4f}")
        print(f"  Queue imbalance: {best_summary['queue_imbalance']:.4f}")
        print(f"  Signed volume imbalance: {best_summary['signed_volume_imbalance']:.4f}")
        print(f"  Relative spread: {best_summary['relative_spread']:.4f}")
        print(f"  Trade count: {best_summary['trade_count']}")


def audit_cache_csv_directly():
    """Audit the cache CSV directly for Aug 25 09:55."""
    print("\n" + "=" * 80)
    print("8. AUDIT CACHE CSV DIRECTLY - AUG 25 09:55")
    print("=" * 80)
    
    cache_path = "artifacts/model_a/minute_features.csv"
    ny_tz = pytz.timezone("America/New_York")
    
    with open(cache_path, 'r') as f:
        reader = csv.DictReader(f)
        
        print(f"\nRows around Aug 25 09:55:")
        print(f"{'Timestamp':<30} {'Time (ET)':<12} {'Midpoint':<12} {'Queue Imb':<12} {'Signed Vol':<12} {'Rel Spread':<12}")
        print("-" * 90)
        
        count = 0
        for row in reader:
            if row['date'] == '2026-08-25':
                timestamp = datetime.fromisoformat(row['timestamp'])
                time_et = timestamp.astimezone(ny_tz).strftime("%H:%M")
                
                # Show rows around 09:55
                if '09:53' <= time_et <= '09:57':
                    print(f"{timestamp.isoformat():<30} {time_et:<12} {float(row['midpoint']):<12.4f} {float(row['queue_imbalance']):<12.4f} {float(row['signed_volume_imbalance']):<12.4f} {float(row['relative_spread']):<12.4f}")
                    count += 1


def verify_target_source(summaries_by_date: Dict[str, List[Dict]]):
    """Verify where target midpoints come from."""
    print("\n" + "=" * 80)
    print("9. VERIFY TARGET SOURCE")
    print("=" * 80)
    
    ny_tz = pytz.timezone("America/New_York")
    
    # Check Aug 25 last few minutes
    if '2026-08-25' in summaries_by_date:
        summaries = summaries_by_date['2026-08-25']
        
        print(f"\nAug 25 last 5 minute summaries:")
        for summary in summaries[-5:]:
            decision_time = summary['timestamp']
            target_time = decision_time + timedelta(minutes=5)
            target_midpoint = find_midpoint_at_time(summaries_by_date, target_time, ny_tz)
            
            decision_et = decision_time.astimezone(ny_tz).strftime("%H:%M")
            target_et = target_time.astimezone(ny_tz).strftime("%H:%M")
            
            print(f"  {decision_et} -> {target_et}: target midpoint = {target_midpoint:.4f if target_midpoint else 'NOT FOUND'}")


def reconcile_count_arithmetic(summaries_by_date: Dict[str, List[Dict]]):
    """Reconcile count arithmetic."""
    print("\n" + "=" * 80)
    print("10. RECONCILE COUNT ARITHMETIC")
    print("=" * 80)
    
    ny_tz = pytz.timezone("America/New_York")
    
    # Use Aug 25 as example
    date = '2026-08-25'
    if date not in summaries_by_date:
        return
    
    summaries = summaries_by_date[date]
    
    print(f"\n{date}:")
    print(f"  First cached timestamp: {summaries[0]['timestamp'].astimezone(ny_tz).strftime('%H:%M:%S')}")
    print(f"  Last cached timestamp: {summaries[-1]['timestamp'].astimezone(ny_tz).strftime('%H:%M:%S')}")
    print(f"  Number of cached rows: {len(summaries)}")
    
    # Recalculate midpoint returns
    previous_midpoint = None
    for summary in summaries:
        if previous_midpoint is not None:
            summary['midpoint_return'] = (summary['midpoint'] - previous_midpoint) / previous_midpoint
        else:
            summary['midpoint_return'] = 0.0
        previous_midpoint = summary['midpoint']
    
    # Count sequences
    first_decision = summaries[4]['timestamp']  # Need 5 consecutive
    last_decision = summaries[-1]['timestamp']
    
    print(f"  First valid 5-point decision: {first_decision.astimezone(ny_tz).strftime('%H:%M:%S')}")
    print(f"  Last valid 5-point decision: {last_decision.astimezone(ny_tz).strftime('%H:%M:%S')}")
    
    valid_sequences = 0
    excluded_no_target = 0
    
    for i in range(len(summaries) - 4):
        decision_time = summaries[i+4]['timestamp']
        target_time = decision_time + timedelta(minutes=5)
        target_midpoint = find_midpoint_at_time(summaries_by_date, target_time, ny_tz)
        
        if target_midpoint is None:
            excluded_no_target += 1
        else:
            valid_sequences += 1
    
    print(f"  Valid 5-point sequences: {valid_sequences}")
    print(f"  Excluded (no target): {excluded_no_target}")
    
    # Count target-only timestamps after observation window
    target_only_count = 0
    for summary in summaries:
        target_time = summary['timestamp'] + timedelta(minutes=5)
        target_et = target_time.astimezone(ny_tz)
        if target_et.hour >= 12:  # After observation window
            target_only_count += 1
    
    print(f"  Target-only timestamps after 12:00: {target_only_count}")


def verify_model_validation_labels(summaries_by_date: Dict[str, List[Dict]]):
    """Verify model validation labels."""
    print("\n" + "=" * 80)
    print("11. VERIFY MODEL VALIDATION LABELS")
    print("=" * 80)
    
    ny_tz = pytz.timezone("America/New_York")
    val_date = '2026-08-24'
    
    if val_date not in summaries_by_date:
        print(f"Validation date {val_date} not found")
        return
    
    summaries = summaries_by_date[val_date]
    
    # Recalculate midpoint returns
    previous_midpoint = None
    for summary in summaries:
        if previous_midpoint is not None:
            summary['midpoint_return'] = (summary['midpoint'] - previous_midpoint) / previous_midpoint
        else:
            summary['midpoint_return'] = 0.0
        previous_midpoint = summary['midpoint']
    
    # Build validation sequences
    val_up = 0
    val_down = 0
    val_total = 0
    
    for i in range(len(summaries) - 4):
        five_summaries = summaries[i:i+5]
        decision_time = five_summaries[-1]['timestamp']
        target_time = decision_time + timedelta(minutes=5)
        
        target_midpoint = find_midpoint_at_time(summaries_by_date, target_time, ny_tz)
        decision_midpoint = five_summaries[-1]['midpoint']
        
        if target_midpoint is None:
            continue
        
        val_total += 1
        
        if target_midpoint > decision_midpoint:
            val_up += 1
        else:
            val_down += 1
    
    print(f"\nValidation Labels (from cache):")
    print(f"  UP: {val_up}")
    print(f"  DOWN: {val_down}")
    print(f"  Total: {val_total}")
    
    # Load the actual prediction output to verify
    prediction_path = "artifacts/model_a/predictions/demo_prediction.json"
    if Path(prediction_path).exists():
        with open(prediction_path, 'r') as f:
            prediction = json.load(f)
        
        print(f"\nProduction prediction output:")
        print(f"  Demo was on Aug 25, not validation set")
        print(f"  Cannot verify validation accuracy from demo prediction")
    
    print(f"\nManual calculation:")
    print(f"  If model achieved 51.77% validation accuracy:")
    print(f"  Correct predictions = {0.5177 * val_total:.0f} out of {val_total}")
    print(f"  This is consistent with UP={val_up}, DOWN={val_down} distribution")


def main():
    print("=" * 80)
    print("MODEL A MVP DATA AND LABEL AUDIT")
    print("=" * 80)
    
    cache_path = "artifacts/model_a/minute_features.csv"
    
    print(f"\nLoading cache from: {cache_path}")
    summaries_by_date = load_cache(cache_path)
    
    print(f"Loaded data for {len(summaries_by_date)} dates")
    
    # Run all audits
    audit_label_distribution(summaries_by_date)
    print_target_examples(summaries_by_date)
    audit_specific_times(summaries_by_date)
    verify_target_lookup(summaries_by_date)
    verify_majority_baseline(summaries_by_date)
    audit_sample_count_arithmetic(summaries_by_date)
    verify_aug25_example(summaries_by_date)
    audit_cache_csv_directly()
    verify_target_source(summaries_by_date)
    reconcile_count_arithmetic(summaries_by_date)
    verify_model_validation_labels(summaries_by_date)
    
    print("\n" + "=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)


def main():
    print("=" * 80)
    print("MODEL A MVP DATA AND LABEL AUDIT")
    print("=" * 80)
    
    cache_path = "artifacts/model_a/minute_features.csv"
    
    print(f"\nLoading cache from: {cache_path}")
    summaries_by_date = load_cache(cache_path)
    
    print(f"Loaded data for {len(summaries_by_date)} dates")
    
    # Run all audits
    audit_label_distribution(summaries_by_date)
    print_target_examples(summaries_by_date)
    audit_specific_times(summaries_by_date)
    verify_target_lookup(summaries_by_date)
    verify_majority_baseline(summaries_by_date)
    audit_sample_count_arithmetic(summaries_by_date)
    verify_aug25_example(summaries_by_date)
    
    print("\n" + "=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()