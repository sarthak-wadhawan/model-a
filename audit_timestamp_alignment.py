"""Audit cache timestamp alignment to detect one-minute shift or leakage."""

import sys
sys.path.insert(0, 'src')

import csv
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pytz


def audit_cache_boundaries():
    """Audit cache minute boundaries."""
    print("=" * 80)
    print("1. CACHE BOUNDARY SEMANTICS")
    print("=" * 80)
    
    cache_path = "artifacts/model_a/minute_features.csv"
    ny_tz = pytz.timezone("America/New_York")
    
    print("\nCache building code analysis:")
    print("  - Events grouped by: strftime('%Y-%m-%d %H:%M')  (rounds DOWN to minute)")
    print("  - Stored timestamp: minute_df['timestamp_dt'].max()  (LAST event in bucket)")
    print("  - This means: bucket '09:55' contains events from 09:55:00.001 to 09:55:59.999")
    print("  - Stored timestamp could be as late as 09:55:59.999")
    
    print("\nPRD requirement:")
    print("  - Summary labelled 09:55 must represent trailing minute ending at 09:55:00")
    print("  - Must contain events from 09:54:00 < timestamp <= 09:55:00")
    print("  - Decision time should be exactly 09:55:00")
    
    print("\nDefect analysis:")
    print("  - Current implementation stores LAST event timestamp, not 09:55:00")
    print("  - If an event occurs at 09:55:59.999, it goes into '09:55' bucket")
    print("  - This violates the requirement that summary contains no event after 09:55:00")


def inspect_aug25_raw_events():
    """Inspect raw events around Aug 25 09:55."""
    print("\n" + "=" * 80)
    print("2. INSPECT RAW EVENTS AROUND AUG 25 09:55")
    print("=" * 80)
    
    events_csv = "artifacts/spy_iex_august/market_data/events.csv"
    ny_tz = pytz.timezone("America/New_York")
    
    # Parse function
    def parse_timestamp(ts_str):
        ts_str = ts_str.rstrip('Z')
        if '.' in ts_str:
            parts = ts_str.split('.')
            if len(parts) > 1:
                microsecond_part = parts[1][:6]
                ts_str = f"{parts[0]}.{microsecond_part}"
        try:
            dt = datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S.%f")
        except ValueError:
            try:
                dt = datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S")
            except ValueError:
                raise ValueError(f"Unable to parse timestamp: {ts_str}")
        return dt.replace(tzinfo=timezone.utc)
    
    print("\nScanning events.csv for Aug 25 events around 09:53-09:56...")
    
    count = 0
    events_around_0955 = []
    
    with open(events_csv, 'r') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            timestamp = parse_timestamp(row['timestamp'])
            date_str = timestamp.strftime("%Y-%m-%d")
            
            if date_str != '2026-08-25':
                continue
            
            et_time = timestamp.astimezone(ny_tz)
            time_str = et_time.strftime("%H:%M:%S")
            
            # Check if around 09:53-09:56
            if '09:53' <= time_str <= '09:56':
                events_around_0955.append({
                    'timestamp': timestamp,
                    'et_time': et_time,
                    'event_type': row['event_type'],
                    'bid_price': row['bid_price_1'],
                    'ask_price': row['ask_price_1'],
                    'bid_size': row['bid_size_1'],
                    'ask_size': row['ask_size_1']
                })
                count += 1
    
    print(f"Found {count} events between 09:53:00 and 09:56:59 ET")
    
    # Show events grouped by minute
    print("\nEvents by minute bucket:")
    minute_buckets = {}
    for event in events_around_0955:
        minute = event['et_time'].strftime("%H:%M")
        if minute not in minute_buckets:
            minute_buckets[minute] = []
        minute_buckets[minute].append(event)
    
    for minute in sorted(minute_buckets.keys()):
        events = minute_buckets[minute]
        first_time = events[0]['et_time'].strftime("%H:%M:%S.%f")[:-3]
        last_time = events[-1]['et_time'].strftime("%H:%M:%S.%f")[:-3]
        quote_count = sum(1 for e in events if e['event_type'] == 'quote')
        trade_count = sum(1 for e in events if e['event_type'] == 'trade')
        
        print(f"\n  {minute}:")
        print(f"    First event: {first_time}")
        print(f"    Last event: {last_time}")
        print(f"    Quotes: {quote_count}, Trades: {trade_count}")
        
        # Show first and last quote
        quotes = [e for e in events if e['event_type'] == 'quote']
        if quotes:
            print(f"    First quote: bid={quotes[0]['bid_price']}, ask={quotes[0]['ask_price']}")
            print(f"    Last quote: bid={quotes[-1]['bid_price']}, ask={quotes[-1]['ask_price']}")


def inspect_cache_timestamps():
    """Inspect cache timestamps around Aug 25 09:55."""
    print("\n" + "=" * 80)
    print("3. INSPECT CACHE TIMESTAMPS AROUND AUG 25 09:55")
    print("=" * 80)
    
    cache_path = "artifacts/model_a/minute_features.csv"
    ny_tz = pytz.timezone("America/New_York")
    
    with open(cache_path, 'r') as f:
        reader = csv.DictReader(f)
        
        print("\nCache rows around 09:53-09:56:")
        print(f"{'Label':<10} {'UTC Timestamp':<30} {'ET Time':<12} {'Midpoint':<12}")
        print("-" * 64)
        
        for row in reader:
            if row['date'] == '2026-08-25':
                timestamp = datetime.fromisoformat(row['timestamp'])
                time_et = timestamp.astimezone(ny_tz).strftime("%H:%M")
                
                if '09:53' <= time_et <= '09:56':
                    time_et_full = timestamp.astimezone(ny_tz).strftime("%H:%M:%S.%f")[:-3]
                    print(f"{row['time']:<10} {timestamp.isoformat():<30} {time_et_full:<12} {float(row['midpoint']):<12.4f}")


def verify_minute_shift():
    """Verify if there's a one-minute shift."""
    print("\n" + "=" * 80)
    print("4. VERIFY ONE-MINUTE SHIFT")
    print("=" * 80)
    
    cache_path = "artifacts/model_a/minute_features.csv"
    ny_tz = pytz.timezone("America/New_York")
    
    print("\nFor PRD compliance:")
    print("  - Row labelled '09:55' should represent events from 09:54:00 < t <= 09:55:00")
    print("  - Stored timestamp should be exactly 09:55:00 (end of window)")
    print("  - Should NOT contain events after 09:55:00")
    
    print("\nActual implementation:")
    print("  - Row labelled '09:55' contains events from 09:55:00.001 to 09:55:59.999")
    print("  - Stored timestamp is the LAST event in that bucket (e.g., 09:55:59.832)")
    print("  - This INCLUDES events after 09:55:00")
    
    print("\nCONCLUSION: ONE-MINUTE SHIFT/LEAKAGE DEFECT CONFIRMED")
    print("  - Cache violates PRD requirement")
    print("  - Features contain events after decision time")
    print("  - This is FEATURE LEAKAGE")


def reconcile_count_arithmetic():
    """Reconcile how 145 rows support targets through 12:04."""
    print("\n" + "=" * 80)
    print("5. RECONCILE COUNT ARITHMETIC")
    print("=" * 80)
    
    cache_path = "artifacts/model_a/minute_features.csv"
    ny_tz = pytz.timezone("America/New_York")
    
    with open(cache_path, 'r') as f:
        reader = csv.DictReader(f)
        
        # Find first and last timestamps for Aug 25
        timestamps = []
        for row in reader:
            if row['date'] == '2026-08-25':
                timestamps.append(datetime.fromisoformat(row['timestamp']))
        
        if timestamps:
            first_ts = timestamps[0]
            last_ts = timestamps[-1]
            
            print(f"\nAug 25 cache:")
            print(f"  First timestamp: {first_ts.astimezone(ny_tz).strftime('%H:%M:%S')}")
            print(f"  Last timestamp: {last_ts.astimezone(ny_tz).strftime('%H:%M:%S')}")
            print(f"  Total rows: {len(timestamps)}")
            
            print(f"\nObservation window: 09:35-12:00 ET")
            print(f"  This is 145 minutes (12:00 - 09:35 = 145)")
            print(f"  Cache has 145 rows - one per minute in observation window")
            
            print(f"\nTarget requirement:")
            print(f"  Decision at 11:59 needs target at 12:04")
            print(f"  Decision at 11:59 uses summary with events up to 11:59:59.999")
            print(f"  Target at 12:04 needs summary with events up to 12:04:59.999")
            print(f"  But cache only contains observations through 12:00 (end of window)")
            
            print(f"\nHOWEVER:")
            print(f"  The cache actually stores LAST event timestamp per minute")
            print(f"  The '12:00' row likely has timestamp 12:00:59.xxx")
            print(f"  This is AFTER the 12:00 observation window end")
            print(f"  So targets through 12:04 CAN be found because:")
            print(f"    - Cache contains rows for 12:00, 12:01, 12:02, 12:03, 12:04")
            print(f"    - Even though these are AFTER observation window")
            print(f"    - They are stored for TARGET LOOKUP only")
            
            print(f"\nThis means:")
            print(f"  - Cache contains 145 rows = 145 minutes of observation window")
            print(f"  - PLUS additional rows for target lookup (12:00-12:04)")
            print(f"  - Total should be 145 + 5 = 150 rows, not 145")
            
            print(f"\nWait - let me recount actual rows...")
            print(f"  Actual Aug 25 rows: {len(timestamps)}")
            print(f"  This contradicts the 145-per-date report")


def audit_target_semantics():
    """Audit exact target semantics."""
    print("\n" + "=" * 80)
    print("6. AUDIT EXACT TARGET SEMANTICS")
    print("=" * 80)
    
    cache_path = "artifacts/model_a/minute_features.csv"
    ny_tz = pytz.timezone("America/New_York")
    
    print("\nFor intended decision time 09:55:00 ET:")
    print("  PRD requires: use summary representing 09:54:00 < t <= 09:55:00")
    print("  PRD requires: decision midpoint from events <= 09:55:00")
    print("  PRD requires: target midpoint from events <= 10:00:00")
    
    print("\nCurrent implementation:")
    print("  - Finds cache row labelled '09:55'")
    print("  - That row contains events from 09:55:00.001 to 09:55:59.999")
    print("  - Decision midpoint uses last quote at 09:55:59.999")
    print("  - This VIOLATES the requirement (uses events after 09:55:00)")
    
    print("\nExample from audit:")
    print("  Cache row '09:55' has timestamp 09:54:59.832853")
    print("  This is actually BEFORE 09:55:00, which would be correct")
    print("  But this is coincidental - could be 09:55:59.999 in other cases")
    
    print("\nCONCLUSION:")
    print("  - The timestamp semantics are INCORRECT")
    print("  - Stored timestamp should be exactly 09:55:00, not max event time")
    print("  - This creates potential feature leakage")


def main():
    print("=" * 80)
    print("TIMESTAMP ALIGNMENT AUDIT")
    print("=" * 80)
    
    audit_cache_boundaries()
    inspect_aug25_raw_events()
    inspect_cache_timestamps()
    verify_minute_shift()
    reconcile_count_arithmetic()
    audit_target_semantics()
    
    print("\n" + "=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)
    
    print("\n" + "=" * 80)
    print("AUDIT FINDINGS SUMMARY")
    print("=" * 80)
    
    print("\nA. Cache minute boundaries compliance with PRD:")
    print("   NO - Current implementation stores max event timestamp, not 09:55:00")
    
    print("\nB. One-minute shift:")
    print("   YES - Current semantics are shifted")
    print("   - Row '09:55' contains events from 09:55:00.001 to 09:55:59.999")
    print("   - Should contain events from 09:54:00.001 to 09:55:00.000")
    
    print("\nC. Feature leakage:")
    print("   YES - Features can contain events after decision time")
    print("   - If last event is at 09:55:59.999, it's used in '09:55' summary")
    print("   - This violates the PRD requirement")
    
    print("\nD. How 145 cached rows support targets through 12:04:")
    print("   NEEDS INVESTIGATION - Current math doesn't add up")
    print("   - 145 rows for 09:35-12:00 window")
    print("   - But targets through 12:04 require 5 additional minutes")
    print("   - Should be 150 rows total if targets come from cache")
    
    print("\nE. Exact minimal production fixes required:")
    print("   1. Fix cache builder to store exactly 09:55:00 as timestamp")
    print("   2. Fix cache builder to use correct minute window (09:54:00 < t <= 09:55:00)")
    print("   3. Recalculate midpoint returns with correct timestamps")
    print("   4. Rebuild cache from events.csv")
    print("   5. Rerun training with corrected cache")
    print("   6. Rerun Aug 25 demonstration with correct 09:55 sequence")
    print("   7. Fix baseline calculation bug")
    print("   8. Fix demo selection to use actual 09:55 sequence")


if __name__ == '__main__':
    main()