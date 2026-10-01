"""Debug the corrected cache builder."""

import sys
sys.path.insert(0, 'src')

import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pytz

from model_a.simple_cache_fixed import SimpleCacheBuilder

def main():
    print("Debugging corrected cache builder...")
    
    events_csv = "artifacts/spy_iex_august/market_data/events.csv"
    
    cache_builder = SimpleCacheBuilder(events_csv)
    cache_builder.required_dates = ['2026-08-25']  # Just one date for debugging
    
    # Load events for Aug 25
    print("Loading Aug 25 events...")
    chunksize = 100000
    all_events = []
    
    for chunk in pd.read_csv(events_csv, chunksize=chunksize):
        chunk['date'] = chunk['timestamp'].str[:10]
        chunk = chunk[chunk['date'].isin(['2026-08-25'])]
        
        if not chunk.empty:
            all_events.append(chunk)
    
    if all_events:
        full_df = pd.concat(all_events, ignore_index=True)
        print(f"Loaded {len(full_df)} events")
        
        # Parse timestamps
        full_df['timestamp_dt'] = full_df['timestamp'].apply(cache_builder._parse_timestamp)
        
        # Check extended window
        full_df['in_extended_window'] = full_df['timestamp_dt'].apply(cache_builder._is_in_extended_window)
        filtered_df = full_df[full_df['in_extended_window']]
        
        print(f"Events in extended window: {len(filtered_df)}")
        
        # Show first few events
        print("\nFirst 5 events in extended window:")
        for i, row in filtered_df.head(5).iterrows():
            dt = row['timestamp_dt']
            et = dt.astimezone(cache_builder.ny_tz)
            print(f"  {et.strftime('%H:%M:%S.%f')[:-3]} ET: {row['event_type']}")
        
        # Group by minute
        filtered_df['minute_key'] = filtered_df['timestamp_dt'].apply(
            lambda x: x.astimezone(cache_builder.ny_tz).strftime("%Y-%m-%d %H:%M"))
        
        print(f"\nUnique minutes: {filtered_df['minute_key'].nunique()}")
        
        # Try processing one minute
        minute_key = "2026-08-25 09:35"
        if minute_key in filtered_df['minute_key'].values:
            minute_df = filtered_df[filtered_df['minute_key'] == minute_key].copy()
            print(f"\nProcessing minute {minute_key}:")
            print(f"  Events: {len(minute_df)}")
            
            # Show events in this minute
            print(f"  First event: {minute_df.iloc[0]['timestamp_dt'].astimezone(cache_builder.ny_tz).strftime('%H:%M:%S.%f')[:-3]}")
            print(f"  Last event: {minute_df.iloc[-1]['timestamp_dt'].astimezone(cache_builder.ny_tz).strftime('%H:%M:%S.%f')[:-3]}")
            
            # Apply window filter
            et_dt = cache_builder.ny_tz.localize(datetime.strptime(minute_key, "%Y-%m-%d %H:%M"))
            boundary_time = et_dt.astimezone(timezone.utc)
            window_start = boundary_time - timedelta(minutes=1)
            
            print(f"  Boundary time (UTC): {boundary_time.isoformat()}")
            print(f"  Window start (UTC): {window_start.isoformat()}")
            
            minute_df_filtered = minute_df[
                (minute_df['timestamp_dt'] > window_start) &
                (minute_df['timestamp_dt'] <= boundary_time)
            ]
            
            print(f"  Events after window filter: {len(minute_df_filtered)}")
            
            if len(minute_df_filtered) > 0:
                print(f"  First filtered event: {minute_df_filtered.iloc[0]['timestamp_dt'].astimezone(cache_builder.ny_tz).strftime('%H:%M:%S.%f')[:-3]}")
                print(f"  Last filtered event: {minute_df_filtered.iloc[-1]['timestamp_dt'].astimezone(cache_builder.ny_tz).strftime('%H:%M:%S.%f')[:-3]}")
            else:
                print("  NO EVENTS PASSED WINDOW FILTER - THIS IS THE PROBLEM")

if __name__ == '__main__':
    main()