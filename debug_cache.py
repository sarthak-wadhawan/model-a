"""Debug cache builder to see what's happening with timestamps."""

import sys
sys.path.insert(0, 'src')

import pandas as pd
from datetime import datetime, timezone
import pytz

from model_a.simple_cache import SimpleCacheBuilder

def main():
    print("Debugging cache builder...")
    
    events_csv = "artifacts/spy_iex_august/market_data/events.csv"
    
    cache_builder = SimpleCacheBuilder(events_csv)
    
    # Load a small sample
    print("Loading sample of events...")
    df = pd.read_csv(events_csv, nrows=1000)
    
    # Parse timestamps
    df['timestamp_dt'] = df['timestamp'].apply(cache_builder._parse_timestamp)
    
    # Check timezone conversion
    ny_tz = pytz.timezone("America/New_York")
    df['timestamp_et'] = df['timestamp_dt'].apply(lambda x: x.astimezone(ny_tz))
    
    print("\nFirst 10 events:")
    for i, row in df.head(10).iterrows():
        print(f"  {row['timestamp']} -> {row['timestamp_et'].strftime('%Y-%m-%d %H:%M:%S %Z')} ET")
        print(f"    Type: {row['event_type']}, In window: {cache_builder._is_in_observation_window(row['timestamp_dt'])}")
    
    # Check Aug 18 events specifically
    print("\nLoading Aug 18 events...")
    chunksize = 100000
    aug18_events = []
    
    for chunk in pd.read_csv(events_csv, chunksize=chunksize):
        chunk['timestamp_parsed'] = chunk['timestamp'].apply(
            lambda x: cache_builder._parse_timestamp(x).strftime("%Y-%m-%d"))
        chunk_filtered = chunk[chunk['timestamp_parsed'] == '2026-08-18']
        if not chunk_filtered.empty:
            aug18_events.append(chunk_filtered)
            if len(aug18_events) >= 1:  # Just get first chunk
                break
    
    if aug18_events:
        aug18_df = pd.concat(aug18_events, ignore_index=True)
        aug18_df['timestamp_dt'] = aug18_df['timestamp'].apply(cache_builder._parse_timestamp)
        aug18_df['timestamp_et'] = aug18_df['timestamp_dt'].apply(lambda x: x.astimezone(ny_tz))
        aug18_df['in_window'] = aug18_df['timestamp_dt'].apply(cache_builder._is_in_observation_window)
        
        print(f"Aug 18 events in sample: {len(aug18_df)}")
        print(f"Events in observation window: {aug18_df['in_window'].sum()}")
        
        print("\nFirst 10 Aug 18 events:")
        for i, row in aug18_df.head(10).iterrows():
            print(f"  {row['timestamp_et'].strftime('%Y-%m-%d %H:%M:%S')} ET")
            print(f"    Type: {row['event_type']}, In window: {row['in_window']}")

if __name__ == '__main__':
    main()