"""Test cache builder with small subset of data."""

import sys
sys.path.insert(0, 'src')

from pathlib import Path
from model_a.cache_builder import IncrementalCacheBuilder

def main():
    print("Testing cache builder with small data subset...")
    
    events_csv = "artifacts/spy_iex_august/market_data/events.csv"
    cache_output = "artifacts/model_a/test_features.csv"
    
    # Create test version that only processes first day
    cache_builder = IncrementalCacheBuilder(events_csv)
    cache_builder.required_dates = ['2026-08-18']  # Only process first day for testing
    
    # Debug: print first few events
    print("\nDebug: Examining first few events...")
    from model_a.data_loader import EventStream
    event_stream = EventStream(Path(events_csv))
    count = 0
    for event in event_stream.events(start_date='2026-08-18', end_date='2026-08-18'):
        if count < 10:
            et_time = event.timestamp.astimezone(cache_builder.ny_tz)
            print(f"  Event {count}: {event.event_type} at {et_time.strftime('%Y-%m-%d %H:%M:%S')} ET")
            count += 1
        else:
            break
    
    output_path = Path(cache_output)
    
    stats = cache_builder.build_cache(output_path)
    
    print(f"\n=== Test Cache Build Complete ===")
    print(f"Total events processed: {stats['total_events_processed']}")
    print(f"Events by date:")
    for date, count in stats['events_by_date'].items():
        print(f"  {date}: {count}")
    print(f"Valid minute summaries by date:")
    for date, count in stats['valid_summaries_by_date'].items():
        print(f"  {date}: {count}")
    print(f"Total valid minute summaries: {stats['total_valid_summaries']}")
    print(f"Cache saved to: {cache_output}")

if __name__ == '__main__':
    main()