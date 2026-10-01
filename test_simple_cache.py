"""Test simple cache builder with pandas."""

import sys
sys.path.insert(0, 'src')

from pathlib import Path
from model_a.simple_cache import SimpleCacheBuilder

def main():
    print("Testing simple cache builder with pandas...")
    
    events_csv = "artifacts/spy_iex_august/market_data/events.csv"
    cache_output = "artifacts/model_a/test_features.csv"
    
    cache_builder = SimpleCacheBuilder(events_csv)
    cache_builder.required_dates = ['2026-08-18']  # Only process first day for testing
    
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