"""Inspect events.csv for actual quote timestamps around decision and target boundaries."""

import sys
sys.path.insert(0, 'src')

import pandas as pd
from datetime import datetime, timezone
import pytz

def main():
    print("Inspecting events.csv for Aug 25 09:55 and 10:00 quote timestamps...")
    
    events_csv = "artifacts/spy_iex_august/market_data/events.csv"
    target_date = "2026-08-25"
    ny_tz = pytz.timezone("America/New_York")
    
    # Read events for Aug 25 only
    chunksize = 100000
    all_events = []
    
    for chunk in pd.read_csv(events_csv, chunksize=chunksize):
        chunk['date'] = chunk['timestamp'].str[:10]
        chunk = chunk[chunk['date'] == target_date]
        if not chunk.empty:
            all_events.append(chunk)
    
    if all_events:
        full_df = pd.concat(all_events, ignore_index=True)
        
        # Parse timestamps
        def parse_ts(ts_str):
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
                    return None
            return dt.replace(tzinfo=timezone.utc)
        
        full_df['timestamp_dt'] = full_df['timestamp'].apply(parse_ts)
        
        # Filter to quotes only
        quotes = full_df[full_df['event_type'] == 'quote'].copy()
        quotes = quotes[
            (quotes['bid_price_1'].notna()) & 
            (quotes['ask_price_1'].notna()) &
            (quotes['bid_size_1'].notna()) &
            (quotes['ask_size_1'].notna())
        ]
        
        # Decision window: (09:54:00, 09:55:00] ET
        decision_start = ny_tz.localize(datetime(2026, 8, 25, 9, 54, 0, 0)).astimezone(timezone.utc)
        decision_end = ny_tz.localize(datetime(2026, 8, 25, 9, 55, 0, 0)).astimezone(timezone.utc)
        
        decision_quotes = quotes[
            (quotes['timestamp_dt'] > decision_start) &
            (quotes['timestamp_dt'] <= decision_end)
        ].sort_values('timestamp_dt')
        
        print(f"\nDecision window (09:54:00, 09:55:00] ET:")
        print(f"  Quotes found: {len(decision_quotes)}")
        
        if not decision_quotes.empty:
            last_decision_quote = decision_quotes.iloc[-1]
            decision_midpoint = (last_decision_quote['bid_price_1'] + last_decision_quote['ask_price_1']) / 2.0
            print(f"  Last quote timestamp (UTC): {last_decision_quote['timestamp_dt']}")
            print(f"  Last quote timestamp (ET): {last_decision_quote['timestamp_dt'].astimezone(ny_tz).strftime('%H:%M:%S.%f')}")
            print(f"  Bid: {last_decision_quote['bid_price_1']}")
            print(f"  Ask: {last_decision_quote['ask_price_1']}")
            print(f"  Calculated midpoint: {decision_midpoint:.4f}")
        else:
            print("  ERROR: No quotes found in decision window")
        
        # Target window: (09:59:00, 10:00:00] ET
        target_start = ny_tz.localize(datetime(2026, 8, 25, 9, 59, 0, 0)).astimezone(timezone.utc)
        target_end = ny_tz.localize(datetime(2026, 8, 25, 10, 0, 0, 0)).astimezone(timezone.utc)
        
        target_quotes = quotes[
            (quotes['timestamp_dt'] > target_start) &
            (quotes['timestamp_dt'] <= target_end)
        ].sort_values('timestamp_dt')
        
        print(f"\nTarget window (09:59:00, 10:00:00] ET:")
        print(f"  Quotes found: {len(target_quotes)}")
        
        if not target_quotes.empty:
            last_target_quote = target_quotes.iloc[-1]
            target_midpoint = (last_target_quote['bid_price_1'] + last_target_quote['ask_price_1']) / 2.0
            print(f"  Last quote timestamp (UTC): {last_target_quote['timestamp_dt']}")
            print(f"  Last quote timestamp (ET): {last_target_quote['timestamp_dt'].astimezone(ny_tz).strftime('%H:%M:%S.%f')}")
            print(f"  Bid: {last_target_quote['bid_price_1']}")
            print(f"  Ask: {last_target_quote['ask_price_1']}")
            print(f"  Calculated midpoint: {target_midpoint:.4f}")
        else:
            print("  ERROR: No quotes found in target window")
        
        print(f"\nVerification:")
        if not decision_quotes.empty and not target_quotes.empty:
            print(f"  Decision midpoint: {decision_midpoint:.4f} (expected: 766.0150)")
            print(f"  Target midpoint: {target_midpoint:.4f} (expected: 765.9550)")
            print(f"  Match: {abs(decision_midpoint - 766.0150) < 0.0001 and abs(target_midpoint - 765.9550) < 0.0001}")

if __name__ == '__main__':
    main()
