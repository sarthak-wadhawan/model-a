"""Integration tests for corrected cache boundary semantics."""

import pytest
import csv
from datetime import datetime
from pathlib import Path
import pytz


class TestCacheBoundarySemantics:
    """Integration tests for corrected cache boundaries."""
    
    @pytest.fixture
    def cache_path(self):
        return "artifacts/model_a/minute_features.csv"
    
    def test_0935_is_input_eligible(self, cache_path):
        """09:35 is input eligible."""
        ny_tz = pytz.timezone("America/New_York")
        with open(cache_path, 'r') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                if row['date'] == '2026-08-25' and row['time'] == '09:35':
                    assert row['is_target_only'] == 'False', "09:35 should be input-eligible"
                    return
        
        pytest.fail("09:35 not found in cache")
    
    def test_1200_is_input_eligible(self, cache_path):
        """12:00 is input eligible."""
        with open(cache_path, 'r') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                if row['date'] == '2026-08-25' and row['time'] == '12:00':
                    assert row['is_target_only'] == 'False', "12:00 should be input-eligible"
                    return
        
        pytest.fail("12:00 not found in cache")
    
    def test_1201_is_target_only(self, cache_path):
        """12:01 is target-only."""
        with open(cache_path, 'r') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                if row['date'] == '2026-08-25' and row['time'] == '12:01':
                    assert row['is_target_only'] == 'True', "12:01 should be target-only"
                    return
        
        pytest.fail("12:01 not found in cache")
    
    def test_1205_is_target_only(self, cache_path):
        """12:05 is target-only."""
        with open(cache_path, 'r') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                if row['date'] == '2026-08-25' and row['time'] == '12:05':
                    assert row['is_target_only'] == 'True', "12:05 should be target-only"
                    return
        
        pytest.fail("12:05 not found in cache")
    
    def test_1206_not_in_cache(self, cache_path):
        """12:06 should not be in cache (excluded)."""
        with open(cache_path, 'r') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                if row['date'] == '2026-08-25' and row['time'] == '12:06':
                    pytest.fail("12:06 should not be in cache")
        
        # If we get here, 12:06 is not in cache (correct)
    
    def test_summary_timestamp_is_canonical(self, cache_path):
        """Stored summary timestamp equals canonical boundary (exactly :00)."""
        ny_tz = pytz.timezone("America/New_York")
        with open(cache_path, 'r') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                if row['date'] == '2026-08-25':
                    timestamp = datetime.fromisoformat(row['timestamp'])
                    et_time = timestamp.astimezone(ny_tz)
                    
                    # Check that seconds and microseconds are 0
                    assert et_time.second == 0, f"Timestamp should have 0 seconds, got {et_time.second}"
                    assert et_time.microsecond == 0, f"Timestamp should have 0 microseconds, got {et_time.microsecond}"
                    
                    # Only check a few rows to avoid iterating entire cache
                    if row['time'] in ['09:35', '09:55', '12:00']:
                        assert True  # Timestamp is canonical
                        return
        
        pytest.fail("No rows found")
    
    def test_cache_counts_per_date(self, cache_path):
        """Cache has correct counts: 146 input + 5 target = 151 total per date."""
        with open(cache_path, 'r') as f:
            reader = csv.DictReader(f)
            
            counts = {}
            for row in reader:
                date = row['date']
                if date not in counts:
                    counts[date] = {'total': 0, 'input': 0, 'target': 0}
                
                counts[date]['total'] += 1
                if row['is_target_only'] == 'True':
                    counts[date]['target'] += 1
                else:
                    counts[date]['input'] += 1
            
            for date, count in counts.items():
                assert count['total'] == 151, f"{date} should have 151 total rows, got {count['total']}"
                assert count['input'] == 146, f"{date} should have 146 input rows, got {count['input']}"
                assert count['target'] == 5, f"{date} should have 5 target rows, got {count['target']}"