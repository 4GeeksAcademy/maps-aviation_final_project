import pandas as pd
from src.app import check_route, hhmm_to_minutes

def test_valid_route():
    df = pd.DataFrame({'origin': ['JFK'], 'destination': ['LAX']})
    route_frequency = {'JFK_LAX': 10}
    
    result_df = check_route(df.copy(), route_frequency)
    
    # Check that it's not None (meaning route was valid)
    assert result_df is not None
    # Check that route_encoded column exists
    assert 'route_encoded' in result_df.columns
    # Check the encoded value is 10
    assert result_df['route_encoded'].iloc[0] == 10

def test_invalid_route():
    df = pd.DataFrame({'origin': ['BIS'], 'destination': ['BIS']})
    route_frequency = {'LGA_ORF': 10}
    
    result = check_route(df.copy(), route_frequency)
    
    # Should return None for invalid route
    assert result is None

def test_hhmm_to_minutes():
    time = "02:30"

    result = hhmm_to_minutes(time)

    assert result is 150
