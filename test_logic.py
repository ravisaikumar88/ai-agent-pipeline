import pytest
from unittest.mock import patch, MagicMock

# Import the functions/components we want to test
from tools import fetch_weather_data
from agent_core import router_chain, RouterDecision, retriever, synthesis_chain

# --- 1. Test External Tool (Weather API) ---

@pytest.fixture
def mock_geopy():
    """Mocks the geopy.geocoders.Nominatim geolocator."""
    # Create a mock location object that has latitude and longitude
    mock_location = MagicMock()
    mock_location.latitude = 51.5074
    mock_location.longitude = -0.1278
    
    # Create a mock geolocator that returns the mock location
    mock_geolocator = MagicMock()
    mock_geolocator.geocode.return_value = mock_location
    
    # Use 'patch' to replace the real Nominatim with our mock
    with patch('tools.Nominatim', return_value=mock_geolocator) as mock:
        yield mock

@pytest.fixture
def mock_requests_get():
    """Mocks the requests.get function to return a sample OpenWeatherMap JSON response."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "cod": 200,
        "weather": [{"main": "Clear", "description": "clear sky"}],
        "main": {"temp": 15.0, "feels_like": 14.0, "humidity": 60}
    }
    # This context manager will replace 'requests.get' with our 'mock_response'
    with patch('tools.requests.get', return_value=mock_response) as mock:
        yield mock

def test_fetch_weather_data_success(mock_geopy, mock_requests_get):
    """
    Tests the fetch_weather_data function with mocked external APIs.
    """
    city = "London, UK"
    result = fetch_weather_data(city)
    
    # Check that our mocks were called
    mock_geopy.return_value.geocode.assert_called_with(city)
    mock_requests_get.assert_called()
    
    # Check that the output is formatted correctly based on our mock data
    assert "Current weather for London, UK" in result
    assert "Condition: Clear (clear sky)" in result
    assert "Temperature: 15.0°C" in result

def test_fetch_weather_data_geocoding_fail(mock_geopy):
    """Tests what happens if the city cannot be found."""
    # Configure the mock geolocator to return None (city not found)
    mock_geopy.return_value.geocode.return_value = None
    
    city = "FakeCity"
    result = fetch_weather_data(city)
    
    assert "Could not find geographic coordinates" in result

# --- 2. Test Router Logic ---

def test_router_decision_weather():
    """Tests if the router correctly identifies a weather query."""
    query = "What is the forecast for Paris?"
    result = router_chain.invoke({"query": query})
    
    assert isinstance(result, RouterDecision)
    assert result.next_action == "WEATHER_API"
    assert "Paris" in result.city_name

def test_router_decision_rag():
    """Tests if the router correctly identifies a RAG query."""
    query = "What does the PDF say about agentic workflows?"
    result = router_chain.invoke({"query": query})
    
    assert isinstance(result, RouterDecision)
    assert result.next_action == "RAG_LOOKUP"

# --- 3. Test Synthesis Logic ---

def test_synthesis_chain():
    """Tests the final response synthesizer with mock context."""
    query = "What is the weather?"
    context = "The weather is 20°C and sunny."
    
    result = synthesis_chain.invoke({"query": query, "context": context})
    
    assert isinstance(result, str)
    assert "20°C" in result or "sunny" in result