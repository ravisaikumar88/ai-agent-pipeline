import pytest
from unittest.mock import patch, MagicMock

# Import the functions/components we want to test
from tools import fetch_weather_data
from agent_core import router_chain, RouterDecision, retriever, synthesis_chain

@pytest.fixture
def mock_geopy():
    mock_location = MagicMock()
    mock_location.latitude = 51.5074
    mock_location.longitude = -0.1278
    
    mock_geolocator = MagicMock()
    mock_geolocator.geocode.return_value = mock_location
    
    with patch('tools.Nominatim', return_value=mock_geolocator) as mock:
        yield mock

@pytest.fixture
def mock_requests_get():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "cod": 200,
        "weather": [{"main": "Clear", "description": "clear sky"}],
        "main": {"temp": 15.0, "feels_like": 14.0, "humidity": 60}
    }
    with patch('tools.requests.get', return_value=mock_response) as mock:
        yield mock

def test_fetch_weather_data_success(mock_geopy, mock_requests_get):
    city = "London, UK"
    result = fetch_weather_data(city)
    
    mock_geopy.return_value.geocode.assert_called_with(city)
    mock_requests_get.assert_called()
    assert "Current weather for London, UK" in result
    assert "Condition: Clear (clear sky)" in result
    assert "Temperature: 15.0°C" in result

def test_fetch_weather_data_geocoding_fail(mock_geopy):
    mock_geopy.return_value.geocode.return_value = None
    
    city = "FakeCity"
    result = fetch_weather_data(city)
    
    assert "Could not find geographic coordinates" in result

def test_router_decision_weather():
    query = "What is the forecast for Paris?"
    result = router_chain.invoke({"query": query})
    
    assert isinstance(result, RouterDecision)
    assert result.next_action == "WEATHER_API"
    assert "Paris" in result.city_name

def test_router_decision_rag():
    query = "What does the PDF say about agentic workflows?"
    result = router_chain.invoke({"query": query})
    
    assert isinstance(result, RouterDecision)
    assert result.next_action == "RAG_LOOKUP"

def test_synthesis_chain():
    query = "What is the weather?"
    context = "The weather is 20°C and sunny."
    
    result = synthesis_chain.invoke({"query": query, "context": context})
    
    assert isinstance(result, str)
    assert "20°C" in result or "sunny" in result