import requests
import os
from dotenv import load_dotenv
from geopy.geocoders import Nominatim
import sys

# --- 1. Load Environment ---
load_dotenv()
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

if not OPENWEATHER_API_KEY:
    print("ERROR: OPENWEATHER_API_KEY not found in .env file.")
    sys.exit()

# --- 2. Geocoding Function (Copied from tools.py) ---
def get_lat_lon(city_name: str) -> tuple[float, float] | None:
    """Converts a city name into latitude and longitude coordinates."""
    print(f"Attempting to geocode '{city_name}'...")
    try:
        geolocator = Nominatim(user_agent="ai_agent_pipeline")
        location = geolocator.geocode(city_name)
        if location:
            print(f"Success: Found {location.address}")
            print(f"Coords: ({location.latitude}, {location.longitude})")
            return (location.latitude, location.longitude)
        else:
            print("Error: Geocoding failed. Location not found.")
            return None
    except Exception as e:
        print(f"Geocoding error for {city_name}: {e}")
        return None

# --- 3. Weather Fetch Function ---
def test_city_weather(city: str):
    """
    Tests the full geocoding and weather fetch process for a single city.
    """
    print(f"\n--- TESTING: {city} ---")
    
    # Step 1: Geocode
    coords = get_lat_lon(city)
    if not coords:
        return

    lat, lon = coords
    
    # Step 2: Fetch Weather
    BASE_URL = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "lat": lat,
        "lon": lon,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric"
    }

    print(f"Calling OpenWeatherMap API for ({lat}, {lon})...")
    try:
        # Using a 15-second timeout, just like in your tools.py
        response = requests.get(BASE_URL, params=params, timeout=15)
        response.raise_for_status()  # Check for HTTP errors

        # If successful
        data = response.json()
        print(f"API CALL SUCCEEDED for {city}.")
        print(f"Data: {data['weather'][0]['description']}, Temp: {data['main']['temp']}°C")

    except requests.exceptions.Timeout:
        print(f"API CALL FAILED for {city}: The request timed out.")
    except requests.exceptions.RequestException as e:
        print(f"API CALL FAILED for {city}: {e}")
    except Exception as e:
        print(f"An unknown error occurred for {city}: {e}")


# --- 4. Run Tests ---
if __name__ == "__main__":
    cities_to_test = ["Bangalore", "Chennai", "New York"]
    
    for city in cities_to_test:
        test_city_weather(city)
        print("-" * 20)