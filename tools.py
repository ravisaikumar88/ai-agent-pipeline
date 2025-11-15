import requests
from geopy.geocoders import Nominatim
from config import OPENWEATHER_API_KEY

def get_lat_lon(city_name: str) -> tuple[float, float] | None:
    """
    Converts a city name into latitude and longitude coordinates.
    Uses Nominatim for geocoding.
    """
    geolocator = Nominatim(user_agent="ai_agent_weather_app")
    try:
        # Use simple try-except structure for geocoding errors
        location = geolocator.geocode(city_name)
        if location:
            return location.latitude, location.longitude
        return None
    except Exception as e:
        print(f"Geocoding error for {city_name}: {e}")
        return None

def fetch_weather_data(city: str) -> str:
    """
    Fetches real-time weather data from OpenWeatherMap API.

    Args:
        city: The name of the city to get weather for (e.g., "London, UK").

    Returns:
        A formatted string summary of the current weather, or an error message.
    """
    lat_lon = get_lat_lon(city)
    if not lat_lon:
        return f"Could not find geographic coordinates for the city: {city}. Please check the spelling."

    lat, lon = lat_lon
    
    BASE_URL = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        'lat': lat,
        'lon': lon,
        'appid': OPENWEATHER_API_KEY,
        'units': 'metric' # Use Celsius
    }

    try:
        response = requests.get(BASE_URL, params=params, timeout=5)
        response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
        data = response.json()

        if data.get("cod") != 200:
            return f"OpenWeatherMap API error: {data.get('message', 'Unknown error')}"

        # Extract and format relevant information
        main_weather = data['weather'][0]['main']
        description = data['weather'][0]['description']
        temp = data['main']['temp']
        feels_like = data['main']['feels_like']
        humidity = data['main']['humidity']

        weather_summary = (
            f"Current weather for {city} (Lat: {lat:.2f}, Lon: {lon:.2f}):\n"
            f"- Condition: {main_weather} ({description})\n"
            f"- Temperature: {temp}°C (Feels like: {feels_like}°C)\n"
            f"- Humidity: {humidity}%\n"
            f"This data is real-time from the OpenWeatherMap API."
        )
        return weather_summary

    except requests.exceptions.RequestException as e:
        return f"Failed to connect to the weather API: {e}"
    except Exception as e:
        return f"An unexpected error occurred while fetching weather data: {e}"


if __name__ == "__main__":
    # Test cases for the tool
    print("--- Testing Weather Tool ---")
    weather_in_london = fetch_weather_data("London, UK")
    print(f"\nQuery: London, UK\nResult:\n{weather_in_london}\n")

    weather_in_tokyo = fetch_weather_data("Tokyo, Japan")
    print(f"Query: Tokyo, Japan\nResult:\n{weather_in_tokyo}\n")
    
    bad_city = fetch_weather_data("NoCityExistsHereABC")
    print(f"Query: NoCityExistsHereABC\nResult:\n{bad_city}")