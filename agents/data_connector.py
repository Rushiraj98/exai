"""
Building Data Connector - Integration with Real Data Sources

Provides interfaces to connect with:
- Building Management Systems (BMS)
- SCADA systems
- Weather APIs
- Energy databases
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests


class BuildingDataConnector:
    """
    Connector for real-time building energy data from BMS/SCADA systems
    """

    def __init__(self, api_endpoint: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize the data connector

        Args:
            api_endpoint: BMS API endpoint URL
            api_key: Authentication key for BMS API
        """
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.cache = {}
        self.cache_ttl = 60  # seconds

    def get_realtime_data(self, building_id: str) -> Dict:
        """
        Fetch real-time energy data for a building

        Args:
            building_id: Building identifier

        Returns:
            Dictionary with current building metrics
        """
        # Check cache first
        cache_key = f"realtime_{building_id}"
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]['data']

        # In production, this would make an actual API call to BMS
        if self.api_endpoint:
            data = self._fetch_from_bms(building_id)
        else:
            # Mock data for demo
            data = self._generate_mock_realtime_data(building_id)

        # Update cache
        self._update_cache(cache_key, data)

        return data

    def get_historical_data(self, building_id: str, start_date: datetime,
                           end_date: datetime, resolution: str = '1H') -> pd.DataFrame:
        """
        Fetch historical energy consumption data

        Args:
            building_id: Building identifier
            start_date: Start of time range
            end_date: End of time range
            resolution: Data resolution ('1H', '15T', '1D', etc.)

        Returns:
            DataFrame with historical consumption data
        """
        if self.api_endpoint:
            df = self._fetch_historical_from_bms(building_id, start_date, end_date, resolution)
        else:
            # Generate mock historical data
            df = self._generate_mock_historical_data(building_id, start_date, end_date, resolution)

        return df

    def execute_command(self, building_id: str, command: Dict) -> Dict:
        """
        Send a control command to building BMS

        Args:
            building_id: Target building
            command: Command dictionary with control parameters

        Returns:
            Execution result
        """
        if self.api_endpoint:
            result = self._send_command_to_bms(building_id, command)
        else:
            # Mock execution
            result = {
                "success": True,
                "building_id": building_id,
                "command": command,
                "timestamp": datetime.now().isoformat(),
                "message": "Command simulated successfully (no actual BMS connection)"
            }

        return result

    def _fetch_from_bms(self, building_id: str) -> Dict:
        """Fetch data from actual BMS API"""
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = requests.get(
                f"{self.api_endpoint}/buildings/{building_id}/realtime",
                headers=headers,
                timeout=5
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching from BMS: {e}")
            return self._generate_mock_realtime_data(building_id)

    def _send_command_to_bms(self, building_id: str, command: Dict) -> Dict:
        """Send command to actual BMS"""
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = requests.post(
                f"{self.api_endpoint}/buildings/{building_id}/commands",
                headers=headers,
                json=command,
                timeout=5
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error sending command to BMS: {e}")
            return {"success": False, "error": str(e)}

    def _generate_mock_realtime_data(self, building_id: str) -> Dict:
        """Generate mock real-time data for demo"""
        return {
            "building_id": building_id,
            "timestamp": datetime.now().isoformat(),
            "energy_kw": np.random.uniform(200, 500),
            "indoor_temp_c": np.random.uniform(22, 26),
            "outdoor_temp_c": np.random.uniform(35, 48),
            "occupancy": np.random.randint(100, 800),
            "hvac_status": "running",
            "chiller_load_percent": np.random.uniform(40, 95)
        }

    def _generate_mock_historical_data(self, building_id: str, start_date: datetime,
                                      end_date: datetime, resolution: str) -> pd.DataFrame:
        """Generate mock historical data"""
        date_range = pd.date_range(start=start_date, end=end_date, freq=resolution)

        # Generate synthetic consumption with daily patterns
        hours = np.array([d.hour for d in date_range])
        base_load = 300
        daily_variation = 100 * np.sin(2 * np.pi * hours / 24)
        noise = np.random.normal(0, 20, len(date_range))
        consumption = base_load + daily_variation + noise

        df = pd.DataFrame({
            'timestamp': date_range,
            'building_id': building_id,
            'energy_kw': consumption,
            'indoor_temp_c': 24 + np.random.normal(0, 1, len(date_range)),
            'outdoor_temp_c': 42 + 6 * np.sin(2 * np.pi * hours / 24) + np.random.normal(0, 2, len(date_range))
        })

        return df

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is still valid"""
        if cache_key not in self.cache:
            return False

        cache_age = (datetime.now() - self.cache[cache_key]['timestamp']).total_seconds()
        return cache_age < self.cache_ttl

    def _update_cache(self, cache_key: str, data: Dict):
        """Update cache with new data"""
        self.cache[cache_key] = {
            'data': data,
            'timestamp': datetime.now()
        }


class WeatherDataConnector:
    """
    Connector for weather data APIs
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://api.openweathermap.org/data/2.5"  # Example

    def get_current_weather(self, location: str = "Dubai") -> Dict:
        """Get current weather conditions"""
        if self.api_key:
            return self._fetch_real_weather(location)
        else:
            return self._generate_mock_weather(location)

    def get_forecast(self, location: str = "Dubai", hours: int = 24) -> pd.DataFrame:
        """Get weather forecast"""
        if self.api_key:
            return self._fetch_real_forecast(location, hours)
        else:
            return self._generate_mock_forecast(location, hours)

    def _fetch_real_weather(self, location: str) -> Dict:
        """Fetch from real weather API"""
        try:
            params = {
                "q": location,
                "appid": self.api_key,
                "units": "metric"
            }
            response = requests.get(f"{self.base_url}/weather", params=params, timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching weather: {e}")
            return self._generate_mock_weather(location)

    def _generate_mock_weather(self, location: str) -> Dict:
        """Generate mock weather data"""
        hour = datetime.now().hour
        base_temp = 42 if 12 <= hour <= 16 else 38

        return {
            "location": location,
            "temperature_c": base_temp + np.random.uniform(-3, 3),
            "humidity_percent": np.random.randint(45, 75),
            "solar_radiation_wm2": np.random.randint(600, 1100) if 8 <= hour <= 18 else 0,
            "wind_speed_kmh": np.random.randint(10, 25),
            "timestamp": datetime.now().isoformat()
        }

    def _generate_mock_forecast(self, location: str, hours: int) -> pd.DataFrame:
        """Generate mock forecast"""
        timestamps = pd.date_range(start=datetime.now(), periods=hours, freq='H')
        hour_values = np.array([t.hour for t in timestamps])

        # Temperature follows daily cycle
        temps = 40 + 8 * np.sin(2 * np.pi * (hour_values - 14) / 24)

        df = pd.DataFrame({
            'timestamp': timestamps,
            'temperature_c': temps + np.random.normal(0, 2, len(timestamps)),
            'humidity_percent': 60 + np.random.normal(0, 10, len(timestamps)),
            'solar_radiation_wm2': np.where((hour_values >= 6) & (hour_values <= 18),
                                           800 * np.sin(np.pi * (hour_values - 6) / 12), 0)
        })

        return df
