"""
Energy Management Tools for Agent System

Provides shared tools that agents can use to interact with:
- Building energy data
- Spatial anomaly detection
- SHAP explainability
- Vector knowledge base (Qdrant)
- Building control systems
- Weather forecasting
"""

from langchain.tools import tool
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from datetime import datetime
import random


class EnergyTools:
    """Shared tools for all agents in the energy management system"""

    def __init__(self):
        """Initialize tools with connections to data sources"""
        self.building_data = pd.DataFrame()  # Will be loaded from actual sources
        # self.qdrant = QdrantClient(host="localhost", port=6333)  # Uncomment when Qdrant is running

    @staticmethod
    @tool
    def query_building_energy(building_id: str, time_range: str = "24h") -> Dict:
        """
        Query real-time energy consumption for a specific building.

        Args:
            building_id: Unique identifier for the building
            time_range: Time range for query (e.g., "24h", "1w")

        Returns:
            Dictionary containing current energy metrics
        """
        # Simulate real building data - replace with actual BMS/SCADA connection
        base_load = random.uniform(100, 500)

        return {
            "building_id": building_id,
            "current_load_kw": round(base_load, 2),
            "predicted_load_kw": round(base_load * random.uniform(1.0, 1.2), 2),
            "temperature_c": round(random.uniform(22, 26), 1),
            "outdoor_temp_c": round(random.uniform(35, 48), 1),
            "occupancy_count": random.randint(50, 500),
            "hvac_status": random.choice(["normal", "high_load", "optimal"]),
            "timestamp": datetime.now().isoformat(),
            "daily_consumption_kwh": round(base_load * 24 * random.uniform(0.8, 1.2), 2)
        }

    @staticmethod
    @tool
    def calculate_spatial_anomaly_score(building_id: str) -> Dict:
        """
        Calculate how anomalous a building's consumption is compared to spatial neighbors.

        Args:
            building_id: Building to analyze

        Returns:
            Anomaly score and comparison data
        """
        # Simulate spatial comparison - replace with actual geospatial analysis
        building_consumption = random.uniform(150, 400)
        neighbor_avg = random.uniform(200, 300)
        neighbor_std = random.uniform(20, 50)

        # Calculate z-score based anomaly
        z_score = abs(building_consumption - neighbor_avg) / neighbor_std
        anomaly_score = min(z_score / 3.0, 1.0)  # Normalize to 0-1

        return {
            "building_id": building_id,
            "anomaly_score": round(anomaly_score, 3),
            "building_consumption_kw": round(building_consumption, 2),
            "neighbor_avg_kw": round(neighbor_avg, 2),
            "neighbor_std_kw": round(neighbor_std, 2),
            "z_score": round(z_score, 2),
            "severity": "critical" if anomaly_score > 0.7 else "high" if anomaly_score > 0.5 else "medium" if anomaly_score > 0.3 else "low",
            "num_neighbors_analyzed": random.randint(5, 15)
        }

    @staticmethod
    @tool
    def run_shap_analysis(building_id: str) -> Dict:
        """
        Run SHAP explainability analysis to identify top factors driving energy consumption.

        Args:
            building_id: Building to analyze

        Returns:
            SHAP values and feature importance rankings
        """
        # Simulate SHAP analysis - replace with actual ML model + SHAP
        features = {
            "outdoor_temperature": round(random.uniform(0.25, 0.40), 3),
            "solar_radiation": round(random.uniform(0.15, 0.25), 3),
            "occupancy_level": round(random.uniform(0.10, 0.20), 3),
            "time_of_day": round(random.uniform(0.08, 0.15), 3),
            "hvac_setpoint": round(random.uniform(0.05, 0.12), 3),
            "building_age": round(random.uniform(0.03, 0.08), 3),
            "window_area": round(random.uniform(0.02, 0.06), 3)
        }

        # Normalize to sum to 1.0
        total = sum(features.values())
        features = {k: round(v/total, 3) for k, v in features.items()}

        # Sort by importance
        sorted_features = dict(sorted(features.items(), key=lambda x: x[1], reverse=True))
        top_3 = list(sorted_features.keys())[:3]

        return {
            "building_id": building_id,
            "top_factors": sorted_features,
            "top_3_features": top_3,
            "explanation": f"High consumption primarily driven by {top_3[0].replace('_', ' ')} ({sorted_features[top_3[0]]*100:.1f}%), {top_3[1].replace('_', ' ')} ({sorted_features[top_3[1]]*100:.1f}%), and {top_3[2].replace('_', ' ')} ({sorted_features[top_3[2]]*100:.1f}%)",
            "model_confidence": round(random.uniform(0.75, 0.95), 2)
        }

    @staticmethod
    @tool
    def query_vector_knowledge(query: str, collection: str = "energy_patterns") -> Dict:
        """
        Query Qdrant vector database for similar historical energy patterns.

        Args:
            query: Natural language query about energy patterns
            collection: Qdrant collection name

        Returns:
            Similar patterns and insights from historical data
        """
        # Simulate vector search - replace with actual Qdrant queries
        similar_patterns = [
            {
                "pattern": "High afternoon peak in glass-facade tower",
                "similarity": 0.89,
                "building": "Marina Tower 2",
                "date": "2024-03-15",
                "resolution": "Implemented automated blind control"
            },
            {
                "pattern": "Excessive cooling load during low occupancy",
                "similarity": 0.82,
                "building": "Business Bay Complex",
                "date": "2024-02-28",
                "resolution": "Adjusted HVAC schedule based on occupancy sensors"
            },
            {
                "pattern": "Pre-cooling inefficiency in hot weather",
                "similarity": 0.76,
                "building": "JLT Tower 5",
                "date": "2024-04-02",
                "resolution": "Optimized pre-cooling start time reduced consumption by 18%"
            }
        ]

        return {
            "query": query,
            "collection": collection,
            "num_results": len(similar_patterns),
            "similar_patterns": similar_patterns,
            "insights": [
                "Similar pattern found in Marina Tower 2 last Tuesday",
                "This matches typical afternoon peak in glass-facade buildings",
                "Historical data suggests pre-cooling optimization reduces this by 18%"
            ]
        }

    @staticmethod
    @tool
    def simulate_intervention(building_id: str, action: str) -> Dict:
        """
        Simulate the impact of a proposed intervention before execution.

        Args:
            building_id: Target building
            action: Intervention type (pre_cooling, blind_adjustment, hvac_optimization, etc.)

        Returns:
            Projected impact metrics
        """
        # Define typical impact ranges for different actions
        actions_impact = {
            "pre_cooling": (-0.15, -0.20),
            "blind_adjustment": (-0.10, -0.15),
            "hvac_optimization": (-0.12, -0.18),
            "occupancy_scheduling": (-0.06, -0.10),
            "setpoint_adjustment": (-0.08, -0.12),
            "thermal_storage": (-0.18, -0.25)
        }

        impact_range = actions_impact.get(action, (-0.03, -0.08))
        impact_pct = random.uniform(impact_range[0], impact_range[1])

        current_load = random.uniform(300, 450)
        projected_load = current_load * (1 + impact_pct)
        savings_kwh = abs(current_load * impact_pct)

        # AED pricing (approximate Dubai DEWA rates)
        cost_per_kwh = 0.38  # AED
        cost_savings = savings_kwh * cost_per_kwh

        return {
            "building_id": building_id,
            "action": action,
            "current_load_kw": round(current_load, 2),
            "projected_load_kw": round(projected_load, 2),
            "savings_kwh": round(savings_kwh, 2),
            "savings_percentage": round(abs(impact_pct * 100), 1),
            "cost_savings_aed": round(cost_savings, 2),
            "annual_savings_aed": round(cost_savings * 365, 2),
            "confidence": round(random.uniform(0.75, 0.92), 2),
            "implementation_complexity": random.choice(["low", "medium", "high"]),
            "estimated_payback_months": random.randint(3, 18)
        }

    @staticmethod
    @tool
    def execute_building_command(building_id: str, command: Dict[str, Any]) -> Dict:
        """
        Execute a control command for a building's BMS/HVAC system.

        Args:
            building_id: Target building
            command: Command dictionary with action details

        Returns:
            Execution status and confirmation
        """
        # In production, this would interface with actual BMS/SCADA systems
        # For demo, we'll log to session state

        execution_record = {
            "building_id": building_id,
            "command": command,
            "timestamp": datetime.now().isoformat(),
            "status": "executed",
            "estimated_impact": f"{random.randint(10, 30)}% reduction",
            "execution_time_seconds": round(random.uniform(0.5, 2.5), 2)
        }

        return {
            "success": True,
            "building_id": building_id,
            "command_type": command.get("type", "unknown"),
            "execution_record": execution_record,
            "message": f"Command successfully executed for {building_id}",
            "monitoring_period_hours": 2,
            "rollback_available": True
        }

    @staticmethod
    @tool
    def get_weather_forecast(location: str = "Dubai") -> Dict:
        """
        Get weather forecast data for energy planning.

        Args:
            location: Location name

        Returns:
            Weather forecast with energy-relevant parameters
        """
        # Simulate Dubai weather - replace with actual weather API
        hour = datetime.now().hour

        # Dubai typical weather patterns
        base_temp = 42 if 12 <= hour <= 16 else 38 if 10 <= hour <= 18 else 32

        return {
            "location": location,
            "current_temperature_c": round(base_temp + random.uniform(-3, 3), 1),
            "feels_like_c": round(base_temp + random.uniform(2, 8), 1),
            "temperature_trend": "rising" if hour < 14 else "falling",
            "peak_temp_time": "14:00",
            "peak_temp_c": round(random.uniform(44, 50), 1),
            "humidity_percent": random.randint(45, 75),
            "solar_radiation_wm2": random.randint(600, 1100) if 8 <= hour <= 18 else 0,
            "solar_radiation_level": "extreme" if 11 <= hour <= 15 else "high" if 8 <= hour <= 18 else "none",
            "dust_storm_risk": random.choice(["low", "low", "low", "medium"]),
            "wind_speed_kmh": random.randint(10, 25),
            "forecast_confidence": 0.92,
            "energy_impact": "High cooling demand expected" if base_temp > 40 else "Moderate cooling demand"
        }

    @staticmethod
    @tool
    def get_building_metadata(building_id: str) -> Dict:
        """
        Retrieve static metadata about a building.

        Args:
            building_id: Building identifier

        Returns:
            Building characteristics and metadata
        """
        building_types = ["Office Tower", "Residential Tower", "Mixed Use", "Hotel", "Retail"]

        return {
            "building_id": building_id,
            "name": f"{building_id.replace('_', ' ').title()}",
            "type": random.choice(building_types),
            "year_built": random.randint(2005, 2023),
            "floors": random.randint(20, 75),
            "total_area_m2": random.randint(15000, 80000),
            "occupancy_capacity": random.randint(500, 3000),
            "hvac_system": random.choice(["VRF", "Chilled Water", "DX Split", "District Cooling"]),
            "glazing_type": random.choice(["Standard", "Low-E", "Double Glazed", "Triple Glazed"]),
            "insulation_rating": random.choice(["Standard", "Good", "Excellent"]),
            "bms_version": f"v{random.randint(2, 5)}.{random.randint(0, 9)}",
            "coordinates": {
                "lat": round(25.0 + random.uniform(0, 0.3), 6),
                "lon": round(55.1 + random.uniform(0, 0.3), 6)
            }
        }


# Create a singleton instance for easy import
energy_tools = EnergyTools()
