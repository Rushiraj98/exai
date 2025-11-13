"""
Energy Scout Agent - Perception Layer

Continuously monitors the energy landscape across buildings,
detecting anomalies and unusual consumption patterns in real-time.
"""

from crewai import Agent, Task
from typing import List, Dict
from .tools import EnergyTools


class EnergyScout:
    """
    The Scout Agent specializes in:
    - Real-time energy consumption monitoring
    - Spatial anomaly detection
    - Pattern recognition across building portfolios
    - Early warning system for unusual energy behavior
    """

    def __init__(self, llm, tools: EnergyTools):
        """
        Initialize the Scout Agent

        Args:
            llm: Language model instance (e.g., ChatGoogleGenerativeAI)
            tools: EnergyTools instance with monitoring capabilities
        """
        self.tools = tools
        self.agent = Agent(
            role='Energy Monitoring Specialist',
            goal='Detect energy anomalies and unusual patterns across Dubai buildings in real-time with high accuracy',
            backstory="""You are an expert energy monitoring specialist with 10+ years of experience
            in building energy systems and anomaly detection. You have deep knowledge of:

            - Building energy consumption patterns across different typologies
            - Spatial correlation of energy usage in building clusters
            - Weather impact on building performance in Dubai's extreme climate
            - HVAC system behaviors and failure modes
            - Occupancy patterns and their energy signatures

            You have a keen eye for detecting anomalies and can quickly identify when buildings
            are consuming energy outside their normal patterns. You understand how outdoor
            temperature, solar radiation, time of day, day of week, and building characteristics
            affect consumption.

            Your specialty is comparing buildings to their spatial neighbors to identify
            outliers that may indicate equipment failures, control issues, or optimization
            opportunities. You always provide clear severity assessments and actionable
            recommendations for follow-up analysis.""",
            verbose=True,
            allow_delegation=False,
            llm=llm,
            tools=[
                tools.query_building_energy,
                tools.calculate_spatial_anomaly_score,
                tools.get_weather_forecast,
                tools.get_building_metadata
            ],
            memory=True
        )

    def create_monitoring_task(self, building_ids: List[str]) -> Task:
        """
        Create a comprehensive monitoring task for multiple buildings

        Args:
            building_ids: List of building IDs to monitor

        Returns:
            CrewAI Task configured for anomaly detection
        """
        return Task(
            description=f"""
            Monitor the following {len(building_ids)} buildings for energy anomalies: {', '.join(building_ids)}

            Your specific objectives:
            1. Query current energy consumption for each building using query_building_energy
            2. Calculate spatial anomaly scores for each building using calculate_spatial_anomaly_score
            3. Get current weather conditions using get_weather_forecast
            4. Identify buildings with unusual patterns (anomaly score > 0.3)
            5. Cross-reference consumption with weather to identify unexplained variations
            6. Rank buildings by severity and impact
            7. Flag any critical issues that need immediate attention (score > 0.7)

            Focus on detecting:
            - Buildings with >30% deviation from spatial neighbors
            - Sudden spikes or drops in consumption
            - Patterns that don't match weather conditions
            - High consumption during low occupancy periods
            - HVAC inefficiencies indicated by temperature/load mismatch

            Return a structured report with:
            - Summary statistics (total buildings, anomalies found, severity distribution)
            - List of anomalous buildings ranked by severity score
            - For each anomaly: building ID, score, current metrics, initial hypothesis
            - Top 3 buildings recommended for deep analysis by the Analyst agent
            - Any critical alerts requiring immediate response
            - Confidence level in your detections

            Format your findings clearly with severity levels:
            - CRITICAL (score > 0.7): Immediate action required
            - HIGH (score 0.5-0.7): Urgent investigation needed
            - MEDIUM (score 0.3-0.5): Schedule analysis
            - LOW (score < 0.3): Normal operation
            """,
            agent=self.agent,
            expected_output="""A structured JSON-style report containing:
            {
                "summary": {
                    "total_buildings_monitored": int,
                    "anomalies_detected": int,
                    "critical_alerts": int,
                    "monitoring_timestamp": str
                },
                "anomalous_buildings": [
                    {
                        "building_id": str,
                        "anomaly_score": float,
                        "severity": str,
                        "current_load_kw": float,
                        "neighbor_avg_kw": float,
                        "deviation_percent": float,
                        "hypothesis": str
                    }
                ],
                "top_3_for_analysis": [str, str, str],
                "critical_alerts": [str],
                "weather_context": dict,
                "confidence": float
            }"""
        )

    def create_single_building_task(self, building_id: str) -> Task:
        """
        Create a focused monitoring task for a single building

        Args:
            building_id: Building ID to monitor

        Returns:
            CrewAI Task for single building monitoring
        """
        return Task(
            description=f"""
            Perform detailed monitoring of building {building_id}.

            Steps:
            1. Query current energy metrics
            2. Calculate spatial anomaly score
            3. Get building metadata
            4. Assess current weather conditions
            5. Determine if building shows anomalous behavior

            Provide:
            - Current consumption vs. expected baseline
            - Anomaly score with severity classification
            - Comparison with similar buildings
            - Weather-adjusted assessment
            - Recommendation (normal monitoring vs. escalate to Analyst)
            """,
            agent=self.agent,
            expected_output="Detailed assessment of single building with anomaly determination and recommendation"
        )

    def create_comparative_monitoring_task(self, building_group: List[str], reference_building: str) -> Task:
        """
        Compare a reference building against a group of similar buildings

        Args:
            building_group: List of similar buildings for comparison
            reference_building: Building to evaluate

        Returns:
            CrewAI Task for comparative analysis
        """
        return Task(
            description=f"""
            Compare building {reference_building} against similar buildings: {', '.join(building_group)}

            Analysis required:
            1. Query energy consumption for all buildings
            2. Calculate statistical baseline from the group
            3. Determine how much {reference_building} deviates from group norm
            4. Account for any building-specific factors (size, age, type)
            5. Identify if deviation is concerning

            Deliver:
            - Group consumption statistics (mean, std, range)
            - Reference building position relative to group (percentile, z-score)
            - Deviation assessment with severity
            - Recommendation for further investigation
            """,
            agent=self.agent,
            expected_output="Comparative analysis showing how reference building performs vs. peer group"
        )

    def create_temporal_monitoring_task(self, building_id: str, time_range: str = "24h") -> Task:
        """
        Monitor temporal patterns for a building over time

        Args:
            building_id: Building to monitor
            time_range: Time period to analyze

        Returns:
            CrewAI Task for temporal pattern detection
        """
        return Task(
            description=f"""
            Analyze temporal energy patterns for {building_id} over {time_range}.

            Investigate:
            1. Query current and recent energy consumption
            2. Identify unusual temporal patterns (e.g., high nighttime load)
            3. Compare current hour to typical baseline for this time
            4. Detect any sudden changes or trend shifts
            5. Assess if patterns align with expected schedules

            Report:
            - Current load vs. typical for this time of day
            - Any unusual temporal behaviors detected
            - Trend direction (increasing, stable, decreasing)
            - Schedule alignment check (occupancy, HVAC schedules)
            - Temporal anomaly score and recommendation
            """,
            agent=self.agent,
            expected_output="Temporal pattern analysis with anomaly detection over specified time range"
        )
