"""
Energy Analyst Agent - Reasoning Layer

Performs deep analysis and explanation generation to understand WHY
energy patterns occur and provides actionable, explainable insights.
"""

from crewai import Agent, Task
from typing import Dict, List
from .tools import EnergyTools


class EnergyAnalyst:
    """
    The Analyst Agent specializes in:
    - Root cause analysis using SHAP and statistical methods
    - Explainable AI for energy consumption patterns
    - Knowledge base queries for similar historical patterns
    - Generating actionable insights for stakeholders
    """

    def __init__(self, llm, tools: EnergyTools):
        """
        Initialize the Analyst Agent

        Args:
            llm: Language model instance (e.g., ChatGoogleGenerativeAI)
            tools: EnergyTools instance with analysis capabilities
        """
        self.tools = tools
        self.agent = Agent(
            role='Energy Data Scientist & Explainability Expert',
            goal='Provide deep insights into WHY energy patterns occur and generate clear, actionable explanations',
            backstory="""You are a senior energy data scientist with a PhD in Building Physics
            and 15+ years of experience in machine learning explainability. Your expertise spans:

            - SHAP (SHapley Additive exPlanations) for model interpretability
            - Building thermodynamics and HVAC system physics
            - Statistical analysis and causal inference
            - Dubai's extreme climate and its impact on building performance
            - Different building archetypes (office, residential, hotel, retail)
            - Occupant behavior patterns and their energy signatures

            You excel at finding root causes of energy issues by combining:
            - Machine learning explainability techniques
            - Domain knowledge of building systems
            - Historical pattern matching from knowledge bases
            - Spatial and temporal context analysis

            Your superpower is translating complex technical findings into clear, actionable
            insights that both technical teams and non-technical stakeholders can understand
            and act upon. You always quantify your confidence levels and provide evidence
            for your conclusions.

            When analyzing anomalies, you consider multiple hypotheses and systematically
            evaluate evidence to identify the true root cause. You understand that correlation
            doesn't imply causation and always look for mechanistic explanations.""",
            verbose=True,
            allow_delegation=False,
            llm=llm,
            tools=[
                tools.run_shap_analysis,
                tools.query_vector_knowledge,
                tools.query_building_energy,
                tools.simulate_intervention,
                tools.get_building_metadata,
                tools.get_weather_forecast
            ],
            memory=True
        )

    def create_analysis_task(self, building_data: Dict) -> Task:
        """
        Create deep analysis task for a building with detected anomaly

        Args:
            building_data: Dictionary with building_id and anomaly information

        Returns:
            CrewAI Task configured for root cause analysis
        """
        building_id = building_data.get('building_id', 'unknown')
        anomaly_score = building_data.get('anomaly_score', 'high')

        return Task(
            description=f"""
            Perform comprehensive root cause analysis on building {building_id}
            which shows anomaly score of {anomaly_score}.

            Your analysis must include:

            1. FEATURE IMPORTANCE ANALYSIS
               - Run SHAP analysis using run_shap_analysis to identify top contributing factors
               - Quantify the impact of each factor (outdoor temp, solar, occupancy, etc.)
               - Determine which factors are controllable vs. environmental

            2. HISTORICAL PATTERN MATCHING
               - Query knowledge base using query_vector_knowledge for similar past patterns
               - Identify how similar issues were resolved previously
               - Extract lessons learned from historical data

            3. SPATIAL CONTEXT ANALYSIS
               - Get building metadata to understand characteristics
               - Consider why THIS building shows anomaly but neighbors don't
               - Analyze building-specific factors (age, type, HVAC system, glazing)

            4. TEMPORAL & WEATHER CONTEXT
               - Get current weather conditions
               - Assess if consumption aligns with weather expectations
               - Check for time-of-day and day-of-week patterns

            5. HYPOTHESIS GENERATION & TESTING
               - Generate 3 specific hypotheses for the anomaly root cause
               - Rank hypotheses by likelihood based on evidence
               - Provide supporting evidence for top hypothesis

            6. INTERVENTION SIMULATION
               - Simulate impact of 3 most promising interventions
               - Compare projected savings and feasibility
               - Recommend best intervention to Optimizer agent

            Provide a structured report with:
            - Executive Summary (2-3 sentences for non-technical stakeholders)
            - Root Cause Analysis with confidence levels
            - Top 3 Contributing Factors with quantified impacts
            - Comparison with similar buildings
            - Natural language explanation of WHY this is happening
            - Evidence supporting your conclusions
            - Recommended interventions with projected impact
            - Confidence level in your analysis (0-1)

            Remember: Your analysis will guide the Optimizer agent's decisions,
            so be thorough, evidence-based, and clear about uncertainties.
            """,
            agent=self.agent,
            expected_output="""A comprehensive analysis report in JSON format:
            {
                "building_id": str,
                "executive_summary": str,
                "root_cause": {
                    "primary_cause": str,
                    "confidence": float,
                    "evidence": [str]
                },
                "contributing_factors": [
                    {
                        "factor": str,
                        "impact_percentage": float,
                        "controllable": bool
                    }
                ],
                "shap_analysis": dict,
                "similar_historical_patterns": [dict],
                "spatial_context": str,
                "weather_impact": str,
                "hypotheses": [
                    {
                        "hypothesis": str,
                        "likelihood": float,
                        "evidence": [str]
                    }
                ],
                "recommended_interventions": [
                    {
                        "intervention": str,
                        "projected_savings_kwh": float,
                        "projected_savings_aed": float,
                        "feasibility": str,
                        "priority": int
                    }
                ],
                "overall_confidence": float
            }"""
        )

    def create_comparative_task(self, building_a: str, building_b: str) -> Task:
        """
        Compare two buildings to explain consumption differences

        Args:
            building_a: First building ID
            building_b: Second building ID

        Returns:
            CrewAI Task for comparative analysis
        """
        return Task(
            description=f"""
            Compare buildings {building_a} and {building_b} to explain their
            energy consumption differences.

            Analysis required:
            1. Query energy profiles for both buildings
            2. Get metadata for both (type, age, size, HVAC system)
            3. Run SHAP analysis on both to identify different factor weights
            4. Identify key differentiating factors
            5. Explain why similar buildings have different patterns
            6. Determine if differences are due to:
               - Building characteristics (inherent)
               - Operational practices (modifiable)
               - Equipment efficiency (upgradeable)
               - Occupancy patterns (schedulable)

            Deliver:
            - Side-by-side comparison table of key metrics
            - Top 3 factors causing differences with quantified impact
            - Classification of differences (inherent vs. modifiable)
            - Actionable insights: Can building A learn from building B?
            - Specific recommendations to align performance
            - Confidence level in your analysis
            """,
            agent=self.agent,
            expected_output="Comparative analysis with clear differentiating factors and actionable recommendations"
        )

    def create_trend_analysis_task(self, building_id: str, time_period: str = "7d") -> Task:
        """
        Analyze energy consumption trends over time

        Args:
            building_id: Building to analyze
            time_period: Period for trend analysis

        Returns:
            CrewAI Task for trend analysis
        """
        return Task(
            description=f"""
            Analyze energy consumption trends for {building_id} over {time_period}.

            Investigation required:
            1. Query current building energy data
            2. Run SHAP analysis to understand current drivers
            3. Query knowledge base for this building's historical patterns
            4. Assess trend direction and magnitude
            5. Identify any inflection points or sudden changes
            6. Correlate trends with:
               - Weather patterns
               - Occupancy changes
               - Operational modifications
               - Equipment degradation

            Provide:
            - Trend characterization (improving, stable, degrading)
            - Rate of change quantification
            - Root causes of trend
            - Projected future state if trend continues
            - Recommended actions to improve trajectory
            - Early warning indicators to monitor
            """,
            agent=self.agent,
            expected_output="Trend analysis report with root causes and forward-looking recommendations"
        )

    def create_portfolio_insight_task(self, building_ids: List[str]) -> Task:
        """
        Generate portfolio-level insights across multiple buildings

        Args:
            building_ids: List of buildings to analyze collectively

        Returns:
            CrewAI Task for portfolio analysis
        """
        return Task(
            description=f"""
            Analyze energy patterns across building portfolio: {', '.join(building_ids[:5])}
            {'and others' if len(building_ids) > 5 else ''}

            Portfolio Analysis:
            1. Query energy data for all buildings
            2. Identify common patterns and outliers
            3. Segment buildings by performance clusters
            4. Find buildings with replicable best practices
            5. Identify systemic issues affecting multiple buildings
            6. Calculate portfolio-wide optimization potential

            Deliver:
            - Portfolio performance distribution
            - Best performers and their characteristics
            - Worst performers and common issues
            - Estimated total optimization potential (kWh and AED)
            - Top 3 systemic improvements applicable across portfolio
            - Building prioritization for interventions (by ROI)
            - Portfolio-level KPIs and benchmarks
            """,
            agent=self.agent,
            expected_output="Portfolio-level insights with optimization opportunities and prioritization"
        )

    def create_explainability_task(self, building_id: str, stakeholder_type: str = "technical") -> Task:
        """
        Generate stakeholder-specific explanations

        Args:
            building_id: Building to explain
            stakeholder_type: Type of stakeholder (technical, executive, occupant)

        Returns:
            CrewAI Task for generating tailored explanations
        """
        return Task(
            description=f"""
            Generate a {stakeholder_type}-appropriate explanation for energy patterns
            in building {building_id}.

            Tailor explanation for {stakeholder_type} stakeholder:
            - Technical: Detailed SHAP values, system interactions, physics
            - Executive: ROI, cost impact, strategic recommendations
            - Occupant: Comfort impact, simple actions, benefits

            Steps:
            1. Run comprehensive analysis (SHAP, metadata, weather)
            2. Identify key insights relevant to stakeholder type
            3. Translate technical findings into appropriate language
            4. Include relevant metrics and visualizations
            5. Provide clear, actionable recommendations

            Output format:
            - Natural language narrative (not just data dumps)
            - Quantified impacts in stakeholder-relevant units
            - Clear cause-effect relationships
            - Specific action items
            - Expected outcomes of recommended actions
            """,
            agent=self.agent,
            expected_output=f"Clear, {stakeholder_type}-appropriate explanation with actionable insights"
        )
