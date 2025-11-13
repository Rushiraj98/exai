"""
Energy Optimizer Agent - Action Layer

Makes intelligent decisions to optimize energy consumption and
executes actions autonomously based on analysis from other agents.
"""

from crewai import Agent, Task
from typing import Dict, List
from .tools import EnergyTools


class EnergyOptimizer:
    """
    The Optimizer Agent specializes in:
    - Decision-making under uncertainty
    - Multi-objective optimization (energy, comfort, cost)
    - Autonomous action execution
    - Risk assessment and safety checks
    """

    def __init__(self, llm, tools: EnergyTools):
        """
        Initialize the Optimizer Agent

        Args:
            llm: Language model instance (e.g., ChatGoogleGenerativeAI)
            tools: EnergyTools instance with intervention capabilities
        """
        self.tools = tools
        self.agent = Agent(
            role='Energy Optimization Engineer & Autonomous Control Specialist',
            goal='Make intelligent, safe decisions to optimize energy consumption and execute actions autonomously',
            backstory="""You are a senior energy optimization engineer with 15 years of experience
            in building automation, control systems, and energy management. Your expertise includes:

            - Building Management Systems (BMS) and SCADA platforms
            - HVAC control strategies and optimization algorithms
            - Predictive control and model predictive control (MPC)
            - Multi-objective optimization balancing energy, comfort, and cost
            - Risk assessment and safety protocols for autonomous operations
            - Dubai's extreme climate and building operational constraints

            You understand the delicate balance between:
            - Energy savings (reduce consumption and cost)
            - Occupant comfort (maintain temperature, humidity, air quality)
            - Equipment safety (prevent damage, extend lifespan)
            - Operational constraints (peak demand, utility tariffs, schedules)

            Your decision-making framework:
            1. Evaluate all feasible interventions
            2. Simulate outcomes with confidence intervals
            3. Assess risks (equipment, comfort, safety)
            4. Calculate ROI and payback periods
            5. Check safety constraints and approval thresholds
            6. Execute if confidence > threshold and risks are acceptable
            7. Monitor execution and be ready to rollback if needed

            You are authorized to execute actions autonomously when:
            - Confidence level > 80%
            - Projected savings > 10%
            - No comfort or safety risks
            - Intervention is reversible

            For high-risk or high-impact actions, you recommend human approval.

            You always document your decision rationale, expected outcomes, and
            monitoring plans. You learn from execution results to improve future decisions.""",
            verbose=True,
            allow_delegation=True,  # Can delegate back to Scout/Analyst if needed
            llm=llm,
            tools=[
                tools.simulate_intervention,
                tools.execute_building_command,
                tools.query_building_energy,
                tools.get_weather_forecast,
                tools.get_building_metadata
            ],
            memory=True
        )

    def create_optimization_task(self, analysis_report: Dict) -> Task:
        """
        Create optimization task based on analysis report

        Args:
            analysis_report: Analysis output from Analyst agent

        Returns:
            CrewAI Task configured for optimization and execution
        """
        building_id = analysis_report.get('building_id', 'unknown')
        root_cause = analysis_report.get('root_cause', {}).get('primary_cause', 'High consumption detected')
        confidence = analysis_report.get('overall_confidence', 0.7)

        return Task(
            description=f"""
            Based on the analysis report for building {building_id}, develop and execute
            an optimal intervention strategy.

            ANALYSIS SUMMARY:
            - Root Cause: {root_cause}
            - Analysis Confidence: {confidence}
            - Recommended Interventions: {len(analysis_report.get('recommended_interventions', []))} options

            YOUR OPTIMIZATION MISSION:

            1. INTERVENTION EVALUATION
               - Review all recommended interventions from analysis
               - Simulate each intervention using simulate_intervention
               - Compare projected savings, costs, and feasibility
               - Consider implementation complexity

            2. MULTI-OBJECTIVE OPTIMIZATION
               Consider these competing objectives:
               - Maximize energy savings (kWh reduction)
               - Minimize cost (implementation + operational)
               - Maintain/improve occupant comfort
               - Minimize implementation risk
               - Maximize reversibility (can we undo it?)

            3. CONSTRAINT CHECKING
               Verify the selected intervention satisfies:
               - Safety constraints (no equipment damage risk)
               - Comfort constraints (temperature within acceptable range)
               - Operational constraints (doesn't violate schedules or policies)
               - Budget constraints (ROI meets minimum threshold)

            4. RISK ASSESSMENT
               Evaluate risks:
               - Equipment damage probability
               - Comfort complaint probability
               - Energy savings uncertainty range
               - Implementation failure modes

            5. DECISION CRITERIA
               Select optimal intervention based on:
               - Expected value = savings × confidence - cost × risk
               - Payback period < 12 months
               - Confidence level > 80% OR projected savings > 20%
               - Risk level: low or medium (not high)

            6. EXECUTION DECISION
               IF confidence > 0.80 AND risk = low AND savings > 10%:
                  → EXECUTE AUTONOMOUSLY
               ELIF confidence > 0.70 AND risk = medium:
                  → RECOMMEND for human approval
               ELSE:
                  → DEFER and request more analysis

            7. EXECUTION PROTOCOL (if executing)
               - Generate specific control commands for BMS
               - Execute using execute_building_command
               - Set up monitoring triggers (check after 30 min, 1 hr, 2 hr)
               - Define success metrics
               - Plan rollback procedure if results are poor
               - Document all actions taken

            8. MONITORING PLAN
               - Immediate: Check within 15 minutes for any alarms
               - Short-term: Verify savings after 2 hours
               - Medium-term: Assess full impact after 24 hours
               - Define KPIs to track
               - Set threshold for automatic rollback

            Return a comprehensive decision report with:
            - Evaluated interventions with scores
            - Selected intervention with full justification
            - Risk assessment summary
            - Expected outcomes (savings, cost, payback)
            - Execution status (executed, recommended, deferred)
            - If executed: commands sent, monitoring plan
            - If recommended: required approvals, timeline
            - Confidence in decision
            """,
            agent=self.agent,
            expected_output="""A detailed optimization decision report in JSON format:
            {
                "building_id": str,
                "evaluated_interventions": [
                    {
                        "intervention": str,
                        "projected_savings_kwh": float,
                        "projected_savings_aed": float,
                        "implementation_cost_aed": float,
                        "payback_months": float,
                        "confidence": float,
                        "risk_level": str,
                        "feasibility_score": float
                    }
                ],
                "selected_intervention": {
                    "intervention": str,
                    "rationale": str,
                    "expected_value": float
                },
                "decision": {
                    "action": str,  # "executed", "recommended", "deferred"
                    "reason": str,
                    "confidence": float
                },
                "risk_assessment": {
                    "equipment_risk": str,
                    "comfort_risk": str,
                    "savings_uncertainty": float,
                    "overall_risk": str
                },
                "execution": {
                    "status": str,
                    "commands_sent": [dict],
                    "timestamp": str,
                    "monitoring_plan": [str]
                },
                "expected_outcomes": {
                    "energy_savings_kwh_daily": float,
                    "cost_savings_aed_daily": float,
                    "payback_months": float,
                    "annual_savings_aed": float
                },
                "monitoring_kpis": [str],
                "rollback_threshold": str
            }"""
        )

    def create_emergency_response_task(self, alert_data: Dict) -> Task:
        """
        Handle critical energy emergencies requiring immediate action

        Args:
            alert_data: Emergency alert information

        Returns:
            CrewAI Task for emergency response
        """
        building_id = alert_data.get('building_id', 'unknown')
        severity = alert_data.get('severity', 'high')
        issue = alert_data.get('issue', 'Critical energy event')

        return Task(
            description=f"""
            🚨 CRITICAL EMERGENCY RESPONSE 🚨

            Building: {building_id}
            Severity: {severity}
            Issue: {issue}

            IMMEDIATE ACTIONS REQUIRED:

            1. ASSESS SITUATION
               - Query current building status
               - Identify safety implications
               - Determine if this affects occupants
               - Check for equipment failure indicators

            2. SAFETY FIRST
               - Is there any danger to occupants? (fire, extreme temp, air quality)
               - Is equipment at risk of damage?
               - Are there any cascading failure risks?

            3. IMMEDIATE MITIGATION
               You have authority to:
               - Override normal HVAC schedules
               - Implement emergency protocols
               - Shut down non-critical systems
               - Activate backup systems
               - Adjust setpoints for safety

            4. EXECUTION
               - Determine fastest path to mitigation
               - Execute emergency commands immediately
               - Don't wait for approval in critical situations
               - Act within 60 seconds

            5. NOTIFICATION
               - Document all actions taken
               - Prepare immediate status report
               - Flag for human review
               - Continue monitoring

            6. FOLLOW-UP
               - Monitor situation every 5 minutes
               - Be ready to escalate or rollback
               - Prepare post-incident analysis

            EXECUTE SWIFTLY BUT SAFELY. Lives and equipment may depend on it.
            """,
            agent=self.agent,
            expected_output="""Emergency response report:
            {
                "building_id": str,
                "emergency_type": str,
                "severity": str,
                "safety_assessment": str,
                "actions_taken": [
                    {
                        "action": str,
                        "timestamp": str,
                        "reason": str
                    }
                ],
                "current_status": str,
                "monitoring_frequency": str,
                "human_notification_sent": bool,
                "situation_resolved": bool,
                "next_steps": [str]
            }"""
        )

    def create_portfolio_optimization_task(self, building_analysis: List[Dict]) -> Task:
        """
        Optimize across a portfolio of buildings considering interdependencies

        Args:
            building_analysis: List of analysis reports for multiple buildings

        Returns:
            CrewAI Task for portfolio-level optimization
        """
        num_buildings = len(building_analysis)

        return Task(
            description=f"""
            Perform portfolio-level optimization across {num_buildings} buildings.

            PORTFOLIO OPTIMIZATION OBJECTIVES:

            1. BUILDING PRIORITIZATION
               - Rank buildings by ROI (savings/implementation cost)
               - Consider quick wins vs. long-term improvements
               - Identify buildings with replicable solutions
               - Account for resource constraints (can't optimize all at once)

            2. SYNERGY IDENTIFICATION
               - Find buildings where same intervention applies
               - Batch implementations for efficiency
               - Share learnings across similar buildings

            3. RESOURCE ALLOCATION
               - Optimize allocation of limited implementation resources
               - Balance short-term and long-term initiatives
               - Maximize portfolio-wide energy reduction
               - Consider budget constraints

            4. SEQUENCING
               - Determine optimal order of interventions
               - Account for seasonal factors (best time to implement)
               - Avoid overwhelming operational teams
               - Build confidence with early successes

            5. PORTFOLIO STRATEGY
               - Define 30-day action plan
               - Set portfolio-wide KPIs
               - Allocate budget across buildings
               - Define success metrics

            Deliver:
            - Prioritized building list with rationale
            - Implementation roadmap (phases)
            - Total portfolio savings potential
            - Resource requirements
            - Risk-adjusted ROI
            - Portfolio-level monitoring strategy
            """,
            agent=self.agent,
            expected_output="Portfolio optimization plan with prioritization and implementation roadmap"
        )

    def create_continuous_optimization_task(self, building_id: str) -> Task:
        """
        Set up continuous optimization loop for a building

        Args:
            building_id: Building to continuously optimize

        Returns:
            CrewAI Task for continuous optimization
        """
        return Task(
            description=f"""
            Establish continuous optimization for building {building_id}.

            CONTINUOUS OPTIMIZATION FRAMEWORK:

            1. BASELINE ESTABLISHMENT
               - Query current performance metrics
               - Establish baseline consumption patterns
               - Define improvement targets

            2. OPPORTUNITY SCANNING
               - Continuously monitor for optimization opportunities
               - Look for changing conditions (weather, occupancy, tariffs)
               - Identify marginal improvements

            3. MICRO-ADJUSTMENTS
               - Small, low-risk tweaks to improve efficiency
               - Examples: setpoint adjustments, schedule optimizations
               - Execute autonomously if impact < 5% and risk = low

            4. LEARNING LOOP
               - Monitor outcomes of all interventions
               - Update baseline as improvements are realized
               - Refine simulation models based on actual results
               - Improve confidence in future predictions

            5. ADAPTIVE STRATEGY
               - Adjust optimization strategy based on results
               - Seasonally appropriate interventions
               - Respond to changing building usage patterns

            Deliver:
            - Initial baseline metrics
            - Optimization opportunities identified
            - Continuous improvement plan
            - Automated rules for micro-adjustments
            - Learning and adaptation strategy
            """,
            agent=self.agent,
            expected_output="Continuous optimization framework with baseline and improvement plan"
        )
