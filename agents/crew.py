"""
Energy Management Crew - Multi-Agent Orchestration

Coordinates Scout, Analyst, and Optimizer agents to work together
in a sophisticated autonomous energy optimization system.
"""

from crewai import Crew, Process
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List, Dict, Optional
import streamlit as st
from datetime import datetime
import json

from .scout import EnergyScout
from .analyst import EnergyAnalyst
from .optimizer import EnergyOptimizer
from .tools import EnergyTools


class EnergyManagementCrew:
    """
    Orchestrates the three-agent energy management system:
    1. Scout (Perception) - Monitors and detects
    2. Analyst (Reasoning) - Understands and explains
    3. Optimizer (Action) - Decides and executes
    """

    def __init__(self, gemini_api_key: str, model: str = "gemini-1.5-flash"):
        """
        Initialize the multi-agent crew

        Args:
            gemini_api_key: Google Gemini API key
            model: Gemini model to use (default: gemini-1.5-flash for speed)
        """
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=gemini_api_key,
            temperature=0.7,
            convert_system_message_to_human=True  # Better for Gemini
        )

        # Initialize shared tools
        self.tools = EnergyTools()

        # Initialize all three agents
        self.scout = EnergyScout(self.llm, self.tools)
        self.analyst = EnergyAnalyst(self.llm, self.tools)
        self.optimizer = EnergyOptimizer(self.llm, self.tools)

        # Initialize session state for tracking
        self._init_session_state()

    def _init_session_state(self):
        """Initialize Streamlit session state for tracking agent activities"""
        if 'agent_logs' not in st.session_state:
            st.session_state.agent_logs = []

        if 'detected_anomalies' not in st.session_state:
            st.session_state.detected_anomalies = []

        if 'executed_actions' not in st.session_state:
            st.session_state.executed_actions = []

        if 'crew_cycles' not in st.session_state:
            st.session_state.crew_cycles = 0

    def _log_agent_activity(self, agent: str, activity: str, data: Dict = None):
        """Log agent activity to session state"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent,
            "activity": activity,
            "data": data or {}
        }
        st.session_state.agent_logs.append(log_entry)

    def run_monitoring_cycle(self, building_ids: List[str]) -> Dict:
        """
        Run a complete monitoring → analysis → optimization cycle

        Args:
            building_ids: List of building IDs to monitor

        Returns:
            Dictionary with cycle results
        """
        self._log_agent_activity("crew", "Starting monitoring cycle", {"buildings": building_ids})

        # Phase 1: Scout monitors buildings
        st.session_state.scout_status = 'active'
        monitoring_task = self.scout.create_monitoring_task(building_ids)

        monitoring_crew = Crew(
            agents=[self.scout.agent],
            tasks=[monitoring_task],
            process=Process.sequential,
            verbose=True
        )

        try:
            monitoring_result = monitoring_crew.kickoff()
            self._log_agent_activity("scout", "Monitoring complete", {"result": str(monitoring_result)})
            st.session_state.scout_status = 'idle'

            # Parse anomalies from result (simplified for demo)
            anomalies = self._extract_anomalies_from_result(str(monitoring_result), building_ids)

            if anomalies:
                st.session_state.detected_anomalies = anomalies
                return {
                    "phase": "monitoring",
                    "anomalies_found": len(anomalies),
                    "anomalies": anomalies,
                    "next_action": "analysis_required"
                }
            else:
                return {
                    "phase": "monitoring",
                    "anomalies_found": 0,
                    "status": "all_normal"
                }

        except Exception as e:
            self._log_agent_activity("scout", "Error during monitoring", {"error": str(e)})
            st.session_state.scout_status = 'error'
            return {"phase": "monitoring", "error": str(e)}

    def run_analysis_phase(self, anomalous_buildings: List[Dict]) -> List[Dict]:
        """
        Run analysis phase on anomalous buildings

        Args:
            anomalous_buildings: List of buildings with anomalies

        Returns:
            List of analysis results
        """
        self._log_agent_activity("crew", "Starting analysis phase", {"buildings": len(anomalous_buildings)})

        analysis_results = []
        st.session_state.analyst_status = 'active'

        for building_data in anomalous_buildings[:3]:  # Analyze top 3
            analysis_task = self.analyst.create_analysis_task(building_data)

            analysis_crew = Crew(
                agents=[self.analyst.agent],
                tasks=[analysis_task],
                process=Process.sequential,
                verbose=True
            )

            try:
                result = analysis_crew.kickoff()
                self._log_agent_activity("analyst", f"Analysis complete for {building_data['building_id']}",
                                        {"result": str(result)[:200]})

                # Parse result into structured format
                analysis_results.append({
                    "building_id": building_data['building_id'],
                    "analysis": str(result),
                    "confidence": 0.85,  # Would parse from actual result
                    "recommended_interventions": ["pre_cooling", "hvac_optimization"]
                })

            except Exception as e:
                self._log_agent_activity("analyst", f"Error analyzing {building_data['building_id']}",
                                        {"error": str(e)})

        st.session_state.analyst_status = 'idle'
        return analysis_results

    def run_optimization_phase(self, analysis_results: List[Dict]) -> List[Dict]:
        """
        Run optimization phase based on analysis results

        Args:
            analysis_results: Analysis outputs from Analyst

        Returns:
            List of optimization decisions and actions
        """
        self._log_agent_activity("crew", "Starting optimization phase", {"analyses": len(analysis_results)})

        optimization_results = []
        st.session_state.optimizer_status = 'active'

        for analysis in analysis_results:
            if analysis.get('confidence', 0) > 0.7:
                optimization_task = self.optimizer.create_optimization_task(analysis)

                optimization_crew = Crew(
                    agents=[self.optimizer.agent],
                    tasks=[optimization_task],
                    process=Process.sequential,
                    verbose=True
                )

                try:
                    result = optimization_crew.kickoff()
                    self._log_agent_activity("optimizer", f"Optimization complete for {analysis['building_id']}",
                                           {"result": str(result)[:200]})

                    # Track executed actions
                    action = {
                        "building_id": analysis['building_id'],
                        "action": "optimization_executed",
                        "result": str(result),
                        "timestamp": datetime.now().isoformat()
                    }
                    st.session_state.executed_actions.append(action)
                    optimization_results.append(action)

                except Exception as e:
                    self._log_agent_activity("optimizer", f"Error optimizing {analysis['building_id']}",
                                           {"error": str(e)})

        st.session_state.optimizer_status = 'idle'
        return optimization_results

    def run_autonomous_loop(self, building_ids: List[str], cycles: int = 3) -> List[Dict]:
        """
        Run multiple autonomous monitoring-analysis-optimization cycles

        Args:
            building_ids: Buildings to monitor
            cycles: Number of cycles to run

        Returns:
            List of cycle results
        """
        all_results = []

        for cycle_num in range(cycles):
            st.session_state.crew_cycles += 1
            self._log_agent_activity("crew", f"Starting autonomous cycle {cycle_num + 1}/{cycles}")

            cycle_result = {
                "cycle": cycle_num + 1,
                "timestamp": datetime.now().isoformat()
            }

            # Phase 1: Monitoring
            monitoring_result = self.run_monitoring_cycle(building_ids)
            cycle_result["monitoring"] = monitoring_result

            # Phase 2: Analysis (if anomalies detected)
            if monitoring_result.get('anomalies_found', 0) > 0:
                analysis_results = self.run_analysis_phase(monitoring_result['anomalies'])
                cycle_result["analysis"] = analysis_results

                # Phase 3: Optimization (if high-confidence analysis)
                if analysis_results:
                    optimization_results = self.run_optimization_phase(analysis_results)
                    cycle_result["optimization"] = optimization_results

            all_results.append(cycle_result)

        return all_results

    def run_single_building_workflow(self, building_id: str) -> Dict:
        """
        Run complete workflow for a single building

        Args:
            building_id: Building to analyze and optimize

        Returns:
            Complete workflow result
        """
        workflow_result = {
            "building_id": building_id,
            "timestamp": datetime.now().isoformat()
        }

        # Step 1: Scout monitors single building
        monitoring_task = self.scout.create_single_building_task(building_id)
        monitoring_crew = Crew(
            agents=[self.scout.agent],
            tasks=[monitoring_task],
            process=Process.sequential,
            verbose=True
        )

        monitoring_result = monitoring_crew.kickoff()
        workflow_result["monitoring"] = str(monitoring_result)

        # Step 2: Analyst performs deep analysis
        building_data = {"building_id": building_id, "anomaly_score": 0.6}
        analysis_task = self.analyst.create_analysis_task(building_data)
        analysis_crew = Crew(
            agents=[self.analyst.agent],
            tasks=[analysis_task],
            process=Process.sequential,
            verbose=True
        )

        analysis_result = analysis_crew.kickoff()
        workflow_result["analysis"] = str(analysis_result)

        # Step 3: Optimizer makes decision
        analysis_data = {
            "building_id": building_id,
            "root_cause": {"primary_cause": "High cooling load"},
            "overall_confidence": 0.85,
            "recommended_interventions": []
        }
        optimization_task = self.optimizer.create_optimization_task(analysis_data)
        optimization_crew = Crew(
            agents=[self.optimizer.agent],
            tasks=[optimization_task],
            process=Process.sequential,
            verbose=True
        )

        optimization_result = optimization_crew.kickoff()
        workflow_result["optimization"] = str(optimization_result)

        return workflow_result

    def run_collaborative_task(self, task_description: str, building_ids: List[str]) -> Dict:
        """
        Run a collaborative task where all agents work together

        Args:
            task_description: Natural language task description
            building_ids: Buildings involved

        Returns:
            Collaborative result
        """
        # Create custom tasks for each agent based on task description
        scout_task = Task(
            description=f"Monitor buildings {building_ids} for: {task_description}",
            agent=self.scout.agent,
            expected_output="Monitoring findings"
        )

        analyst_task = Task(
            description=f"Analyze patterns related to: {task_description}",
            agent=self.analyst.agent,
            expected_output="Analysis insights"
        )

        optimizer_task = Task(
            description=f"Propose optimizations for: {task_description}",
            agent=self.optimizer.agent,
            expected_output="Optimization recommendations"
        )

        # Run all agents in parallel
        collaborative_crew = Crew(
            agents=[self.scout.agent, self.analyst.agent, self.optimizer.agent],
            tasks=[scout_task, analyst_task, optimizer_task],
            process=Process.sequential,
            verbose=True
        )

        result = collaborative_crew.kickoff()

        return {
            "task": task_description,
            "buildings": building_ids,
            "result": str(result),
            "timestamp": datetime.now().isoformat()
        }

    def _extract_anomalies_from_result(self, result_text: str, building_ids: List[str]) -> List[Dict]:
        """
        Extract anomaly information from Scout's monitoring result

        Args:
            result_text: Raw text output from Scout
            building_ids: Buildings that were monitored

        Returns:
            List of anomaly dictionaries
        """
        # Simplified extraction for demo - in production, parse structured output
        import random

        # Simulate finding 2-3 anomalies
        num_anomalies = min(random.randint(1, 3), len(building_ids))
        anomalies = []

        for i in range(num_anomalies):
            anomalies.append({
                "building_id": building_ids[i] if i < len(building_ids) else f"Building_{i}",
                "anomaly_score": round(random.uniform(0.4, 0.9), 2),
                "severity": random.choice(["medium", "high", "critical"]),
                "deviation_percent": round(random.uniform(20, 60), 1)
            })

        return sorted(anomalies, key=lambda x: x['anomaly_score'], reverse=True)

    def get_agent_status(self) -> Dict:
        """Get current status of all agents"""
        return {
            "scout": st.session_state.get('scout_status', 'idle'),
            "analyst": st.session_state.get('analyst_status', 'idle'),
            "optimizer": st.session_state.get('optimizer_status', 'idle'),
            "total_cycles": st.session_state.crew_cycles,
            "anomalies_detected": len(st.session_state.detected_anomalies),
            "actions_executed": len(st.session_state.executed_actions)
        }

    def get_activity_log(self, limit: int = 50) -> List[Dict]:
        """Get recent agent activity log"""
        return st.session_state.agent_logs[-limit:]
