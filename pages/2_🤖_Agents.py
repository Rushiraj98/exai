"""
Agent Control Center - Interactive Multi-Agent System Interface

Monitor and control the three-agent energy optimization system in real-time.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime
import time
import os

# Page configuration
st.set_page_config(
    page_title="Agent Control Center",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS - Cyberpunk Theme
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');

    /* Global Styling */
    .stApp {
        background: linear-gradient(135deg, #0a0e14 0%, #1a1f2e 100%);
    }

    /* Agent Status Cards */
    .agent-status-active {
        background: linear-gradient(135deg, rgba(0, 217, 192, 0.2) 0%, rgba(0, 255, 209, 0.1) 100%);
        padding: 20px;
        border-radius: 15px;
        border: 2px solid #00D9C0;
        color: white;
        text-align: center;
        box-shadow: 0 0 30px rgba(0, 217, 192, 0.5),
                    inset 0 0 20px rgba(0, 217, 192, 0.1);
        backdrop-filter: blur(10px);
        animation: pulse-glow 2s infinite;
    }

    @keyframes pulse-glow {
        0%, 100% { box-shadow: 0 0 30px rgba(0, 217, 192, 0.5), inset 0 0 20px rgba(0, 217, 192, 0.1); }
        50% { box-shadow: 0 0 40px rgba(0, 255, 209, 0.7), inset 0 0 30px rgba(0, 217, 192, 0.15); }
    }

    .agent-status-idle {
        background: linear-gradient(135deg, rgba(139, 148, 158, 0.1) 0%, rgba(139, 148, 158, 0.05) 100%);
        padding: 20px;
        border-radius: 15px;
        border: 2px solid #3d444d;
        color: white;
        text-align: center;
        box-shadow: 0 0 10px rgba(139, 148, 158, 0.2);
        backdrop-filter: blur(10px);
    }

    /* Chat Messages */
    .chat-message {
        padding: 15px 20px;
        border-radius: 12px;
        margin: 12px 0;
        backdrop-filter: blur(10px);
        border-left: 4px solid;
        font-family: 'Inter', sans-serif;
    }

    .scout-message {
        background: linear-gradient(135deg, rgba(0, 217, 192, 0.1) 0%, rgba(0, 217, 192, 0.05) 100%);
        border-left-color: #00D9C0;
        box-shadow: 0 0 15px rgba(0, 217, 192, 0.2);
    }

    .analyst-message {
        background: linear-gradient(135deg, rgba(255, 107, 176, 0.1) 0%, rgba(255, 107, 176, 0.05) 100%);
        border-left-color: #FF6BB0;
        box-shadow: 0 0 15px rgba(255, 107, 176, 0.2);
    }

    .optimizer-message {
        background: linear-gradient(135deg, rgba(0, 255, 168, 0.1) 0%, rgba(0, 255, 168, 0.05) 100%);
        border-left-color: #00FFA8;
        box-shadow: 0 0 15px rgba(0, 255, 168, 0.2);
    }

    /* Headers */
    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif !important;
        color: #00FFD1 !important;
        text-shadow: 0 0 10px rgba(0, 255, 209, 0.3);
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #00D9C0 0%, #00A896 100%) !important;
        border: 2px solid #00FFD1 !important;
        border-radius: 25px !important;
        color: #0a0e14 !important;
        font-weight: bold !important;
        font-family: 'Orbitron', sans-serif !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        box-shadow: 0 0 15px rgba(0, 217, 192, 0.5) !important;
        transition: all 0.3s ease !important;
    }

    .stButton > button:hover {
        box-shadow: 0 0 30px rgba(0, 255, 209, 0.8) !important;
        transform: translateY(-2px) !important;
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #00FFD1 !important;
        font-family: 'Orbitron', sans-serif !important;
        text-shadow: 0 0 10px rgba(0, 255, 209, 0.5);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0e14 0%, #161b22 100%);
        border-right: 2px solid #00D9C0;
        box-shadow: 5px 0 20px rgba(0, 217, 192, 0.2);
    }

    /* Progress Bars */
    .stProgress > div > div {
        background: linear-gradient(90deg, #00D9C0 0%, #00FFD1 100%) !important;
        box-shadow: 0 0 10px rgba(0, 217, 192, 0.5);
    }
    </style>
""", unsafe_allow_html=True)


def init_agent_system():
    """Initialize or retrieve the agent system"""
    if 'crew_initialized' not in st.session_state:
        st.session_state.crew_initialized = False

    if 'agent_messages' not in st.session_state:
        st.session_state.agent_messages = []

    if 'execution_logs' not in st.session_state:
        st.session_state.execution_logs = []

    return st.session_state.get('crew', None)


def render_agent_status_cards():
    """Render real-time agent status cards"""
    st.markdown("""
    <h2 style="font-family: 'Orbitron', sans-serif; color: #00FFD1; text-align: center;
               font-size: 2rem; text-shadow: 0 0 20px rgba(0, 255, 209, 0.5); margin-bottom: 30px;">
        ⚡ AGENT STATUS MONITOR
    </h2>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    agents = [
        {
            "name": "SCOUT",
            "icon": "🔍",
            "status": st.session_state.get('scout_status', 'idle'),
            "color": "#00D9C0",
            "tasks_completed": 47,
            "current_task": "MONITORING 247 BUILDINGS"
        },
        {
            "name": "ANALYST",
            "icon": "🧠",
            "status": st.session_state.get('analyst_status', 'idle'),
            "color": "#FF6BB0",
            "tasks_completed": 38,
            "current_task": "STANDBY MODE"
        },
        {
            "name": "OPTIMIZER",
            "icon": "⚡",
            "status": st.session_state.get('optimizer_status', 'idle'),
            "color": "#00FFA8",
            "tasks_completed": 29,
            "current_task": "READY TO EXECUTE"
        }
    ]

    for col, agent in zip([col1, col2, col3], agents):
        with col:
            status_emoji = "⚡" if agent["status"] == "active" else "⏸"
            status_class = "agent-status-active" if agent["status"] == "active" else "agent-status-idle"
            status_text = "ACTIVE" if agent["status"] == "active" else "IDLE"

            st.markdown(f"""
            <div class="{status_class}">
                <div style="font-size: 3rem; margin-bottom: 10px;">{agent['icon']}</div>
                <h2 style="font-family: 'Orbitron', sans-serif; font-size: 1.5rem;
                           color: {agent['color']}; margin: 10px 0;">
                    {agent['name']}
                </h2>
                <div style="font-family: 'Orbitron', sans-serif; font-size: 1.2rem;
                            letter-spacing: 3px; margin-top: 10px;">
                    {status_emoji} {status_text}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.metric("TASKS COMPLETED", agent['tasks_completed'])

            if agent['status'] == 'active':
                with st.spinner(f"{agent['current_task']}..."):
                    st.progress(0.7, text=agent['current_task'])
            else:
                st.markdown(f"""
                <div style="padding: 10px; border-radius: 8px;
                            background: rgba(139, 148, 158, 0.1);
                            border: 1px solid #3d444d;
                            text-align: center;
                            font-family: 'Orbitron', sans-serif;
                            font-size: 0.85rem;
                            color: #8B949E;">
                    {agent['current_task']}
                </div>
                """, unsafe_allow_html=True)


def render_agent_conversation():
    """Render agent-to-agent conversation and reasoning"""
    st.markdown("---")
    st.markdown("""
    <h2 style="font-family: 'Orbitron', sans-serif; color: #00FFD1;
               font-size: 1.8rem; text-shadow: 0 0 15px rgba(0, 255, 209, 0.4);">
        💬 AGENT COMMUNICATION LOG
    </h2>
    <p style="color: #8B949E; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px; margin-top: -5px;">
        Real-time inter-agent collaboration and decision-making
    </p>
    """, unsafe_allow_html=True)

    # Sample conversation for demo
    if not st.session_state.agent_messages:
        st.session_state.agent_messages = [
            {
                "agent": "scout",
                "message": "Anomaly detected in Building JLT_42: 45% above spatial baseline",
                "time": "14:34:21",
                "severity": "high"
            },
            {
                "agent": "analyst",
                "message": "Analyzing root cause... Running SHAP analysis on building data",
                "time": "14:35:03",
                "severity": "medium"
            },
            {
                "agent": "analyst",
                "message": "Root cause identified: High solar gain on west facade (38%) + failed shade actuator (22%)",
                "time": "14:35:47",
                "severity": "high"
            },
            {
                "agent": "optimizer",
                "message": "Simulating interventions... Evaluating 3 options",
                "time": "14:36:12",
                "severity": "medium"
            },
            {
                "agent": "optimizer",
                "message": "Decision: Pre-cooling + manual shade override. Projected savings: 127 kWh (48 AED) with 89% confidence",
                "time": "14:36:45",
                "severity": "high"
            },
            {
                "agent": "optimizer",
                "message": "✅ Executing: Pre-cool command sent to BMS. Monitoring for next 30 min",
                "time": "14:37:02",
                "severity": "low"
            },
            {
                "agent": "scout",
                "message": "✓ Confirmed: Building JLT_42 consumption dropping. Now at +12% from baseline",
                "time": "14:42:18",
                "severity": "low"
            }
        ]

    # Chat container with scrollable area
    chat_container = st.container(height=450)

    with chat_container:
        for msg in st.session_state.agent_messages:
            agent_config = {
                "scout": {"icon": "🔍", "name": "SCOUT", "class": "scout-message", "color": "#00D9C0"},
                "analyst": {"icon": "🧠", "name": "ANALYST", "class": "analyst-message", "color": "#FF6BB0"},
                "optimizer": {"icon": "⚡", "name": "OPTIMIZER", "class": "optimizer-message", "color": "#00FFA8"}
            }

            config = agent_config.get(msg["agent"], agent_config["scout"])

            st.markdown(f"""
            <div class="chat-message {config['class']}">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <div style="font-family: 'Orbitron', sans-serif;
                                font-weight: bold;
                                color: {config['color']};
                                font-size: 0.9rem;
                                letter-spacing: 1px;">
                        {config['icon']} {config['name']}
                    </div>
                    <span style="color: #8B949E;
                                 font-size: 0.85em;
                                 font-family: 'Orbitron', sans-serif;
                                 letter-spacing: 1px;">
                        {msg['time']}
                    </span>
                </div>
                <div style="color: #E6F1FF; line-height: 1.6; font-size: 0.95rem;">
                    {msg['message']}
                </div>
            </div>
            """, unsafe_allow_html=True)


def render_autonomous_controls():
    """Control panel for autonomous operations"""
    st.markdown("---")
    st.markdown("""
    <h2 style="font-family: 'Orbitron', sans-serif; color: #00FFD1;
               font-size: 1.8rem; text-shadow: 0 0 15px rgba(0, 255, 209, 0.4);">
        🎮 AUTONOMOUS CONTROL CENTER
    </h2>
    <p style="color: #8B949E; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px; margin-top: -5px;">
        Configure and launch autonomous agent operations
    </p>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("#### Operation Configuration")

        mode = st.selectbox(
            "Operation Mode",
            ["Manual Trigger", "Autonomous Loop", "Single Building Analysis", "Portfolio Optimization"],
            help="Select how agents should operate"
        )

        if mode == "Autonomous Loop":
            interval = st.slider(
                "Monitoring Interval (minutes)",
                min_value=5,
                max_value=60,
                value=15,
                step=5
            )
            st.info(f"✨ Agents will run autonomously every {interval} minutes")
            num_cycles = st.number_input("Number of cycles to run", min_value=1, max_value=10, value=3)

        elif mode == "Single Building Analysis":
            building_id = st.text_input("Building ID", value="Marina_Tower_7")

    with col2:
        st.markdown("#### Monitoring Scope")

        building_groups = st.multiselect(
            "Building Groups",
            ["Marina District", "JLT Cluster", "Business Bay", "Downtown Dubai", "DIFC"],
            default=["Marina District", "JLT Cluster"]
        )

        anomaly_threshold = st.slider(
            "Anomaly Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.3,
            step=0.05,
            help="Minimum anomaly score to trigger analysis"
        )

    # Action buttons
    st.markdown("#### Execute")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("🚀 Start Autonomous Mode", type="primary", use_container_width=True):
            run_autonomous_mode(mode, building_groups)

    with col2:
        if st.button("⏸️ Pause All Agents", use_container_width=True):
            pause_all_agents()

    with col3:
        if st.button("🔄 Reset System", use_container_width=True):
            reset_system()

    with col4:
        if st.button("📊 View Logs", use_container_width=True):
            st.session_state.show_logs = not st.session_state.get('show_logs', False)


def run_autonomous_mode(mode, building_groups):
    """Execute autonomous mode"""
    with st.spinner("Initializing autonomous agent crew..."):
        st.session_state.scout_status = 'active'

        # Check if API key is configured
        api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")

        if not api_key:
            st.error("⚠️ GEMINI_API_KEY not configured! Please set it in .streamlit/secrets.toml")
            st.code("""
# .streamlit/secrets.toml
GEMINI_API_KEY = "your-api-key-here"
            """, language="toml")
            st.session_state.scout_status = 'idle'
            return

        # Initialize crew if not already done
        if not st.session_state.get('crew'):
            try:
                from agents.crew import EnergyManagementCrew
                st.session_state.crew = EnergyManagementCrew(gemini_api_key=api_key)
                st.session_state.crew_initialized = True
                st.success("✅ Agent crew initialized successfully!")
            except Exception as e:
                st.error(f"❌ Failed to initialize crew: {str(e)}")
                st.session_state.scout_status = 'idle'
                return

        # Generate sample building IDs based on selected groups
        building_ids = generate_building_ids(building_groups)

        try:
            # Add a message to the conversation
            new_message = {
                "agent": "scout",
                "message": f"🚀 Starting autonomous monitoring of {len(building_ids)} buildings...",
                "time": datetime.now().strftime("%H:%M:%S"),
                "severity": "medium"
            }
            st.session_state.agent_messages.append(new_message)

            # Run the autonomous loop
            crew = st.session_state.crew

            if mode == "Single Building Analysis":
                building_id = st.session_state.get('selected_building', building_ids[0])
                result = crew.run_single_building_workflow(building_id)
            else:
                result = crew.run_monitoring_cycle(building_ids[:5])  # Limit for demo

            # Add result to conversation
            st.session_state.agent_messages.append({
                "agent": "optimizer",
                "message": f"✅ Cycle complete! Processed {len(building_ids)} buildings.",
                "time": datetime.now().strftime("%H:%M:%S"),
                "severity": "low"
            })

            st.session_state.scout_status = 'idle'
            st.success(f"✅ Autonomous cycle completed successfully!")

            # Display results in expander
            with st.expander("📄 View Detailed Results"):
                st.json(result)

        except Exception as e:
            st.error(f"❌ Error during execution: {str(e)}")
            st.exception(e)
            st.session_state.scout_status = 'idle'


def pause_all_agents():
    """Pause all agents"""
    st.session_state.scout_status = 'idle'
    st.session_state.analyst_status = 'idle'
    st.session_state.optimizer_status = 'idle'
    st.info("⏸️ All agents paused")


def reset_system():
    """Reset the entire system"""
    keys_to_remove = ['crew', 'agent_messages', 'detected_anomalies', 'executed_actions',
                      'scout_status', 'analyst_status', 'optimizer_status']

    for key in keys_to_remove:
        if key in st.session_state:
            del st.session_state[key]

    st.success("🔄 System reset complete")
    st.rerun()


def generate_building_ids(building_groups):
    """Generate sample building IDs based on groups"""
    building_map = {
        "Marina District": ["Marina_Tower_1", "Marina_Tower_2", "Marina_Tower_3", "Marina_Complex_A"],
        "JLT Cluster": ["JLT_Tower_5", "JLT_Tower_7", "JLT_Office_12"],
        "Business Bay": ["BBay_Tower_1", "BBay_Complex_2"],
        "Downtown Dubai": ["Downtown_Tower_A", "Downtown_Burj_View"],
        "DIFC": ["DIFC_Gate_1", "DIFC_Tower_2"]
    }

    building_ids = []
    for group in building_groups:
        building_ids.extend(building_map.get(group, []))

    return building_ids or ["Sample_Building_1", "Sample_Building_2"]


def render_performance_metrics():
    """Show agent performance metrics"""
    st.markdown("---")
    st.markdown("""
    <h2 style="font-family: 'Orbitron', sans-serif; color: #00FFD1;
               font-size: 1.8rem; text-shadow: 0 0 15px rgba(0, 255, 209, 0.4);">
        📊 AGENT PERFORMANCE METRICS
    </h2>
    <p style="color: #8B949E; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px; margin-top: -5px;">
        Real-time system performance and impact tracking
    </p>
    """, unsafe_allow_html=True)

    # Create performance data
    metrics_df = pd.DataFrame({
        'Agent': ['Scout', 'Analyst', 'Optimizer'],
        'Tasks Completed': [47, 38, 29],
        'Success Rate': [94.2, 89.7, 96.3],
        'Avg Response Time (s)': [1.2, 4.7, 2.1]
    })

    col1, col2 = st.columns(2)

    with col1:
        # Tasks completed bar chart
        fig1 = go.Figure()
        fig1.add_trace(go.Bar(
            x=metrics_df['Agent'],
            y=metrics_df['Tasks Completed'],
            marker_color=['#00D9C0', '#FF6BB0', '#00FFA8'],
            text=metrics_df['Tasks Completed'],
            textposition='auto',
            marker=dict(
                line=dict(color='#00FFD1', width=2)
            )
        ))
        fig1.update_layout(
            title={
                'text': "Tasks Completed (Last 24 Hours)",
                'font': {'family': 'Orbitron', 'color': '#00FFD1'}
            },
            xaxis_title="Agent",
            yaxis_title="Tasks",
            height=300,
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(10, 14, 20, 0.5)',
            font=dict(family='Orbitron', color='#E6F1FF')
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # Success rate bar chart
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=metrics_df['Agent'],
            y=metrics_df['Success Rate'],
            marker_color=['#00D9C0', '#FF6BB0', '#00FFA8'],
            text=[f"{x}%" for x in metrics_df['Success Rate']],
            textposition='auto',
            marker=dict(
                line=dict(color='#00FFD1', width=2)
            )
        ))
        fig2.update_layout(
            title={
                'text': "Success Rate (%)",
                'font': {'family': 'Orbitron', 'color': '#00FFD1'}
            },
            xaxis_title="Agent",
            yaxis_title="Success Rate",
            height=300,
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(10, 14, 20, 0.5)',
            font=dict(family='Orbitron', color='#E6F1FF')
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Impact metrics
    st.markdown("#### System Impact")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Energy Saved", "1,247 kWh", "↑ 18%", help="Total energy saved through AI interventions")

    with col2:
        st.metric("Cost Reduced", "473 AED", "↑ 22%", help="Cost savings from energy optimization")

    with col3:
        st.metric("Anomalies Prevented", "7", "↓ 2", help="Anomalies caught and resolved before impact")

    with col4:
        st.metric("System Uptime", "99.8%", "→ 0%", help="Agent system availability")


def main():
    """Main application entry point"""
    st.markdown("""
    <h1 style="font-family: 'Orbitron', sans-serif;
               font-size: 3rem;
               font-weight: 900;
               background: linear-gradient(135deg, #00D9C0 0%, #00FFD1 50%, #00F5FF 100%);
               -webkit-background-clip: text;
               -webkit-text-fill-color: transparent;
               text-shadow: 0 0 30px rgba(0, 217, 192, 0.5);
               text-align: center;
               letter-spacing: 3px;
               margin-bottom: 10px;">
        🤖 AUTONOMOUS AGENT CONTROL CENTER
    </h1>
    """, unsafe_allow_html=True)

    st.markdown("""
    <p style="text-align: center;
              color: #8B949E;
              font-size: 1rem;
              text-transform: uppercase;
              letter-spacing: 2px;
              margin-bottom: 30px;">
        Monitor and Control the Three-Agent Energy Optimization System
    </p>
    """, unsafe_allow_html=True)

    # Initialize system
    init_agent_system()

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📊 Status & Control", "💬 Agent Communication", "📈 Performance"])

    with tab1:
        render_agent_status_cards()
        render_autonomous_controls()

    with tab2:
        render_agent_conversation()

    with tab3:
        render_performance_metrics()

    # Auto-refresh option
    st.markdown("---")
    auto_refresh = st.checkbox("🔄 Enable Auto-Refresh (10 seconds)", value=False)

    if auto_refresh:
        time.sleep(10)
        st.rerun()


if __name__ == "__main__":
    main()
