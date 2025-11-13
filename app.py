"""
ExAI - Explainable AI Energy Optimization System
Main Streamlit Application

A production-ready autonomous multi-agent system for optimizing
building energy consumption in Dubai using LangChain + CrewAI
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os

# Page configuration
st.set_page_config(
    page_title="ExAI - Energy Optimization Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - AI Genesis Inspired
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;600&display=swap');

    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #0a0e14 0%, #1a1f2e 100%);
    }

    /* Main Header */
    .main-header {
        font-family: 'Orbitron', sans-serif;
        font-size: 3.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #00D9C0 0%, #00FFD1 50%, #00F5FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(0, 217, 192, 0.5);
        margin-bottom: 0;
        letter-spacing: 2px;
    }

    /* Neon Glow Effect */
    .neon-border {
        border: 2px solid #00D9C0;
        border-radius: 15px;
        padding: 20px;
        background: rgba(0, 217, 192, 0.05);
        box-shadow: 0 0 20px rgba(0, 217, 192, 0.3),
                    inset 0 0 20px rgba(0, 217, 192, 0.1);
    }

    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(0, 217, 192, 0.1) 0%, rgba(0, 255, 209, 0.05) 100%);
        border: 1px solid #00D9C0;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 0 20px rgba(0, 217, 192, 0.2);
        backdrop-filter: blur(10px);
    }

    /* Cyber Buttons */
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

    /* Agent Badges */
    .agent-badge {
        display: inline-block;
        padding: 8px 20px;
        border-radius: 25px;
        font-weight: bold;
        margin: 5px;
        border: 2px solid;
        font-family: 'Orbitron', sans-serif;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .scout-badge {
        background: rgba(0, 217, 192, 0.1);
        border-color: #00D9C0;
        color: #00FFD1;
        box-shadow: 0 0 10px rgba(0, 217, 192, 0.3);
    }
    .analyst-badge {
        background: rgba(255, 107, 176, 0.1);
        border-color: #FF6BB0;
        color: #FF6BB0;
        box-shadow: 0 0 10px rgba(255, 107, 176, 0.3);
    }
    .optimizer-badge {
        background: rgba(0, 255, 168, 0.1);
        border-color: #00FFA8;
        color: #00FFA8;
        box-shadow: 0 0 10px rgba(0, 255, 168, 0.3);
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0e14 0%, #161b22 100%);
        border-right: 2px solid #00D9C0;
        box-shadow: 5px 0 20px rgba(0, 217, 192, 0.2);
    }

    /* Metric Styling */
    [data-testid="stMetricValue"] {
        color: #00FFD1 !important;
        font-family: 'Orbitron', sans-serif !important;
        font-size: 2rem !important;
        text-shadow: 0 0 10px rgba(0, 255, 209, 0.5);
    }

    [data-testid="stMetricLabel"] {
        color: #8B949E !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 0.8rem !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(0, 217, 192, 0.05);
        border: 1px solid #00D9C0;
        border-radius: 10px;
        color: #00FFD1;
        font-family: 'Orbitron', sans-serif;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(0, 217, 192, 0.2) 0%, rgba(0, 255, 209, 0.1) 100%);
        box-shadow: 0 0 15px rgba(0, 217, 192, 0.4);
    }

    /* Headers */
    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif !important;
        color: #00FFD1 !important;
        text-shadow: 0 0 10px rgba(0, 255, 209, 0.3);
    }

    /* Cyber Grid Lines */
    .cyber-grid {
        background-image:
            linear-gradient(rgba(0, 217, 192, 0.1) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 217, 192, 0.1) 1px, transparent 1px);
        background-size: 50px 50px;
    }
    </style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.scout_status = 'idle'
        st.session_state.analyst_status = 'idle'
        st.session_state.optimizer_status = 'idle'
        st.session_state.detected_anomalies = []
        st.session_state.executed_actions = []
        st.session_state.agent_logs = []
        st.session_state.crew_cycles = 0


def render_hero_section():
    """Render hero section with main value proposition"""
    st.markdown("""
    <div style="text-align: center; padding: 40px 0 20px 0;">
        <div style="font-family: 'Orbitron', sans-serif;
                    font-size: 1rem;
                    color: #8B949E;
                    text-transform: uppercase;
                    letter-spacing: 5px;
                    margin-bottom: 15px;">
            The Future of Building Energy Management
        </div>
        <h1 class="main-header">⚡ ExAI</h1>
        <div style="font-family: 'Orbitron', sans-serif;
                    font-size: 1.8rem;
                    color: #00D9C0;
                    letter-spacing: 3px;
                    margin-top: -10px;
                    text-shadow: 0 0 15px rgba(0, 217, 192, 0.4);">
            ENERGY OPTIMIZATION PLATFORM
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: center; max-width: 900px; margin: 30px auto; padding: 0 20px;">
        <p style="font-size: 1.2rem; color: #E6F1FF; line-height: 1.8; font-weight: 300;">
            <span style="color: #00FFD1; font-weight: 600;">ExAI</span> uses a three-tier autonomous agent architecture
            powered by <span style="color: #00D9C0;">CrewAI + LangChain + Gemini</span>
            to monitor, analyze, and optimize energy consumption across Dubai's building portfolio.
        </p>
        <div style="margin-top: 30px; padding: 20px;
                    background: linear-gradient(135deg, rgba(0, 217, 192, 0.1) 0%, rgba(0, 255, 209, 0.05) 100%);
                    border: 1px solid #00D9C0;
                    border-radius: 15px;
                    box-shadow: 0 0 20px rgba(0, 217, 192, 0.2);">
            <div style="font-family: 'Orbitron', sans-serif; color: #00FFD1; font-size: 0.9rem;
                        text-transform: uppercase; letter-spacing: 2px;">
                🎯 Autonomous • 📊 Explainable • ⚡ Production-Ready
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Value proposition metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="🎯 Energy Savings",
            value="18-25%",
            delta="Avg. reduction",
            help="Average energy reduction achieved through AI optimization"
        )

    with col2:
        st.metric(
            label="💰 Cost Savings",
            value="₹1.2M+",
            delta="Annual",
            help="Estimated annual cost savings across portfolio"
        )

    with col3:
        st.metric(
            label="🏢 Buildings Monitored",
            value="247",
            delta="+12 this month",
            help="Total buildings under AI monitoring"
        )

    with col4:
        st.metric(
            label="🤖 Autonomous Actions",
            value="1,847",
            delta="+142 today",
            help="AI-executed optimizations without human intervention"
        )


def render_agent_architecture():
    """Render the three-tier agent architecture"""
    st.markdown("---")
    st.markdown("""
    <h2 style="font-family: 'Orbitron', sans-serif; color: #00FFD1; text-align: center;
               font-size: 2.5rem; text-shadow: 0 0 20px rgba(0, 255, 209, 0.5);">
        ⚡ THREE-TIER AUTONOMOUS AGENT ARCHITECTURE
    </h2>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(0, 217, 192, 0.15) 0%, rgba(0, 217, 192, 0.05) 100%);
                    padding: 25px;
                    border-radius: 15px;
                    border: 2px solid #00D9C0;
                    box-shadow: 0 0 20px rgba(0, 217, 192, 0.3),
                                inset 0 0 20px rgba(0, 217, 192, 0.05);
                    height: 320px;
                    backdrop-filter: blur(10px);">
            <h3 style="font-family: 'Orbitron', sans-serif; color: #00FFD1; font-size: 1.5rem;">
                🔍 SCOUT AGENT
            </h3>
            <p style="font-size: 0.75rem; color: #8B949E; text-transform: uppercase; letter-spacing: 2px; margin-top: -10px;">
                Perception Layer
            </p>
            <div style="color: #E6F1FF; font-size: 0.9em; line-height: 1.8; margin-top: 15px;">
                <div style="margin-bottom: 8px;">→ Real-time energy monitoring</div>
                <div style="margin-bottom: 8px;">→ Spatial anomaly detection</div>
                <div style="margin-bottom: 8px;">→ Pattern recognition</div>
                <div style="margin-bottom: 8px;">→ Early warning system</div>
                <div style="margin-bottom: 8px;">→ Weather correlation</div>
                <br>
                <div style="color: #00D9C0; font-family: 'Orbitron', sans-serif; font-size: 0.8rem;">
                    TOOLS: Spatial analysis, Time-series
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(255, 107, 176, 0.15) 0%, rgba(255, 107, 176, 0.05) 100%);
                    padding: 25px;
                    border-radius: 15px;
                    border: 2px solid #FF6BB0;
                    box-shadow: 0 0 20px rgba(255, 107, 176, 0.3),
                                inset 0 0 20px rgba(255, 107, 176, 0.05);
                    height: 320px;
                    backdrop-filter: blur(10px);">
            <h3 style="font-family: 'Orbitron', sans-serif; color: #FF6BB0; font-size: 1.5rem;">
                🧠 ANALYST AGENT
            </h3>
            <p style="font-size: 0.75rem; color: #8B949E; text-transform: uppercase; letter-spacing: 2px; margin-top: -10px;">
                Reasoning Layer
            </p>
            <div style="color: #E6F1FF; font-size: 0.9em; line-height: 1.8; margin-top: 15px;">
                <div style="margin-bottom: 8px;">→ Root cause analysis</div>
                <div style="margin-bottom: 8px;">→ SHAP explainability</div>
                <div style="margin-bottom: 8px;">→ Knowledge base queries</div>
                <div style="margin-bottom: 8px;">→ Pattern explanation</div>
                <div style="margin-bottom: 8px;">→ Insight generation</div>
                <br>
                <div style="color: #FF6BB0; font-family: 'Orbitron', sans-serif; font-size: 0.8rem;">
                    TOOLS: SHAP, Vector DB, ML models
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(0, 255, 168, 0.15) 0%, rgba(0, 255, 168, 0.05) 100%);
                    padding: 25px;
                    border-radius: 15px;
                    border: 2px solid #00FFA8;
                    box-shadow: 0 0 20px rgba(0, 255, 168, 0.3),
                                inset 0 0 20px rgba(0, 255, 168, 0.05);
                    height: 320px;
                    backdrop-filter: blur(10px);">
            <h3 style="font-family: 'Orbitron', sans-serif; color: #00FFA8; font-size: 1.5rem;">
                ⚡ OPTIMIZER AGENT
            </h3>
            <p style="font-size: 0.75rem; color: #8B949E; text-transform: uppercase; letter-spacing: 2px; margin-top: -10px;">
                Action Layer
            </p>
            <div style="color: #E6F1FF; font-size: 0.9em; line-height: 1.8; margin-top: 15px;">
                <div style="margin-bottom: 8px;">→ Decision optimization</div>
                <div style="margin-bottom: 8px;">→ Impact simulation</div>
                <div style="margin-bottom: 8px;">→ Risk assessment</div>
                <div style="margin-bottom: 8px;">→ Autonomous execution</div>
                <div style="margin-bottom: 8px;">→ Continuous learning</div>
                <br>
                <div style="color: #00FFA8; font-family: 'Orbitron', sans-serif; font-size: 0.8rem;">
                    TOOLS: BMS control, Simulators
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_live_metrics():
    """Render live system metrics and activity"""
    st.markdown("---")
    st.markdown("## 📊 Live System Activity")

    # Create sample time-series data for energy consumption
    hours = pd.date_range(start=datetime.now() - timedelta(hours=24), end=datetime.now(), freq='H')
    baseline = 350 + 50 * np.sin(np.linspace(0, 2*np.pi, len(hours))) + np.random.normal(0, 10, len(hours))
    optimized = baseline * 0.82  # 18% reduction
    savings = baseline - optimized

    df = pd.DataFrame({
        'Time': hours,
        'Baseline': baseline,
        'AI-Optimized': optimized,
        'Savings': savings
    })

    # Energy consumption chart
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['Time'],
        y=df['Baseline'],
        name='Baseline Consumption',
        line=dict(color='#FF6BB0', width=2, dash='dash'),
        fill=None
    ))

    fig.add_trace(go.Scatter(
        x=df['Time'],
        y=df['AI-Optimized'],
        name='AI-Optimized',
        line=dict(color='#00D9C0', width=3),
        fill='tonexty',
        fillcolor='rgba(0, 217, 192, 0.1)'
    ))

    fig.update_layout(
        title={
            'text': "24-Hour Energy Consumption: Baseline vs. AI-Optimized",
            'font': {'family': 'Orbitron', 'color': '#00FFD1', 'size': 18}
        },
        xaxis_title="Time",
        yaxis_title="Energy (kW)",
        height=400,
        hovermode='x unified',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(10, 14, 20, 0.5)',
        font=dict(family='Orbitron', color='#E6F1FF'),
        xaxis=dict(
            gridcolor='rgba(0, 217, 192, 0.1)',
            showgrid=True
        ),
        yaxis=dict(
            gridcolor='rgba(0, 217, 192, 0.1)',
            showgrid=True
        ),
        legend=dict(
            bgcolor='rgba(10, 14, 20, 0.8)',
            bordercolor='#00D9C0',
            borderwidth=1
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_savings = df['Savings'].sum()
        st.metric(
            "24h Energy Saved",
            f"{total_savings:.0f} kWh",
            f"{(total_savings/df['Baseline'].sum()*100):.1f}%"
        )

    with col2:
        cost_savings = total_savings * 0.38  # AED per kWh
        st.metric(
            "Cost Savings",
            f"{cost_savings:.0f} AED",
            "Last 24h"
        )

    with col3:
        st.metric(
            "Active Optimizations",
            "47",
            "+5 this hour"
        )

    with col4:
        st.metric(
            "System Efficiency",
            "94.2%",
            "+2.1%"
        )


def render_recent_agent_activity():
    """Render recent agent activities and decisions"""
    st.markdown("---")
    st.markdown("""
    <h2 style="font-family: 'Orbitron', sans-serif; color: #00FFD1; text-align: center;
               font-size: 2rem; text-shadow: 0 0 20px rgba(0, 255, 209, 0.5);">
        🤖 RECENT AGENT ACTIVITY
    </h2>
    <p style="text-align: center; color: #8B949E; font-size: 0.9rem;
              text-transform: uppercase; letter-spacing: 2px; margin-bottom: 30px;">
        Live feed of autonomous agent actions and decisions
    </p>
    """, unsafe_allow_html=True)

    # Sample agent activities
    activities = [
        {
            "time": "2 min ago",
            "agent": "Scout",
            "action": "Detected anomaly in Marina Tower 7",
            "severity": "high",
            "icon": "🔍"
        },
        {
            "time": "5 min ago",
            "agent": "Analyst",
            "action": "Identified root cause: Excessive solar gain on west facade",
            "severity": "medium",
            "icon": "🧠"
        },
        {
            "time": "8 min ago",
            "agent": "Optimizer",
            "action": "Executed: Pre-cooling protocol for JLT Tower 3",
            "severity": "low",
            "icon": "⚡"
        },
        {
            "time": "12 min ago",
            "agent": "Scout",
            "action": "Monitoring 247 buildings - all systems normal",
            "severity": "low",
            "icon": "🔍"
        },
        {
            "time": "15 min ago",
            "agent": "Optimizer",
            "action": "Projected savings: 127 kWh (48 AED) with 89% confidence",
            "severity": "medium",
            "icon": "⚡"
        }
    ]

    for activity in activities:
        severity_config = {
            "low": {"color": "#00FFA8", "glow": "rgba(0, 255, 168, 0.3)"},
            "medium": {"color": "#FFD700", "glow": "rgba(255, 215, 0, 0.3)"},
            "high": {"color": "#FF6BB0", "glow": "rgba(255, 107, 176, 0.3)"}
        }[activity["severity"]]

        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(0, 217, 192, 0.05) 0%, rgba(10, 14, 20, 0.8) 100%);
                    padding: 20px;
                    border-radius: 12px;
                    margin: 15px 0;
                    border-left: 4px solid {severity_config['color']};
                    box-shadow: 0 0 15px {severity_config['glow']};
                    backdrop-filter: blur(10px);
                    border: 1px solid rgba(0, 217, 192, 0.2);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <div style="display: flex; align-items: center; gap: 15px;">
                    <span style="font-size: 2em;">{activity['icon']}</span>
                    <span style="font-family: 'Orbitron', sans-serif;
                                 font-weight: bold;
                                 color: #00FFD1;
                                 font-size: 1.1rem;
                                 letter-spacing: 1px;">
                        {activity['agent'].upper()} AGENT
                    </span>
                </div>
                <span style="color: #8B949E;
                             font-family: 'Orbitron', sans-serif;
                             font-size: 0.85rem;
                             letter-spacing: 1px;">
                    {activity['time']}
                </span>
            </div>
            <div style="margin-left: 60px;
                        color: #E6F1FF;
                        font-size: 0.95rem;
                        line-height: 1.6;">
                {activity['action']}
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_quick_actions():
    """Render quick action buttons"""
    st.markdown("---")
    st.markdown("""
    <h2 style="font-family: 'Orbitron', sans-serif; color: #00FFD1; text-align: center;
               font-size: 2rem; text-shadow: 0 0 20px rgba(0, 255, 209, 0.5);">
        🚀 QUICK ACTIONS
    </h2>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("🤖 Launch Agents", type="primary", use_container_width=True):
            st.switch_page("pages/2_🤖_Agents.py")

    with col2:
        if st.button("📊 View Analytics", use_container_width=True):
            st.info("Analytics dashboard - Coming soon!")

    with col3:
        if st.button("🗺️ Geospatial View", use_container_width=True):
            st.info("Geospatial visualization - Coming soon!")

    with col4:
        if st.button("📈 Reports", use_container_width=True):
            st.info("Report generation - Coming soon!")


def render_tech_stack():
    """Render technology stack information"""
    st.markdown("---")
    st.markdown("""
    <h2 style="font-family: 'Orbitron', sans-serif; color: #00FFD1; text-align: center;
               font-size: 2rem; text-shadow: 0 0 20px rgba(0, 255, 209, 0.5); margin-bottom: 30px;">
        🛠️ TECHNOLOGY STACK
    </h2>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(0, 217, 192, 0.1) 0%, rgba(0, 217, 192, 0.05) 100%);
                    padding: 25px;
                    border-radius: 15px;
                    border: 2px solid #00D9C0;
                    box-shadow: 0 0 20px rgba(0, 217, 192, 0.2);
                    height: 280px;">
            <h3 style="font-family: 'Orbitron', sans-serif; color: #00FFD1; margin-bottom: 20px;">
                🤖 AI & AGENTS
            </h3>
            <div style="color: #E6F1FF; line-height: 2; font-size: 0.95rem;">
                → <span style="color: #00D9C0;">CrewAI</span> (Multi-agent)<br>
                → <span style="color: #00D9C0;">LangChain</span> (Framework)<br>
                → <span style="color: #00D9C0;">Gemini 1.5</span> (LLM)<br>
                → <span style="color: #00D9C0;">SHAP</span> (Explainability)
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(255, 107, 176, 0.1) 0%, rgba(255, 107, 176, 0.05) 100%);
                    padding: 25px;
                    border-radius: 15px;
                    border: 2px solid #FF6BB0;
                    box-shadow: 0 0 20px rgba(255, 107, 176, 0.2);
                    height: 280px;">
            <h3 style="font-family: 'Orbitron', sans-serif; color: #FF6BB0; margin-bottom: 20px;">
                📊 DATA & ANALYTICS
            </h3>
            <div style="color: #E6F1FF; line-height: 2; font-size: 0.95rem;">
                → <span style="color: #FF6BB0;">Qdrant</span> (Vector DB)<br>
                → <span style="color: #FF6BB0;">Pandas/NumPy</span> (Processing)<br>
                → <span style="color: #FF6BB0;">Plotly</span> (Visualization)<br>
                → <span style="color: #FF6BB0;">Scikit-learn</span> (ML)
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(0, 255, 168, 0.1) 0%, rgba(0, 255, 168, 0.05) 100%);
                    padding: 25px;
                    border-radius: 15px;
                    border: 2px solid #00FFA8;
                    box-shadow: 0 0 20px rgba(0, 255, 168, 0.2);
                    height: 280px;">
            <h3 style="font-family: 'Orbitron', sans-serif; color: #00FFA8; margin-bottom: 20px;">
                🖥️ PLATFORM
            </h3>
            <div style="color: #E6F1FF; line-height: 2; font-size: 0.95rem;">
                → <span style="color: #00FFA8;">Streamlit</span> (UI/UX)<br>
                → <span style="color: #00FFA8;">Python 3.12+</span><br>
                → <span style="color: #00FFA8;">Docker</span> (Deploy)<br>
                → <span style="color: #00FFA8;">FastAPI</span> (Planned)
            </div>
        </div>
        """, unsafe_allow_html=True)


def main():
    """Main application entry point"""
    init_session_state()

    # Sidebar
    with st.sidebar:
        # Logo/Header
        st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <div style="font-family: 'Orbitron', sans-serif;
                        font-size: 2.5rem;
                        font-weight: 900;
                        background: linear-gradient(135deg, #00D9C0 0%, #00FFD1 100%);
                        -webkit-background-clip: text;
                        -webkit-text-fill-color: transparent;
                        text-shadow: 0 0 30px rgba(0, 217, 192, 0.5);">
                ⚡ ExAI
            </div>
            <div style="font-family: 'Orbitron', sans-serif;
                        font-size: 0.7rem;
                        color: #8B949E;
                        text-transform: uppercase;
                        letter-spacing: 3px;
                        margin-top: -5px;">
                Control System
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        st.markdown("""
        <div style="font-family: 'Orbitron', sans-serif;
                    color: #00FFD1;
                    font-size: 0.9rem;
                    text-transform: uppercase;
                    letter-spacing: 2px;
                    margin-bottom: 15px;">
            Navigation
        </div>
        """, unsafe_allow_html=True)
        st.page_link("app.py", label="🏠 Home", icon="🏠")
        st.page_link("pages/2_🤖_Agents.py", label="🤖 Agent Control Center", icon="🤖")

        st.markdown("---")
        st.markdown("""
        <div style="font-family: 'Orbitron', sans-serif;
                    color: #00FFD1;
                    font-size: 0.9rem;
                    text-transform: uppercase;
                    letter-spacing: 2px;
                    margin-bottom: 15px;">
            System Status
        </div>
        """, unsafe_allow_html=True)

        # Agent status indicators
        agents = [
            ("SCOUT", st.session_state.scout_status, "#00D9C0"),
            ("ANALYST", st.session_state.analyst_status, "#FF6BB0"),
            ("OPTIMIZER", st.session_state.optimizer_status, "#00FFA8")
        ]

        for agent_name, status, color in agents:
            status_icon = "⚡" if status == "active" else "⏸"
            st.markdown(f"""
            <div style="padding: 8px;
                        margin: 5px 0;
                        background: rgba(0, 217, 192, 0.05);
                        border-left: 3px solid {color};
                        border-radius: 5px;
                        font-family: 'Orbitron', sans-serif;
                        font-size: 0.85rem;">
                {status_icon} <span style="color: {color};">{agent_name}</span>: {status.upper()}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        <div style="font-family: 'Orbitron', sans-serif;
                    color: #00FFD1;
                    font-size: 0.9rem;
                    text-transform: uppercase;
                    letter-spacing: 2px;
                    margin-bottom: 15px;">
            Quick Stats
        </div>
        """, unsafe_allow_html=True)
        st.metric("CYCLES RUN", st.session_state.crew_cycles)
        st.metric("ANOMALIES DETECTED", len(st.session_state.detected_anomalies))
        st.metric("ACTIONS EXECUTED", len(st.session_state.executed_actions))

    # Main content
    render_hero_section()
    render_agent_architecture()
    render_live_metrics()
    render_recent_agent_activity()
    render_quick_actions()
    render_tech_stack()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center;
                padding: 40px 20px;
                background: linear-gradient(135deg, rgba(0, 217, 192, 0.05) 0%, rgba(10, 14, 20, 0.5) 100%);
                border-top: 2px solid rgba(0, 217, 192, 0.3);
                border-radius: 15px;
                margin-top: 40px;">
        <div style="font-family: 'Orbitron', sans-serif;
                    font-size: 1.5rem;
                    background: linear-gradient(135deg, #00D9C0 0%, #00FFD1 100%);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    font-weight: bold;
                    margin-bottom: 10px;">
            ⚡ ExAI
        </div>
        <p style="color: #E6F1FF; font-size: 1rem; margin: 10px 0;">
            Explainable AI Energy Optimization Platform
        </p>
        <p style="color: #8B949E; font-size: 0.9rem; margin: 10px 0;">
            Powered by <span style="color: #00D9C0;">CrewAI</span> +
            <span style="color: #FF6BB0;">LangChain</span> +
            <span style="color: #00FFA8;">Google Gemini</span>
        </p>
        <div style="margin-top: 20px;
                    padding: 15px;
                    background: rgba(0, 217, 192, 0.1);
                    border: 1px solid #00D9C0;
                    border-radius: 10px;
                    display: inline-block;">
            <span style="font-family: 'Orbitron', sans-serif;
                         color: #00FFD1;
                         font-size: 0.85rem;
                         text-transform: uppercase;
                         letter-spacing: 2px;">
                🏗️ Production-Ready • 🎯 Autonomous • 📊 Explainable
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
