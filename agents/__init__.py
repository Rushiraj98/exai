"""
Autonomous Multi-Agent Energy Optimization System

Three-tier autonomous agent architecture:
1. Perception Layer (Scout) - Monitors and detects anomalies
2. Reasoning Layer (Analyst) - Understands and explains patterns
3. Action Layer (Optimizer) - Decides and executes interventions
"""

from .tools import EnergyTools
from .scout import EnergyScout
from .analyst import EnergyAnalyst
from .optimizer import EnergyOptimizer
from .crew import EnergyManagementCrew

__all__ = [
    'EnergyTools',
    'EnergyScout',
    'EnergyAnalyst',
    'EnergyOptimizer',
    'EnergyManagementCrew'
]

__version__ = '0.1.0'
