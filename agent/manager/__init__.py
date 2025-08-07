"""
Manager Agent Package
Orchestrates workflow between frontend, backend, and deployment agents.
"""

from .agent import manager_agent, ManagerAgentRunner, ManagerRequest, ManagerResponse
from .workflow_manager import WorkflowManager
from .task_analyzer import TaskAnalyzer
from .coordination_tools import CoordinationTool
from .validation_tools import ValidationTool
from .config import ManagerAgentConfig

__version__ = "1.0.0"
__author__ = "Manager Agent Development Team"
__description__ = "Manager Agent for orchestrating multi-agent workflows"

# Export root agent for ADK
root_agent = manager_agent

__all__ = [
    "manager_agent",
    "root_agent", 
    "ManagerAgentRunner",
    "ManagerRequest",
    "ManagerResponse",
    "WorkflowManager",
    "TaskAnalyzer",
    "CoordinationTool",
    "ValidationTool",
    "ManagerAgentConfig"
]

