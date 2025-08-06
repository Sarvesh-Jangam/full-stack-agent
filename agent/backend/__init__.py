"""
Backend Agent Package for, API, authentication, and file processing capabilities.
"""

from .agent import backend_agent, BackendAgentRunner
from .tools.database_tools import DatabaseTool
from .tools.api_tools import APIIntegrationTool
from .tools.auth_tools import AuthenticationTool
from .tools.file_tools import FileProcessingTool

__version__ = "1.0.0"
__author__ = "Backend Agent Development Team"
__description__ = "Backend Agent for Google ADK with comprehensive server-side capabilities"

# Export main components
__all__ = [
    "backend_agent",
    "BackendAgentRunner", 
    "DatabaseTool",
    "APIIntegrationTool",
    "AuthenticationTool",
    "FileProcessingTool"
]

# Package-level configuration
PACKAGE_NAME = "backend_agent"
DEFAULT_MODEL = "gemini-2.0-flash"

