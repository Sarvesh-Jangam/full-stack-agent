"""
Configuration for Manager Agent
Manages settings and configuration for the multi-agent workflow system.
"""

import os
from typing import Dict, Any, List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field, validator
from datetime import timedelta

class ManagerAgentConfig(BaseSettings):
    """Configuration settings for Manager Agent"""
    
    # Agent Settings
    agent_name: str = Field(
        default="manager_agent",
        env="MANAGER_AGENT_NAME",
        description="Name of the manager agent"
    )
    
    agent_description: str = Field(
        default="Manager agent that orchestrates multi-agent workflows",
        env="MANAGER_AGENT_DESCRIPTION",
        description="Description of the manager agent"
    )
    
    # Workflow Settings
    max_iterations: int = Field(
        default=5,
        env="MAX_WORKFLOW_ITERATIONS",
        description="Maximum number of workflow iterations"
    )
    
    validation_threshold: float = Field(
        default=0.8,
        env="VALIDATION_THRESHOLD",
        description="Minimum validation score to consider workflow complete"
    )
    
    iteration_timeout_minutes: int = Field(
        default=30,
        env="ITERATION_TIMEOUT_MINUTES",
        description="Timeout for each workflow iteration in minutes"
    )
    
    workflow_timeout_hours: int = Field(
        default=4,
        env="WORKFLOW_TIMEOUT_HOURS",
        description="Maximum workflow duration in hours"
    )
    
    # Agent Coordination Settings
    task_timeout_minutes: int = Field(
        default=15,
        env="TASK_TIMEOUT_MINUTES",
        description="Default timeout for individual tasks"
    )
    
    max_parallel_tasks: int = Field(
        default=3,
        env="MAX_PARALLEL_TASKS",
        description="Maximum number of tasks to run in parallel"
    )
    
    task_retry_attempts: int = Field(
        default=2,
        env="TASK_RETRY_ATTEMPTS",
        description="Number of retry attempts for failed tasks"
    )
    
    task_retry_delay_seconds: int = Field(
        default=30,
        env="TASK_RETRY_DELAY_SECONDS",
        description="Delay between task retry attempts"
    )
    
    # Validation Settings
    enable_continuous_validation: bool = Field(
        default=True,
        env="ENABLE_CONTINUOUS_VALIDATION",
        description="Enable validation after each iteration"
    )
    
    validation_strictness: str = Field(
        default="medium",
        env="VALIDATION_STRICTNESS",
        description="Validation strictness level: low, medium, high"
    )
    
    require_user_approval: bool = Field(
        default=False,
        env="REQUIRE_USER_APPROVAL",
        description="Require user approval before proceeding to next iteration"
    )
    
    # Agent Communication Settings
    agent_communication_timeout: int = Field(
        default=60,
        env="AGENT_COMMUNICATION_TIMEOUT",
        description="Timeout for agent-to-agent communication in seconds"
    )
    
    enable_agent_feedback: bool = Field(
        default=True,
        env="ENABLE_AGENT_FEEDBACK",
        description="Enable feedback collection from sub-agents"
    )
    
    # Quality Thresholds
    minimum_frontend_score: float = Field(
        default=0.7,
        env="MINIMUM_FRONTEND_SCORE",
        description="Minimum acceptable frontend quality score"
    )
    
    minimum_backend_score: float = Field(
        default=0.7,
        env="MINIMUM_BACKEND_SCORE",
        description="Minimum acceptable backend quality score"
    )
    
    minimum_integration_score: float = Field(
        default=0.6,
        env="MINIMUM_INTEGRATION_SCORE",
        description="Minimum acceptable integration quality score"
    )
    
    # Resource Management
    memory_limit_mb: int = Field(
        default=1024,
        env="MEMORY_LIMIT_MB",
        description="Memory limit for workflow execution in MB"
    )
    
    disk_space_limit_mb: int = Field(
        default=5120,
        env="DISK_SPACE_LIMIT_MB",
        description="Disk space limit for artifacts in MB"
    )
    
    cleanup_artifacts_after_hours: int = Field(
        default=24,
        env="CLEANUP_ARTIFACTS_AFTER_HOURS",
        description="Hours after which to cleanup old artifacts"
    )
    
    # Logging and Monitoring
    log_level: str = Field(
        default="INFO",
        env="LOG_LEVEL",
        description="Logging level"
    )
    
    enable_metrics: bool = Field(
        default=True,
        env="ENABLE_METRICS",
        description="Enable metrics collection"
    )
    
    metrics_export_interval: int = Field(
        default=300,
        env="METRICS_EXPORT_INTERVAL",
        description="Metrics export interval in seconds"
    )
    
    # Development Settings
    debug_mode: bool = Field(
        default=False,
        env="DEBUG_MODE",
        description="Enable debug mode"
    )
    
    mock_agents: bool = Field(
        default=False,
        env="MOCK_AGENTS",
        description="Use mock agents for testing"
    )
    
    save_intermediate_results: bool = Field(
        default=True,
        env="SAVE_INTERMEDIATE_RESULTS",
        description="Save intermediate workflow results"
    )
    
    class Config:
        env_file = ".env"
        case_sensitive = False
    
    @validator('max_iterations')
    def validate_max_iterations(cls, v):
        if v < 1 or v > 10:
            raise ValueError("max_iterations must be between 1 and 10")
        return v
    
    @validator('validation_threshold')
    def validate_validation_threshold(cls, v):
        if v < 0.0 or v > 1.0:
            raise ValueError("validation_threshold must be between 0.0 and 1.0")
        return v
    
    @validator('validation_strictness')
    def validate_validation_strictness(cls, v):
        if v not in ['low', 'medium', 'high']:
            raise ValueError("validation_strictness must be 'low', 'medium', or 'high'")
        return v
    
    def get_timeout_settings(self) -> Dict[str, timedelta]:
        """Get timeout settings as timedelta objects"""
        return {
            "iteration_timeout": timedelta(minutes=self.iteration_timeout_minutes),
            "workflow_timeout": timedelta(hours=self.workflow_timeout_hours),
            "task_timeout": timedelta(minutes=self.task_timeout_minutes),
            "communication_timeout": timedelta(seconds=self.agent_communication_timeout)
        }
    
    def get_quality_thresholds(self) -> Dict[str, float]:
        """Get quality threshold settings"""
        return {
            "frontend": self.minimum_frontend_score,
            "backend": self.minimum_backend_score,
            "integration": self.minimum_integration_score,
            "overall": self.validation_threshold
        }
    
    def get_resource_limits(self) -> Dict[str, int]:
        """Get resource limit settings"""
        return {
            "memory_mb": self.memory_limit_mb,
            "disk_space_mb": self.disk_space_limit_mb,
            "max_parallel_tasks": self.max_parallel_tasks
        }
    
    def get_retry_settings(self) -> Dict[str, int]:
        """Get retry configuration settings"""
        return {
            "max_attempts": self.task_retry_attempts,
            "delay_seconds": self.task_retry_delay_seconds
        }
    
    def is_development_mode(self) -> bool:
        """Check if running in development mode"""
        return self.debug_mode or self.mock_agents
    
    def get_agent_config(self, agent_name: str) -> Dict[str, Any]:
        """Get configuration specific to a sub-agent"""
        base_config = {
            "timeout_minutes": self.task_timeout_minutes,
            "retry_attempts": self.task_retry_attempts,
            "enable_feedback": self.enable_agent_feedback,
            "debug_mode": self.debug_mode
        }
        
        # Agent-specific configurations
        if agent_name == "frontend":
            base_config.update({
                "minimum_quality_score": self.minimum_frontend_score,
                "enable_responsive_validation": True,
                "enable_accessibility_checks": True
            })
        elif agent_name == "backend":
            base_config.update({
                "minimum_quality_score": self.minimum_backend_score,
                "enable_security_validation": True,
                "enable_performance_checks": True
            })
        elif agent_name == "deployment":
            base_config.update({
                "enable_monitoring_setup": True,
                "enable_security_scanning": True
            })
        
        return base_config
