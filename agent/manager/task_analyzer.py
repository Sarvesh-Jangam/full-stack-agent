"""
Task Analyzer for Manager Agent
Analyzes user requirements and determines which agents need to be involved.
"""

import logging
import re
from typing import Dict, Any, List, Optional, Set
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
from .config import ManagerAgentConfig
logger = logging.getLogger(__name__)

class TaskType(str, Enum):
    FRONTEND_ONLY = "frontend_only"
    BACKEND_ONLY = "backend_only"
    DEPLOYMENT_ONLY = "deployment_only"
    FRONTEND_BACKEND = "frontend_backend"
    FULL_STACK = "full_stack"
    MAINTENANCE = "maintenance"
    ANALYSIS = "analysis"

class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

class TaskRequirement(BaseModel):
    """Individual task requirement"""
    requirement_id: str = Field(..., description="Unique requirement ID")
    description: str = Field(..., description="Requirement description")
    task_type: TaskType = Field(..., description="Type of task")
    priority: Priority = Field(..., description="Task priority")
    agents_required: List[str] = Field(..., description="List of agents required")
    estimated_complexity: int = Field(..., description="Complexity score 1-10")
    dependencies: List[str] = Field(default_factory=list, description="Dependent requirement IDs")

class AnalysisResult(BaseModel):
    """Result of task analysis"""
    user_intent: str = Field(..., description="Interpreted user intent")
    project_type: str = Field(..., description="Type of project")
    requirements: List[TaskRequirement] = Field(..., description="Breakdown of requirements")
    workflow_stages: List[str] = Field(..., description="Ordered workflow stages")
    estimated_duration: str = Field(..., description="Estimated completion time")
    success_criteria: List[str] = Field(..., description="Success criteria")

class TaskAnalyzer:
    """Analyzes high-level tasks and breaks them down into smaller steps."""

    def __init__(self, config: ManagerAgentConfig):
        """
        Initializes the TaskAnalyzer with a configuration.

        Args:
            config: The configuration object for the manager agent.
        """
        self.config = config
        logger.info("Task Analyzer initialized")

    async def analyze_task_requirements(self, task_description: str) -> Dict[str, Any]:
        """
        Analyzes the task description and breaks it down.
        In a real scenario, this would involve an LLM call.
        """
        logger.info(f"Analyzing task: {task_description}")

        # Mock analysis based on keywords
        if "frontend" in task_description.lower() and "backend" in task_description.lower():
            plan = {
                "steps": [
                    {"step": 1, "agent": "backend", "task": "Design database schema"},
                    {"step": 2, "agent": "backend", "task": "Create API endpoints"},
                    {"step": 3, "agent": "frontend", "task": "Build UI components"},
                    {"step": 4, "agent": "frontend", "task": "Connect UI to backend API"},
                    {"step": 5, "agent": "manager", "task": "Run integration tests"}
                ],
                "artifacts": ["database_schema.sql", "api_spec.yaml", "component_library.js", "final_app.zip"],
                "dependencies": {
                    "3": [1, 2],
                    "4": [2, 3],
                    "5": [4]
                }
            }
        elif "frontend" in task_description.lower():
            plan = {
                "steps": [
                    {"step": 1, "agent": "frontend", "task": "Build UI components"},
                    {"step": 2, "agent": "frontend", "task": "Implement UI logic"}
                ],
                "artifacts": ["ui_components.js", "app.js"],
                "dependencies": {}
            }
        else:
            plan = {
                "steps": [{"step": 1, "agent": "backend", "task": "Perform backend task"}],
                "artifacts": ["backend_result.json"],
                "dependencies": {}
            }

        return {
            "status": "success",
            "plan": plan,
            "message": "Task analysis complete."
        }

    def _preprocess_input(self, user_input: str) -> str:
        """Clean and preprocess user input"""
        # Remove extra whitespace, normalize case
        cleaned = ' '.join(user_input.strip().split())
        return cleaned.lower()

    def _extract_user_intent(self, input_text: str) -> str:
        """Extract the main user intent from input"""
        intent_patterns = {
            "create_application": r"(create|build|develop|make).*(app|application|system|platform)",
            "add_feature": r"(add|implement|include).*(feature|functionality|capability)",
            "fix_issue": r"(fix|resolve|debug|solve).*(issue|problem|bug|error)",
            "improve_performance": r"(improve|optimize|enhance|speed up).*(performance|speed)",
            "deploy_application": r"(deploy|host|publish|launch).*(app|application|site)",
            "design_interface": r"(design|create|build).*(ui|interface|frontend|page)",
            "setup_backend": r"(setup|create|build).*(backend|api|server|database)",
            "integrate_services": r"(integrate|connect|link).*(service|api|system)"
        }
        
        for intent, pattern in intent_patterns.items():
            if re.search(pattern, input_text):
                return intent.replace('_', ' ').title()
        
        return "General Development Task"

    def _determine_project_type(self, input_text: str) -> str:
        """Determine the type of project based on keywords"""
        project_types = {
            "E-commerce": ["shop", "store", "cart", "payment", "product", "order"],
            "Social Media": ["social", "post", "follow", "feed", "chat", "message"],
            "Dashboard": ["dashboard", "analytics", "chart", "graph", "report", "metric"],
            "Blog/CMS": ["blog", "article", "content", "cms", "publish", "editor"],
            "Authentication System": ["login", "register", "auth", "user", "profile"],
            "API Service": ["api", "endpoint", "service", "integration", "webhook"],
            "Landing Page": ["landing", "marketing", "showcase", "portfolio"],
            "Web Application": ["webapp", "application", "interactive", "dynamic"]
        }
        
        for project_type, keywords in project_types.items():
            if any(keyword in input_text for keyword in keywords):
                return project_type
        
        return "Custom Web Application"

    def _identify_required_agents(self, input_text: str) -> Set[str]:
        """Identify which agents are needed based on requirements"""
        required_agents = set()
        
        # Check for frontend requirements
        if any(keyword in input_text for keyword in self.frontend_keywords):
            required_agents.add("frontend")
        
        # Check for backend requirements
        if any(keyword in input_text for keyword in self.backend_keywords):
            required_agents.add("backend")
        
        # Check for deployment requirements
        if any(keyword in input_text for keyword in self.deployment_keywords):
            required_agents.add("deployment")
        
        # If no specific agents identified, default to full stack
        if not required_agents:
            required_agents = {"frontend", "backend", "deployment"}
        
        return required_agents

    def _break_down_requirements(self, input_text: str, required_agents: Set[str]) -> List[TaskRequirement]:
        """Break down user input into specific requirements"""
        requirements = []
        req_id_counter = 1
        
        # Frontend requirements
        if "frontend" in required_agents:
            frontend_reqs = self._extract_frontend_requirements(input_text)
            for req in frontend_reqs:
                requirements.append(TaskRequirement(
                    requirement_id=f"FRONTEND_{req_id_counter:03d}",
                    description=req["description"],
                    task_type=TaskType.FRONTEND_ONLY,
                    priority=req.get("priority", Priority.MEDIUM),
                    agents_required=["frontend"],
                    estimated_complexity=req.get("complexity", 5),
                    dependencies=req.get("dependencies", [])
                ))
                req_id_counter += 1
        
        # Backend requirements
        if "backend" in required_agents:
            backend_reqs = self._extract_backend_requirements(input_text)
            for req in backend_reqs:
                requirements.append(TaskRequirement(
                    requirement_id=f"BACKEND_{req_id_counter:03d}",
                    description=req["description"],
                    task_type=TaskType.BACKEND_ONLY,
                    priority=req.get("priority", Priority.MEDIUM),
                    agents_required=["backend"],
                    estimated_complexity=req.get("complexity", 6),
                    dependencies=req.get("dependencies", [])
                ))
                req_id_counter += 1
        
        # Deployment requirements
        if "deployment" in required_agents:
            deployment_reqs = self._extract_deployment_requirements(input_text)
            for req in deployment_reqs:
                requirements.append(TaskRequirement(
                    requirement_id=f"DEPLOY_{req_id_counter:03d}",
                    description=req["description"],
                    task_type=TaskType.DEPLOYMENT_ONLY,
                    priority=req.get("priority", Priority.LOW),
                    agents_required=["deployment"],
                    estimated_complexity=req.get("complexity", 4),
                    dependencies=req.get("dependencies", [])
                ))
                req_id_counter += 1
        
        # Integration requirements (multi-agent)
        if len(required_agents) > 1:
            integration_reqs = self._extract_integration_requirements(input_text, required_agents)
            for req in integration_reqs:
                requirements.append(TaskRequirement(
                    requirement_id=f"INTEGRATION_{req_id_counter:03d}",
                    description=req["description"],
                    task_type=TaskType.FULL_STACK,
                    priority=req.get("priority", Priority.HIGH),
                    agents_required=list(required_agents),
                    estimated_complexity=req.get("complexity", 8),
                    dependencies=req.get("dependencies", [])
                ))
                req_id_counter += 1
        
        return requirements

    def _extract_frontend_requirements(self, input_text: str) -> List[Dict[str, Any]]:
        """Extract frontend-specific requirements"""
        requirements = []
        
        # UI/Interface requirements
        if any(word in input_text for word in ["ui", "interface", "design", "layout"]):
            requirements.append({
                "description": "Design and implement user interface components",
                "priority": Priority.HIGH,
                "complexity": 6
            })
        
        # Responsive design
        if any(word in input_text for word in ["responsive", "mobile", "tablet"]):
            requirements.append({
                "description": "Implement responsive design for all screen sizes",
                "priority": Priority.MEDIUM,
                "complexity": 5
            })
        
        # Forms and user input
        if any(word in input_text for word in ["form", "input", "submit", "validation"]):
            requirements.append({
                "description": "Create interactive forms with client-side validation",
                "priority": Priority.MEDIUM,
                "complexity": 4
            })
        
        # Dashboard/Analytics
        if any(word in input_text for word in ["dashboard", "chart", "graph", "analytics"]):
            requirements.append({
                "description": "Build interactive dashboard with data visualization",
                "priority": Priority.HIGH,
                "complexity": 8
            })
        
        # Default frontend requirement if none specific
        if not requirements:
            requirements.append({
                "description": "Create modern, responsive web interface",
                "priority": Priority.MEDIUM,
                "complexity": 5
            })
        
        return requirements

    def _extract_backend_requirements(self, input_text: str) -> List[Dict[str, Any]]:
        """Extract backend-specific requirements"""
        requirements = []
        
        # API development
        if any(word in input_text for word in ["api", "endpoint", "rest", "graphql"]):
            requirements.append({
                "description": "Develop RESTful API endpoints with proper documentation",
                "priority": Priority.HIGH,
                "complexity": 7
            })
        
        # Database operations
        if any(word in input_text for word in ["database", "data", "storage", "crud"]):
            requirements.append({
                "description": "Set up database schema and implement CRUD operations",
                "priority": Priority.HIGH,
                "complexity": 6
            })
        
        # Authentication
        if any(word in input_text for word in ["auth", "login", "register", "jwt", "oauth"]):
            requirements.append({
                "description": "Implement secure authentication and authorization system",
                "priority": Priority.HIGH,
                "complexity": 8
            })
        
        # File processing
        if any(word in input_text for word in ["file", "upload", "process", "csv", "pdf"]):
            requirements.append({
                "description": "Handle file uploads and processing functionality",
                "priority": Priority.MEDIUM,
                "complexity": 6
            })
        
        # External integrations
        if any(word in input_text for word in ["integration", "webhook", "third-party", "api"]):
            requirements.append({
                "description": "Integrate with external APIs and services",
                "priority": Priority.MEDIUM,
                "complexity": 7
            })
        
        # Default backend requirement if none specific
        if not requirements:
            requirements.append({
                "description": "Create robust backend API with database integration",
                "priority": Priority.HIGH,
                "complexity": 6
            })
        
        return requirements

    def _extract_deployment_requirements(self, input_text: str) -> List[Dict[str, Any]]:
        """Extract deployment-specific requirements"""
        requirements = []
        
        # Cloud deployment
        if any(word in input_text for word in ["cloud", "aws", "azure", "gcp", "heroku"]):
            requirements.append({
                "description": "Deploy application to cloud infrastructure",
                "priority": Priority.MEDIUM,
                "complexity": 5
            })
        
        # Containerization
        if any(word in input_text for word in ["docker", "container", "kubernetes"]):
            requirements.append({
                "description": "Containerize application with Docker and orchestration",
                "priority": Priority.MEDIUM,
                "complexity": 6
            })
        
        # CI/CD pipeline
        if any(word in input_text for word in ["ci/cd", "pipeline", "automation", "build"]):
            requirements.append({
                "description": "Set up CI/CD pipeline for automated deployments",
                "priority": Priority.LOW,
                "complexity": 7
            })
        
        # Monitoring and logging
        if any(word in input_text for word in ["monitoring", "logging", "metrics", "alerts"]):
            requirements.append({
                "description": "Implement monitoring, logging, and alerting systems",
                "priority": Priority.LOW,
                "complexity": 5
            })
        
        # Default deployment requirement if none specific
        if not requirements:
            requirements.append({
                "description": "Deploy application to production environment",
                "priority": Priority.MEDIUM,
                "complexity": 4
            })
        
        return requirements

    def _extract_integration_requirements(self, input_text: str, required_agents: Set[str]) -> List[Dict[str, Any]]:
        """Extract requirements that need multiple agents working together"""
        requirements = []
        
        if "frontend" in required_agents and "backend" in required_agents:
            requirements.append({
                "description": "Integrate frontend with backend API endpoints",
                "priority": Priority.HIGH,
                "complexity": 7,
                "dependencies": ["FRONTEND_001", "BACKEND_001"]
            })
        
        if "backend" in required_agents and "deployment" in required_agents:
            requirements.append({
                "description": "Configure backend services for production deployment",
                "priority": Priority.MEDIUM,
                "complexity": 6,
                "dependencies": ["BACKEND_001"]
            })
        
        if len(required_agents) == 3:
            requirements.append({
                "description": "End-to-end testing and integration verification",
                "priority": Priority.HIGH,
                "complexity": 8,
                "dependencies": ["FRONTEND_001", "BACKEND_001", "DEPLOY_001"]
            })
        
        return requirements

    def _define_workflow_stages(self, requirements: List[TaskRequirement], required_agents: Set[str]) -> List[str]:
        """Define the workflow stages based on requirements"""
        stages = []
        
        # Analysis stage
        stages.append("Requirements Analysis & Planning")
        
        # Backend first (if needed)
        if "backend" in required_agents:
            stages.append("Backend Development")
            stages.append("Database Setup & Configuration")
            stages.append("API Development & Testing")
        
        # Frontend development
        if "frontend" in required_agents:
            stages.append("Frontend Development")
            stages.append("UI/UX Implementation")
            if "backend" in required_agents:
                stages.append("Frontend-Backend Integration")
        
        # Testing stage
        stages.append("Testing & Quality Assurance")
        
        # Deployment
        if "deployment" in required_agents:
            stages.append("Production Deployment")
            stages.append("Monitoring & Health Checks")
        
        # Final validation
        stages.append("User Acceptance & Refinement")
        
        return stages

    def _estimate_duration(self, requirements: List[TaskRequirement]) -> str:
        """Estimate project duration based on complexity"""
        total_complexity = sum(req.estimated_complexity for req in requirements)
        
        # Rough estimation: 1 complexity point = 2 hours of work
        total_hours = total_complexity * 2
        
        if total_hours < 8:
            return "Less than 1 day"
        elif total_hours < 40:
            return f"{total_hours // 8} days"
        elif total_hours < 160:
            return f"{total_hours // 40} weeks"
        else:
            return f"{total_hours // 160} months"

    def _define_success_criteria(self, user_intent: str, requirements: List[TaskRequirement]) -> List[str]:
        """Define success criteria for the project"""
        criteria = []
        
        # Basic criteria
        criteria.append("All functional requirements are implemented and working")
        criteria.append("Code passes quality and security checks")
        criteria.append("User interface is responsive and accessible")
        
        # Specific criteria based on requirements
        if any("api" in req.description.lower() for req in requirements):
            criteria.append("API endpoints are documented and tested")
        
        if any("database" in req.description.lower() for req in requirements):
            criteria.append("Data is stored securely and efficiently")
        
        if any("deploy" in req.description.lower() for req in requirements):
            criteria.append("Application is successfully deployed and accessible")
        
        # Performance criteria
        criteria.append("Application performance meets acceptable standards")
        criteria.append("User experience is smooth and intuitive")
        
        return criteria

    async def validate_requirements(self, requirements: List[TaskRequirement]) -> Dict[str, Any]:
        """Validate requirements for completeness and feasibility"""
        validation_result = {
            "is_valid": True,
            "warnings": [],
            "suggestions": [],
            "missing_dependencies": []
        }
        
        # Check for circular dependencies
        req_ids = {req.requirement_id for req in requirements}
        for req in requirements:
            for dep in req.dependencies:
                if dep not in req_ids:
                    validation_result["missing_dependencies"].append(dep)
                    validation_result["warnings"].append(
                        f"Requirement {req.requirement_id} depends on missing requirement {dep}"
                    )
        
        # Check for high-complexity requirements
        high_complexity_reqs = [req for req in requirements if req.estimated_complexity >= 8]
        if high_complexity_reqs:
            validation_result["suggestions"].append(
                f"Consider breaking down {len(high_complexity_reqs)} high-complexity requirements"
            )
        
        # Check for balanced workload across agents
        agent_workload = {}
        for req in requirements:
            for agent in req.agents_required:
                agent_workload[agent] = agent_workload.get(agent, 0) + req.estimated_complexity
        
        if agent_workload:
            max_workload = max(agent_workload.values())
            min_workload = min(agent_workload.values())
            if max_workload - min_workload > 20:
                validation_result["suggestions"].append(
                    "Consider redistributing workload for better balance across agents"
                )
        
        return validation_result
