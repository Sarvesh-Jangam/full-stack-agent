"""
Coordination Tools for Manager Agent
Handles communication and task delegation between agents.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from enum import Enum
from pydantic import BaseModel, Field
import json
from .config import ManagerAgentConfig
logger = logging.getLogger(__name__)

class AgentStatus(str, Enum):
    IDLE = "idle"
    WORKING = "working"
    WAITING = "waiting"
    ERROR = "error"
    COMPLETED = "completed"

class TaskStatus(str, Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class AgentTask(BaseModel):
    """Task assigned to an agent"""
    task_id: str = Field(..., description="Unique task identifier")
    agent_name: str = Field(..., description="Target agent name")
    task_type: str = Field(..., description="Type of task")
    instructions: str = Field(..., description="Detailed task instructions")
    input_data: Dict[str, Any] = Field(default_factory=dict, description="Input data for the task")
    priority: str = Field("medium", description="Task priority")
    dependencies: List[str] = Field(default_factory=list, description="Task dependencies")
    expected_output: str = Field(..., description="Expected output description")
    timeout_minutes: int = Field(30, description="Task timeout in minutes")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    assigned_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: TaskStatus = Field(TaskStatus.PENDING, description="Current task status")

class AgentResponse(BaseModel):
    """Response from an agent"""
    task_id: str = Field(..., description="Task identifier")
    agent_name: str = Field(..., description="Agent name")
    status: TaskStatus = Field(..., description="Task status")
    result_data: Any = Field(None, description="Task result data")
    message: str = Field("", description="Response message")
    execution_time: Optional[float] = Field(None, description="Execution time in seconds")
    artifacts: List[str] = Field(default_factory=list, description="Generated artifacts/files")
    feedback: str = Field("", description="Agent feedback or suggestions")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class WorkflowState(BaseModel):
    """Current state of the workflow"""
    workflow_id: str = Field(..., description="Workflow identifier")
    current_stage: str = Field(..., description="Current workflow stage")
    active_tasks: List[str] = Field(default_factory=list, description="Currently active tasks")
    completed_tasks: List[str] = Field(default_factory=list, description="Completed tasks")
    failed_tasks: List[str] = Field(default_factory=list, description="Failed tasks")
    agent_status: Dict[str, AgentStatus] = Field(default_factory=dict, description="Status of each agent")
    artifacts: Dict[str, List[str]] = Field(default_factory=dict, description="Generated artifacts by agent")
    start_time: datetime = Field(default_factory=datetime.utcnow)
    last_updated: datetime = Field(default_factory=datetime.utcnow)

class CoordinationTool:
    """Handles delegation of tasks to specialist agents."""

    def __init__(self, config: ManagerAgentConfig):
        """
        Initializes the CoordinationTool with a configuration.

        Args:
            config: The configuration object for the manager agent.
        """
        self.config = config
        # In a real app, you would have clients to communicate with other agents
        # self.frontend_agent_client = ...
        # self.backend_agent_client = ...
        logger.info("Coordination tool initialized")

    async def delegate_task(self, agent_name: str, task: str, dependencies: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Delegates a task to a specified agent.
        """
        logger.info(f"Delegating task '{task}' to agent '{agent_name}'")

        # Mock delegation
        if agent_name not in self.config.specialist_agents:
            return {"status": "error", "message": f"Agent '{agent_name}' not found."}

        # Simulate calling the agent and getting a result
        await asyncio.sleep(2) # Simulate network latency
        artifact = f"artifact_for_{task.replace(' ', '_')}.txt"

        return {
            "status": "completed",
            "agent": agent_name,
            "task": task,
            "artifact_id": artifact,
            "message": f"Task '{task}' completed by '{agent_name}'."
        }

    async def get_agent_status(self, agent_name: str) -> Dict[str, Any]:
        """
        Checks the status of a specialist agent.
        """
        if agent_name not in self.config.specialist_agents:
            return {"status": "error", "message": f"Agent '{agent_name}' not found."}

        return {"agent": agent_name, "status": "idle", "health": "ok"}

    async def create_workflow(self, workflow_id: str) -> WorkflowState:
        """Create a new workflow instance"""
        workflow = WorkflowState(
            workflow_id=workflow_id,
            current_stage="initialization",
            agent_status={
                "frontend": AgentStatus.IDLE,
                "backend": AgentStatus.IDLE,
                "deployment": AgentStatus.IDLE
            }
        )
        self.active_workflows[workflow_id] = workflow
        logger.info(f"Created new workflow: {workflow_id}")
        return workflow

    async def assign_task(self, 
                         workflow_id: str,
                         agent_name: str, 
                         task_type: str,
                         instructions: str,
                         input_data: Dict[str, Any] = None,
                         dependencies: List[str] = None,
                         priority: str = "medium") -> AgentTask:
        """Assign a task to a specific agent"""
        try:
            # Generate unique task ID
            task_id = f"{agent_name}_{task_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            # Create task
            task = AgentTask(
                task_id=task_id,
                agent_name=agent_name,
                task_type=task_type,
                instructions=instructions,
                input_data=input_data or {},
                dependencies=dependencies or [],
                priority=priority,
                expected_output=self._generate_expected_output(agent_name, task_type)
            )
            
            # Add to queue
            if agent_name in self.task_queue:
                self.task_queue[agent_name].append(task)
                task.status = TaskStatus.ASSIGNED
                task.assigned_at = datetime.utcnow()
            
            # Update workflow state
            if workflow_id in self.active_workflows:
                workflow = self.active_workflows[workflow_id]
                workflow.active_tasks.append(task_id)
                workflow.agent_status[agent_name] = AgentStatus.WORKING
                workflow.last_updated = datetime.utcnow()
            
            logger.info(f"Task assigned: {task_id} to {agent_name}")
            return task
            
        except Exception as e:
            logger.error(f"Error assigning task: {str(e)}")
            raise

    async def delegate_to_agent(self, agent_name: str, task: AgentTask) -> AgentResponse:
        """Delegate a specific task to an agent and wait for response"""
        try:
            logger.info(f"Delegating task {task.task_id} to {agent_name}")
            
            # Mark task as in progress
            task.status = TaskStatus.IN_PROGRESS
            task.started_at = datetime.utcnow()
            
            # Simulate agent execution (In real implementation, this would call the actual agent)
            response = await self._execute_agent_task(agent_name, task)
            
            # Update task status
            if response.status == TaskStatus.COMPLETED:
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.utcnow()
            elif response.status == TaskStatus.FAILED:
                task.status = TaskStatus.FAILED
            
            return response
            
        except Exception as e:
            logger.error(f"Error delegating to {agent_name}: {str(e)}")
            return AgentResponse(
                task_id=task.task_id,
                agent_name=agent_name,
                status=TaskStatus.FAILED,
                message=f"Delegation failed: {str(e)}"
            )

    async def _execute_agent_task(self, agent_name: str, task: AgentTask) -> AgentResponse:
        """Execute task on specific agent (mock implementation)"""
        
        # This is a mock implementation. In reality, you would:
        # 1. Import the actual agent (frontend_agent, backend_agent, etc.)
        # 2. Call the agent with the task instructions
        # 3. Return the actual response
        
        try:
            # Simulate processing time
            await asyncio.sleep(2)  # Remove in real implementation
            
            # Mock response based on agent capabilities
            if agent_name == "frontend":
                result_data = await self._mock_frontend_execution(task)
            elif agent_name == "backend":
                result_data = await self._mock_backend_execution(task)
            elif agent_name == "deployment":
                result_data = await self._mock_deployment_execution(task)
            else:
                raise ValueError(f"Unknown agent: {agent_name}")
            
            return AgentResponse(
                task_id=task.task_id,
                agent_name=agent_name,
                status=TaskStatus.COMPLETED,
                result_data=result_data,
                message=f"Task completed successfully by {agent_name}",
                execution_time=2.0,  # Mock execution time
                artifacts=result_data.get("artifacts", []),
                feedback=result_data.get("feedback", "")
            )
            
        except Exception as e:
            return AgentResponse(
                task_id=task.task_id,
                agent_name=agent_name,
                status=TaskStatus.FAILED,
                message=f"Task execution failed: {str(e)}"
            )

    async def _mock_frontend_execution(self, task: AgentTask) -> Dict[str, Any]:
        """Mock frontend agent execution"""
        task_type = task.task_type.lower()
        
        if "html" in task_type or "interface" in task_type:
            return {
                "html_code": "<html><!-- Generated HTML code --></html>",
                "css_code": "/* Generated CSS styles */",
                "js_code": "// Generated JavaScript code",
                "artifacts": ["index.html", "styles.css", "script.js"],
                "components": ["header", "main", "footer"],
                "feedback": "HTML structure created with responsive design"
            }
        elif "dashboard" in task_type:
            return {
                "dashboard_components": ["sidebar", "main-content", "charts"],
                "charts": ["bar-chart", "line-chart", "pie-chart"],
                "artifacts": ["dashboard.html", "dashboard.css", "dashboard.js"],
                "feedback": "Interactive dashboard with data visualization components"
            }
        else:
            return {
                "ui_components": ["forms", "buttons", "navigation"],
                "artifacts": ["components.html", "styles.css"],
                "feedback": f"Frontend {task_type} completed successfully"
            }

    async def _mock_backend_execution(self, task: AgentTask) -> Dict[str, Any]:
        """Mock backend agent execution"""
        task_type = task.task_type.lower()
        
        if "api" in task_type:
            return {
                "endpoints": ["/api/users", "/api/data", "/api/auth"],
                "methods": ["GET", "POST", "PUT", "DELETE"],
                "documentation": "OpenAPI specification generated",
                "artifacts": ["api.py", "models.py", "routes.py"],
                "feedback": "RESTful API endpoints created with proper documentation"
            }
        elif "database" in task_type:
            return {
                "tables": ["users", "data", "sessions"],
                "relationships": ["one-to-many", "many-to-many"],
                "migrations": ["001_initial.sql", "002_add_indexes.sql"],
                "artifacts": ["schema.sql", "migrations/"],
                "feedback": "Database schema designed with proper relationships"
            }
        elif "auth" in task_type:
            return {
                "auth_methods": ["JWT", "OAuth"],
                "endpoints": ["/login", "/register", "/refresh"],
                "security_features": ["password-hashing", "rate-limiting"],
                "artifacts": ["auth.py", "middleware.py"],
                "feedback": "Secure authentication system implemented"
            }
        else:
            return {
                "services": ["data-processing", "validation"],
                "artifacts": ["services.py", "utils.py"],
                "feedback": f"Backend {task_type} completed successfully"
            }

    async def _mock_deployment_execution(self, task: AgentTask) -> Dict[str, Any]:
        """Mock deployment agent execution"""
        task_type = task.task_type.lower()
        
        if "docker" in task_type or "container" in task_type:
            return {
                "containers": ["frontend", "backend", "database"],
                "images": ["node:18", "python:3.9", "postgres:14"],
                "networks": ["app-network"],
                "artifacts": ["Dockerfile", "docker-compose.yml"],
                "feedback": "Application containerized with Docker"
            }
        elif "cloud" in task_type or "deploy" in task_type:
            return {
                "services": ["app-service", "database-service"],
                "infrastructure": ["load-balancer", "auto-scaling"],
                "monitoring": ["health-checks", "logging"],
                "artifacts": ["deploy.yml", "config/"],
                "feedback": "Application deployed to cloud with monitoring"
            }
        else:
            return {
                "deployment_config": ["production", "staging"],
                "artifacts": ["deployment.yml"],
                "feedback": f"Deployment {task_type} completed successfully"
            }

    def _generate_expected_output(self, agent_name: str, task_type: str) -> str:
        """Generate expected output description for a task"""
        output_templates = {
            "frontend": {
                "html_generation": "Complete HTML files with semantic structure",
                "css_styling": "CSS stylesheets with responsive design",
                "javascript_development": "JavaScript files with interactive functionality",
                "ui_design": "User interface components and layouts",
                "dashboard_creation": "Interactive dashboard with data visualization"
            },
            "backend": {
                "api_development": "RESTful API endpoints with documentation",
                "database_operations": "Database schema and CRUD operations",
                "authentication": "Secure authentication and authorization system",
                "file_processing": "File upload and processing functionality",
                "data_validation": "Input validation and sanitization logic"
            },
            "deployment": {
                "cloud_deployment": "Deployed application on cloud infrastructure",
                "containerization": "Docker containers and orchestration files",
                "ci_cd_setup": "Automated CI/CD pipeline configuration",
                "monitoring_setup": "Monitoring and alerting system",
                "infrastructure_management": "Infrastructure as Code templates"
            }
        }
        
        if agent_name in output_templates and task_type in output_templates[agent_name]:
            return output_templates[agent_name][task_type]
        else:
            return f"Completed {task_type} deliverables for {agent_name}"

    async def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get current status of a workflow"""
        if workflow_id not in self.active_workflows:
            return {"error": "Workflow not found"}
        
        workflow = self.active_workflows[workflow_id]
        
        # Calculate progress
        total_tasks = len(workflow.active_tasks) + len(workflow.completed_tasks) + len(workflow.failed_tasks)
        completed_tasks = len(workflow.completed_tasks)
        progress_percentage = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
        
        return {
            "workflow_id": workflow_id,
            "current_stage": workflow.current_stage,
            "progress_percentage": round(progress_percentage, 2),
            "active_tasks": workflow.active_tasks,
            "completed_tasks": workflow.completed_tasks,
            "failed_tasks": workflow.failed_tasks,
            "agent_status": workflow.agent_status,
            "artifacts": workflow.artifacts,
            "duration": str(datetime.utcnow() - workflow.start_time),
            "last_updated": workflow.last_updated.isoformat()
        }

    async def update_workflow_stage(self, workflow_id: str, new_stage: str):
        """Update the current stage of a workflow"""
        if workflow_id in self.active_workflows:
            workflow = self.active_workflows[workflow_id]
            workflow.current_stage = new_stage
            workflow.last_updated = datetime.utcnow()
            logger.info(f"Workflow {workflow_id} moved to stage: {new_stage}")

    async def handle_task_completion(self, workflow_id: str, task_id: str, response: AgentResponse):
        """Handle the completion of a task"""
        if workflow_id not in self.active_workflows:
            return
        
        workflow = self.active_workflows[workflow_id]
        
        if response.status == TaskStatus.COMPLETED:
            # Move task from active to completed
            if task_id in workflow.active_tasks:
                workflow.active_tasks.remove(task_id)
                workflow.completed_tasks.append(task_id)
            
            # Store artifacts
            if response.artifacts:
                if response.agent_name not in workflow.artifacts:
                    workflow.artifacts[response.agent_name] = []
                workflow.artifacts[response.agent_name].extend(response.artifacts)
            
            # Update agent status to idle if no more tasks
            if response.agent_name in self.task_queue:
                if not self.task_queue[response.agent_name]:
                    workflow.agent_status[response.agent_name] = AgentStatus.IDLE
        
        elif response.status == TaskStatus.FAILED:
            # Move task from active to failed
            if task_id in workflow.active_tasks:
                workflow.active_tasks.remove(task_id)
                workflow.failed_tasks.append(task_id)
            
            workflow.agent_status[response.agent_name] = AgentStatus.ERROR
        
        workflow.last_updated = datetime.utcnow()
        logger.info(f"Task {task_id} completion handled: {response.status}")

    async def check_dependencies(self, task: AgentTask, workflow_id: str) -> bool:
        """Check if all task dependencies are satisfied"""
        if not task.dependencies:
            return True
        
        if workflow_id not in self.active_workflows:
            return False
        
        workflow = self.active_workflows[workflow_id]
        
        # Check if all dependencies are completed
        for dep_id in task.dependencies:
            if dep_id not in workflow.completed_tasks:
                return False
        
        return True

    async def get_next_available_task(self, agent_name: str) -> Optional[AgentTask]:
        """Get the next available task for an agent"""
        if agent_name not in self.task_queue:
            return None
        
        # Sort by priority and creation time
        priority_order = {"urgent": 0, "high": 1, "medium": 2, "low": 3}
        available_tasks = [
            task for task in self.task_queue[agent_name]
            if task.status == TaskStatus.ASSIGNED
        ]
        
        if not available_tasks:
            return None
        
        # Sort by priority, then by creation time
        available_tasks.sort(
            key=lambda t: (priority_order.get(t.priority, 4), t.created_at)
        )
        
        return available_tasks[0]

    async def cancel_task(self, task_id: str, reason: str = "Cancelled by manager"):
        """Cancel a pending or in-progress task"""
        for agent_queue in self.task_queue.values():
            for task in agent_queue:
                if task.task_id == task_id:
                    task.status = TaskStatus.CANCELLED
                    logger.info(f"Task {task_id} cancelled: {reason}")
                    return True
        
        return False

    async def get_coordination_metrics(self) -> Dict[str, Any]:
        """Get coordination metrics and statistics"""
        total_tasks = sum(len(queue) for queue in self.task_queue.values())
        completed_tasks = 0
        failed_tasks = 0
        
        for queue in self.task_queue.values():
            completed_tasks += sum(1 for task in queue if task.status == TaskStatus.COMPLETED)
            failed_tasks += sum(1 for task in queue if task.status == TaskStatus.FAILED)
        
        return {
            "total_workflows": len(self.active_workflows),
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "success_rate": (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0,
            "agent_utilization": {
                agent: len(queue) for agent, queue in self.task_queue.items()
            },
            "average_task_completion_time": self._calculate_average_completion_time()
        }

    def _calculate_average_completion_time(self) -> float:
        """Calculate average task completion time"""
        completed_tasks = []
        for queue in self.task_queue.values():
            for task in queue:
                if task.status == TaskStatus.COMPLETED and task.started_at and task.completed_at:
                    completed_tasks.append(task)
        
        if not completed_tasks:
            return 0.0
        
        total_time = sum(
            (task.completed_at - task.started_at).total_seconds()
            for task in completed_tasks
        )
        
        return total_time / len(completed_tasks)


