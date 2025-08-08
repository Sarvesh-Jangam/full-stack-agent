import os
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import time

import asyncio

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService, VertexAiSessionService
from google.genai import types
from pydantic import BaseModel, Field
from google.adk.events import Event
from google.adk.events.event_actions import EventActions


from .workflow_manager import WorkflowManager, IterationResult
from .task_analyzer import TaskAnalyzer
from .coordination_tools import CoordinationTool
from .validation_tools import ValidationTool
from .config import ManagerAgentConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic Models for Request and Response
class ManagerRequest(BaseModel):
    action: str
    payload: Dict[str, Any] = Field(default_factory=dict)

class ManagerResponse(BaseModel):
    status: str
    workflow_id: str
    data: Optional[Any] = None
    message: str
    next_actions: List[str] = Field(default_factory=list)
    artifacts: Dict[str, Any] = Field(default_factory=dict)
    validation_score: Optional[float] = None
    progress_percentage: Optional[int] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# Initialize Tools
config = ManagerAgentConfig()
task_analyzer = TaskAnalyzer(config)
coordination_tool = CoordinationTool(config)
validation_tool = ValidationTool(config)

all_tools = [
    task_analyzer.analyze_task_requirements,
    coordination_tool.delegate_task,
    coordination_tool.get_agent_status,
    validation_tool.validate_artifact,
    validation_tool.run_integration_tests
]

# Define the Manager Agent
manager_agent = Agent(
    model=LiteLlm(os.getenv('DEFAULT_MODEL', 'gemini-pro')),
    name="manager_agent",
    description="Coordinates tasks between frontend and backend agents.",
    instruction="""
    You are the Manager Agent. Your role is to receive a high-level task,
    break it down into smaller, actionable steps, and delegate these steps
    to the appropriate specialist agents (frontend_agent or backend_agent).
    You must validate the artifacts produced by the agents and manage the
    overall workflow until the task is complete.
    """,
    tools=all_tools,
)

class ManagerAgentTool:
    def __init__(self):
        self.workflow_manager = WorkflowManager(Runner(agent=manager_agent))
        logger.info("Manager Agent Tool initialized")

    async def start_workflow(self, task_description: str) -> Dict[str, Any]:
        result = await self.workflow_manager.start_new_workflow(task_description)
        return result.model_dump()

    async def execute_iteration(self, workflow_id: str) -> Dict[str, Any]:
        result = await self.workflow_manager.run_iteration(workflow_id)
        return result.model_dump()

class ManagerAgentRunner:
    """Runner class for the manager agent"""

    def __init__(self,
                 session_service: Optional[Union[InMemorySessionService, VertexAiSessionService]] = None,
                 app_name: str = "manager_agent_app"):
        self.app_name = app_name
        self.session_service = session_service or InMemorySessionService()
        self.runner = Runner(
            agent=manager_agent,
            app_name=app_name,
            session_service=self.session_service
        )
        self.workflow_manager = WorkflowManager(self.runner)
        logger.info(f"Manager Agent Runner initialized with app_name: {app_name}")


# ... other necessary imports from your file

    async def process_request(
                        self,
                        request: ManagerRequest,
                        user_id: str = "default_user",
                        session_id: str = "default_session"
                                                            ) -> ManagerResponse:
        start_time = datetime.utcnow()
        try:
            logger.info(f"Processing manager request: {request.action} for user: {user_id}")

            session_state = await self.get_session_state(user_id, session_id)

            if not session_state and request.action != "start_workflow":
                raise ValueError(f"Session not found: {session_id}")

            if request.action == "start_workflow":
                task_description = request.payload.get("task_description")
                if not task_description:
                    raise ValueError("task_description is required for start_workflow")

                iteration_result = await self.workflow_manager.start_new_workflow(task_description)

                # --- FIX: Update session state using an event with state_delta ---
                actions_with_update = EventActions(state_delta={"workflow_id": iteration_result.workflow_id})

                system_event = Event(
                    invocation_id="workflow_start_update",
                    author="system",
                    actions=actions_with_update,
                    timestamp=time.time()
                )

                # Get the current session
                session = await self.session_service.get_session(
                    app_name=self.app_name,
                    user_id=user_id,
                    session_id=session_id
                )

                # Append the event to update the session state
                await self.session_service.append_event(session, system_event)

            else:
                workflow_id = session_state.get("workflow_id")
                if not workflow_id:
                    raise ValueError("workflow_id not found in session")

                iteration_result = await self.workflow_manager.run_iteration(workflow_id)

            return ManagerResponse(
                status=iteration_result.status,
                workflow_id=iteration_result.workflow_id,
                message=iteration_result.message,
                next_actions=iteration_result.next_actions,
                artifacts=iteration_result.artifacts,
                validation_score=iteration_result.validation_score,
                progress_percentage=iteration_result.progress_percentage,
                timestamp=datetime.utcnow()
            )

        except Exception as e:
            logger.error(f"Error processing manager request: {str(e)}")
            return ManagerResponse(
                status="error",
                workflow_id="unknown",
                message=f"Error processing request: {str(e)}",
                next_actions=[],
                artifacts={},
                timestamp=datetime.utcnow()
            )

    async def create_session(self, user_id: str, session_id: str, initial_state: Dict[str, Any] = None):
        """Create a new session"""
        try:
            await self.session_service.create_session(
                app_name=self.app_name,
                user_id=user_id,
                session_id=session_id,
                state=initial_state or {}
            )
            logger.info(f"Session created: {session_id} for user: {user_id}")
        except Exception as e:
            logger.error(f"Error creating session: {str(e)}")
            raise

    async def get_session_state(self, user_id: str, session_id: str) -> Dict[str, Any]:
        """Get current session state"""
        try:
            session = await self.session_service.get_session(
                app_name=self.app_name,
                user_id=user_id,
                session_id=session_id
            )
            return session.state if session else {}
        except Exception as e:
            logger.error(f"Error getting session state: {str(e)}")
            return {}

# Create default runner instance
default_runner = ManagerAgentRunner()

# Convenience functions
async def process_manager_request(request: ManagerRequest,
                                user_id: str = "default_user",
                                session_id: str = "default_session") -> ManagerResponse:
    """Convenience function to process a manager request"""
    return await default_runner.process_request(request, user_id, session_id)

