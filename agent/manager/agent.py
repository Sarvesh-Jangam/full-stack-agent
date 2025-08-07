"""
Main Manager Agent Implementation
Orchestrates multi-agent workflows with iterative refinement.
"""

import os
import logging
import asyncio
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from pydantic import BaseModel, Field

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService, VertexAiSessionService
from google.genai import types

from .workflow_manager import WorkflowManager, IterationResult
from .task_analyzer import TaskAnalyzer, AnalysisResult
from .coordination_tools import CoordinationTool, AgentTask, AgentResponse
from .validation_tools import ValidationTool
from .config import ManagerAgentConfig

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('manager_agent.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class ManagerRequest(BaseModel):
    """Request model for manager agent operations"""
    action: str = Field(..., description="The action to perform")
    user_input: str = Field(..., description="User requirements and instructions")
    workflow_id: Optional[str] = Field(None, description="Workflow ID for continuing workflows")
    preferences: Dict[str, Any] = Field(default_factory=dict, description="User preferences")
    feedback: Optional[str] = Field(None, description="User feedback for iterative improvement")

class ManagerResponse(BaseModel):
    """Response model for manager agent operations"""
    status: str = Field(..., description="Response status")
    workflow_id: str = Field(..., description="Workflow identifier")
    data: Any = Field(None, description="Response data")
    message: str = Field("", description="Response message")
    next_actions: List[str] = Field(default_factory=list, description="Suggested next actions")
    artifacts: Dict[str, List[str]] = Field(default_factory=dict, description="Generated artifacts")
    validation_score: Optional[float] = Field(None, description="Current validation score")
    progress_percentage: Optional[float] = Field(None, description="Workflow progress percentage")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# Initialize configuration
config = ManagerAgentConfig()

class ManagerAgentTool:
    """Main tool class for the manager agent"""
    
    def __init__(self):
        self.config = config
        self.workflow_manager = WorkflowManager()
        self.task_analyzer = TaskAnalyzer()
        self.coordination_tool = CoordinationTool()
        self.validation_tool = ValidationTool()
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        logger.info("Manager Agent Tool initialized")

    async def process_user_request(self, request: ManagerRequest) -> ManagerResponse:
        """
        Process user request and orchestrate multi-agent workflow
        
        Args:
            request: Manager request with user input and action
            
        Returns:
            Manager response with workflow results
        """
        try:
            action = request.action.lower()
            
            if action == "start_workflow":
                return await self._start_new_workflow(request)
            elif action == "continue_workflow":
                return await self._continue_workflow(request)
            elif action == "get_status":
                return await self._get_workflow_status(request)
            elif action == "provide_feedback":
                return await self._process_user_feedback(request)
            elif action == "finalize_workflow":
                return await self._finalize_workflow(request)
            elif action == "analyze_requirements":
                return await self._analyze_requirements_only(request)
            else:
                return ManagerResponse(
                    status="error",
                    workflow_id="unknown",
                    message=f"Unknown action: {action}",
                    next_actions=["Use valid actions: start_workflow, continue_workflow, get_status, provide_feedback, finalize_workflow"]
                )
                
        except Exception as e:
            logger.error(f"Error processing user request: {str(e)}")
            return ManagerResponse(
                status="error",
                workflow_id=request.workflow_id or "unknown",
                message=f"Error processing request: {str(e)}",
                next_actions=["Please try again with a valid request"]
            )

    async def _start_new_workflow(self, request: ManagerRequest) -> ManagerResponse:
        """Start a new multi-agent workflow"""
        try:
            # Generate workflow ID
            workflow_id = f"workflow_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            logger.info(f"Starting new workflow: {workflow_id}")
            
            # Start workflow
            workflow_result = await self.workflow_manager.start_workflow(
                workflow_id=workflow_id,
                user_input=request.user_input,
                user_preferences=request.preferences
            )
            
            # Store session info
            self.active_sessions[workflow_id] = {
                "start_time": datetime.utcnow(),
                "user_input": request.user_input,
                "preferences": request.preferences,
                "status": "active"
            }
            
            # Execute first iteration
            logger.info(f"Executing first iteration for workflow: {workflow_id}")
            iteration_result = await self.workflow_manager.execute_workflow_iteration(workflow_id)
            
            return ManagerResponse(
                status="started",
                workflow_id=workflow_id,
                data={
                    "analysis": workflow_result["analysis"],
                    "execution_plan": workflow_result["execution_plan"],
                    "first_iteration": iteration_result.model_dump()
                },
                message=f"Workflow started successfully. Completed iteration {iteration_result.iteration_number}.",
                artifacts=iteration_result.artifacts_generated,
                validation_score=iteration_result.validation_score,
                progress_percentage=min((iteration_result.iteration_number / self.config.max_iterations) * 100, 100),
                next_actions=[
                    "Review the generated artifacts",
                    "Provide feedback if needed",
                    "Continue with next iteration if validation score is low",
                    "Finalize workflow if satisfied with results"
                ]
            )
            
        except Exception as e:
            logger.error(f"Error starting workflow: {str(e)}")
            return ManagerResponse(
                status="error",
                workflow_id=request.workflow_id or "unknown",
                message=f"Failed to start workflow: {str(e)}",
                next_actions=["Please check your requirements and try again"]
            )

    async def _continue_workflow(self, request: ManagerRequest) -> ManagerResponse:
        """Continue an existing workflow with next iteration"""
        try:
            workflow_id = request.workflow_id
            if not workflow_id or workflow_id not in self.active_sessions:
                return ManagerResponse(
                    status="error",
                    workflow_id=workflow_id or "unknown",
                    message="Invalid or missing workflow ID",
                    next_actions=["Start a new workflow or provide valid workflow ID"]
                )
            
            logger.info(f"Continuing workflow: {workflow_id}")
            
            # Check if workflow is still active
            workflow_progress = await self.workflow_manager.get_workflow_progress(workflow_id)
            current_iteration = workflow_progress.get("current_iteration", 0)
            
            if current_iteration >= self.config.max_iterations:
                return ManagerResponse(
                    status="completed",
                    workflow_id=workflow_id,
                    message="Workflow has reached maximum iterations",
                    next_actions=["Finalize workflow or start a new one"]
                )
            
            # Execute next iteration
            iteration_result = await self.workflow_manager.execute_workflow_iteration(workflow_id)
            
            # Check if workflow should continue
            should_continue = (
                iteration_result.validation_score < self.config.validation_threshold and
                iteration_result.iteration_number < self.config.max_iterations and
                len(iteration_result.refinements_needed) > 0
            )
            
            next_actions = []
            if should_continue:
                next_actions.extend([
                    "Review the improvements made in this iteration",
                    "Continue with next iteration to further refine results",
                    "Provide specific feedback for targeted improvements"
                ])
            else:
                next_actions.extend([
                    "Workflow appears to be complete",
                    "Review final artifacts and validation results",
                    "Finalize workflow if satisfied"
                ])
            
            return ManagerResponse(
                status="continued",
                workflow_id=workflow_id,
                data={
                    "iteration_result": iteration_result.model_dump(),
                    "should_continue": should_continue
                },
                message=f"Completed iteration {iteration_result.iteration_number}. Validation score: {iteration_result.validation_score:.2f}",
                artifacts=iteration_result.artifacts_generated,
                validation_score=iteration_result.validation_score,
                progress_percentage=min((iteration_result.iteration_number / self.config.max_iterations) * 100, 100),
                next_actions=next_actions
            )
            
        except Exception as e:
            logger.error(f"Error continuing workflow: {str(e)}")
            return ManagerResponse(
                status="error",
                workflow_id=request.workflow_id or "unknown",
                message=f"Failed to continue workflow: {str(e)}",
                next_actions=["Check workflow status or start a new workflow"]
            )

    async def _get_workflow_status(self, request: ManagerRequest) -> ManagerResponse:
        """Get current status of a workflow"""
        try:
            workflow_id = request.workflow_id
            if not workflow_id:
                return ManagerResponse(
                    status="error",
                    workflow_id="unknown",
                    message="Workflow ID is required",
                    next_actions=["Provide a valid workflow ID"]
                )
            
            # Get comprehensive status
            workflow_progress = await self.workflow_manager.get_workflow_progress(workflow_id)
            coordination_metrics = await self.coordination_tool.get_coordination_metrics()
            
            if "error" in workflow_progress:
                return ManagerResponse(
                    status="error",
                    workflow_id=workflow_id,
                    message=workflow_progress["error"],
                    next_actions=["Start a new workflow"]
                )
            
            latest_score = workflow_progress.get("latest_validation_score", 0.0)
            current_iteration = workflow_progress.get("current_iteration", 0)
            
            # Determine workflow status
            if workflow_progress.get("phase") == "completion":
                status = "completed"
                next_actions = ["Review final results", "Start a new workflow if needed"]
            elif current_iteration >= self.config.max_iterations:
                status = "max_iterations_reached"
                next_actions = ["Finalize current results", "Start a new workflow for improvements"]
            elif latest_score >= self.config.validation_threshold:
                status = "ready_for_finalization"
                next_actions = ["Finalize workflow", "Continue for further improvements"]
            else:
                status = "in_progress"
                next_actions = ["Continue workflow", "Provide feedback for improvements"]
            
            return ManagerResponse(
                status=status,
                workflow_id=workflow_id,
                data={
                    "workflow_progress": workflow_progress,
                    "coordination_metrics": coordination_metrics,
                    "detailed_status": {
                        "phase": workflow_progress.get("phase"),
                        "current_iteration": current_iteration,
                        "max_iterations": self.config.max_iterations,
                        "validation_score": latest_score,
                        "validation_threshold": self.config.validation_threshold
                    }
                },
                message=f"Workflow {workflow_id} is {status}",
                validation_score=latest_score,
                progress_percentage=workflow_progress.get("progress_percentage", 0.0),
                next_actions=next_actions
            )
            
        except Exception as e:
            logger.error(f"Error getting workflow status: {str(e)}")
            return ManagerResponse(
                status="error",
                workflow_id=request.workflow_id or "unknown",
                message=f"Failed to get workflow status: {str(e)}",
                next_actions=["Check if workflow ID is valid"]
            )

    async def _process_user_feedback(self, request: ManagerRequest) -> ManagerResponse:
        """Process user feedback and adjust workflow"""
        try:
            workflow_id = request.workflow_id
            feedback = request.feedback
            
            if not workflow_id or not feedback:
                return ManagerResponse(
                    status="error",
                    workflow_id=workflow_id or "unknown",
                    message="Both workflow ID and feedback are required",
                    next_actions=["Provide workflow ID and feedback"]
                )
            
            # Process feedback through workflow manager
            feedback_data = {
                "action": "request_specific_changes",
                "specific_requests": [feedback],
                "user_satisfaction": "needs_improvement"
            }
            
            feedback_result = await self.workflow_manager.process_user_feedback(workflow_id, feedback_data)
            
            return ManagerResponse(
                status="feedback_processed",
                workflow_id=workflow_id,
                data=feedback_result,
                message=f"Feedback processed: {feedback_result.get('message', 'Feedback incorporated')}",
                next_actions=[
                    "Continue workflow with incorporated feedback",
                    "Review how feedback affects next iteration"
                ]
            )
            
        except Exception as e:
            logger.error(f"Error processing user feedback: {str(e)}")
            return ManagerResponse(
                status="error",
                workflow_id=request.workflow_id or "unknown",
                message=f"Failed to process feedback: {str(e)}",
                next_actions=["Try providing feedback again"]
            )

    async def _finalize_workflow(self, request: ManagerRequest) -> ManagerResponse:
        """Finalize a completed workflow"""
        try:
            workflow_id = request.workflow_id
            if not workflow_id:
                return ManagerResponse(
                    status="error",
                    workflow_id="unknown",
                    message="Workflow ID is required for finalization",
                    next_actions=["Provide a valid workflow ID"]
                )
            
            # Get final workflow status
            workflow_progress = await self.workflow_manager.get_workflow_progress(workflow_id)
            
            if "error" in workflow_progress:
                return ManagerResponse(
                    status="error",
                    workflow_id=workflow_id,
                    message="Cannot finalize: " + workflow_progress["error"],
                    next_actions=["Check workflow status"]
                )
            
            # Mark session as completed
            if workflow_id in self.active_sessions:
                self.active_sessions[workflow_id]["status"] = "completed"
                self.active_sessions[workflow_id]["completion_time"] = datetime.utcnow()
            
            # Get final artifacts
            final_artifacts = workflow_progress.get("coordination_status", {}).get("artifacts", {})
            latest_score = workflow_progress.get("latest_validation_score", 0.0)
            
            return ManagerResponse(
                status="finalized",
                workflow_id=workflow_id,
                data={
                    "final_progress": workflow_progress,
                    "completion_summary": {
                        "total_iterations": workflow_progress.get("current_iteration", 0),
                        "final_validation_score": latest_score,
                        "artifacts_generated": sum(len(files) for files in final_artifacts.values()),
                        "agents_used": list(final_artifacts.keys()),
                        "completion_time": datetime.utcnow().isoformat()
                    }
                },
                message=f"Workflow {workflow_id} finalized successfully with validation score {latest_score:.2f}",
                artifacts=final_artifacts,
                validation_score=latest_score,
                progress_percentage=100.0,
                next_actions=[
                    "Review all generated artifacts",
                    "Download or deploy the generated applications",
                    "Start a new workflow for additional projects"
                ]
            )
            
        except Exception as e:
            logger.error(f"Error finalizing workflow: {str(e)}")
            return ManagerResponse(
                status="error",
                workflow_id=request.workflow_id or "unknown",
                message=f"Failed to finalize workflow: {str(e)}",
                next_actions=["Check workflow status and try again"]
            )

    async def _analyze_requirements_only(self, request: ManagerRequest) -> ManagerResponse:
        """Analyze user requirements without starting full workflow"""
        try:
            logger.info("Analyzing user requirements")
            
            # Analyze requirements
            analysis_result = await self.task_analyzer.analyze_requirements(request.user_input)
            
            # Validate requirements
            validation_result = await self.task_analyzer.validate_requirements(analysis_result.requirements)
            
            return ManagerResponse(
                status="analysis_complete",
                workflow_id="analysis_only",
                data={
                    "analysis": analysis_result.model_dump(),
                    "validation": validation_result,
                    "estimated_effort": {
                        "total_requirements": len(analysis_result.requirements),
                        "estimated_duration": analysis_result.estimated_duration,
                        "agents_required": list(set(agent for req in analysis_result.requirements for agent in req.agents_required)),
                        "complexity_breakdown": {
                            "low": len([r for r in analysis_result.requirements if r.estimated_complexity <= 3]),
                            "medium": len([r for r in analysis_result.requirements if 4 <= r.estimated_complexity <= 6]),
                            "high": len([r for r in analysis_result.requirements if r.estimated_complexity >= 7])
                        }
                    }
                },
                message="Requirements analysis completed successfully",
                next_actions=[
                    "Review the analysis results",
                    "Start workflow to begin implementation",
                    "Modify requirements if needed"
                ]
            )
            
        except Exception as e:
            logger.error(f"Error analyzing requirements: {str(e)}")
            return ManagerResponse(
                status="error",
                workflow_id="analysis_error",
                message=f"Failed to analyze requirements: {str(e)}",
                next_actions=["Check your requirements and try again"]
            )

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on manager agent"""
        try:
            return {
                "status": "healthy",
                "active_workflows": len(self.active_sessions),
                "config_loaded": self.config is not None,
                "components_status": {
                    "workflow_manager": "active",
                    "task_analyzer": "active",
                    "coordination_tool": "active",
                    "validation_tool": "active"
                },
                "resource_usage": {
                    "memory_limit_mb": self.config.memory_limit_mb,
                    "max_iterations": self.config.max_iterations,
                    "validation_threshold": self.config.validation_threshold
                }
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

# Create the manager agent tool instance
manager_tool = ManagerAgentTool()

# Create the main manager agent
manager_agent = Agent(
    model=LiteLlm(os.getenv('GOOGLE_MODEL', 'gemini-2.0-flash')),
    name=config.agent_name,
    description=config.agent_description,
    instruction=f"""
You are an advanced Manager Agent responsible for orchestrating multi-agent workflows to build complete web applications. Your role is to coordinate between Frontend, Backend, and Deployment agents to deliver high-quality solutions that meet user requirements.

## Your Capabilities:

**1. Requirements Analysis**
- Analyze user input to understand project requirements
- Break down complex requirements into manageable tasks
- Identify which agents (frontend, backend, deployment) are needed
- Create execution plans with proper task dependencies

**2. Workflow Orchestration**
- Start and manage multi-agent workflows
- Coordinate task execution between agents
- Handle task dependencies and parallel execution
- Monitor progress and handle errors gracefully

**3. Iterative Refinement**
- Execute workflows in iterations (max {config.max_iterations})
- Validate results after each iteration
- Identify areas needing improvement
- Refine and improve results based on validation feedback

**4. Quality Assurance** 
- Validate all generated artifacts and code
- Ensure compliance with user requirements
- Maintain quality standards (validation threshold: {config.validation_threshold})
- Provide recommendations for improvements

**5. User Interaction**
- Process user feedback and incorporate changes
- Provide clear status updates and progress reports
- Explain technical decisions and recommendations
- Guide users through the development process

## Available Actions:
- **start_workflow**: Begin a new multi-agent workflow
- **continue_workflow**: Execute next iteration of existing workflow
- **get_status**: Get current workflow status and progress
- **provide_feedback**: Process user feedback for improvements
- **finalize_workflow**: Complete and finalize a workflow
- **analyze_requirements**: Analyze requirements without starting workflow

## Agent Coordination:
- **Frontend Agent**: Handles UI/UX design, HTML, CSS, JavaScript, responsive design
- **Backend Agent**: Manages APIs, databases, authentication, business logic, data processing
- **Deployment Agent**: Handles containerization, cloud deployment, CI/CD, monitoring

## Quality Standards:
- Minimum validation score: {config.validation_threshold}
- Frontend quality threshold: {config.minimum_frontend_score}
- Backend quality threshold: {config.minimum_backend_score}
- Integration quality threshold: {config.minimum_integration_score}

## Process Flow:
1. Analyze user requirements and create execution plan
2. Execute workflow iterations with agent coordination
3. Validate results after each iteration
4. Refine based on validation feedback and user input
5. Continue until quality standards are met or max iterations reached
6. Finalize and deliver complete solution

Always provide clear explanations of what you're doing, why you're doing it, and what the user can expect next. Focus on delivering high-quality, complete solutions that fully meet the user's requirements.
""",
    tools=[manager_tool.process_user_request],
    input_schema=ManagerRequest,
    output_key="manager_result"
)

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
        logger.info(f"Manager Agent Runner initialized with app_name: {app_name}")

    async def process_request(self, 
                            request: ManagerRequest,
                            user_id: str = "default_user",
                            session_id: str = "default_session") -> ManagerResponse:
        """Process a manager agent request"""
        start_time = datetime.utcnow()
        
        try:
            logger.info(f"Processing manager request: {request.action} for user: {user_id}")
            
            # Create content for the agent
            request_content = request.model_dump_json()
            user_content = types.Content(
                role='user',
                parts=[types.Part(text=request_content)]
            )
            
            # Process through agent
            final_response = None
            async for event in self.runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=user_content
            ):
                if event.is_final_response() and event.content and event.content.parts:
                    final_response = event.content.parts.text
                    break
            
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            if final_response:
                logger.info(f"Manager request processed successfully in {execution_time:.2f}s")
                return ManagerResponse(
                    status="success",
                    workflow_id=request.workflow_id or "processed",
                    data=final_response,
                    message="Request processed successfully"
                )
            else:
                logger.error("No response received from manager agent")
                return ManagerResponse(
                    status="error",
                    workflow_id=request.workflow_id or "unknown",
                    message="No response received from manager agent"
                )
                
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            logger.error(f"Error processing manager request: {str(e)}")
            return ManagerResponse(
                status="error",
                workflow_id=request.workflow_id or "unknown",
                message=f"Error processing request: {str(e)}"
            )

# Create default runner instance
default_runner = ManagerAgentRunner()

# Convenience functions
async def process_manager_request(request: ManagerRequest, 
                                user_id: str = "default_user",
                                session_id: str = "default_session") -> ManagerResponse:
    """Convenience function to process a manager request"""
    return await default_runner.process_request(request, user_id, session_id)

def get_manager_agent_info() -> Dict[str, Any]:
    """Get information about the manager agent"""
    return {
        "name": manager_agent.name,
        "description": manager_agent.description,
        "model": str(manager_agent.model),
        "max_iterations": config.max_iterations,
        "validation_threshold": config.validation_threshold,
        "version": "1.0.0"
    }

if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def main():
        # Test the manager agent
        test_request = ManagerRequest(
            action="analyze_requirements",
            user_input="Create a modern todo list application with user authentication and responsive design"
        )
        
        response = await process_manager_request(test_request)
        print(f"Manager Agent Response: {response}")
        
        # Health check
        health = manager_tool.health_check()
        print(f"Health Check: {health}")

    asyncio.run(main())

