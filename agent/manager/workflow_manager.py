"""
Workflow Manager for Manager Agent
Orchestrates multi-agent workflows with iterative refinement.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
from pydantic import BaseModel, Field
import uuid


from google.adk.runners import Runner

from .task_analyzer import TaskAnalyzer, AnalysisResult, TaskRequirement
from .coordination_tools import CoordinationTool, AgentTask, AgentResponse, WorkflowState
from .validation_tools import ValidationTool

logger = logging.getLogger(__name__)

class WorkflowPhase(str, Enum):
    ANALYSIS = "analysis"
    PLANNING = "planning"
    EXECUTION = "execution"
    VALIDATION = "validation"
    REFINEMENT = "refinement"
    COMPLETION = "completion"

class IterationResult(BaseModel):
    workflow_id: str
    status: str
    message: str
    next_actions: List[str] = Field(default_factory=list)
    artifacts: Dict[str, Any] = Field(default_factory=dict)
    validation_score: float = 0.0
    progress_percentage: int = 0

class WorkflowState(BaseModel):
    workflow_id: str
    task_description: str
    plan: Dict[str, Any] = Field(default_factory=dict)
    completed_steps: List[int] = Field(default_factory=list)
    artifacts: Dict[str, Any] = Field(default_factory=dict)
    status: str = "pending"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class WorkflowManager:
    """Manages the lifecycle of a development workflow."""

    def __init__(self, runner: Runner):
        """
        Initializes the WorkflowManager.

        Args:
            runner: The ADK Runner instance to interact with the agent.
        """
        self.runner = runner
        self.workflows: Dict[str, WorkflowState] = {}
        logger.info("Workflow Manager initialized")

    async def start_new_workflow(self, task_description: str) -> IterationResult:
        """Creates and starts a new workflow."""
        workflow_id = str(uuid.uuid4())
        state = WorkflowState(workflow_id=workflow_id, task_description=task_description)
        self.workflows[workflow_id] = state
        logger.info(f"Started new workflow {workflow_id} for task: {task_description}")
        return await self.run_iteration(workflow_id)

    async def run_iteration(self, workflow_id: str) -> IterationResult:
        """Runs the next logical step in the workflow."""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found.")

        state = self.workflows[workflow_id]
        state.updated_at = datetime.utcnow()

        # 1. Analyze task if no plan exists
        if not state.plan:
            logger.info(f"[{workflow_id}] Analyzing task...")
            # This would be a call to the TaskAnalyzer tool via the agent
            # For now, we mock the result
            analysis_result = {
                "plan": {
                    "steps": [
                        {"step": 1, "agent": "backend", "task": "Setup database"},
                        {"step": 2, "agent": "frontend", "task": "Create UI mockups"},
                    ]
                }
            }
            state.plan = analysis_result["plan"]
            state.status = "planning_complete"
            return IterationResult(
                workflow_id=workflow_id,
                status="planning_complete",
                message="Task analyzed. Plan created.",
                next_actions=["delegate_next_task"],
                progress_percentage=10
            )

        # 2. Find and delegate the next task
        next_step = self._get_next_step(state)
        if next_step:
            logger.info(f"[{workflow_id}] Delegating step {next_step['step']}: {next_step['task']}")
            # This would be a call to the CoordinationTool via the agent
            # For now, we mock the result
            delegation_result = {
                "status": "completed",
                "artifact_id": f"artifact_step_{next_step['step']}.txt"
            }
            state.completed_steps.append(next_step['step'])
            state.artifacts[f"step_{next_step['step']}"] = delegation_result["artifact_id"]
            state.status = f"step_{next_step['step']}_complete"
            progress = int((len(state.completed_steps) / len(state.plan.get("steps", []))) * 100)

            return IterationResult(
                workflow_id=workflow_id,
                status=state.status,
                message=f"Step {next_step['step']} completed.",
                artifacts=state.artifacts,
                next_actions=["delegate_next_task", "validate_artifacts"],
                progress_percentage=progress
            )

        # 3. If all steps are done, finalize
        state.status = "completed"
        logger.info(f"[{workflow_id}] Workflow completed.")
        return IterationResult(
            workflow_id=workflow_id,
            status="completed",
            message="All workflow steps completed successfully.",
            artifacts=state.artifacts,
            progress_percentage=100
        )

    def _get_next_step(self, state: WorkflowState) -> Optional[Dict[str, Any]]:
        """Determines the next step to execute based on the plan and completed steps."""
        all_steps = state.plan.get("steps", [])
        for step in all_steps:
            if step["step"] not in state.completed_steps:
                # In a real app, you would also check dependencies here
                return step
        return None

    async def start_workflow(self, 
                           workflow_id: str,
                           user_input: str,
                           user_preferences: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Start a new multi-agent workflow
        
        Args:
            workflow_id: Unique identifier for the workflow
            user_input: User requirements and instructions
            user_preferences: User preferences for the workflow
            
        Returns:
            Initial workflow status and plan
        """
        try:
            logger.info(f"Starting workflow {workflow_id}")
            
            # Phase 1: Analyze requirements
            analysis_result = await self.task_analyzer.analyze_requirements(user_input)
            
            # Create workflow state
            workflow_state = await self.coordination_tool.create_workflow(workflow_id)
            
            # Initialize workflow tracking
            self.active_workflows[workflow_id] = {
                "analysis": analysis_result,
                "preferences": user_preferences or {},
                "iterations": [],
                "current_iteration": 0,
                "phase": WorkflowPhase.ANALYSIS,
                "start_time": datetime.utcnow(),
                "user_input": user_input,
                "artifacts": {},
                "validation_history": [],
                "refinement_requests": []
            }
            
            # Phase 2: Create execution plan
            execution_plan = await self._create_execution_plan(workflow_id, analysis_result)
            
            # Update workflow
            self.active_workflows[workflow_id]["execution_plan"] = execution_plan
            self.active_workflows[workflow_id]["phase"] = WorkflowPhase.PLANNING
            
            logger.info(f"Workflow {workflow_id} initialized with {len(analysis_result.requirements)} requirements")
            
            return {
                "workflow_id": workflow_id,
                "status": "started",
                "analysis": analysis_result.model_dump(),
                "execution_plan": execution_plan,
                "estimated_iterations": min(len(analysis_result.requirements) // 2 + 1, self.max_iterations),
                "next_phase": WorkflowPhase.EXECUTION
            }
            
        except Exception as e:
            logger.error(f"Error starting workflow {workflow_id}: {str(e)}")
            raise

    async def execute_workflow_iteration(self, workflow_id: str) -> IterationResult:
        """
        Execute one iteration of the workflow
        
        Args:
            workflow_id: Workflow identifier
            
        Returns:
            Results of the iteration
        """
        try:
            if workflow_id not in self.active_workflows:
                raise ValueError(f"Workflow {workflow_id} not found")
            
            workflow = self.active_workflows[workflow_id]
            workflow["current_iteration"] += 1
            current_iteration = workflow["current_iteration"]
            
            logger.info(f"Executing iteration {current_iteration} for workflow {workflow_id}")
            
            # Phase 3: Execute tasks
            workflow["phase"] = WorkflowPhase.EXECUTION
            execution_results = await self._execute_workflow_tasks(workflow_id)
            
            # Phase 4: Validate results
            workflow["phase"] = WorkflowPhase.VALIDATION
            validation_results = await self._validate_iteration_results(workflow_id, execution_results)
            
            # Phase 5: Determine refinements needed
            workflow["phase"] = WorkflowPhase.REFINEMENT
            refinements = await self._analyze_refinements_needed(workflow_id, validation_results)
            
            # Create iteration result
            iteration_result = IterationResult(
                iteration_number=current_iteration,
                phase=WorkflowPhase.EXECUTION,
                tasks_completed=execution_results.get("completed_tasks", []),
                artifacts_generated=execution_results.get("artifacts", {}),
                validation_score=validation_results.get("overall_score", 0.0),
                refinements_needed=refinements.get("refinements", []),
                next_actions=refinements.get("next_actions", [])
            )
            
            # Store iteration result
            workflow["iterations"].append(iteration_result)
            workflow["artifacts"].update(execution_results.get("artifacts", {}))
            workflow["validation_history"].append(validation_results)
            
            # Check if workflow is complete
            if (validation_results.get("overall_score", 0) >= self.validation_threshold or 
                current_iteration >= self.max_iterations or
                not refinements.get("refinements")):
                workflow["phase"] = WorkflowPhase.COMPLETION
                await self._finalize_workflow(workflow_id)
            
            logger.info(f"Iteration {current_iteration} completed for workflow {workflow_id}")
            return iteration_result
            
        except Exception as e:
            logger.error(f"Error in workflow iteration {workflow_id}: {str(e)}")
            raise

    async def _create_execution_plan(self, workflow_id: str, analysis: AnalysisResult) -> Dict[str, Any]:
        """Create detailed execution plan for the workflow"""
        try:
            plan = {
                "phases": [],
                "task_dependencies": {},
                "parallel_tasks": [],
                "critical_path": [],
                "resource_allocation": {}
            }
            
            # Group requirements by agent
            agent_requirements = {
                "frontend": [],
                "backend": [],
                "deployment": []
            }
            
            for req in analysis.requirements:
                for agent in req.agents_required:
                    if agent in agent_requirements:
                        agent_requirements[agent].append(req)
            
            # Create phases based on dependencies
            phases = []
            
            # Phase 1: Backend foundation (if needed)
            if agent_requirements["backend"]:
                backend_tasks = [req for req in agent_requirements["backend"] 
                               if not req.dependencies]
                if backend_tasks:
                    phases.append({
                        "name": "Backend Foundation",
                        "description": "Set up backend services and APIs",
                        "tasks": [task.requirement_id for task in backend_tasks],
                        "agents": ["backend"],
                        "estimated_duration": "2-4 hours"
                    })
            
            # Phase 2: Frontend development
            if agent_requirements["frontend"]:
                phases.append({
                    "name": "Frontend Development",
                    "description": "Create user interface and interactions",
                    "tasks": [req.requirement_id for req in agent_requirements["frontend"]],
                    "agents": ["frontend"],
                    "estimated_duration": "3-6 hours"
                })
            
            # Phase 3: Integration
            integration_tasks = [req for req in analysis.requirements 
                               if len(req.agents_required) > 1]
            if integration_tasks:
                phases.append({
                    "name": "Integration & Testing",
                    "description": "Connect frontend and backend components",
                    "tasks": [task.requirement_id for task in integration_tasks],
                    "agents": ["frontend", "backend"],
                    "estimated_duration": "2-3 hours"
                })
            
            # Phase 4: Deployment
            if agent_requirements["deployment"]:
                phases.append({
                    "name": "Deployment & Configuration",
                    "description": "Deploy to production environment",
                    "tasks": [req.requirement_id for req in agent_requirements["deployment"]],
                    "agents": ["deployment"],
                    "estimated_duration": "1-2 hours"
                })
            
            plan["phases"] = phases
            
            # Identify parallel tasks (tasks that can run simultaneously)
            parallel_tasks = []
            for phase in phases:
                if len(phase["agents"]) == 1:
                    # Tasks within the same agent can potentially run in parallel
                    phase_tasks = phase["tasks"]
                    if len(phase_tasks) > 1:
                        parallel_tasks.append(phase_tasks)
            
            plan["parallel_tasks"] = parallel_tasks
            
            # Resource allocation
            plan["resource_allocation"] = {
                "frontend": len(agent_requirements["frontend"]),
                "backend": len(agent_requirements["backend"]),
                "deployment": len(agent_requirements["deployment"])
            }
            
            return plan
            
        except Exception as e:
            logger.error(f"Error creating execution plan: {str(e)}")
            raise

    async def _execute_workflow_tasks(self, workflow_id: str) -> Dict[str, Any]:
        """Execute all tasks for current workflow iteration"""
        try:
            workflow = self.active_workflows[workflow_id]
            analysis = workflow["analysis"]
            execution_plan = workflow["execution_plan"]
            
            completed_tasks = []
            artifacts = {}
            task_results = {}
            
            # Execute phases in order
            for phase in execution_plan["phases"]:
                logger.info(f"Executing phase: {phase['name']}")
                
                # Update workflow stage
                await self.coordination_tool.update_workflow_stage(workflow_id, phase['name'])
                
                # Execute tasks in parallel for each agent in the phase
                phase_tasks = []
                for agent in phase["agents"]:
                    agent_tasks = await self._create_agent_tasks_for_phase(
                        workflow_id, agent, phase, analysis
                    )
                    phase_tasks.extend(agent_tasks)
                
                # Execute phase tasks
                phase_results = await self._execute_phase_tasks(workflow_id, phase_tasks)
                
                # Collect results
                for result in phase_results:
                    if result.status.value == "completed":
                        completed_tasks.append(result.task_id)
                        task_results[result.task_id] = result
                        
                        # Collect artifacts
                        if result.agent_name not in artifacts:
                            artifacts[result.agent_name] = []
                        artifacts[result.agent_name].extend(result.artifacts)
                
                # Brief pause between phases
                await asyncio.sleep(1)
            
            return {
                "completed_tasks": completed_tasks,
                "artifacts": artifacts,
                "task_results": task_results,
                "execution_summary": f"Completed {len(completed_tasks)} tasks across {len(execution_plan['phases'])} phases"
            }
            
        except Exception as e:
            logger.error(f"Error executing workflow tasks: {str(e)}")
            raise

    async def _create_agent_tasks_for_phase(self, 
                                          workflow_id: str, 
                                          agent_name: str, 
                                          phase: Dict[str, Any],
                                          analysis: AnalysisResult) -> List[AgentTask]:
        """Create specific tasks for an agent in a phase"""
        tasks = []
        
        # Find requirements for this agent in this phase
        phase_requirements = [
            req for req in analysis.requirements 
            if req.requirement_id in phase["tasks"] and agent_name in req.agents_required
        ]
        
        for req in phase_requirements:
            # Create detailed instructions for the agent
            instructions = await self._generate_agent_instructions(agent_name, req, analysis)
            
            # Create agent task
            task = await self.coordination_tool.assign_task(
                workflow_id=workflow_id,
                agent_name=agent_name,
                task_type=req.task_type.value,
                instructions=instructions,
                input_data={
                    "requirement": req.model_dump(),
                    "user_intent": analysis.user_intent,
                    "project_type": analysis.project_type,
                    "phase": phase["name"]
                },
                dependencies=req.dependencies,
                priority=req.priority.value
            )
            
            tasks.append(task)
        
        return tasks

    async def _generate_agent_instructions(self, 
                                         agent_name: str, 
                                         requirement: TaskRequirement,
                                         analysis: AnalysisResult) -> str:
        """Generate detailed instructions for a specific agent"""
        
        base_context = f"""
You are the {agent_name} agent working on a {analysis.project_type} project.

Project Goal: {analysis.user_intent}

Current Requirement: {requirement.description}
Priority: {requirement.priority}
Complexity: {requirement.estimated_complexity}/10

Success Criteria:
{chr(10).join('- ' + criteria for criteria in analysis.success_criteria)}
"""
        
        if agent_name == "frontend":
            specific_instructions = f"""
Frontend-Specific Instructions:
- Create modern, responsive web interfaces using HTML5, CSS (Tailwind CSS), and JavaScript
- Ensure all components are accessible and follow UI/UX best practices
- Generate clean, semantic HTML structure
- Implement interactive features using vanilla JavaScript or modern frameworks
- Make the interface mobile-friendly and cross-browser compatible
- Focus on user experience and visual appeal

Expected Deliverables:
- HTML files with semantic structure
- CSS stylesheets (preferably using Tailwind CSS)
- JavaScript files with interactive functionality
- Any additional assets (images, icons, fonts)

Technical Requirements:
- Use semantic HTML5 elements
- Implement responsive design principles
- Follow accessibility guidelines (WCAG)
- Optimize for performance and loading speed
"""
        
        elif agent_name == "backend":
            specific_instructions = f"""
Backend-Specific Instructions:
- Develop robust server-side logic and APIs
- Implement secure authentication and authorization
- Design and set up database schemas and relationships
- Create RESTful API endpoints with proper documentation
- Handle data validation, sanitization, and business logic
- Implement error handling and logging

Expected Deliverables:
- API endpoint implementations
- Database schema and migration files
- Authentication and authorization logic
- Data models and validation logic
- API documentation (OpenAPI/Swagger)
- Configuration files and environment setup

Technical Requirements:
- Use secure coding practices
- Implement proper error handling
- Follow REST API conventions
- Ensure data integrity and consistency
- Optimize database queries and performance
"""
        
        elif agent_name == "deployment":
            specific_instructions = f"""
Deployment-Specific Instructions:
- Prepare application for production deployment
- Set up containerization and orchestration
- Configure cloud infrastructure and services
- Implement CI/CD pipelines and automation
- Set up monitoring, logging, and alerting
- Ensure security and compliance requirements

Expected Deliverables:
- Containerization files (Dockerfile, docker-compose.yml)
- Cloud infrastructure configuration
- CI/CD pipeline configuration
- Monitoring and logging setup
- Security configurations
- Deployment documentation

Technical Requirements:
- Implement infrastructure as code
- Set up automated deployment processes
- Configure load balancing and scaling
- Implement security best practices
- Set up backup and disaster recovery
"""
        
        else:
            specific_instructions = f"Complete the {requirement.description} according to best practices for {agent_name}."
        
        return base_context + "\n" + specific_instructions

    async def _execute_phase_tasks(self, workflow_id: str, tasks: List[AgentTask]) -> List[AgentResponse]:
        """Execute a list of tasks, handling dependencies and parallelization"""
        results = []
        pending_tasks = tasks.copy()
        
        while pending_tasks:
            # Find tasks ready to execute (dependencies satisfied)
            ready_tasks = []
            for task in pending_tasks:
                if await self.coordination_tool.check_dependencies(task, workflow_id):
                    ready_tasks.append(task)
            
            if not ready_tasks:
                logger.warning("No ready tasks found, breaking potential deadlock")
                break
            
            # Execute ready tasks in parallel (up to 3 at a time)
            batch_size = min(3, len(ready_tasks))
            task_batch = ready_tasks[:batch_size]
            
            # Execute batch
            batch_results = await asyncio.gather(*[
                self.coordination_tool.delegate_to_agent(task.agent_name, task)
                for task in task_batch
            ])
            
            # Process results
            for result in batch_results:
                results.append(result)
                await self.coordination_tool.handle_task_completion(workflow_id, result.task_id, result)
            
            # Remove completed tasks from pending
            completed_task_ids = {result.task_id for result in batch_results}
            pending_tasks = [task for task in pending_tasks if task.task_id not in completed_task_ids]
            
            # Brief pause between batches
            await asyncio.sleep(0.5)
        
        return results

    async def _validate_iteration_results(self, workflow_id: str, execution_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the results of an iteration"""
        try:
            workflow = self.active_workflows[workflow_id]
            analysis = workflow["analysis"]
            
            # Collect all artifacts and results
            artifacts = execution_results.get("artifacts", {})
            task_results = execution_results.get("task_results", {})
            
            # Validate against success criteria
            validation_results = await self.validation_tool.validate_workflow_results(
                workflow_id=workflow_id,
                success_criteria=analysis.success_criteria,
                artifacts=artifacts,
                task_results=task_results,
                user_input=workflow["user_input"]
            )
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error validating iteration results: {str(e)}")
            return {
                "overall_score": 0.0,
                "validation_errors": [str(e)],
                "recommendations": ["Fix validation errors and retry"]
            }

    async def _analyze_refinements_needed(self, workflow_id: str, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze what refinements are needed based on validation results"""
        try:
            refinements = []
            next_actions = []
            
            overall_score = validation_results.get("overall_score", 0.0)
            
            # If validation score is low, identify specific areas for improvement
            if overall_score < self.validation_threshold:
                validation_details = validation_results.get("validation_details", {})
                
                for category, score in validation_details.items():
                    if score < 0.7:  # Category needs improvement
                        if "frontend" in category.lower():
                            refinements.append(f"Improve {category}: enhance UI/UX design and responsiveness")
                            next_actions.append("frontend: refine user interface based on validation feedback")
                        elif "backend" in category.lower():
                            refinements.append(f"Improve {category}: optimize API performance and data handling")
                            next_actions.append("backend: enhance API endpoints and data processing")
                        elif "deployment" in category.lower():
                            refinements.append(f"Improve {category}: optimize deployment configuration and monitoring")
                            next_actions.append("deployment: improve infrastructure and monitoring setup")
                        else:
                            refinements.append(f"Improve {category}")
                            next_actions.append(f"Address issues in {category}")
            
            # Check for specific validation errors
            validation_errors = validation_results.get("validation_errors", [])
            for error in validation_errors:
                refinements.append(f"Fix validation error: {error}")
                next_actions.append(f"Resolve: {error}")
            
            # If no specific refinements identified but score is low, provide general guidance
            if not refinements and overall_score < self.validation_threshold:
                refinements.append("General quality improvements needed")
                next_actions.append("Review all components and enhance based on user requirements")
            
            return {
                "refinements": refinements,
                "next_actions": next_actions,
                "priority_level": "high" if overall_score < 0.5 else "medium" if overall_score < 0.8 else "low"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing refinements: {str(e)}")
            return {
                "refinements": ["Error in refinement analysis"],
                "next_actions": ["Investigate and fix analysis errors"],
                "priority_level": "high"
            }

    async def _finalize_workflow(self, workflow_id: str):
        """Finalize a completed workflow"""
        try:
            workflow = self.active_workflows[workflow_id]
            workflow["phase"] = WorkflowPhase.COMPLETION
            workflow["completion_time"] = datetime.utcnow()
            workflow["total_duration"] = workflow["completion_time"] - workflow["start_time"]
            
            # Generate final summary
            final_summary = await self._generate_workflow_summary(workflow_id)
            workflow["final_summary"] = final_summary
            
            logger.info(f"Workflow {workflow_id} finalized successfully")
            
        except Exception as e:
            logger.error(f"Error finalizing workflow {workflow_id}: {str(e)}")

    async def _generate_workflow_summary(self, workflow_id: str) -> Dict[str, Any]:
        """Generate a comprehensive summary of the completed workflow"""
        workflow = self.active_workflows[workflow_id]
        
        total_tasks = len(workflow.get("iterations", []))
        if total_tasks > 0:
            last_iteration = workflow["iterations"][-1]
            final_score = last_iteration.validation_score
        else:
            final_score = 0.0
        
        return {
            "workflow_id": workflow_id,
            "project_type": workflow["analysis"].project_type,
            "user_intent": workflow["analysis"].user_intent,
            "total_iterations": workflow["current_iteration"],
            "final_validation_score": final_score,
            "total_duration": str(workflow.get("total_duration", "Unknown")),
            "agents_used": list(workflow.get("artifacts", {}).keys()),
            "total_artifacts": sum(len(artifacts) for artifacts in workflow.get("artifacts", {}).values()),
            "success_criteria_met": final_score >= self.validation_threshold,
            "completion_status": "successful" if final_score >= self.validation_threshold else "completed_with_issues"
        }

    async def get_workflow_progress(self, workflow_id: str) -> Dict[str, Any]:
        """Get current progress of a workflow"""
        if workflow_id not in self.active_workflows:
            return {"error": "Workflow not found"}
        
        workflow = self.active_workflows[workflow_id]
        coordination_status = await self.coordination_tool.get_workflow_status(workflow_id)
        
        progress_data = {
            "workflow_id": workflow_id,
            "phase": workflow["phase"],
            "current_iteration": workflow["current_iteration"],
            "max_iterations": self.max_iterations,
            "progress_percentage": min((workflow["current_iteration"] / self.max_iterations) * 100, 100),
            "coordination_status": coordination_status,
            "validation_history": workflow.get("validation_history", []),
            "latest_artifacts": workflow.get("artifacts", {})
        }
        
        # Add latest validation score if available
        if workflow.get("validation_history"):
            latest_validation = workflow["validation_history"][-1]
            progress_data["latest_validation_score"] = latest_validation.get("overall_score", 0.0)
        
        return progress_data

    async def request_user_feedback(self, workflow_id: str) -> Dict[str, Any]:
        """Request feedback from user on current workflow state"""
        if workflow_id not in self.active_workflows:
            return {"error": "Workflow not found"}
        
        workflow = self.active_workflows[workflow_id]
        
        # Prepare current state summary for user review
        current_artifacts = workflow.get("artifacts", {})
        latest_iteration = workflow["iterations"][-1] if workflow["iterations"] else None
        
        feedback_request = {
            "workflow_id": workflow_id,
            "current_phase": workflow["phase"],
            "iteration": workflow["current_iteration"],
            "artifacts_available": {
                agent: len(files) for agent, files in current_artifacts.items()
            },
            "validation_score": latest_iteration.validation_score if latest_iteration else 0.0,
            "questions_for_user": [
                "Are you satisfied with the current progress?",
                "Do the generated artifacts meet your expectations?",
                "What specific improvements would you like to see?",
                "Should we continue with more iterations or finalize the current state?"
            ],
            "feedback_options": [
                "continue_iterations",
                "request_specific_changes", 
                "finalize_current_state",
                "restart_with_modifications"
            ]
        }
        
        return feedback_request

    async def process_user_feedback(self, workflow_id: str, user_feedback: Dict[str, Any]) -> Dict[str, Any]:
        """Process user feedback and adjust workflow accordingly"""
        if workflow_id not in self.active_workflows:
            return {"error": "Workflow not found"}
        
        workflow = self.active_workflows[workflow_id]
        
        # Store user feedback
        if "user_feedback_history" not in workflow:
            workflow["user_feedback_history"] = []
        
        workflow["user_feedback_history"].append({
            "timestamp": datetime.utcnow(),
            "feedback": user_feedback
        })
        
        # Process feedback and determine next actions
        feedback_action = user_feedback.get("action", "continue_iterations")
        
        if feedback_action == "continue_iterations":
            return {"status": "continuing", "message": "Continuing with next iteration"}
        
        elif feedback_action == "request_specific_changes":
            # Add specific refinement requests
            specific_requests = user_feedback.get("specific_requests", [])
            workflow["refinement_requests"].extend(specific_requests)
            return {"status": "refining", "message": f"Added {len(specific_requests)} specific refinement requests"}
        
        elif feedback_action == "finalize_current_state":
            await self._finalize_workflow(workflow_id)
            return {"status": "finalized", "message": "Workflow finalized based on user request"}
        
        elif feedback_action == "restart_with_modifications":
            # Create a new workflow with modifications
            modifications = user_feedback.get("modifications", {})
            return {"status": "restarting", "message": "Workflow restart requested", "modifications": modifications}
        
        return {"status": "processed", "message": "User feedback processed successfully"}
