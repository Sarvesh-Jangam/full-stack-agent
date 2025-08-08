import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from agent.manager.agent import ManagerRequest, process_manager_request, default_runner

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the FastAPI app
app = FastAPI(
    title="Full-Stack Agent Server",
    description="An API to interact with the multi-agent system.",
    version="1.0.0"
)

# --- Pydantic models for API input and output ---
class StartWorkflowRequest(BaseModel):
    task_description: str
    user_id: str = "default_user"
    session_id: str = "default_session"

class WorkflowStatusResponse(BaseModel):
    workflow_id: str
    status: str
    message: str
    progress: int

# --- API Endpoints ---

@app.post("/workflow/start", response_model=WorkflowStatusResponse)
async def start_workflow(request: StartWorkflowRequest):
    """
    Starts a new development workflow based on a user's task description.
    """
    try:
        logging.info(f"Received request to start workflow for user: {request.user_id}")

        # 1. Create a session for the new workflow
        await default_runner.create_session(user_id=request.user_id, session_id=request.session_id)

        # 2. Create the initial request for the manager agent
        manager_request = ManagerRequest(
            action="start_workflow",
            payload={"task_description": request.task_description}
        )

        # 3. Process the request to kick off the workflow
        response = await process_manager_request(
            request=manager_request,
            user_id=request.user_id,
            session_id=request.session_id
        )

        logging.info(f"Workflow started successfully. Workflow ID: {response.workflow_id}")

        return WorkflowStatusResponse(
            workflow_id=response.workflow_id,
            status=response.status,
            message=response.message,
            progress=response.progress_percentage
        )

    except Exception as e:
        logging.error(f"Error starting workflow: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/workflow/status/{workflow_id}")
async def get_workflow_status(workflow_id: str):
    """
    (This is a placeholder) In a real app, you would fetch the workflow's
    current state from your WorkflowManager.
    """
    # This is a simplified example. In a real application, you would look up
    # the workflow_id in your workflow_manager's state.
    return {"workflow_id": workflow_id, "status": "in_progress", "message": "Fetching status is not fully implemented in this example."}


# --- How to Run This Server ---
# Open your terminal in the root of the 'full-stack-agent' project and run:
# uvicorn agent.main:app --reload
if __name__ == "__main__":
    uvicorn.run("agent.main:app", host="0.0.0.0", port=8000, reload=True)
