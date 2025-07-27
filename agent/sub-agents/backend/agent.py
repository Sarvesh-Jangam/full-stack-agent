"""
Main Backend Agent Implementation
Core agent that orchestrates all backend operations including database, API, auth, and file processing.
"""

import os
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService, VertexAiSessionService
from google.genai import types
from pydantic import BaseModel, Field

from .tools.database_tools import DatabaseTool
from .tools.api_tools import APIIntegrationTool
from .tools.auth_tools import AuthenticationTool
from .tools.file_tools import FileProcessingTool
from .config.database import DatabaseConfig
from .config.auth_config import AuthConfig
from .config.api_config import APIConfig

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format=os.getenv('LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
    handlers=[
        logging.FileHandler(os.getenv('LOG_FILE', 'backend_agent.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class BackendAgentRequest(BaseModel):
    """Request model for backend agent operations"""
    action: str = Field(..., description="The action to perform")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Request payload")
    user_id: Optional[str] = Field(None, description="User ID for authentication")
    session_id: Optional[str] = Field(None, description="Session ID")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class BackendAgentResponse(BaseModel):
    """Response model for backend agent operations"""
    status: str = Field(..., description="Response status")
    data: Any = Field(None, description="Response data")
    message: str = Field("", description="Response message")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    execution_time: Optional[float] = Field(None, description="Execution time in seconds")

# Initialize configuration
db_config = DatabaseConfig()
auth_config = AuthConfig()
api_config = APIConfig()

# Initialize tools
database_tool = DatabaseTool(db_config)
api_tool = APIIntegrationTool(api_config)
auth_tool = AuthenticationTool(auth_config)
file_tool = FileProcessingTool()

# Define the backend agent
backend_agent = Agent(
    model=LiteLlm(os.getenv('DEFAULT_MODEL', 'gemini-2.0-flash')),
    name=os.getenv('AGENT_NAME', 'backend_agent'),
    description=os.getenv('AGENT_DESCRIPTION', 'Backend agent for server-side operations and data management'),
    instruction="""
    You are an expert backend agent responsible for handling all server-side operations. Your capabilities include:

    1. DATABASE OPERATIONS:
       - Execute SQL queries safely with parameterization
       - Perform CRUD operations on various data models
       - Handle database transactions and rollbacks
       - Manage database connections and connection pooling
       - Execute complex joins and aggregations

    2. API INTEGRATIONS:
       - Make HTTP requests to external APIs
       - Handle different authentication methods (API keys, OAuth, JWT)
       - Parse and transform API responses
       - Implement retry logic and error handling
       - Rate limit API calls appropriately

    3. AUTHENTICATION & AUTHORIZATION:
       - Generate and validate JWT tokens
       - Handle OAuth 2.0 flows
       - Manage user sessions and permissions
       - Implement role-based access control
       - Secure password hashing and verification

    4. FILE PROCESSING:
       - Handle file uploads and downloads
       - Process various file formats (CSV, JSON, XML, PDF)
       - Extract data from documents
       - Generate reports and export data
       - Manage file storage and cleanup

    5. BUSINESS LOGIC:
       - Implement complex workflows and business rules
       - Handle data validation and transformation
       - Coordinate between different services
       - Process batch operations
       - Generate analytics and insights

    Always:
    - Use proper error handling and return meaningful error messages
    - Log important operations for debugging and monitoring
    - Validate input data before processing
    - Use database transactions for data consistency
    - Implement security best practices
    - Return structured responses with status, data, and messages

    For database operations, always use parameterized queries to prevent SQL injection.
    For API calls, implement proper timeout and retry mechanisms.
    For authentication, follow OAuth 2.0 and JWT best practices.
    For file processing, validate file types and sizes before processing.
    """,
    tools=[database_tool, api_tool, auth_tool, file_tool],
    input_schema=BackendAgentRequest,
    output_key="backend_result"
)

class BackendAgentRunner:
    """Runner class for the backend agent with additional functionality"""
    
    def __init__(self, 
                 session_service: Optional[Union[InMemorySessionService, VertexAiSessionService]] = None,
                 app_name: str = "backend_agent_app"):
        self.app_name = app_name
        self.session_service = session_service or InMemorySessionService()
        self.runner = Runner(
            agent=backend_agent,
            app_name=app_name,
            session_service=self.session_service
        )
        logger.info(f"Backend Agent Runner initialized with app_name: {app_name}")

    async def process_request(self, 
                            request: BackendAgentRequest,
                            user_id: str = "default_user",
                            session_id: str = "default_session") -> BackendAgentResponse:
        """
        Process a backend agent request
        
        Args:
            request: The request to process
            user_id: User ID for the session
            session_id: Session ID
            
        Returns:
            BackendAgentResponse with the result
        """
        start_time = datetime.utcnow()
        
        try:
            logger.info(f"Processing request: {request.action} for user: {user_id}")
            
            # Create user content for the agent
            request_content = request.model_dump_json()
            user_content = types.Content(
                role='user',
                parts=[types.Part(text=request_content)]
            )
            
            # Process the request through the agent
            final_response = None
            async for event in self.runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=user_content
            ):
                if event.is_final_response() and event.content and event.content.parts:
                    final_response = event.content.parts[0].text
                    break
            
            # Calculate execution time
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            if final_response:
                logger.info(f"Request processed successfully in {execution_time:.2f}s")
                return BackendAgentResponse(
                    status="success",
                    data=final_response,
                    message="Request processed successfully",
                    execution_time=execution_time
                )
            else:
                logger.error("No response received from agent")
                return BackendAgentResponse(
                    status="error",
                    message="No response received from agent",
                    execution_time=execution_time
                )
                
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            logger.error(f"Error processing request: {str(e)}")
            return BackendAgentResponse(
                status="error",
                message=f"Error processing request: {str(e)}",
                execution_time=execution_time
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

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on the backend agent"""
        try:
            return {
                "status": "healthy",
                "agent_name": backend_agent.name,
                "tools_count": len(backend_agent.tools),
                "timestamp": datetime.utcnow().isoformat(),
                "database_status": database_tool.health_check(),
                "api_status": api_tool.health_check(),
                "auth_status": auth_tool.health_check(),
                "file_status": file_tool.health_check()
            }
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

# Create default runner instance
default_runner = BackendAgentRunner()

# Convenience functions
async def process_backend_request(request: BackendAgentRequest, 
                                user_id: str = "default_user",
                                session_id: str = "default_session") -> BackendAgentResponse:
    """Convenience function to process a backend request"""
    return await default_runner.process_request(request, user_id, session_id)

def get_backend_agent_info() -> Dict[str, Any]:
    """Get information about the backend agent"""
    return {
        "name": backend_agent.name,
        "description": backend_agent.description,
        "model": str(backend_agent.model),
        "tools": [tool.__class__.__name__ for tool in backend_agent.tools],
        "version": "1.0.0"
    }

if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def main():
        # Test the backend agent
        test_request = BackendAgentRequest(
            action="health_check",
            payload={}
        )
        
        response = await process_backend_request(test_request)
        print(f"Backend Agent Response: {response}")
        
        # Health check
        health = default_runner.health_check()
        print(f"Health Check: {health}")

    asyncio.run(main())
