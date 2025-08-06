"""
API Integration Tools for Backend Agent
Handles external API calls, authentication, and response processing.
"""

import os
import logging
import asyncio
import json
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from urllib.parse import urljoin, urlparse
from enum import Enum

import aiohttp
import httpx
from pydantic import BaseModel, Field, validator
import backoff

from ..config.api_config import APIConfig

logger = logging.getLogger(__name__)

class HTTPMethod(str, Enum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"

class APIRequest(BaseModel):
    """Model for API request configuration"""
    url: str = Field(..., description="API endpoint URL")
    method: HTTPMethod = Field(HTTPMethod.GET, description="HTTP method")
    headers: Dict[str, str] = Field(default_factory=dict, description="Request headers")
    params: Dict[str, Any] = Field(default_factory=dict, description="URL parameters")
    data: Optional[Dict[str, Any]] = Field(None, description="Request body data")
    timeout: int = Field(30, description="Request timeout in seconds")
    retry_attempts: int = Field(3, description="Number of retry attempts")

class APIResponse(BaseModel):
    """Model for API response"""
    success: bool = Field(..., description="Whether the request was successful")
    status_code: int = Field(..., description="HTTP status code")
    data: Any = Field(None, description="Response data")
    headers: Dict[str, str] = Field(default_factory=dict, description="Response headers")
    message: str = Field("", description="Response message")
    execution_time: Optional[float] = Field(None, description="Request execution time")
    url: str = Field("", description="Request URL")

class RateLimiter:
    """Simple rate limiter for API calls"""
    
    def __init__(self, calls_per_minute: int = 60):
        self.calls_per_minute = calls_per_minute
        self.calls = []
    
    async def wait_if_needed(self):
        """Wait if rate limit would be exceeded"""
        now = datetime.utcnow()
        # Remove calls older than 1 minute
        self.calls = [call_time for call_time in self.calls 
                     if now - call_time < timedelta(minutes=1)]
        
        if len(self.calls) >= self.calls_per_minute:
            wait_time = 60 - (now - self.calls[0]).total_seconds()
            if wait_time > 0:
                logger.info(f"Rate limit reached, waiting {wait_time:.1f} seconds")
                await asyncio.sleep(wait_time)
        
        self.calls.append(now)

class APIIntegrationTool:
    """Advanced API integration tool with authentication, retries, and rate limiting"""
    
    def __init__(self, config: APIConfig):
        self.config = config
        self.rate_limiter = RateLimiter(calls_per_minute=60)
        self._session = None
        logger.info("API integration tool initialized")

    async def _get_session(self) -> httpx.AsyncClient:
        """Get or create HTTP client session"""
        if self._session is None or self._session.is_closed:
            self._session = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0),
                verify=True,
                follow_redirects=True
            )
        return self._session

    async def _add_authentication(self, headers: Dict[str, str], auth_config: Dict[str, Any] = None):
        """Add authentication headers to the request"""
        if not auth_config:
            auth_config = self.config.get_default_auth()
        
        auth_type = auth_config.get('type', '').lower()
        
        if auth_type == 'api_key':
            key_name = auth_config.get('key_name', 'X-API-Key')
            api_key = auth_config.get('api_key')
            if api_key:
                headers[key_name] = api_key
        
        elif auth_type == 'bearer':
            token = auth_config.get('token')
            if token:
                headers['Authorization'] = f"Bearer {token}"
        
        elif auth_type == 'basic':
            username = auth_config.get('username')
            password = auth_config.get('password')
            if username and password:
                import base64
                credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
                headers['Authorization'] = f"Basic {credentials}"

    @backoff.on_exception(
        backoff.expo,
        (httpx.RequestError, httpx.HTTPStatusError),
        max_tries=3,
        max_time=300
    )
    async def make_request(self, request: APIRequest, auth_config: Dict[str, Any] = None) -> APIResponse:
        """
        Make HTTP request with retry logic and error handling
        
        Args:
            request: API request configuration
            auth_config: Authentication configuration
            
        Returns:
            APIResponse with the result
        """
        start_time = datetime.utcnow()
        
        try:
            # Rate limiting
            await self.rate_limiter.wait_if_needed()
            
            # Setup headers
            headers = request.headers.copy()
            headers.setdefault('Content-Type', 'application/json')
            headers.setdefault('User-Agent', 'Backend-Agent/1.0')
            
            # Add authentication
            await self._add_authentication(headers, auth_config)
            
            # Get session
            session = await self._get_session()
            
            # Prepare request data
            request_kwargs = {
                'method': request.method.value,
                'url': request.url,
                'headers': headers,
                'params': request.params if request.params else None,
                'timeout': request.timeout
            }
            
            # Add request body if present
            if request.data is not None:
                if headers.get('Content-Type') == 'application/json':
                    request_kwargs['json'] = request.data
                else:
                    request_kwargs['data'] = request.data
            
            logger.info(f"Making {request.method} request to {request.url}")
            
            # Make the request
            response = await session.request(**request_kwargs)
            
            # Calculate execution time
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Parse response data
            try:
                if response.headers.get('content-type', '').startswith('application/json'):
                    response_data = response.json()
                else:
                    response_data = response.text
            except Exception:
                response_data = response.text
            
            # Create response object
            api_response = APIResponse(
                success=response.is_success,
                status_code=response.status_code,
                data=response_data,
                headers=dict(response.headers),
                message=f"Request completed with status {response.status_code}",
                execution_time=execution_time,
                url=str(response.url)
            )
            
            if response.is_success:
                logger.info(f"Request successful: {response.status_code} in {execution_time:.3f}s")
            else:
                logger.warning(f"Request failed: {response.status_code} - {response.text}")
                response.raise_for_status()  # This will be caught by backoff decorator
            
            return api_response
            
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            logger.error(f"Request failed: {str(e)}")
            
            return APIResponse(
                success=False,
                status_code=0,
                message=f"Request failed: {str(e)}",
                execution_time=execution_time,
                url=request.url
            )

    async def get(self, url: str, params: Dict[str, Any] = None, headers: Dict[str, str] = None) -> APIResponse:
        """Make GET request"""
        request = APIRequest(
            url=url,
            method=HTTPMethod.GET,
            params=params or {},
            headers=headers or {}
        )
        return await self.make_request(request)

    async def post(self, url: str, data: Dict[str, Any] = None, headers: Dict[str, str] = None) -> APIResponse:
        """Make POST request"""
        request = APIRequest(
            url=url,
            method=HTTPMethod.POST,
            data=data,
            headers=headers or {}
        )
        return await self.make_request(request)

    async def put(self, url: str, data: Dict[str, Any] = None, headers: Dict[str, str] = None) -> APIResponse:
        """Make PUT request"""
        request = APIRequest(
            url=url,
            method=HTTPMethod.PUT,
            data=data,
            headers=headers or {}
        )
        return await self.make_request(request)

    async def delete(self, url: str, headers: Dict[str, str] = None) -> APIResponse:
        """Make DELETE request"""
        request = APIRequest(
            url=url,
            method=HTTPMethod.DELETE,
            headers=headers or {}
        )
        return await self.make_request(request)

    async def batch_requests(self, requests: List[APIRequest]) -> List[APIResponse]:
        """Execute multiple API requests concurrently"""
        try:
            logger.info(f"Executing {len(requests)} batch requests")
            
            # Create tasks for concurrent execution
            tasks = [self.make_request(request) for request in requests]
            
            # Execute all requests concurrently
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Convert exceptions to error responses
            final_responses = []
            for i, response in enumerate(responses):
                if isinstance(response, Exception):
                    final_responses.append(APIResponse(
                        success=False,
                        status_code=0,
                        message=f"Batch request failed: {str(response)}",
                        url=requests[i].url
                    ))
                else:
                    final_responses.append(response)
            
            logger.info(f"Batch requests completed: {sum(1 for r in final_responses if r.success)}/{len(final_responses)} successful")
            
            return final_responses
            
        except Exception as e:
            logger.error(f"Batch requests failed: {str(e)}")
            return [APIResponse(
                success=False,
                status_code=0,
                message=f"Batch execution failed: {str(e)}",
                url=""
            ) for _ in requests]

    async def upload_file(self, url: str, file_path: str, field_name: str = "file", 
                         additional_data: Dict[str, Any] = None) -> APIResponse:
        """Upload file to API endpoint"""
        try:
            session = await self._get_session()
            
            # Prepare multipart form data
            with open(file_path, 'rb') as f:
                files = {field_name: f}
                data = additional_data or {}
                
                response = await session.post(url, files=files, data=data)
                
                return APIResponse(
                    success=response.is_success,
                    status_code=response.status_code,
                    data=response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text,
                    message=f"File upload completed with status {response.status_code}",
                    url=url
                )
                
        except Exception as e:
            logger.error(f"File upload failed: {str(e)}")
            return APIResponse(
                success=False,
                status_code=0,
                message=f"File upload failed: {str(e)}",
                url=url
            )

    async def download_file(self, url: str, file_path: str) -> APIResponse:
        """Download file from API endpoint"""
        try:
            session = await self._get_session()
            
            response = await session.get(url)
            response.raise_for_status()
            
            # Save file
            with open(file_path, 'wb') as f:
                f.write(response.content)
            
            return APIResponse(
                success=True,
                status_code=response.status_code,
                data={"file_path": file_path, "size": len(response.content)},
                message=f"File downloaded successfully to {file_path}",
                url=url
            )
            
        except Exception as e:
            logger.error(f"File download failed: {str(e)}")
            return APIResponse(
                success=False,
                status_code=0,
                message=f"File download failed: {str(e)}",
                url=url
            )

    async def webhook_handler(self, webhook_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming webhook data"""
        try:
            logger.info(f"Processing webhook data: {webhook_data}")
            
            # Extract webhook metadata
            event_type = webhook_data.get('event_type', 'unknown')
            timestamp = webhook_data.get('timestamp', datetime.utcnow().isoformat())
            payload = webhook_data.get('payload', {})
            
            # Process webhook based on event type
            if event_type == 'user_created':
                result = await self._handle_user_created(payload)
            elif event_type == 'payment_completed':
                result = await self._handle_payment_completed(payload)
            elif event_type == 'data_updated':
                result = await self._handle_data_updated(payload)
            else:
                result = await self._handle_generic_webhook(payload)
            
            return {
                "success": True,
                "event_type": event_type,
                "processed_at": datetime.utcnow().isoformat(),
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Webhook processing failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "processed_at": datetime.utcnow().isoformat()
            }

    async def _handle_user_created(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle user created webhook"""
        # Implement user creation logic
        return {"action": "user_processed", "user_id": payload.get("user_id")}

    async def _handle_payment_completed(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle payment completed webhook"""
        # Implement payment processing logic
        return {"action": "payment_processed", "transaction_id": payload.get("transaction_id")}

    async def _handle_data_updated(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle data updated webhook"""
        # Implement data update logic
        return {"action": "data_synchronized", "record_id": payload.get("record_id")}

    async def _handle_generic_webhook(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle generic webhook"""
        # Implement generic webhook processing
        return {"action": "webhook_received", "payload_keys": list(payload.keys())}

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on API integration"""
        try:
            return {
                "status": "healthy",
                "session_active": self._session is not None and not self._session.is_closed,
                "rate_limiter_calls": len(self.rate_limiter.calls),
                "config_loaded": self.config is not None
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    async def close(self):
        """Close HTTP session"""
        try:
            if self._session and not self._session.is_closed:
                await self._session.aclose()
                logger.info("API session closed")
        except Exception as e:
            logger.error(f"Error closing API session: {str(e)}")

# Factory function to create API tools
def create_api_tools(config: APIConfig = None) -> List:
    """Create and return API tools for the agent"""
    if not config:
        config = APIConfig()
    
    api_tool = APIIntegrationTool(config)
    
    # Define individual tool functions that the agent can use
    async def make_api_request(url: str, method: str = "GET", data: dict = None, headers: dict = None) -> dict:
        """Make HTTP API request"""
        request = APIRequest(
            url=url,
            method=HTTPMethod(method.upper()),
            data=data,
            headers=headers or {}
        )
        response = await api_tool.make_request(request)
        return response.model_dump()
    
    async def get_api_data(url: str, params: dict = None) -> dict:
        """Make GET request to API"""
        response = await api_tool.get(url, params)
        return response.model_dump()
    
    async def post_api_data(url: str, data: dict) -> dict:
        """Make POST request to API"""
        response = await api_tool.post(url, data)
        return response.model_dump()
    
    async def process_webhook(webhook_data: dict) -> dict:
        """Process incoming webhook data"""
        return await api_tool.webhook_handler(webhook_data)
    
    return [
        make_api_request,
        get_api_data,
        post_api_data,
        process_webhook
    ]

if __name__ == "__main__":
    # Test the API tool
    async def test_api_tool():
        config = APIConfig()
        api_tool = APIIntegrationTool(config)
        
        # Test health check
        health = api_tool.health_check()
        print(f"Health check: {health}")
        
        # Test GET request
        response = await api_tool.get("https://httpbin.org/get")
        print(f"Test GET response: {response}")
        
        await api_tool.close()
    
    asyncio.run(test_api_tool())

