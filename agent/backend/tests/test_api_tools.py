"""
Tests for API Integration Tools
"""

import pytest
import asyncio
from unittest.mock import Mock, patch

from backend_agent.tools.api_tools import APIIntegrationTool, APIRequest, HTTPMethod, create_api_tools
from backend_agent.config.api_config import APIConfig

class TestAPITools:
    """Test suite for API Integration Tools"""
    
    def setup_method(self):
        """Setup test environment"""
        config = APIConfig()
        self.api_tool = APIIntegrationTool(config)
    
    def teardown_method(self):
        """Cleanup"""
        asyncio.run(self.api_tool.close())
    
    def test_health_check(self):
        """Test API tool health check"""
        health = self.api_tool.health_check()
        assert "status" in health
        assert "session_active" in health
    
    def test_api_request_creation(self):
        """Test API request model creation"""
        request = APIRequest(
            url="https://httpbin.org/get",
            method=HTTPMethod.GET,
            timeout=30
        )
        assert request.url == "https://httpbin.org/get"
        assert request.method == HTTPMethod.GET
        assert request.timeout == 30
    
    @pytest.mark.asyncio
    async def test_rate_limiter(self):
        """Test rate limiting functionality"""
        # Test that rate limiter doesn't block initially
        await self.api_tool.rate_limiter.wait_if_needed()
        assert len(self.api_tool.rate_limiter.calls) == 1
    
    @pytest.mark.asyncio
    @patch('httpx.AsyncClient.request')
    async def test_successful_request(self, mock_request):
        """Test successful API request"""
        # Mock successful response
        mock_response = Mock()
        mock_response.is_success = True
        mock_response.status_code = 200
        mock_response.json.return_value = {"test": "data"}
        mock_response.headers = {"content-type": "application/json"}
        mock_response.url = "https://httpbin.org/get"
        mock_request.return_value = mock_response
        
        request = APIRequest(
            url="https://httpbin.org/get",
            method=HTTPMethod.GET
        )
        
        response = await self.api_tool.make_request(request)
        assert response.success is True
        assert response.status_code == 200
        assert response.data == {"test": "data"}
    
    @pytest.mark.asyncio
    @patch('httpx.AsyncClient.request')
    async def test_failed_request(self, mock_request):
        """Test failed API request"""
        # Mock failed response
        mock_request.side_effect = Exception("Connection error")
        
        request = APIRequest(
            url="https://httpbin.org/get",
            method=HTTPMethod.GET
        )
        
        response = await self.api_tool.make_request(request)
        assert response.success is False
        assert "Connection error" in response.message
    
    @pytest.mark.asyncio
    async def test_batch_requests(self):
        """Test batch request processing"""
        requests = [
            APIRequest(url="https://httpbin.org/get"),
            APIRequest(url="https://httpbin.org/post", method=HTTPMethod.POST)
        ]
        
        with patch('httpx.AsyncClient.request') as mock_request:
            mock_response = Mock()
            mock_response.is_success = True
            mock_response.status_code = 200
            mock_response.json.return_value = {"test": "data"}
            mock_response.headers = {"content-type": "application/json"}
            mock_response.url = "https://httpbin.org/get"
            mock_request.return_value = mock_response
            
            responses = await self.api_tool.batch_requests(requests)
            assert len(responses) == 2
    
    @pytest.mark.asyncio
    async def test_webhook_handler(self):
        """Test webhook processing"""
        webhook_data = {
            "event_type": "user_created",
            "payload": {"user_id": "123", "email": "test@example.com"},
            "timestamp": "2023-01-01T00:00:00Z"
        }
        
        result = await self.api_tool.webhook_handler(webhook_data)
        assert result["success"] is True
        assert result["event_type"] == "user_created"
    
    def test_tool_factory(self):
        """Test API tools factory"""
        config = APIConfig()
        tools = create_api_tools(config)
        assert len(tools) > 0
        assert callable(tools[0])

