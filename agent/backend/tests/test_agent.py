"""
Tests for the main Backend Agent
"""

import pytest
import asyncio
from datetime import datetime

from backend_agent.agent import backend_agent, BackendAgentRunner, BackendAgentRequest
from backend_agent.config.database import DatabaseConfig
from backend_agent.config.auth_config import AuthConfig
from backend_agent.config.api_config import APIConfig

class TestBackendAgent:
    """Test suite for Backend Agent"""
    
    def setup_method(self):
        """Setup test environment"""
        self.runner = BackendAgentRunner()
    
    def test_agent_initialization(self):
        """Test agent initialization"""
        assert backend_agent.name == "backend_agent"
        assert len(backend_agent.tools) > 0
        assert backend_agent.description is not None
    
    def test_agent_info(self):
        """Test agent information retrieval"""
        from backend_agent.agent import get_backend_agent_info
        
        info = get_backend_agent_info()
        assert "name" in info
        assert "description" in info
        assert "tools" in info
        assert "version" in info
    
    def test_health_check(self):
        """Test agent health check"""
        health = self.runner.health_check()
        assert "status" in health
        assert "agent_name" in health
        assert "tools_count" in health
    
    @pytest.mark.asyncio
    async def test_simple_request(self):
        """Test simple agent request"""
        request = BackendAgentRequest(
            action="health_check",
            payload={}
        )
        
        response = await self.runner.process_request(request)
        assert response.status in ["success", "error"]
        assert response.execution_time is not None
    
    @pytest.mark.asyncio
    async def test_session_management(self):
        """Test session creation and management"""
        user_id = "test_user"
        session_id = "test_session"
        
        # Create session
        await self.runner.create_session(user_id, session_id, {"test": "data"})
        
        # Get session state
        state = await self.runner.get_session_state(user_id, session_id)
        assert isinstance(state, dict)
    
    def test_configuration_loading(self):
        """Test configuration loading"""
        db_config = DatabaseConfig()
        auth_config = AuthConfig()
        api_config = APIConfig()
        
        assert db_config.database_url is not None
        assert auth_config.jwt_secret_key is not None
        assert api_config.external_api_timeout > 0
