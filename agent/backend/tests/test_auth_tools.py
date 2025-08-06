"""
Tests for Authentication Tools
"""

import pytest
import asyncio
from datetime import datetime, timedelta

from backend_agent.tools.auth_tools import AuthenticationTool, UserCredentials, create_auth_tools
from backend_agent.config.auth_config import AuthConfig

class TestAuthTools:
    """Test suite for Authentication Tools"""
    
    def setup_method(self):
        """Setup test environment"""
        config = AuthConfig()
        self.auth_tool = AuthenticationTool(config)
    
    def test_health_check(self):
        """Test auth tool health check"""
        health = self.auth_tool.health_check()
        assert "status" in health
        assert "oauth_sessions" in health
        assert "encryption_enabled" in health
    
    def test_password_hashing(self):
        """Test password hashing and verification"""
        password = "testpassword123"
        hashed = self.auth_tool._hash_password(password)
        
        assert hashed != password
        assert self.auth_tool._verify_password(password, hashed) is True
        assert self.auth_tool._verify_password("wrongpassword", hashed) is False
    
    def test_jwt_token_generation(self):
        """Test JWT token generation"""
        user_id = "test_user_123"
        token = self.auth_tool.generate_jwt_token(user_id)
        
        assert isinstance(token, str)
        assert len(token) > 0
    
    def test_jwt_token_verification(self):
        """Test JWT token verification"""
        user_id = "test_user_123"
        token = self.auth_tool.generate_jwt_token(user_id)
        
        result = self.auth_tool.verify_jwt_token(token)
        assert result["valid"] is True
        assert result["user_id"] == user_id
    
    def test_jwt_token_expiration(self):
        """Test JWT token expiration"""
        user_id = "test_user_123"
        
        # Create expired token (simulate by setting past expiration)
        import jwt
        from datetime import datetime, timedelta
        
        payload = {
            "user_id": user_id,
            "exp": datetime.utcnow() - timedelta(hours=1)  # Expired 1 hour ago
        }
        
        expired_token = jwt.encode(
            payload,
            self.auth_tool.config.jwt_secret_key,
            algorithm=self.auth_tool.config.jwt_algorithm
        )
        
        result = self.auth_tool.verify_jwt_token(expired_token)
        assert result["valid"] is False
        assert result["error_code"] == "TOKEN_EXPIRED"
    
    @pytest.mark.asyncio
    async def test_user_authentication_success(self):
        """Test successful user authentication"""
        credentials = UserCredentials(
            email="admin@example.com",
            password="admin123"
        )
        
        result = await self.auth_tool.authenticate_user(credentials)
        assert result.success is True
        assert result.user_profile is not None
        assert result.tokens is not None
    
    @pytest.mark.asyncio
    async def test_user_authentication_failure(self):
        """Test failed user authentication"""
        credentials = UserCredentials(
            email="nonexistent@example.com",
            password="wrongpassword"
        )
        
        result = await self.auth_tool.authenticate_user(credentials)
        assert result.success is False
        assert result.user_profile is None
        assert result.tokens is None
    
    @pytest.mark.asyncio
    async def test_token_refresh(self):
        """Test token refresh functionality"""
        # First authenticate to get tokens
        credentials = UserCredentials(
            email="admin@example.com",
            password="admin123"
        )
        
        auth_result = await self.auth_tool.authenticate_user(credentials)
        assert auth_result.success is True
        
        # Then refresh the token
        refresh_result = await self.auth_tool.refresh_access_token(
            auth_result.tokens.refresh_token
        )
        assert refresh_result.success is True
        assert refresh_result.tokens is not None
    
    @pytest.mark.asyncio
    async def test_oauth_flow_initiation(self):
        """Test OAuth flow initiation"""
        result = await self.auth_tool.initiate_oauth_flow("google")
        
        assert result["success"] is True
        assert "authorization_url" in result
        assert "state" in result
        assert "provider" in result
    
    def test_permission_checking(self):
        """Test permission checking"""
        # Test admin permissions
        admin_roles = ["admin"]
        required_permissions = ["read", "write", "delete"]
        
        result = self.auth_tool.check_permissions(admin_roles, required_permissions)
        assert result is True
        
        # Test user permissions
        user_roles = ["user"]
        admin_permissions = ["delete", "admin"]
        
        result = self.auth_tool.check_permissions(user_roles, admin_permissions)
        assert result is False
    
    def test_data_encryption(self):
        """Test data encryption and decryption"""
        if self.auth_tool.cipher:
            sensitive_data = "sensitive information"
            encrypted = self.auth_tool.encrypt_sensitive_data(sensitive_data)
            decrypted = self.auth_tool.decrypt_sensitive_data(encrypted)
            
            assert encrypted != sensitive_data
            assert decrypted == sensitive_data
    
    def test_tool_factory(self):
        """Test authentication tools factory"""
        config = AuthConfig()
        tools = create_auth_tools(config)
        assert len(tools) > 0
        assert callable(tools[0])
