"""
Authentication Tools for Backend Agent
Handles JWT tokens, OAuth flows, session management, and user authentication.
"""

import os
import logging
import asyncio
import hashlib
import secrets
import base64
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from urllib.parse import urlencode, parse_qs

import jwt
import bcrypt
import httpx
from pydantic import BaseModel, Field, EmailStr
from cryptography.fernet import Fernet

from ..config.auth_config import AuthConfig

logger = logging.getLogger(__name__)

class UserCredentials(BaseModel):
    """Model for user authentication credentials"""
    username: Optional[str] = None
    email: Optional[EmailStr] = None
    password: str = Field(..., min_length=8)

class TokenPair(BaseModel):
    """Model for JWT token pair"""
    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="JWT refresh token")
    token_type: str = Field("Bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")

class UserProfile(BaseModel):
    """Model for user profile information"""
    user_id: str = Field(..., description="Unique user identifier")
    username: Optional[str] = None
    full_name: Optional[str] = None
    roles: List[str] = Field(default_factory=list, description="User roles")
    permissions: List[str] = Field(default_factory=list, description="User permissions")
    is_active: bool = Field(True, description="Whether user is active")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None

class AuthResult(BaseModel):
    """Model for authentication results"""
    success: bool = Field(..., description="Whether authentication was successful")
    user_profile: Optional[UserProfile] = None
    tokens: Optional[TokenPair] = None
    message: str = Field("", description="Authentication message")
    expires_at: Optional[datetime] = None

class AuthenticationTool:
    """Advanced authentication tool with JWT, OAuth, and session management"""
    
    def __init__(self, config: AuthConfig):
        self.config = config
        self.cipher = Fernet(self.config.encryption_key.encode()) if self.config.encryption_key else None
        self._oauth_sessions = {}  # Store OAuth state sessions
        logger.info("Authentication tool initialized")

    def _hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')

    def _verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

    def generate_jwt_token(self, 
                          user_id: str, 
                          user_data: Dict[str, Any] = None,
                          token_type: str = "access") -> str:
        """Generate JWT token for user"""
        try:
            now = datetime.utcnow()
            
            # Set expiration based on token type
            if token_type == "access":
                expires_delta = timedelta(minutes=self.config.jwt_access_token_expire_minutes)
            else:  # refresh token
                expires_delta = timedelta(days=self.config.jwt_refresh_token_expire_days)
            
            # Create payload
            payload = {
                "user_id": user_id,
                "token_type": token_type,
                "iat": now,
                "exp": now + expires_delta,
                "jti": secrets.token_urlsafe(32)  # JWT ID for token revocation
            }
            
            # Add additional user data
            if user_data:
                payload.update(user_data)
            
            # Generate token
            token = jwt.encode(
                payload, 
                self.config.jwt_secret_key, 
                algorithm=self.config.jwt_algorithm
            )
            
            logger.info(f"Generated {token_type} token for user {user_id}")
            return token
            
        except Exception as e:
            logger.error(f"Error generating JWT token: {str(e)}")
            raise

    def verify_jwt_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode JWT token"""
        try:
            # Decode token
            payload = jwt.decode(
                token, 
                self.config.jwt_secret_key, 
                algorithms=[self.config.jwt_algorithm]
            )
            
            # Check if token is expired
            exp = payload.get('exp')
            if exp and datetime.utcnow() > datetime.fromtimestamp(exp):
                return {
                    "valid": False,
                    "error": "Token has expired",
                    "error_code": "TOKEN_EXPIRED"
                }
            
            logger.info(f"Token verified for user {payload.get('user_id')}")
            return {
                "valid": True,
                "payload": payload,
                "user_id": payload.get('user_id'),
                "token_type": payload.get('token_type', 'access')
            }
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token verification failed: Token expired")
            return {
                "valid": False,
                "error": "Token has expired",
                "error_code": "TOKEN_EXPIRED"
            }
        except jwt.InvalidTokenError as e:
            logger.warning(f"Token verification failed: {str(e)}")
            return {
                "valid": False,
                "error": "Invalid token",
                "error_code": "INVALID_TOKEN"
            }
        except Exception as e:
            logger.error(f"Error verifying JWT token: {str(e)}")
            return {
                "valid": False,
                "error": f"Token verification error: {str(e)}",
                "error_code": "VERIFICATION_ERROR"
            }

    async def authenticate_user(self, credentials: UserCredentials) -> AuthResult:
        """Authenticate user with credentials"""
        try:
            # This would typically involve database lookup
            # For demo purposes, we'll simulate authentication
            
            # Validate credentials format
            if not credentials.password:
                return AuthResult(
                    success=False,
                    message="Password is required"
                )
            
            # Simulate user lookup (replace with actual database query)
            user_data = await self._lookup_user(credentials)
            
            if not user_data:
                return AuthResult(
                    success=False,
                    message="Invalid credentials"
                )
            
            # Verify password
            if not self._verify_password(credentials.password, user_data["password_hash"]):
                return AuthResult(
                    success=False,
                    message="Invalid credentials"
                )
            
            # Create user profile
            user_profile = UserProfile(
                user_id=user_data["user_id"],
                username=user_data.get("username"),
                email=user_data.get("email"),
                full_name=user_data.get("full_name"),
                roles=user_data.get("roles", []),
                permissions=user_data.get("permissions", []),
                last_login=datetime.utcnow()
            )
            
            # Generate token pair
            access_token = self.generate_jwt_token(
                user_profile.user_id,
                {
                    "username": user_profile.username,
                    "roles": user_profile.roles
                },
                "access"
            )
            
            refresh_token = self.generate_jwt_token(
                user_profile.user_id,
                {},
                "refresh"
            )
            
            tokens = TokenPair(
                access_token=access_token,
                refresh_token=refresh_token,
                expires_in=self.config.jwt_access_token_expire_minutes * 60
            )
            
            logger.info(f"User authenticated successfully: {user_profile.user_id}")
            
            return AuthResult(
                success=True,
                user_profile=user_profile,
                tokens=tokens,
                message="Authentication successful",
                expires_at=datetime.utcnow() + timedelta(minutes=self.config.jwt_access_token_expire_minutes)
            )
            
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            return AuthResult(
                success=False,
                message=f"Authentication failed: {str(e)}"
            )

    async def _lookup_user(self, credentials: UserCredentials) -> Optional[Dict[str, Any]]:
        """Lookup user in database (mock implementation)"""
        # This should be replaced with actual database lookup
        mock_users = {
            "admin@example.com": {
                "user_id": "user_001",
                "username": "admin",
                "email": "admin@example.com",
                "full_name": "System Administrator",
                "password_hash": self._hash_password("admin123"),
                "roles": ["admin", "user"],
                "permissions": ["read", "write", "delete", "admin"]
            },
            "user@example.com": {
                "user_id": "user_002", 
                "username": "user",
                "email": "user@example.com",
                "full_name": "Regular User",
                "password_hash": self._hash_password("user123"),
                "roles": ["user"],
                "permissions": ["read", "write"]
            }
        }
        
        # Lookup by email or username
        lookup_key = credentials.email or credentials.username
        return mock_users.get(lookup_key)

    async def refresh_access_token(self, refresh_token: str) -> AuthResult:
        """Refresh access token using refresh token"""
        try:
            # Verify refresh token
            token_data = self.verify_jwt_token(refresh_token)
            
            if not token_data["valid"]:
                return AuthResult(
                    success=False,
                    message="Invalid refresh token"
                )
            
            # Check token type
            if token_data["payload"].get("token_type") != "refresh":
                return AuthResult(
                    success=False,
                    message="Invalid token type"
                )
            
            user_id = token_data["user_id"]
            
            # Generate new access token
            access_token = self.generate_jwt_token(user_id, {}, "access")
            
            tokens = TokenPair(
                access_token=access_token,
                refresh_token=refresh_token,  # Keep the same refresh token
                expires_in=self.config.jwt_access_token_expire_minutes * 60
            )
            
            return AuthResult(
                success=True,
                tokens=tokens,
                message="Token refreshed successfully",
                expires_at=datetime.utcnow() + timedelta(minutes=self.config.jwt_access_token_expire_minutes)
            )
            
        except Exception as e:
            logger.error(f"Token refresh error: {str(e)}")
            return AuthResult(
                success=False,
                message=f"Token refresh failed: {str(e)}"
            )

    async def initiate_oauth_flow(self, provider: str, redirect_uri: str = None) -> Dict[str, Any]:
        """Initiate OAuth 2.0 authorization flow"""
        try:
            # Generate state parameter for CSRF protection
            state = secrets.token_urlsafe(32)
            
            # Store OAuth session
            self._oauth_sessions[state] = {
                "provider": provider,
                "created_at": datetime.utcnow(),
                "redirect_uri": redirect_uri
            }
            
            # Build authorization URL based on provider
            if provider == "google":
                auth_url = "https://accounts.google.com/o/oauth2/v2/auth"
                params = {
                    "client_id": self.config.oauth_client_id,
                    "redirect_uri": redirect_uri or self.config.oauth_redirect_uri,
                    "scope": "openid email profile",
                    "response_type": "code",
                    "state": state
                }
            elif provider == "github":
                auth_url = "https://github.com/login/oauth/authorize"
                params = {
                    "client_id": self.config.oauth_client_id,
                    "redirect_uri": redirect_uri or self.config.oauth_redirect_uri,
                    "scope": "user:email",
                    "state": state
                }
            else:
                return {
                    "success": False,
                    "error": f"Unsupported OAuth provider: {provider}"
                }
            
            authorization_url = f"{auth_url}?{urlencode(params)}"
            
            logger.info(f"OAuth flow initiated for provider: {provider}")
            
            return {
                "success": True,
                "authorization_url": authorization_url,
                "state": state,
                "provider": provider
            }
            
        except Exception as e:
            logger.error(f"OAuth initiation error: {str(e)}")
            return {
                "success": False,
                "error": f"OAuth initiation failed: {str(e)}"
            }

    async def handle_oauth_callback(self, code: str, state: str) -> AuthResult:
        """Handle OAuth callback and exchange code for tokens"""
        try:
            # Verify state parameter
            if state not in self._oauth_sessions:
                return AuthResult(
                    success=False,
                    message="Invalid OAuth state parameter"
                )
            
            session = self._oauth_sessions[state]
            provider = session["provider"]
            
            # Exchange authorization code for access token
            token_data = await self._exchange_oauth_code(provider, code)
            
            if not token_data:
                return AuthResult(
                    success=False,
                    message="Failed to exchange OAuth code"
                )
            
            # Get user profile from OAuth provider
            user_info = await self._get_oauth_user_info(provider, token_data["access_token"])
            
            if not user_info:
                return AuthResult(
                    success=False,
                    message="Failed to get user information"
                )
            
            # Create or update user profile
            user_profile = UserProfile(
                user_id=f"{provider}_{user_info['id']}",
                username=user_info.get("login") or user_info.get("name"),
                email=user_info.get("email"),
                full_name=user_info.get("name"),
                roles=["user"],
                permissions=["read", "write"]
            )
            
            # Generate JWT tokens
            access_token = self.generate_jwt_token(
                user_profile.user_id,
                {
                    "username": user_profile.username,
                    "provider": provider
                },
                "access"
            )
            
            refresh_token = self.generate_jwt_token(user_profile.user_id, {}, "refresh")
            
            tokens = TokenPair(
                access_token=access_token,
                refresh_token=refresh_token,
                expires_in=self.config.jwt_access_token_expire_minutes * 60
            )
            
            # Clean up OAuth session
            del self._oauth_sessions[state]
            
            logger.info(f"OAuth authentication successful for user: {user_profile.user_id}")
            
            return AuthResult(
                success=True,
                user_profile=user_profile,
                tokens=tokens,
                message="OAuth authentication successful"
            )
            
        except Exception as e:
            logger.error(f"OAuth callback error: {str(e)}")
            return AuthResult(
                success=False,
                message=f"OAuth authentication failed: {str(e)}"
            )

    async def _exchange_oauth_code(self, provider: str, code: str) -> Optional[Dict[str, Any]]:
        """Exchange OAuth authorization code for access token"""
        try:
            if provider == "google":
                token_url = "https://oauth2.googleapis.com/token"
            elif provider == "github":
                token_url = "https://github.com/login/oauth/access_token"
            else:
                return None
            
            data = {
                "client_id": self.config.oauth_client_id,
                "client_secret": self.config.oauth_client_secret,
                "code": code,
                "grant_type": "authorization_code"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    token_url,
                    data=data,
                    headers={"Accept": "application/json"}
                )
                
                if response.is_success:
                    return response.json()
                else:
                    logger.error(f"OAuth token exchange failed: {response.text}")
                    return None
                    
        except Exception as e:
            logger.error(f"OAuth code exchange error: {str(e)}")
            return None

    async def _get_oauth_user_info(self, provider: str, access_token: str) -> Optional[Dict[str, Any]]:
        """Get user information from OAuth provider"""
        try:
            if provider == "google":
                user_url = "https://www.googleapis.com/oauth2/v2/userinfo"
            elif provider == "github":
                user_url = "https://api.github.com/user"
            else:
                return None
            
            headers = {"Authorization": f"Bearer {access_token}"}
            
            async with httpx.AsyncClient() as client:
                response = await client.get(user_url, headers=headers)
                
                if response.is_success:
                    return response.json()
                else:
                    logger.error(f"OAuth user info failed: {response.text}")
                    return None
                    
        except Exception as e:
            logger.error(f"OAuth user info error: {str(e)}")
            return None

    def check_permissions(self, user_roles: List[str], required_permissions: List[str]) -> bool:
        """Check if user has required permissions"""
        try:
            # Define role-permission mapping
            role_permissions = {
                "admin": ["read", "write", "delete", "admin"],
                "moderator": ["read", "write", "moderate"],
                "user": ["read", "write"],
                "viewer": ["read"]
            }
            
            # Get all permissions for user roles
            user_permissions = set()
            for role in user_roles:
                if role in role_permissions:
                    user_permissions.update(role_permissions[role])
            
            # Check if all required permissions are present
            return all(perm in user_permissions for perm in required_permissions)
            
        except Exception as e:
            logger.error(f"Permission check error: {str(e)}")
            return False

    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        try:
            if not self.cipher:
                logger.warning("Encryption not configured, returning data as-is")
                return data
            
            encrypted = self.cipher.encrypt(data.encode())
            return base64.urlsafe_b64encode(encrypted).decode()
            
        except Exception as e:
            logger.error(f"Encryption error: {str(e)}")
            raise

    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        try:
            if not self.cipher:
                logger.warning("Decryption not configured, returning data as-is")
                return encrypted_data
            
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted = self.cipher.decrypt(encrypted_bytes)
            return decrypted.decode()
            
        except Exception as e:
            logger.error(f"Decryption error: {str(e)}")
            raise

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on authentication system"""
        try:
            return {
                "status": "healthy",
                "oauth_sessions": len(self._oauth_sessions),
                "encryption_enabled": self.cipher is not None,
                "jwt_algorithm": self.config.jwt_algorithm
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

# Factory function to create authentication tools
def create_auth_tools(config: AuthConfig = None) -> List:
    """Create and return authentication tools for the agent"""
    if not config:
        config = AuthConfig()
    
    auth_tool = AuthenticationTool(config)
    
    # Define individual tool functions that the agent can use
    async def authenticate_user_credentials(username: str = None, email: str = None, password: str = None) -> dict:
        """Authenticate user with username/email and password"""
        credentials = UserCredentials(
            username=username,
            email=email,
            password=password
        )
        result = await auth_tool.authenticate_user(credentials)
        return result.model_dump()
    
    async def verify_access_token(token: str) -> dict:
        """Verify JWT access token"""
        return auth_tool.verify_jwt_token(token)
    
    async def refresh_user_token(refresh_token: str) -> dict:
        """Refresh user access token"""
        result = await auth_tool.refresh_access_token(refresh_token)
        return result.model_dump()
    
    async def start_oauth_flow(provider: str, redirect_uri: str = None) -> dict:
        """Initiate OAuth flow"""
        return await auth_tool.initiate_oauth_flow(provider, redirect_uri)
    
    async def complete_oauth_flow(code: str, state: str) -> dict:
        """Complete OAuth flow with callback code"""
        result = await auth_tool.handle_oauth_callback(code, state)
        return result.model_dump()
    
    def check_user_permissions(user_roles: list, required_permissions: list) -> bool:
        """Check if user has required permissions"""
        return auth_tool.check_permissions(user_roles, required_permissions)
    
    return [
        authenticate_user_credentials,
        verify_access_token,
        refresh_user_token,
        start_oauth_flow,
        complete_oauth_flow,
        check_user_permissions
    ]

if __name__ == "__main__":
    # Test the authentication tool
    async def test_auth_tool():
        config = AuthConfig()
        auth_tool = AuthenticationTool(config)
        
        # Test health check
        health = auth_tool.health_check()
        print(f"Health check: {health}")
        
        # Test user authentication
        credentials = UserCredentials(
            email="admin@example.com",
            password="admin123"
        )
        
        result = await auth_tool.authenticate_user(credentials)
        print(f"Authentication result: {result}")
        
        if result.success and result.tokens:
            # Test token verification
            token_check = auth_tool.verify_jwt_token(result.tokens.access_token)
            print(f"Token verification: {token_check}")
    
    asyncio.run(test_auth_tool())
