"""
Authentication Configuration for Backend Agent
Handles JWT, OAuth, and security settings.
"""

import os
import secrets
from typing import Optional, Dict, Any, List
from pydantic import BaseSettings, Field, validator
from cryptography.fernet import Fernet

class AuthConfig(BaseSettings):
    """Authentication configuration settings"""
    
    # JWT Settings
    jwt_secret_key: str = Field(
        default_factory=lambda: secrets.token_urlsafe(32),
        env="JWT_SECRET_KEY",
        description="Secret key for JWT token signing"
    )
    
    jwt_algorithm: str = Field(
        default="HS256",
        env="JWT_ALGORITHM",
        description="JWT signing algorithm"
    )
    
    jwt_access_token_expire_minutes: int = Field(
        default=1440,  # 24 hours
        env="JWT_ACCESS_TOKEN_EXPIRE_MINUTES",
        description="Access token expiration time in minutes"
    )
    
    jwt_refresh_token_expire_days: int = Field(
        default=30,
        env="JWT_REFRESH_TOKEN_EXPIRE_DAYS",
        description="Refresh token expiration time in days"
    )
    
    # OAuth Settings
    oauth_client_id: str = Field(
        default="",
        env="OAUTH_CLIENT_ID",
        description="OAuth client ID"
    )
    
    oauth_client_secret: str = Field(
        default="",
        env="OAUTH_CLIENT_SECRET",
        description="OAuth client secret"
    )
    
    oauth_redirect_uri: str = Field(
        default="http://localhost:8000/auth/callback",
        env="OAUTH_REDIRECT_URI",
        description="OAuth redirect URI"
    )
    
    oauth_state_expire_minutes: int = Field(
        default=10,
        env="OAUTH_STATE_EXPIRE_MINUTES",
        description="OAuth state parameter expiration in minutes"
    )
    
    # Password Settings
    password_min_length: int = Field(
        default=8,
        env="PASSWORD_MIN_LENGTH",
        description="Minimum password length"
    )
    
    password_require_uppercase: bool = Field(
        default=True,
        env="PASSWORD_REQUIRE_UPPERCASE",
        description="Require uppercase letters in password"
    )
    
    password_require_lowercase: bool = Field(
        default=True,
        env="PASSWORD_REQUIRE_LOWERCASE",
        description="Require lowercase letters in password"
    )
    
    password_require_numbers: bool = Field(
        default=True,
        env="PASSWORD_REQUIRE_NUMBERS",
        description="Require numbers in password"
    )
    
    password_require_symbols: bool = Field(
        default=True,
        env="PASSWORD_REQUIRE_SYMBOLS",
        description="Require symbols in password"
    )
    
    # Session Settings
    session_expire_hours: int = Field(
        default=24,
        env="SESSION_EXPIRE_HOURS",
        description="Session expiration time in hours"
    )
    
    session_refresh_threshold_hours: int = Field(
        default=6,
        env="SESSION_REFRESH_THRESHOLD_HOURS",
        description="Hours before expiration to refresh session"
    )
    
    # Security Settings
    max_login_attempts: int = Field(
        default=5,
        env="MAX_LOGIN_ATTEMPTS",
        description="Maximum login attempts before lockout"
    )
    
    lockout_duration_minutes: int = Field(
        default=15,
        env="LOCKOUT_DURATION_MINUTES",
        description="Account lockout duration in minutes"
    )
    
    encryption_key: str = Field(
        default_factory=lambda: Fernet.generate_key().decode(),
        env="ENCRYPTION_KEY",
        description="Key for encrypting sensitive data"
    )
    
    # CORS Settings
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        env="CORS_ORIGINS",
        description="Allowed CORS origins"
    )
    
    cors_allow_credentials: bool = Field(
        default=True,
        env="CORS_ALLOW_CREDENTIALS",
        description="Allow credentials in CORS requests"
    )
    
    # Rate Limiting
    rate_limit_per_minute: int = Field(
        default=60,
        env="RATE_LIMIT_PER_MINUTE",
        description="API rate limit per minute per user"
    )
    
    rate_limit_burst: int = Field(
        default=10,
        env="RATE_LIMIT_BURST",
        description="Rate limit burst allowance"
    )
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        
        @classmethod
        def parse_env_var(cls, field_name: str, raw_val: str) -> Any:
            if field_name == 'cors_origins':
                return [origin.strip() for origin in raw_val.split(',')]
            return cls.json_loads(raw_val)
    
    @validator('jwt_secret_key')
    def validate_jwt_secret_key(cls, v):
        """Validate JWT secret key strength"""
        if len(v) < 32:
            raise ValueError("JWT secret key must be at least 32 characters long")
        return v
    
    @validator('jwt_algorithm')
    def validate_jwt_algorithm(cls, v):
        """Validate JWT algorithm"""
        allowed_algorithms = ['HS256', 'HS384', 'HS512', 'RS256', 'RS384', 'RS512']
        if v not in allowed_algorithms:
            raise ValueError(f"JWT algorithm must be one of: {', '.join(allowed_algorithms)}")
        return v
    
    @validator('password_min_length')
    def validate_password_min_length(cls, v):
        """Validate minimum password length"""
        if v < 6:
            raise ValueError("Password minimum length cannot be less than 6")
        if v > 128:
            raise ValueError("Password minimum length cannot exceed 128")
        return v
    
    @validator('encryption_key')
    def validate_encryption_key(cls, v):
        """Validate encryption key"""
        try:
            # Test if key is valid for Fernet
            Fernet(v.encode() if isinstance(v, str) else v)
            return v
        except Exception:
            raise ValueError("Invalid encryption key format")
    
    def get_password_policy(self) -> Dict[str, Any]:
        """Get password policy configuration"""
        return {
            "min_length": self.password_min_length,
            "require_uppercase": self.password_require_uppercase,
            "require_lowercase": self.password_require_lowercase,
            "require_numbers": self.password_require_numbers,
            "require_symbols": self.password_require_symbols
        }
    
    def get_jwt_config(self) -> Dict[str, Any]:
        """Get JWT configuration"""
        return {
            "secret_key": self.jwt_secret_key,
            "algorithm": self.jwt_algorithm,
            "access_token_expire_minutes": self.jwt_access_token_expire_minutes,
            "refresh_token_expire_days": self.jwt_refresh_token_expire_days
        }
    
    def get_oauth_config(self) -> Dict[str, Any]:
        """Get OAuth configuration"""
        return {
            "client_id": self.oauth_client_id,
            "client_secret": self.oauth_client_secret,
            "redirect_uri": self.oauth_redirect_uri,
            "state_expire_minutes": self.oauth_state_expire_minutes
        }
    
    def get_security_config(self) -> Dict[str, Any]:
        """Get security configuration"""
        return {
            "max_login_attempts": self.max_login_attempts,
            "lockout_duration_minutes": self.lockout_duration_minutes,
            "session_expire_hours": self.session_expire_hours,
            "rate_limit_per_minute": self.rate_limit_per_minute
        }
    
    def get_cors_config(self) -> Dict[str, Any]:
        """Get CORS configuration"""
        return {
            "origins": self.cors_origins,
            "allow_credentials": self.cors_allow_credentials
        }

    def get_default_auth(self) -> Dict[str, Any]:
        """Get default authentication configuration"""
        return {
            "type": "bearer",
            "token": ""  # This would be set by the application
        }
