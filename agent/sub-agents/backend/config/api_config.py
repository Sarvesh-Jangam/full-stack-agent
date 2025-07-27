"""
API Configuration for Backend Agent
Handles external API integration settings.
"""

import os
from typing import Optional, Dict, Any, List
from pydantic import BaseSettings, Field, validator, HttpUrl

class APIConfig(BaseSettings):
    """API integration configuration settings"""
    
    # External API Settings
    external_api_base_url: str = Field(
        default="https://api.example.com",
        env="EXTERNAL_API_BASE_URL",
        description="Base URL for external API"
    )
    
    external_api_key: str = Field(
        default="",
        env="EXTERNAL_API_KEY",
        description="API key for external API"
    )
    
    external_api_timeout: int = Field(
        default=30,
        env="EXTERNAL_API_TIMEOUT",
        description="API request timeout in seconds"
    )
    
    external_api_retries: int = Field(
        default=3,
        env="EXTERNAL_API_RETRIES",
        description="Number of API request retries"
    )
    
    external_api_retry_delay: float = Field(
        default=1.0,
        env="EXTERNAL_API_RETRY_DELAY",
        description="Delay between API retries in seconds"
    )
    
    # Rate Limiting
    api_rate_limit_per_minute: int = Field(
        default=60,
        env="API_RATE_LIMIT_PER_MINUTE",
        description="API requests per minute limit"
    )
    
    api_rate_limit_burst: int = Field(
        default=10,
        env="API_RATE_LIMIT_BURST", 
        description="API rate limit burst allowance"
    )
    
    # Webhook Settings
    webhook_secret: str = Field(
        default="",
        env="WEBHOOK_SECRET",
        description="Secret for webhook verification"
    )
    
    webhook_timeout: int = Field(
        default=30,
        env="WEBHOOK_TIMEOUT",
        description="Webhook processing timeout in seconds"
    )
    
    # HTTP Client Settings
    http_pool_connections: int = Field(
        default=10,
        env="HTTP_POOL_CONNECTIONS",
        description="HTTP connection pool size"  
    )
    
    http_pool_maxsize: int = Field(
        default=20,
        env="HTTP_POOL_MAXSIZE",
        description="Maximum HTTP connections in pool"
    )
    
    http_max_retries: int = Field(
        default=3,
        env="HTTP_MAX_RETRIES",
        description="Maximum HTTP request retries"
    )
    
    # SSL/TLS Settings
    verify_ssl: bool = Field(
        default=True,
        env="VERIFY_SSL",
        description="Verify SSL certificates"
    )
    
    ssl_cert_path: Optional[str] = Field(
        default=None,
        env="SSL_CERT_PATH",
        description="Path to SSL certificate file"
    )
    
    ssl_key_path: Optional[str] = Field(
        default=None,
        env="SSL_KEY_PATH",
        description="Path to SSL private key file"
    )
    
    # Proxy Settings
    http_proxy: Optional[str] = Field(
        default=None,
        env="HTTP_PROXY",
        description="HTTP proxy URL"
    )
    
    https_proxy: Optional[str] = Field(
        default=None,
        env="HTTPS_PROXY",
        description="HTTPS proxy URL"
    )
    
    # API Provider Specific Settings
    openai_api_key: str = Field(
        default="",
        env="OPENAI_API_KEY",
        description="OpenAI API key"
    )
    
    stripe_api_key: str = Field(
        default="",
        env="STRIPE_API_KEY",
        description="Stripe API key"
    )
    
    sendgrid_api_key: str = Field(
        default="",
        env="SENDGRID_API_KEY",
        description="SendGrid API key"
    )
    
    twilio_account_sid: str = Field(
        default="",
        env="TWILIO_ACCOUNT_SID",
        description="Twilio Account SID"
    )
    
    twilio_auth_token: str = Field(
        default="",
        env="TWILIO_AUTH_TOKEN",
        description="Twilio Auth Token"
    )
    
    # API Documentation
    enable_api_docs: bool = Field(
        default=True,
        env="ENABLE_API_DOCS",
        description="Enable API documentation endpoints"
    )
    
    api_docs_url: str = Field(
        default="/docs",
        env="API_DOCS_URL",
        description="API documentation URL path"
    )
    
    redoc_url: str = Field(
        default="/redoc",
        env="REDOC_URL",
        description="ReDoc documentation URL path"
    )
    
    # Monitoring and Logging
    log_api_requests: bool = Field(
        default=True,
        env="LOG_API_REQUESTS",
        description="Log API requests and responses"
    )
    
    log_api_response_body: bool = Field(
        default=False,
        env="LOG_API_RESPONSE_BODY",
        description="Log API response bodies"
    )
    
    api_metrics_enabled: bool = Field(
        default=True,
        env="API_METRICS_ENABLED",
        description="Enable API metrics collection"
    )
    
    class Config:
        env_file = ".env"
        case_sensitive = False
    
    @validator('external_api_timeout')
    def validate_timeout(cls, v):
        """Validate API timeout"""
        if v < 1:
            raise ValueError("API timeout must be at least 1 second")
        if v > 300:
            raise ValueError("API timeout cannot exceed 300 seconds")
        return v
    
    @validator('api_rate_limit_per_minute')
    def validate_rate_limit(cls, v):
        """Validate rate limit"""
        if v < 1:
            raise ValueError("Rate limit must be at least 1 request per minute")
        if v > 10000:
            raise ValueError("Rate limit cannot exceed 10000 requests per minute")
        return v
    
    @validator('http_pool_connections', 'http_pool_maxsize')
    def validate_pool_size(cls, v):
        """Validate connection pool sizes"""
        if v < 1:
            raise ValueError("Pool size must be at least 1")
        if v > 100:
            raise ValueError("Pool size cannot exceed 100")
        return v
    
    def get_http_client_config(self) -> Dict[str, Any]:
        """Get HTTP client configuration"""
        config = {
            "timeout": self.external_api_timeout,
            "verify": self.verify_ssl,
            "pool_connections": self.http_pool_connections,
            "pool_maxsize": self.http_pool_maxsize,
            "max_retries": self.http_max_retries
        }
        
        # Add proxy settings if configured
        if self.http_proxy or self.https_proxy:
            config["proxies"] = {}
            if self.http_proxy:
                config["proxies"]["http"] = self.http_proxy
            if self.https_proxy:
                config["proxies"]["https"] = self.https_proxy
        
        # Add SSL settings if configured
        if self.ssl_cert_path and self.ssl_key_path:
            config["cert"] = (self.ssl_cert_path, self.ssl_key_path)
        
        return config
    
    def get_rate_limit_config(self) -> Dict[str, Any]:
        """Get rate limiting configuration"""
        return {
            "per_minute": self.api_rate_limit_per_minute,
            "burst": self.api_rate_limit_burst
        }
    
    def get_provider_config(self, provider: str) -> Dict[str, Any]:
        """Get configuration for specific API provider"""
        provider_configs = {
            "openai": {
                "api_key": self.openai_api_key,
                "base_url": "https://api.openai.com/v1"
            },
            "stripe": {
                "api_key": self.stripe_api_key,
                "base_url": "https://api.stripe.com/v1"
            },
            "sendgrid": {
                "api_key": self.sendgrid_api_key,
                "base_url": "https://api.sendgrid.com/v3"
            },
            "twilio": {
                "account_sid": self.twilio_account_sid,
                "auth_token": self.twilio_auth_token,
                "base_url": "https://api.twilio.com/2010-04-01"
            }
        }
        
        return provider_configs.get(provider.lower(), {})
    
    def get_webhook_config(self) -> Dict[str, Any]:
        """Get webhook configuration"""
        return {
            "secret": self.webhook_secret,
            "timeout": self.webhook_timeout
        }
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get API logging configuration"""
        return {
            "log_requests": self.log_api_requests,
            "log_response_body": self.log_api_response_body,
            "metrics_enabled": self.api_metrics_enabled
        }
    
    def get_docs_config(self) -> Dict[str, Any]:
        """Get API documentation configuration"""
        return {
            "enabled": self.enable_api_docs,
            "docs_url": self.api_docs_url,
            "redoc_url": self.redoc_url
        }

    def get_default_auth(self) -> Dict[str, Any]:
        """Get default authentication configuration"""
        return {
            "type": "api_key",
            "key_name": "X-API-Key",
            "api_key": self.external_api_key
        }

