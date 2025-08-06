"""
Database Configuration for Backend Agent
Handles database connection settings and configuration.
"""

import os
from typing import Optional, Dict, Any
from urllib.parse import urlparse
from pydantic_settings import BaseSettings
from pydantic import Field, validator


class DatabaseConfig(BaseSettings):
    """Database configuration settings"""
    
    # Database connection settings
    database_url: str = Field(
        default="postgresql://localhost:5432/backend_agent_db",
        env="DATABASE_URL",
        description="Database connection URL"
    )
    
    database_host: str = Field(
        default="localhost",
        env="DATABASE_HOST",
        description="Database host"
    )
    
    database_port: int = Field(
        default=5432,
        env="DATABASE_PORT",
        description="Database port"
    )
    
    database_name: str = Field(
        default="backend_agent_db",
        env="DATABASE_NAME",
        description="Database name"
    )
    
    database_user: str = Field(
        default="backend_agent_user",
        env="DATABASE_USER",
        description="Database username"
    )
    
    database_password: str = Field(
        default="",
        env="DATABASE_PASSWORD",
        description="Database password"
    )
    
    database_ssl_mode: str = Field(
        default="prefer",
        env="DATABASE_SSL_MODE",
        description="SSL mode for database connection"
    )
    
    # Connection pool settings
    pool_size: int = Field(
        default=20,
        env="DATABASE_POOL_SIZE",
        description="Database connection pool size"
    )
    
    max_overflow: int = Field(
        default=0,
        env="DATABASE_MAX_OVERFLOW",
        description="Maximum connection pool overflow"
    )
    
    pool_timeout: int = Field(
        default=30,
        env="DATABASE_POOL_TIMEOUT",
        description="Connection pool timeout in seconds"
    )
    
    pool_recycle: int = Field(
        default=3600,
        env="DATABASE_POOL_RECYCLE",
        description="Connection recycle time in seconds"
    )
    
    # Query settings
    query_timeout: int = Field(
        default=30,
        env="DATABASE_QUERY_TIMEOUT",
        description="Default query timeout in seconds"
    )
    
    # Development settings
    debug: bool = Field(
        default=False,
        env="DEBUG",
        description="Enable database query logging"
    )
    
    echo_queries: bool = Field(
        default=False,
        env="DATABASE_ECHO_QUERIES",
        description="Echo SQL queries to console"
    )
    
    class Config:
        env_file = ".env"
        case_sensitive = False
    
    @validator('database_url')
    def validate_database_url(cls, v):
        """Validate database URL format"""
        try:
            parsed = urlparse(v)
            if not parsed.scheme:
                raise ValueError("Database URL must include a scheme")
            if parsed.scheme not in ['postgresql', 'mysql', 'sqlite']:
                raise ValueError(f"Unsupported database scheme: {parsed.scheme}")
            return v
        except Exception as e:
            raise ValueError(f"Invalid database URL: {str(e)}")
    
    @validator('pool_size')
    def validate_pool_size(cls, v):
        """Validate pool size"""
        if v < 1:
            raise ValueError("Pool size must be at least 1")
        if v > 100:
            raise ValueError("Pool size cannot exceed 100")
        return v
    
    def get_sync_url(self) -> str:
        """Get synchronous database URL"""
        if self.database_url.startswith('postgresql://'):
            return self.database_url.replace('postgresql://', 'postgresql+psycopg2://', 1)
        elif self.database_url.startswith('mysql://'):
            return self.database_url.replace('mysql://', 'mysql+pymysql://', 1)
        return self.database_url
    
    def get_async_url(self) -> str:
        """Get asynchronous database URL"""
        if self.database_url.startswith('postgresql://'):
            return self.database_url.replace('postgresql://', 'postgresql+asyncpg://', 1)
        elif self.database_url.startswith('mysql://'):
            return self.database_url.replace('mysql://', 'mysql+aiomysql://', 1)
        return self.database_url
    
    def get_connection_params(self) -> Dict[str, Any]:
        """Get connection parameters dictionary"""
        return {
            "host": self.database_host,
            "port": self.database_port,
            "database": self.database_name,
            "user": self.database_user,
            "password": self.database_password,
            "sslmode": self.database_ssl_mode
        }
    
    def get_engine_options(self) -> Dict[str, Any]:
        """Get SQLAlchemy engine options"""
        return {
            "pool_size": self.pool_size,
            "max_overflow": self.max_overflow,
            "pool_timeout": self.pool_timeout,
            "pool_recycle": self.pool_recycle,
            "echo": self.echo_queries or self.debug,
            "pool_pre_ping": True
        }

    def is_sqlite(self) -> bool:
        """Check if database is SQLite"""
        return self.database_url.startswith('sqlite')
    
    def is_postgresql(self) -> bool:
        """Check if database is PostgreSQL"""
        return self.database_url.startswith('postgresql')
    
    def is_mysql(self) -> bool:
        """Check if database is MySQL"""
        return self.database_url.startswith('mysql')
