"""
Schema package for Backend Agent
Contains data models and API specifications.
"""

from .data_models import (
    User,
    UserCreate,
    UserUpdate,
    UserResponse,
    LoginRequest,
    TokenResponse,
    APIResponse,
    PaginatedResponse,
    FileUpload,
    ProcessingJob
)

from .api_schemas import get_openapi_schema

__all__ = [
    "User",
    "UserCreate", 
    "UserUpdate",
    "UserResponse",
    "LoginRequest",
    "TokenResponse",
    "APIResponse",
    "PaginatedResponse",
    "FileUpload",
    "ProcessingJob",
    "get_openapi_schema"
]

