"""
Data Models for Backend Agent
Pydantic models for request/response validation and database schemas.
"""

from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, EmailStr, validator, root_validator
from pydantic.types import UUID4

class UserRole(str, Enum):
    """User role enumeration"""
    ADMIN = "admin"
    MODERATOR = "moderator"
    USER = "user"
    VIEWER = "viewer"

class UserStatus(str, Enum):
    """User status enumeration"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING = "pending"

class ProcessingStatus(str, Enum):
    """Processing status enumeration"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

# Base Models
class BaseModel(BaseModel):
    """Base model with common configuration"""
    
    class Config:
        use_enum_values = True
        validate_assignment = True
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class TimestampMixin(BaseModel):
    """Mixin for timestamp fields"""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None

# User Models
class UserBase(BaseModel):
    """Base user model"""
    username: str = Field(..., min_length=3, max_length=50, regex=r'^[a-zA-Z0-9_-]+$')
    email: EmailStr = Field(..., description="User email address")
    full_name: Optional[str] = Field(None, max_length=100)
    is_active: bool = Field(True, description="Whether the user account is active")
    
    @validator('username')
    def validate_username(cls, v):
        """Validate username format"""
        if v.lower() in ['admin', 'root', 'system', 'api']:
            raise ValueError('Username cannot be a reserved word')
        return v.lower()

class UserCreate(UserBase):
    """User creation model"""
    password: str = Field(..., min_length=8, max_length=128)
    confirm_password: str = Field(..., description="Password confirmation")
    roles: List[UserRole] = Field(default=[UserRole.USER], description="User roles")
    
    @validator('password')
    def validate_password(cls, v):
        """Validate password strength"""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        
        has_upper = any(c.isupper() for c in v)
        has_lower = any(c.islower() for c in v)
        has_digit = any(c.isdigit() for c in v)
        has_symbol = any(c in '!@#$%^&*()_+-=[]{}|;:,.<>?' for c in v)
        
        if not all([has_upper, has_lower, has_digit, has_symbol]):
            raise ValueError('Password must contain uppercase, lowercase, digit, and symbol')
        
        return v
    
    @root_validator
    def validate_passwords_match(cls, values):
        """Validate that passwords match"""
        password = values.get('password')
        confirm_password = values.get('confirm_password')
        
        if password and confirm_password and password != confirm_password:
            raise ValueError('Passwords do not match')
        
        return values

class UserUpdate(BaseModel):
    """User update model"""
    full_name: Optional[str] = Field(None, max_length=100)
    is_active: Optional[bool] = None
    roles: Optional[List[UserRole]] = None

class User(UserBase, TimestampMixin):
    """Complete user model"""
    id: UUID4 = Field(..., description="User unique identifier")
    roles: List[UserRole] = Field(default_factory=list)
    status: UserStatus = Field(default=UserStatus.ACTIVE)
    last_login: Optional[datetime] = None
    login_count: int = Field(default=0)
    
    class Config:
        from_attributes = True

class UserResponse(BaseModel):
    """User response model (without sensitive data)"""
    id: UUID4
    username: str
    email: EmailStr
    full_name: Optional[str]
    roles: List[UserRole]
    status: UserStatus
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime]

# Authentication Models
class LoginRequest(BaseModel):
    """User login request"""
    username_or_email: str = Field(..., description="Username or email")
    password: str = Field(..., description="User password")
    remember_me: bool = Field(False, description="Remember login session")

class TokenResponse(BaseModel):
    """Token response model"""
    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="JWT refresh token")
    token_type: str = Field("Bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration in seconds")
    user: UserResponse = Field(..., description="User information")

class RefreshTokenRequest(BaseModel):
    """Refresh token request"""
    refresh_token: str = Field(..., description="Refresh token")

class PasswordResetRequest(BaseModel):
    """Password reset request"""
    email: EmailStr = Field(..., description="User email")

class PasswordResetConfirm(BaseModel):
    """Password reset confirmation"""
    token: str = Field(..., description="Reset token")
    new_password: str = Field(..., min_length=8)
    confirm_password: str = Field(..., description="Password confirmation")
    
    @root_validator
    def validate_passwords_match(cls, values):
        """Validate that passwords match"""
        password = values.get('new_password')
        confirm_password = values.get('confirm_password')
        
        if password and confirm_password and password != confirm_password:
            raise ValueError('Passwords do not match')
        
        return values

# API Response Models
class APIResponse(BaseModel):
    """Standard API response wrapper"""
    success: bool = Field(..., description="Whether the request was successful")
    message: str = Field("", description="Response message")
    data: Any = Field(None, description="Response data")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = Field(None, description="Request tracking ID")

class PaginatedResponse(BaseModel):
    """Paginated response model"""
    items: List[Any] = Field(..., description="List of items")
    total: int = Field(..., description="Total number of items")
    page: int = Field(..., description="Current page number")
    size: int = Field(..., description="Items per page")
    pages: int = Field(..., description="Total number of pages")
    has_next: bool = Field(..., description="Whether there is a next page")
    has_prev: bool = Field(..., description="Whether there is a previous page")

class ErrorResponse(BaseModel):
    """Error response model"""
    success: bool = Field(False)
    error_code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# File Processing Models
class FileUpload(BaseModel):
    """File upload model"""
    filename: str = Field(..., description="Original filename")
    content_type: str = Field(..., description="File MIME type")
    size: int = Field(..., description="File size in bytes")
    checksum: Optional[str] = Field(None, description="File checksum")

class FileInfo(BaseModel, TimestampMixin):
    """File information model"""
    id: UUID4 = Field(..., description="File unique identifier")
    filename: str = Field(..., description="Original filename")
    stored_filename: str = Field(..., description="Stored filename")
    file_path: str = Field(..., description="File storage path")
    content_type: str = Field(..., description="File MIME type")
    size: int = Field(..., description="File size in bytes")
    checksum: str = Field(..., description="File checksum")
    uploaded_by: UUID4 = Field(..., description="User who uploaded the file")
    is_processed: bool = Field(False, description="Whether file has been processed")

class ProcessingJob(BaseModel, TimestampMixin):
    """Processing job model"""
    id: UUID4 = Field(..., description="Job unique identifier")
    job_type: str = Field(..., description="Type of processing job")
    status: ProcessingStatus = Field(default=ProcessingStatus.PENDING)
    input_data: Dict[str, Any] = Field(..., description="Job input data")
    output_data: Optional[Dict[str, Any]] = Field(None, description="Job output data")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    progress: int = Field(0, ge=0, le=100, description="Job progress percentage")
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_by: UUID4 = Field(..., description="User who created the job")

# Database Query Models
class QueryRequest(BaseModel):
    """Database query request"""
    query: str = Field(..., description="SQL query to execute")
    params: Dict[str, Any] = Field(default_factory=dict, description="Query parameters")
    limit: Optional[int] = Field(None, ge=1, le=10000, description="Result limit")

class QueryResponse(BaseModel):
    """Database query response"""
    success: bool = Field(..., description="Whether query was successful")
    data: List[Dict[str, Any]] = Field(default_factory=list, description="Query results")
    row_count: int = Field(..., description="Number of rows returned")
    execution_time: float = Field(..., description="Query execution time in seconds")
    columns: List[str] = Field(default_factory=list, description="Column names")

# API Integration Models
class APICallRequest(BaseModel):
    """External API call request"""
    url: str = Field(..., description="API endpoint URL")
    method: str = Field("GET", regex=r'^(GET|POST|PUT|PATCH|DELETE)$')
    headers: Dict[str, str] = Field(default_factory=dict, description="Request headers")
    params: Dict[str, Any] = Field(default_factory=dict, description="URL parameters")
    data: Optional[Dict[str, Any]] = Field(None, description="Request body")
    timeout: int = Field(30, ge=1, le=300, description="Request timeout in seconds")

class APICallResponse(BaseModel):
    """External API call response"""
    success: bool = Field(..., description="Whether API call was successful")
    status_code: int = Field(..., description="HTTP status code")
    data: Any = Field(None, description="Response data")
    headers: Dict[str, str] = Field(default_factory=dict, description="Response headers")
    execution_time: float = Field(..., description="Request execution time")

# Webhook Models
class WebhookEvent(BaseModel):
    """Webhook event model"""
    id: UUID4 = Field(..., description="Event unique identifier")
    event_type: str = Field(..., description="Type of webhook event")
    payload: Dict[str, Any] = Field(..., description="Event payload")
    signature: Optional[str] = Field(None, description="Webhook signature")
    source: str = Field(..., description="Event source")
    received_at: datetime = Field(default_factory=datetime.utcnow)

class WebhookResponse(BaseModel):
    """Webhook processing response"""
    success: bool = Field(..., description="Whether webhook was processed successfully")
    message: str = Field("", description="Processing message")
    processed_at: datetime = Field(default_factory=datetime.utcnow)

# Analytics Models
class AnalyticsQuery(BaseModel):
    """Analytics query model"""
    metric: str = Field(..., description="Metric to analyze")
    dimensions: List[str] = Field(default_factory=list, description="Analysis dimensions")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Query filters")
    date_range: Dict[str, str] = Field(..., description="Date range for analysis")
    aggregation: str = Field("count", description="Aggregation method")

class AnalyticsResult(BaseModel):
    """Analytics result model"""
    metric: str = Field(..., description="Analyzed metric")
    value: Union[int, float, str] = Field(..., description="Metric value")
    breakdown: Dict[str, Any] = Field(default_factory=dict, description="Metric breakdown")
    period: str = Field(..., description="Analysis period")
    generated_at: datetime = Field(default_factory=datetime.utcnow)

# Health Check Models
class HealthCheck(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="Overall health status")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field("1.0.0", description="Application version")
    uptime: float = Field(..., description="Uptime in seconds")
    services: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Service health status")

class ServiceHealth(BaseModel):
    """Individual service health model"""
    status: str = Field(..., description="Service status")
    response_time: Optional[float] = Field(None, description="Service response time")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional service details")
    last_checked: datetime = Field(default_factory=datetime.utcnow)

