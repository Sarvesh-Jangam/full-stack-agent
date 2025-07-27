"""
OpenAPI Schema Definitions for Backend Agent
Contains OpenAPI specifications for automatic tool generation.
"""

from typing import Dict, Any

def get_openapi_schema() -> Dict[str, Any]:
    """Get OpenAPI 3.0 schema for backend agent APIs"""
    
    return {
        "openapi": "3.0.0",
        "info": {
            "title": "Backend Agent API",
            "description": "Comprehensive backend API for database operations, authentication, file processing, and external integrations",
            "version": "1.0.0",
            "contact": {
                "name": "Backend Agent Team",
                "email": "support@backendagent.com"
            },
            "license": {
                "name": "MIT",
                "url": "https://opensource.org/licenses/MIT"
            }
        },
        "servers": [
            {
                "url": "http://localhost:8000",
                "description": "Development server"
            },
            {
                "url": "https://api.backendagent.com",
                "description": "Production server"
            }
        ],
        "security": [
            {
                "BearerAuth": []
            },
            {
                "ApiKeyAuth": []
            }
        ],
        "components": {
            "securitySchemes": {
                "BearerAuth": {
                    "type": "http",
                    "scheme": "bearer",
                    "bearerFormat": "JWT"
                },
                "ApiKeyAuth": {
                    "type": "apiKey",
                    "in": "header",
                    "name": "X-API-Key"
                }
            },
            "schemas": {
                "User": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "format": "uuid"},
                        "username": {"type": "string"},
                        "email": {"type": "string", "format": "email"},
                        "full_name": {"type": "string"},
                        "roles": {
                            "type": "array",
                            "items": {"type": "string", "enum": ["admin", "moderator", "user", "viewer"]}
                        },
                        "is_active": {"type": "boolean"},
                        "created_at": {"type": "string", "format": "date-time"}
                    }
                },
                "APIResponse": {
                    "type": "object",
                    "properties": {
                        "success": {"type": "boolean"},
                        "message": {"type": "string"},
                        "data": {"type": "object"},
                        "timestamp": {"type": "string", "format": "date-time"}
                    }
                },
                "ErrorResponse": {
                    "type": "object",
                    "properties": {
                        "success": {"type": "boolean", "example": False},
                        "error_code": {"type": "string"},
                        "message": {"type": "string"},
                        "details": {"type": "object"},
                        "timestamp": {"type": "string", "format": "date-time"}
                    }
                }
            }
        },
        "paths": {
            "/auth/login": {
                "post": {
                    "tags": ["Authentication"],
                    "summary": "User login",
                    "description": "Authenticate user with username/email and password",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "username_or_email": {"type": "string"},
                                        "password": {"type": "string"},
                                        "remember_me": {"type": "boolean", "default": False}
                                    },
                                    "required": ["username_or_email", "password"]
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Login successful",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "access_token": {"type": "string"},
                                            "refresh_token": {"type": "string"},
                                            "token_type": {"type": "string", "default": "Bearer"},
                                            "expires_in": {"type": "integer"},
                                            "user": {"$ref": "#/components/schemas/User"}
                                        }
                                    }
                                }
                            }
                        },
                        "401": {
                            "description": "Authentication failed",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                                }
                            }
                        }
                    }
                }
            },
            "/database/query": {
                "post": {
                    "tags": ["Database"],
                    "summary": "Execute database query",
                    "description": "Execute SQL query with parameters",
                    "security": [{"BearerAuth": []}],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "query": {"type": "string"},
                                        "params": {"type": "object"},
                                        "limit": {"type": "integer", "minimum": 1, "maximum": 10000}
                                    },
                                    "required": ["query"]
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Query executed successfully",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "success": {"type": "boolean"},
                                            "data": {"type": "array"},
                                            "row_count": {"type": "integer"},
                                            "execution_time": {"type": "number"},
                                            "columns": {"type": "array", "items": {"type": "string"}}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/files/upload": {
                "post": {
                    "tags": ["File Processing"],
                    "summary": "Upload file",
                    "description": "Upload and process file",
                    "security": [{"BearerAuth": []}],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "multipart/form-data": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "file": {"type": "string", "format": "binary"},
                                        "process_type": {"type": "string", "enum": ["auto", "csv", "excel", "pdf", "image"]}
                                    },
                                    "required": ["file"]
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "File uploaded and processed successfully",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/APIResponse"}
                                }
                            }
                        }
                    }
                }
            },
            "/api/call": {
                "post": {
                    "tags": ["API Integration"],
                    "summary": "Make external API call",
                    "description": "Make HTTP request to external API",
                    "security": [{"BearerAuth": []}],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "url": {"type": "string", "format": "uri"},
                                        "method": {"type": "string", "enum": ["GET", "POST", "PUT", "PATCH", "DELETE"]},
                                        "headers": {"type": "object"},
                                        "params": {"type": "object"},
                                        "data": {"type": "object"},
                                        "timeout": {"type": "integer", "minimum": 1, "maximum": 300}
                                    },
                                    "required": ["url"]
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "API call completed",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "success": {"type": "boolean"},
                                            "status_code": {"type": "integer"},
                                            "data": {"type": "object"},
                                            "headers": {"type": "object"},
                                            "execution_time": {"type": "number"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/health": {
                "get": {
                    "tags": ["System"],
                    "summary": "Health check",
                    "description": "Get system health status",
                    "responses": {
                        "200": {
                            "description": "System health status",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "status": {"type": "string"},
                                            "timestamp": {"type": "string", "format": "date-time"},
                                            "version": {"type": "string"},
                                            "uptime": {"type": "number"},
                                            "services": {"type": "object"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "tags": [
            {
                "name": "Authentication",
                "description": "User authentication and authorization operations"
            },
            {
                "name": "Database",
                "description": "Database operations and queries"
            },
            {
                "name": "File Processing",
                "description": "File upload and processing operations"
            },
            {
                "name": "API Integration",
                "description": "External API integration operations"
            },
            {
                "name": "System",
                "description": "System health and monitoring operations"
            }
        ]
    }

def get_webhook_schema() -> Dict[str, Any]:
    """Get OpenAPI schema for webhook endpoints"""
    
    return {
        "openapi": "3.0.0",
        "info": {
            "title": "Backend Agent Webhook API",
            "description": "Webhook endpoints for receiving external events",
            "version": "1.0.0"
        },
        "paths": {
            "/webhooks/github": {
                "post": {
                    "tags": ["Webhooks"],
                    "summary": "GitHub webhook",
                    "description": "Handle GitHub webhook events",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "action": {"type": "string"},
                                        "repository": {"type": "object"},
                                        "sender": {"type": "object"}
                                    }
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Webhook processed successfully"
                        }
                    }
                }
            },
            "/webhooks/stripe": {
                "post": {
                    "tags": ["Webhooks"],
                    "summary": "Stripe webhook",
                    "description": "Handle Stripe payment events",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "id": {"type": "string"},
                                        "object": {"type": "string"},
                                        "type": {"type": "string"},
                                        "data": {"type": "object"}
                                    }
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Webhook processed successfully"
                        }
                    }
                }
            }
        }
    }

