"""
Validation Tools for Manager Agent
Validates workflow results and provides quality assessments.
"""

import logging
import re
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pydantic import BaseModel, Field
from .config import ManagerAgentConfig
logger = logging.getLogger(__name__)

class ValidationCriteria(BaseModel):
    """Validation criteria for different aspects"""
    name: str = Field(..., description="Criteria name")
    weight: float = Field(..., description="Weight in overall score (0-1)")
    description: str = Field(..., description="Criteria description")
    validation_rules: List[str] = Field(..., description="List of validation rules")

class ValidationResult(BaseModel):
    """Result of a validation check"""
    criteria_name: str = Field(..., description="Name of validated criteria")
    score: float = Field(..., description="Score (0-1)")
    passed: bool = Field(..., description="Whether validation passed")
    issues: List[str] = Field(default_factory=list, description="List of issues found")
    recommendations: List[str] = Field(default_factory=list, description="Improvement recommendations")
    details: Dict[str, Any] = Field(default_factory=dict, description="Detailed validation info")

class ValidationTool:
    """Validates artifacts produced by specialist agents."""

    def __init__(self, config: ManagerAgentConfig):
        """
        Initializes the ValidationTool with a configuration.

        Args:
            config: The configuration object for the manager agent.
        """
        self.config = config
        logger.info("Validation tool initialized")

    async def validate_artifact(self, artifact_id: str, validation_criteria: str) -> Dict[str, Any]:
        """
        Validates a given artifact against specified criteria.
        """
        logger.info(f"Validating artifact '{artifact_id}' with criteria: {validation_criteria}")

        # Mock validation
        if "error" in artifact_id:
            score = 0.2
            message = "Validation failed: Artifact contains errors."
        else:
            score = 0.95
            message = "Validation successful."

        return {
            "artifact_id": artifact_id,
            "validation_score": score,
            "is_valid": score > self.config.validation_threshold,
            "message": message
        }

    async def run_integration_tests(self) -> Dict[str, Any]:
        """
        Runs integration tests on the completed application.
        """
        logger.info("Running integration tests...")
        # Mock integration test run
        await asyncio.sleep(3) # Simulate test execution time

        return {
            "status": "success",
            "tests_passed": 128,
            "tests_failed": 2,
            "coverage": "88%",
            "message": "Integration tests completed with minor failures."
        }


    def _initialize_validation_criteria(self) -> Dict[str, ValidationCriteria]:
        """Initialize validation criteria for different aspects"""
        return {
            "frontend_quality": ValidationCriteria(
                name="Frontend Quality",
                weight=0.3,
                description="Quality of frontend components and user interface",
                validation_rules=[
                    "HTML structure is semantic and valid",
                    "CSS follows responsive design principles",
                    "JavaScript functionality works correctly",
                    "User interface is intuitive and accessible",
                    "Cross-browser compatibility is maintained"
                ]
            ),
            "backend_functionality": ValidationCriteria(
                name="Backend Functionality",
                weight=0.3,
                description="Backend API and data processing functionality",
                validation_rules=[
                    "API endpoints are implemented correctly",
                    "Database operations work as expected",
                    "Authentication and authorization are secure",
                    "Error handling is comprehensive",
                    "Data validation is implemented"
                ]
            ),
            "integration_quality": ValidationCriteria(
                name="Integration Quality", 
                weight=0.2,
                description="Quality of frontend-backend integration",
                validation_rules=[
                    "Frontend communicates properly with backend",
                    "Data flow between components is correct",
                    "Error states are handled gracefully",
                    "User workflows function end-to-end"
                ]
            ),
            "deployment_readiness": ValidationCriteria(
                name="Deployment Readiness",
                weight=0.1,
                description="Readiness for production deployment",
                validation_rules=[
                    "Application can be deployed successfully",
                    "Configuration is production-ready",
                    "Monitoring and logging are implemented",
                    "Security configurations are in place"
                ]
            ),
            "user_requirements": ValidationCriteria(
                name="User Requirements Compliance",
                weight=0.1,
                description="Compliance with original user requirements",
                validation_rules=[
                    "All specified features are implemented",
                    "User workflow matches requirements",
                    "Performance meets expectations",
                    "Functionality aligns with user intent"
                ]
            )
        }

    async def validate_workflow_results(self,
                                      workflow_id: str,
                                      success_criteria: List[str],
                                      artifacts: Dict[str, List[str]],
                                      task_results: Dict[str, Any],
                                      user_input: str) -> Dict[str, Any]:
        """
        Validate complete workflow results
        
        Args:
            workflow_id: Workflow identifier
            success_criteria: List of success criteria
            artifacts: Generated artifacts by agent
            task_results: Results from completed tasks
            user_input: Original user requirements
            
        Returns:
            Comprehensive validation results
        """
        try:
            logger.info(f"Validating workflow results for {workflow_id}")
            
            validation_results = {}
            overall_scores = []
            
            # Validate each criteria
            for criteria_name, criteria in self.validation_criteria.items():
                result = await self._validate_criteria(
                    criteria, artifacts, task_results, user_input, success_criteria
                )
                validation_results[criteria_name] = result
                overall_scores.append(result.score * criteria.weight)
            
            # Calculate overall score
            overall_score = sum(overall_scores)
            
            # Determine quality level
            quality_level = self._determine_quality_level(overall_score)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(validation_results, overall_score)
            
            # Compile final results
            final_results = {
                "workflow_id": workflow_id,
                "overall_score": round(overall_score, 3),
                "quality_level": quality_level,
                "validation_details": {name: result.model_dump() for name, result in validation_results.items()},
                "recommendations": recommendations,
                "validation_passed": overall_score >= 0.7,
                "timestamp": datetime.utcnow().isoformat(),
                "summary": self._generate_validation_summary(validation_results, overall_score)
            }
            
            logger.info(f"Validation completed for {workflow_id}: {quality_level} ({overall_score:.3f})")
            return final_results
            
        except Exception as e:
            logger.error(f"Error validating workflow results: {str(e)}")
            return {
                "workflow_id": workflow_id,
                "overall_score": 0.0,
                "quality_level": "error",
                "validation_errors": [str(e)],
                "validation_passed": False,
                "timestamp": datetime.utcnow().isoformat()
            }

    async def _validate_criteria(self,
                               criteria: ValidationCriteria,
                               artifacts: Dict[str, List[str]],
                               task_results: Dict[str, Any],
                               user_input: str,
                               success_criteria: List[str]) -> ValidationResult:
        """Validate a specific criteria"""
        try:
            if criteria.name == "Frontend Quality":
                return await self._validate_frontend_quality(artifacts, task_results, user_input)
            elif criteria.name == "Backend Functionality":
                return await self._validate_backend_functionality(artifacts, task_results, user_input)
            elif criteria.name == "Integration Quality":
                return await self._validate_integration_quality(artifacts, task_results, user_input)
            elif criteria.name == "Deployment Readiness":
                return await self._validate_deployment_readiness(artifacts, task_results, user_input)
            elif criteria.name == "User Requirements Compliance":
                return await self._validate_user_requirements(artifacts, task_results, user_input, success_criteria)
            else:
                return ValidationResult(
                    criteria_name=criteria.name,
                    score=0.5,
                    passed=False,
                    issues=["Unknown validation criteria"],
                    recommendations=["Implement validation for this criteria"]
                )
                
        except Exception as e:
            logger.error(f"Error validating {criteria.name}: {str(e)}")
            return ValidationResult(
                criteria_name=criteria.name,
                score=0.0,
                passed=False,
                issues=[f"Validation error: {str(e)}"],
                recommendations=["Fix validation errors"]
            )

    async def _validate_frontend_quality(self,
                                       artifacts: Dict[str, List[str]],
                                       task_results: Dict[str, Any],
                                       user_input: str) -> ValidationResult:
        """Validate frontend quality"""
        issues = []
        recommendations = []
        score_components = []
        details = {}
        
        frontend_artifacts = artifacts.get("frontend", [])
        
        # Check if frontend artifacts exist
        if not frontend_artifacts:
            issues.append("No frontend artifacts generated")
            score_components.append(0.0)
        else:
            details["artifacts_count"] = len(frontend_artifacts)
            
            # Check for essential files
            has_html = any("html" in artifact.lower() for artifact in frontend_artifacts)
            has_css = any("css" in artifact.lower() for artifact in frontend_artifacts)
            has_js = any("js" in artifact.lower() or "javascript" in artifact.lower() for artifact in frontend_artifacts)
            
            if has_html:
                score_components.append(0.4)
                details["has_html"] = True
            else:
                issues.append("Missing HTML files")
                recommendations.append("Generate HTML structure files")
                score_components.append(0.0)
            
            if has_css:
                score_components.append(0.3)
                details["has_css"] = True
            else:
                issues.append("Missing CSS styling files")
                recommendations.append("Create CSS stylesheets for responsive design")
                score_components.append(0.0)
            
            if has_js:
                score_components.append(0.3)
                details["has_js"] = True
            else:
                # JavaScript might not be required for all projects 
                score_components.append(0.2)
                details["has_js"] = False
        
        # Check frontend task results
        frontend_tasks = {k: v for k, v in task_results.items() if "frontend" in k.lower()}
        if frontend_tasks:
            successful_tasks = sum(1 for task in frontend_tasks.values() 
                                 if hasattr(task, 'status') and task.status.value == "completed")
            task_success_rate = successful_tasks / len(frontend_tasks)
            score_components.append(task_success_rate * 0.3)
            details["task_success_rate"] = task_success_rate
        
        # Check responsiveness mentions in user input
        if any(keyword in user_input.lower() for keyword in ["responsive", "mobile", "tablet"]):
            if any("responsive" in artifact.lower() for artifact in frontend_artifacts):
                details["responsive_design"] = True
            else:
                issues.append("Responsive design requirements not clearly addressed")
                recommendations.append("Ensure responsive design implementation")
        
        final_score = min(sum(score_components), 1.0)
        
        return ValidationResult(
            criteria_name="Frontend Quality",
            score=final_score,
            passed=final_score >= 0.7,
            issues=issues,
            recommendations=recommendations,
            details=details
        )

    async def _validate_backend_functionality(self,
                                            artifacts: Dict[str, List[str]],
                                            task_results: Dict[str, Any],
                                            user_input: str) -> ValidationResult:
        """Validate backend functionality"""
        issues = []
        recommendations = []
        score_components = []
        details = {}
        
        backend_artifacts = artifacts.get("backend", [])
        
        # Check if backend artifacts exist
        if not backend_artifacts:
            issues.append("No backend artifacts generated")
            score_components.append(0.0)
        else:
            details["artifacts_count"] = len(backend_artifacts)
            
            # Check for essential backend components
            has_api = any("api" in artifact.lower() for artifact in backend_artifacts)
            has_models = any("model" in artifact.lower() for artifact in backend_artifacts)
            has_auth = any("auth" in artifact.lower() for artifact in backend_artifacts)
            has_database = any(db_keyword in artifact.lower() 
                             for db_keyword in ["database", "schema", "migration"] 
                             for artifact in backend_artifacts)
            
            component_score = 0
            if has_api:
                component_score += 0.3
                details["has_api"] = True
            else:
                issues.append("Missing API implementation")
                recommendations.append("Implement API endpoints")
            
            if has_models:
                component_score += 0.2
                details["has_models"] = True
            
            if has_auth and any(keyword in user_input.lower() for keyword in ["auth", "login", "register"]):
                component_score += 0.3
                details["has_auth"] = True
            elif any(keyword in user_input.lower() for keyword in ["auth", "login", "register"]):
                issues.append("Authentication requirements not addressed")
                recommendations.append("Implement authentication system")
            
            if has_database:
                component_score += 0.2
                details["has_database"] = True
            
            score_components.append(component_score)
        
        # Check backend task results
        backend_tasks = {k: v for k, v in task_results.items() if "backend" in k.lower()}
        if backend_tasks:
            successful_tasks = sum(1 for task in backend_tasks.values() 
                                 if hasattr(task, 'status') and task.status.value == "completed")
            task_success_rate = successful_tasks / len(backend_tasks)
            score_components.append(task_success_rate * 0.4)
            details["task_success_rate"] = task_success_rate
        
        final_score = min(sum(score_components), 1.0)
        
        return ValidationResult(
            criteria_name="Backend Functionality",
            score=final_score,
            passed=final_score >= 0.7,
            issues=issues,
            recommendations=recommendations,
            details=details
        )

    async def _validate_integration_quality(self,
                                          artifacts: Dict[str, List[str]],
                                          task_results: Dict[str, Any],
                                          user_input: str) -> ValidationResult:
        """Validate integration between components"""
        issues = []
        recommendations = []
        score_components = []
        details = {}
        
        has_frontend = "frontend" in artifacts and len(artifacts["frontend"]) > 0
        has_backend = "backend" in artifacts and len(artifacts["backend"]) > 0
        
        details["has_frontend"] = has_frontend
        details["has_backend"] = has_backend
        
        if not has_frontend or not has_backend:
            # Single-component projects don't need integration
            if has_frontend and not has_backend:
                details["integration_type"] = "frontend-only"
                score_components.append(0.8)  # Good score for frontend-only
            elif has_backend and not has_frontend:
                details["integration_type"] = "backend-only"
                score_components.append(0.8)  # Good score for backend-only
            else:
                issues.append("Neither frontend nor backend components found")
                score_components.append(0.0)
        else:
            # Multi-component project - check integration
            details["integration_type"] = "full-stack"
            
            # Look for integration tasks
            integration_tasks = {k: v for k, v in task_results.items() 
                               if "integration" in k.lower() or "full_stack" in k.lower()}
            
            if integration_tasks:
                successful_integrations = sum(1 for task in integration_tasks.values() 
                                            if hasattr(task, 'status') and task.status.value == "completed")
                integration_success_rate = successful_integrations / len(integration_tasks)
                score_components.append(integration_success_rate * 0.6)
                details["integration_success_rate"] = integration_success_rate
            else:
                issues.append("No explicit integration tasks found")
                recommendations.append("Implement frontend-backend integration")
                score_components.append(0.3)  # Partial score
            
            # Check for API communication indicators
            frontend_artifacts = artifacts.get("frontend", [])
            has_api_calls = any("api" in artifact.lower() or "fetch" in artifact.lower() 
                              for artifact in frontend_artifacts)
            
            if has_api_calls:
                score_components.append(0.4)
                details["has_api_communication"] = True
            else:
                recommendations.append("Implement API communication between frontend and backend")
                details["has_api_communication"] = False
        
        final_score = min(sum(score_components), 1.0)
        
        return ValidationResult(
            criteria_name="Integration Quality",
            score=final_score,
            passed=final_score >= 0.7,
            issues=issues,
            recommendations=recommendations,
            details=details
        )

    async def _validate_deployment_readiness(self,
                                           artifacts: Dict[str, List[str]],
                                           task_results: Dict[str, Any],
                                           user_input: str) -> ValidationResult:
        """Validate deployment readiness"""
        issues = []
        recommendations = []
        score_components = []
        details = {}
        
        deployment_artifacts = artifacts.get("deployment", [])
        
        # Check if deployment is required
        deployment_required = any(keyword in user_input.lower() 
                                for keyword in ["deploy", "production", "host", "cloud"])
        
        details["deployment_required"] = deployment_required
        
        if not deployment_required:
            # If deployment wasn't requested, give good score
            score_components.append(0.9)
            details["deployment_type"] = "not_required"
        else:
            if not deployment_artifacts:
                issues.append("Deployment artifacts missing despite deployment requirements")
                recommendations.append("Create deployment configuration files")
                score_components.append(0.0)
            else:
                details["artifacts_count"] = len(deployment_artifacts)
                
                # Check for deployment essentials
                has_docker = any("docker" in artifact.lower() for artifact in deployment_artifacts)
                has_config = any("config" in artifact.lower() or "yml" in artifact.lower() 
                                for artifact in deployment_artifacts)
                has_cicd = any("ci" in artifact.lower() or "pipeline" in artifact.lower() 
                             for artifact in deployment_artifacts)
                
                component_score = 0
                if has_docker:
                    component_score += 0.4
                    details["has_containerization"] = True
                else:
                    recommendations.append("Add containerization with Docker")
                
                if has_config:
                    component_score += 0.3
                    details["has_config"] = True
                else:
                    recommendations.append("Create deployment configuration files")
                
                if has_cicd:
                    component_score += 0.3
                    details["has_cicd"] = True
                
                score_components.append(component_score)
        
        # Check deployment task results
        deployment_tasks = {k: v for k, v in task_results.items() if "deploy" in k.lower()}
        if deployment_tasks:
            successful_tasks = sum(1 for task in deployment_tasks.values() 
                                 if hasattr(task, 'status') and task.status.value == "completed")
            task_success_rate = successful_tasks / len(deployment_tasks)
            score_components.append(task_success_rate * 0.3)
            details["task_success_rate"] = task_success_rate
        
        final_score = min(sum(score_components), 1.0)
        
        return ValidationResult(
            criteria_name="Deployment Readiness",
            score=final_score,
            passed=final_score >= 0.7,
            issues=issues,
            recommendations=recommendations,
            details=details
        )

    async def _validate_user_requirements(self,
                                        artifacts: Dict[str, List[str]],
                                        task_results: Dict[str, Any],
                                        user_input: str,
                                        success_criteria: List[str]) -> ValidationResult:
        """Validate compliance with user requirements"""
        issues = []
        recommendations = []
        score_components = []
        details = {}
        
        # Analyze user input for key requirements
        user_keywords = self._extract_user_keywords(user_input)
        details["user_keywords"] = user_keywords
        
        # Check if artifacts address user keywords
        all_artifacts = []
        for agent_artifacts in artifacts.values():
            all_artifacts.extend(agent_artifacts)
        
        artifact_text = " ".join(all_artifacts).lower()
        
        addressed_keywords = []
        missing_keywords = []
        
        for keyword in user_keywords:
            if keyword in artifact_text or any(keyword in result_text for result_text in task_results.values() if isinstance(result_text, str)):
                addressed_keywords.append(keyword)
            else:
                missing_keywords.append(keyword)
        
        # Calculate coverage score
        if user_keywords:
            coverage_score = len(addressed_keywords) / len(user_keywords)
            score_components.append(coverage_score * 0.6)
            details["keyword_coverage"] = coverage_score
            details["addressed_keywords"] = addressed_keywords
            details["missing_keywords"] = missing_keywords
        else:
            score_components.append(0.5)  # Neutral score if no specific keywords
        
        # Check success criteria compliance
        if success_criteria:
            # Simple heuristic: check if task results suggest success criteria are met
            successful_tasks = sum(1 for task in task_results.values() 
                                 if hasattr(task, 'status') and task.status.value == "completed")
            total_tasks = len(task_results)
            
            if total_tasks > 0:
                criteria_score = successful_tasks / total_tasks
                score_components.append(criteria_score * 0.4)
                details["criteria_compliance"] = criteria_score
            
            # Add recommendations for missing keywords
            for keyword in missing_keywords:
                recommendations.append(f"Address user requirement: {keyword}")
        
        # If few artifacts generated, might not meet requirements
        total_artifacts = sum(len(agent_artifacts) for agent_artifacts in artifacts.values())
        if total_artifacts < 2:
            issues.append("Limited artifacts generated - may not fully address user requirements")
            recommendations.append("Generate more comprehensive deliverables")
        
        final_score = min(sum(score_components), 1.0)
        
        return ValidationResult(
            criteria_name="User Requirements Compliance",
            score=final_score,
            passed=final_score >= 0.7,
            issues=issues,
            recommendations=recommendations,
            details=details
        )

    def _extract_user_keywords(self, user_input: str) -> List[str]:
        """Extract key requirements from user input"""
        # Common requirement keywords
        requirement_patterns = [
            r'\b(create|build|develop|make|implement|design)\s+(?:a|an)?\s*(\w+(?:\s+\w+)*)',
            r'\b(add|include|integrate|connect)\s+(\w+(?:\s+\w+)*)',
            r'\b(with|using|for)\s+(\w+(?:\s+\w+)*)',
            r'\b(\w+)\s+(?:page|system|app|application|component|feature)'
        ]
        
        keywords = set()
        user_input_lower = user_input.lower()
        
        for pattern in requirement_patterns:
            matches = re.findall(pattern, user_input_lower)
            for match in matches:
                if isinstance(match, tuple):
                    keywords.update(word.strip() for word in match if word.strip())
                else:
                    keywords.add(match.strip())
        
        # Filter out common words
        common_words = {'a', 'an', 'the', 'and', 'or', 'but', 'with', 'for', 'to', 'of', 'in', 'on', 'at'}
        filtered_keywords = [kw for kw in keywords if kw not in common_words and len(kw) > 2]
        
        return filtered_keywords[:10]  # Limit to top 10 keywords

    def _determine_quality_level(self, overall_score: float) -> str:
        """Determine quality level based on score"""
        for level, threshold in self.quality_thresholds.items():
            if overall_score >= threshold:
                return level
        return "poor"

    def _generate_recommendations(self, 
                                validation_results: Dict[str, ValidationResult], 
                                overall_score: float) -> List[str]:
        """Generate overall recommendations"""
        recommendations = []
        
        # Collect recommendations from all validation results
        for result in validation_results.values():
            recommendations.extend(result.recommendations)
        
        # Add overall recommendations based on score
        if overall_score < 0.5:
            recommendations.append("Consider restarting the workflow with clearer requirements")
        elif overall_score < 0.7:
            recommendations.append("Focus on improving the lowest-scoring components")
        elif overall_score < 0.9:
            recommendations.append("Fine-tune existing components for better quality")
        
        # Remove duplicates and limit
        unique_recommendations = list(dict.fromkeys(recommendations))
        return unique_recommendations[:8]  # Limit to 8 recommendations

    def _generate_validation_summary(self, 
                                   validation_results: Dict[str, ValidationResult], 
                                   overall_score: float) -> str:
        """Generate a human-readable validation summary"""
        quality_level = self._determine_quality_level(overall_score)
        
        passed_count = sum(1 for result in validation_results.values() if result.passed)
        total_count = len(validation_results)
        
        summary = f"Validation Summary: {quality_level.upper()} quality ({overall_score:.1%})\n"
        summary += f"Passed {passed_count}/{total_count} validation criteria.\n"
        
        # Highlight areas needing attention
        low_scoring = [name for name, result in validation_results.items() if result.score < 0.7]
        if low_scoring:
            summary += f"Areas needing improvement: {', '.join(low_scoring)}"
        
        return summary

    async def quick_quality_check(self, artifacts: Dict[str, List[str]]) -> Dict[str, Any]:
        """Perform a quick quality check on artifacts"""
        try:
            total_artifacts = sum(len(agent_artifacts) for agent_artifacts in artifacts.values())
            agent_coverage = len(artifacts.keys())
            
            # Quick scoring based on artifact counts and coverage
            artifact_score = min(total_artifacts / 5.0, 1.0)  # Expect at least 5 artifacts for full score
            coverage_score = min(agent_coverage / 3.0, 1.0)  # Expect 3 agents for full coverage
            
            quick_score = (artifact_score + coverage_score) / 2
            
            return {
                "quick_score": quick_score,
                "total_artifacts": total_artifacts,
                "agent_coverage": agent_coverage,
                "quality_indicator": self._determine_quality_level(quick_score),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in quick quality check: {str(e)}")
            return {
                "quick_score": 0.0,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
