"""
JSON Schema definitions for HIL feedback and trajectory data.

This module defines data structures and validation logic for:
- Trajectories sent to frontend (without error info)
- Feedback received from frontend (with error info)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union


@dataclass
class ErrorInfo:
    """Error information structure from human feedback.
    
    Attributes:
        type: "real-time" for discrete timesteps, "calibrated" for intervals
        data: List[int] for real-time, List[List[int]] for calibrated intervals
    """
    type: str  # "real-time" or "calibrated"
    data: Union[List[int], List[List[int]]]
    
    def get_timesteps(self) -> List[int]:
        """Convert error data to flat list of timesteps.
        
        Returns:
            List of error timesteps (expanded from intervals if calibrated)
        """
        if self.type == "real-time":
            # data: [2, 5, 100, 190, 320, 350]
            return list(self.data)
        elif self.type == "calibrated":
            # data: [[2, 5], [100, 190], [320, 350]]
            timesteps = []
            for interval in self.data:
                if len(interval) == 2:
                    start, end = interval
                    timesteps.extend(range(start, end + 1))
            return timesteps
        else:
            raise ValueError(f"Unknown error type: {self.type}")


@dataclass
class FeedbackData:
    """Complete feedback data structure from frontend.
    
    Attributes:
        static_info: Static environment information
        dynamic_state: Dynamic state at specific timestep
        error_info: List of exactly 2 error information objects
                   [0]: real-time version
                   [1]: calibrated version
        raw_data: Original JSON data for debugging
    """
    static_info: Dict[str, Any]
    dynamic_state: Dict[str, Any]
    error_info: List[ErrorInfo]  # Exactly 2 elements: real-time and calibrated
    raw_data: Dict[str, Any]
    
    def get_realtime_timesteps(self) -> List[int]:
        """Get real-time error timesteps.
        
        Searches error_info list for the entry with type="real-time".
        
        Returns:
            Sorted list of real-time error timesteps
        """
        for error in self.error_info:
            if error.type == "real-time":
                return sorted(list(set(error.get_timesteps())))
        return []
    
    def get_calibrated_timesteps(self) -> List[int]:
        """Get calibrated error timesteps.
        
        Searches error_info list for the entry with type="calibrated".
        
        Returns:
            Sorted list of calibrated error timesteps
        """
        for error in self.error_info:
            if error.type == "calibrated":
                return sorted(list(set(error.get_timesteps())))
        return []
    
    def get_all_error_timesteps(self) -> List[int]:
        """Get all error timesteps from both real-time and calibrated versions.
        
        Note: This merges both versions. For separate access, use
        get_realtime_timesteps() or get_calibrated_timesteps().
        
        Returns:
            Sorted list of unique error timesteps from both versions
        """
        all_timesteps = []
        for error in self.error_info:
            all_timesteps.extend(error.get_timesteps())
        return sorted(list(set(all_timesteps)))


@dataclass
class TrajectoryData:
    """Trajectory data structure sent to frontend (no error info).
    
    Attributes:
        static_info: Static environment information
        dynamic_states: List of dynamic states for each timestep
    """
    static_info: Dict[str, Any]
    dynamic_states: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization.
        
        Note: Key is "dynamicState" (singular) even though value is a list.
        This matches the frontend expectation where:
        - Key: "dynamicState" (consistent naming with feedback)
        - Value: list of state dictionaries (one per timestep)
        """
        return {
            "staticInfo": self.static_info,
            "dynamicState": self.dynamic_states,  # Singular key, list value
        }


def validate_feedback_schema(data: Dict[str, Any]) -> bool:
    """Validate feedback JSON schema.
    
    Args:
        data: JSON data dictionary
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    required_keys = ["staticInfo", "dynamicState", "errorInfo"]
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing required key: {key}")
    
    # Validate errorInfo structure (must be list of exactly 2 elements)
    error_info = data.get("errorInfo", [])
    if not isinstance(error_info, list):
        raise ValueError("errorInfo must be a list")
    
    if len(error_info) != 2:
        raise ValueError(f"errorInfo must have exactly 2 elements (real-time and calibrated), got {len(error_info)}")
    
    # Validate each error info element
    for idx, error in enumerate(error_info):
        if not isinstance(error, dict):
            raise ValueError(f"errorInfo[{idx}] must be a dictionary")
        if "type" not in error:
            raise ValueError(f"errorInfo[{idx}] missing 'type' field")
        if "data" not in error:
            raise ValueError(f"errorInfo[{idx}] missing 'data' field")
        
        error_type = error["type"]
        if error_type not in ["real-time", "calibrated"]:
            raise ValueError(f"errorInfo[{idx}] has invalid type: {error_type}")
    
    # Validate that we have exactly one of each type
    types = [error["type"] for error in error_info]
    if "real-time" not in types:
        raise ValueError("errorInfo must contain one 'real-time' type entry")
    if "calibrated" not in types:
        raise ValueError("errorInfo must contain one 'calibrated' type entry")
    
    # Check for duplicates
    if types.count("real-time") > 1:
        raise ValueError("errorInfo cannot have multiple 'real-time' entries")
    if types.count("calibrated") > 1:
        raise ValueError("errorInfo cannot have multiple 'calibrated' entries")
    
    return True


def validate_trajectory_schema(data: Dict[str, Any]) -> bool:
    """Validate trajectory JSON schema.
    
    Trajectory JSON structure:
    {
        "staticInfo": {...},
        "dynamicState": [  # Singular key!
            {...},  # timestep 0
            {...},  # timestep 1
            ...
        ]
    }
    
    Args:
        data: JSON data dictionary
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    required_keys = ["staticInfo", "dynamicState"]  # Changed to singular
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing required key: {key}")
    
    dynamic_state = data.get("dynamicState", [])
    if not isinstance(dynamic_state, list):
        raise ValueError("dynamicState must be a list")
    
    return True


def parse_feedback_json(data: Dict[str, Any]) -> FeedbackData:
    """Parse JSON dictionary to FeedbackData object.
    
    Args:
        data: JSON data dictionary
        
    Returns:
        FeedbackData object with 2 ErrorInfo objects (real-time and calibrated)
    """
    validate_feedback_schema(data)
    
    # Parse error_info list (exactly 2 elements)
    error_info_list = []
    for error in data.get("errorInfo", []):
        error_info_list.append(
            ErrorInfo(
                type=error["type"],
                data=error["data"]
            )
        )
    
    return FeedbackData(
        static_info=data.get("staticInfo", {}),
        dynamic_state=data.get("dynamicState", {}),
        error_info=error_info_list,
        raw_data=data
    )

