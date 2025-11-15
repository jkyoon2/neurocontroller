"""
FeedbackLoader: Load and parse human feedback from JSON files.

This module handles:
1. Loading feedback JSON files from filesystem
2. Parsing error information (real-time vs calibrated)
3. Converting to FeedbackData objects
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

from zsceval.utils.feedback.feedback_schema import (
    FeedbackData,
    parse_feedback_json,
    validate_feedback_schema,
)


class FeedbackLoader:
    """Loader for human feedback JSON files.
    
    This class is used by the Learner process to load feedback
    from the human_interface/data/feedback_from_human/ directory.
    """
    
    def __init__(self, feedback_dir: Optional[str] = None):
        """Initialize FeedbackLoader.
        
        Args:
            feedback_dir: Directory containing feedback JSON files.
                         If None, uses default 'human_interface/data/feedback_from_human/'
        """
        if feedback_dir is None:
            # Default to workspace root / human_interface / data / feedback_from_human
            self.feedback_dir = Path("human_interface/data/feedback_from_human")
        else:
            self.feedback_dir = Path(feedback_dir)
        
        # Create directory if it doesn't exist
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"FeedbackLoader initialized with directory: {self.feedback_dir}")
    
    def load_feedback(self, filepath: str) -> Optional[FeedbackData]:
        """Load feedback from JSON file.
        
        Args:
            filepath: Path to feedback JSON file (relative or absolute)
            
        Returns:
            FeedbackData object if successful, None otherwise
        """
        try:
            path = Path(filepath)
            if not path.is_absolute():
                path = self.feedback_dir / path
            
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"[FeedbackLoader] Successfully loaded feedback from {path}")
            
            # Validate and parse
            feedback_data = parse_feedback_json(data)
            
            # Log error timesteps for debugging
            error_timesteps = feedback_data.get_all_error_timesteps()
            logger.debug(f"[FeedbackLoader] Extracted {len(error_timesteps)} error timesteps: {error_timesteps[:10]}...")
            
            return feedback_data
            
        except FileNotFoundError:
            logger.error(f"[FeedbackLoader] Feedback file not found: {filepath}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"[FeedbackLoader] Failed to decode JSON from {filepath}: {e}")
            return None
        except ValueError as e:
            logger.error(f"[FeedbackLoader] Schema validation failed for {filepath}: {e}")
            return None
        except Exception as e:
            logger.error(f"[FeedbackLoader] Unexpected error loading {filepath}: {e}")
            return None
    
    def load_feedback_by_id(self, episode_id: int) -> Optional[FeedbackData]:
        """Load feedback by episode ID using standard naming convention.
        
        Args:
            episode_id: Episode number
            
        Returns:
            FeedbackData object if successful, None otherwise
        """
        filename = f"feedback_{episode_id}.json"
        return self.load_feedback(filename)
    
    def check_feedback_exists(self, filepath: str) -> bool:
        """Check if feedback file exists.
        
        Args:
            filepath: Path to feedback JSON file
            
        Returns:
            True if file exists, False otherwise
        """
        path = Path(filepath)
        if not path.is_absolute():
            path = self.feedback_dir / path
        
        return path.exists()
    
    def wait_for_feedback(
        self, 
        filepath: str, 
        timeout: Optional[float] = None,
        check_interval: float = 1.0
    ) -> Optional[FeedbackData]:
        """Wait for feedback file to appear and load it.
        
        This is a blocking operation that polls for file existence.
        
        Args:
            filepath: Path to feedback JSON file
            timeout: Maximum seconds to wait (None = wait forever)
            check_interval: Seconds between file existence checks
            
        Returns:
            FeedbackData object if successful, None if timeout
        """
        import time
        
        path = Path(filepath)
        if not path.is_absolute():
            path = self.feedback_dir / path
        
        logger.info(f"[FeedbackLoader] Waiting for feedback file: {path}")
        
        start_time = time.time()
        while True:
            if path.exists():
                logger.info(f"[FeedbackLoader] Feedback file detected: {path}")
                # Small delay to ensure file write is complete
                time.sleep(0.5)
                return self.load_feedback(str(path))
            
            if timeout is not None and (time.time() - start_time) > timeout:
                logger.warning(f"[FeedbackLoader] Timeout waiting for feedback: {path}")
                return None
            
            time.sleep(check_interval)
    
    def list_available_feedback(self) -> List[str]:
        """List all feedback JSON files in the feedback directory.
        
        Returns:
            List of feedback file paths
        """
        if not self.feedback_dir.exists():
            return []
        
        feedback_files = sorted(self.feedback_dir.glob("feedback_*.json"))
        return [str(f.name) for f in feedback_files]
    
    def parse_error_timesteps(self, feedback_data: FeedbackData) -> List[int]:
        """Extract error timesteps from FeedbackData.
        
        This is a convenience method that wraps FeedbackData.get_all_error_timesteps()
        
        Args:
            feedback_data: FeedbackData object
            
        Returns:
            Sorted list of unique error timesteps
        """
        return feedback_data.get_all_error_timesteps()

