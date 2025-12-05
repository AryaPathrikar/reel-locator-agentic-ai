# rl_agents/refinement_agent.py
"""
Refinement Loop Agent for iterative location improvement

This module implements a loop refinement pattern that iteratively improves
location predictions until a confidence threshold is met or confidence
stops improving.
"""

from typing import Dict, Any, List
import logging

logger = logging.getLogger("reel_locator_mcp")


class RefinementLoop:
    """
    Sequential refinement loop that iteratively improves location predictions.
    
    The loop continues until:
    - Confidence threshold is met, OR
    - Confidence stops improving, OR
    - Maximum iterations reached
    
    This ensures stability and gradual improvement of location accuracy.
    """
    
    def __init__(self, threshold: float = 0.70, max_iters: int = 3):
        """
        Initialize refinement loop with stopping criteria.
        
        Args:
            threshold: Confidence threshold to stop early (default: 0.70)
            max_iters: Maximum number of iterations (default: 3)
        """
        self.threshold = threshold
        self.max_iters = max_iters

    def refine(self, raw_vision, geo_agent):
        """
        Execute sequential refinement loop to improve location accuracy.
        
        The loop iteratively refines location metadata using the geo_agent.
        Stops early if confidence threshold is met or confidence stops improving.
        
        Args:
            raw_vision: Initial location result from vision agents
            geo_agent: GeoAgent instance for refinement
            
        Returns:
            Tuple of (final_location_info, iterations_used):
            - final_location_info: Refined location dictionary
            - iterations_used: Number of iterations performed
        """
        logger.info("[LOOP] Starting refinement loop")
        logger.info(f"[LOOP] Threshold={self.threshold}, Max Iterations={self.max_iters}")

        # Initialize with raw vision result
        current = raw_vision
        last_conf = raw_vision.get("confidence", 0)
        logger.info(f"[LOOP] Initial confidence = {last_conf}")

        iters = 0

        # Iterate up to max_iters times
        for i in range(self.max_iters):
            iters += 1
            logger.info(f"[LOOP] Iteration {i+1}/{self.max_iters}")

            # Refine location using geo agent
            refined = geo_agent.refine_location(current)
            new_conf = refined.get("confidence", 0)
            logger.info(f"[LOOP] Confidence after iteration {i+1}: {new_conf}")

            # Stop if confidence meets threshold (early exit)
            if new_conf >= self.threshold:
                logger.info("[LOOP] Threshold met → stopping loop early")
                return refined, iters

            # Stop if confidence dropped or didn't improve (no progress)
            if new_conf <= last_conf:
                logger.info("[LOOP] Confidence dropped or did not improve → stopping loop")
                return refined, iters

            # Continue refinement with improved result
            current = refined
            last_conf = new_conf

        # Reached maximum iterations
        logger.info("[LOOP] Max iterations reached")
        return current, iters
