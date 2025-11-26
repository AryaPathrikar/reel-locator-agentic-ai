# rl_agents/refinement_agent.py

from typing import Dict, Any, List
import logging

logger = logging.getLogger("reel_locator_mcp")


class RefinementLoop:
    def __init__(self, threshold: float = 0.70, max_iters: int = 3):
        self.threshold = threshold
        self.max_iters = max_iters

    def refine(self, raw_vision, geo_agent):
        """
        Sequential refinement loop.
        MUST return exactly:
        - final_location_info (dict)
        - iterations_used (int)

        👉 Added:
           - Detailed loop logs
           - Confidence tracking logs
           - Early stop logs
        """

        logger.info("[LOOP] Starting refinement loop")
        logger.info(f"[LOOP] Threshold={self.threshold}, Max Iterations={self.max_iters}")

        current = raw_vision
        last_conf = raw_vision.get("confidence", 0)
        logger.info(f"[LOOP] Initial confidence = {last_conf}")

        iters = 0

        for i in range(self.max_iters):
            iters += 1
            logger.info(f"[LOOP] Iteration {i+1}/{self.max_iters}")

            refined = geo_agent.refine_location(current)
            new_conf = refined.get("confidence", 0)
            logger.info(f"[LOOP] Confidence after iteration {i+1}: {new_conf}")

            # Stop if confidence meets threshold OR confidence stopped improving
            if new_conf >= self.threshold:
                logger.info("[LOOP] Threshold met → stopping loop early")
                return refined, iters

            if new_conf <= last_conf:
                logger.info("[LOOP] Confidence dropped or did not improve → stopping loop")
                return refined, iters

            # Continue refinement
            current = refined
            last_conf = new_conf

        logger.info("[LOOP] Max iterations reached")
        return current, iters
