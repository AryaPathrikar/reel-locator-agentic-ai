# rl_agents/parallel_vision.py
"""
Parallel Vision Engine for redundant and robust location detection

This module runs multiple VisionAgent instances in parallel on the same frames,
providing redundancy and robustness. Results are merged based on confidence scores.
"""

import asyncio
import logging
from typing import List, Dict, Any

from rl_agents.vision_agent import VisionAgent

logger = logging.getLogger("reel_locator_mcp")


class ParallelVisionEngine:
    """
    Runs multiple VisionAgent instances in parallel on the same frames
    and merges the results based on confidence ranking.
    
    This provides redundancy and robustness - if one agent makes an error,
    the others can still provide accurate results. The best result (highest
    average confidence) is selected as the final output.
    """

    def __init__(self, num_agents: int = 3):
        """
        Initialize parallel vision engine with multiple agents.
        
        Args:
            num_agents: Number of vision agents to run in parallel (default: 3)
        """
        self.num_agents = num_agents
        # Create multiple independent vision agent instances
        self.agents = [VisionAgent() for _ in range(num_agents)]

    async def analyze(self, frame_paths: List[str]) -> Dict[str, Any]:
        """
        Analyze frames using multiple agents in parallel and merge results.
        
        All agents analyze the same frames simultaneously. The result with the
        highest average confidence score is selected as the final output.
        
        Args:
            frame_paths: List of file paths to image frames
            
        Returns:
            Dictionary with location info from the agent with highest confidence
            
        Raises:
            ValueError: If no valid results are returned
        """
        logger.info(
            f"[PARALLEL] Starting parallel vision with {self.num_agents} agents "
            f"for {len(frame_paths)} frame(s)"
        )

        # Create tasks for all agents (all agents run on SAME frames)
        tasks = []
        for i, agent in enumerate(self.agents):
            logger.info(f"[PARALLEL] Dispatching frames to Agent {i}")
            tasks.append(self._run_agent(agent, frame_paths, agent_id=i))

        # Run all agents concurrently using asyncio.gather
        results = await asyncio.gather(*tasks, return_exceptions=False)
        logger.info(f"[PARALLEL] Completed parallel inference → {len(results)} results")

        # Aggregate Results: Select the result with highest average confidence
        best = None
        best_score = -1

        for idx, r in enumerate(results):
            # Calculate average confidence from all landmarks
            confs = [lm.get("confidence", 0.0) for lm in r.get("landmarks", [])]
            avg_conf = sum(confs) / len(confs) if confs else 0
            logger.info(f"[PARALLEL] Agent {idx} avg confidence = {avg_conf}")

            # Track the best result
            if avg_conf > best_score:
                best = r
                best_score = avg_conf

        if best is None:
            raise ValueError("ParallelVisionEngine: No valid results returned")

        # Add average confidence to the final result
        best["avg_confidence"] = best_score
        logger.info(
            f"[PARALLEL] FINAL merged result → city={best.get('city')}, "
            f"country={best.get('country')}, avg_conf={best_score}"
        )

        return best

    async def _run_agent(self, agent: VisionAgent, frames: List[str], agent_id: int):
        """
        Run a single vision agent in an executor to avoid blocking.
        
        Args:
            agent: VisionAgent instance to run
            frames: List of frame paths to analyze
            agent_id: Identifier for logging purposes
            
        Returns:
            Analysis result dictionary from the agent
        """
        logger.info(f"[PARALLEL] Agent {agent_id} starting inference on {len(frames)} frames")

        # Run synchronous agent.analyze_frames in executor to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, agent.analyze_frames, frames)

        logger.info(f"[PARALLEL] Agent {agent_id} finished inference")
        return result
