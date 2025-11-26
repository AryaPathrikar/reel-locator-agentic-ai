# rl_agents/parallel_vision.py

import asyncio
import logging
from typing import List, Dict, Any

from rl_agents.vision_agent import VisionAgent

logger = logging.getLogger("reel_locator_mcp")


class ParallelVisionEngine:
    """
    Runs multiple VisionAgent instances in parallel on the same frames
    and merges the results based on confidence ranking.
    (Original functionality preserved, now with observability logging.)
    """

    def __init__(self, num_agents: int = 3):
        self.num_agents = num_agents
        self.agents = [VisionAgent() for _ in range(num_agents)]

    async def analyze(self, frame_paths: List[str]) -> Dict[str, Any]:
        logger.info(
            f"[PARALLEL] Starting parallel vision with {self.num_agents} agents "
            f"for {len(frame_paths)} frame(s)"
        )

        # Create tasks (all agents run on SAME frames)
        tasks = []
        for i, agent in enumerate(self.agents):
            logger.info(f"[PARALLEL] Dispatching frames to Agent {i}")
            tasks.append(self._run_agent(agent, frame_paths, agent_id=i))

        # Run all agents concurrently
        results = await asyncio.gather(*tasks, return_exceptions=False)
        logger.info(f"[PARALLEL] Completed parallel inference → {len(results)} results")

        # ---- Aggregate Results (original logic kept intact) ----
        best = None
        best_score = -1

        for idx, r in enumerate(results):
            confs = [lm.get("confidence", 0.0) for lm in r.get("landmarks", [])]
            avg_conf = sum(confs) / len(confs) if confs else 0
            logger.info(f"[PARALLEL] Agent {idx} avg confidence = {avg_conf}")

            if avg_conf > best_score:
                best = r
                best_score = avg_conf

        if best is None:
            raise ValueError("ParallelVisionEngine: No valid results returned")

        best["avg_confidence"] = best_score
        logger.info(
            f"[PARALLEL] FINAL merged result → city={best.get('city')}, "
            f"country={best.get('country')}, avg_conf={best_score}"
        )

        return best

    async def _run_agent(self, agent: VisionAgent, frames: List[str], agent_id: int):
        logger.info(f"[PARALLEL] Agent {agent_id} starting inference on {len(frames)} frames")

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, agent.analyze_frames, frames)

        logger.info(f"[PARALLEL] Agent {agent_id} finished inference")
        return result
