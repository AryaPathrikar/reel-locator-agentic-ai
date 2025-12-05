# mcp_server/mcp_server.py
"""
MCP (Model Context Protocol) Server for Reel Locator

This module exposes MCP tools that the root agent can call:
- analyze_reel: Extracts frames and runs parallel vision analysis
- fetch_city_places: Queries Google Places API for attractions
- plan_itinerary_from_reel: Full pipeline orchestrator
- get_observability_metrics: Returns performance metrics

The server runs as a subprocess via stdio and communicates with the ADK agent
using the MCP JSON-RPC protocol.
"""

import os
import sys
import logging
from typing import Any, Dict, List
import json

# Add project root to Python path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# Import frame extraction utility
from tools.extract_frames import extract_key_frames

# Import specialized agents for the pipeline
from rl_agents.vision_agent import VisionAgent
from rl_agents.geo_agent import GeoAgent
from rl_agents.itinerary_agent import ItineraryAgent
from rl_agents.parallel_vision import ParallelVisionEngine
from rl_agents.refinement_agent import RefinementLoop

# Observability utilities for metrics and timing
from observability.obs import timer, inc, record_latency, get_metrics
from observability.dashboard import format_observability_dashboard
import requests  # For Google Places API calls

# Configure logging to STDERR to keep STDOUT clean for MCP JSON-RPC protocol
# MCP uses STDOUT for JSON-RPC communication, so logs must go to STDERR
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("reel_locator_mcp")

load_dotenv()

# Initialize FastMCP server instance
mcp = FastMCP("reel_locator")


def _project_root() -> str:
    """Get the absolute path to the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _default_video_path() -> str:
    """Get the default path for input video files."""
    return os.path.join(_project_root(), "data", "input", "reel.mp4")


def _frames_dir() -> str:
    """Get the directory path where extracted frames are stored."""
    return os.path.join(_project_root(), "data", "frames")


def _google_places_search(
    city: str,
    place_type: str = "tourist_attraction",
    max_results: int = 15,
) -> List[Dict[str, Any]]:
    """
    Search Google Places API for attractions in a given city.
    
    Args:
        city: City name to search in
        place_type: Type of place to search for (default: tourist_attraction)
        max_results: Maximum number of results to return
        
    Returns:
        List of place dictionaries with name, address, rating, location, and types
        
    Raises:
        RuntimeError: If GOOGLE_PLACES_API_KEY is not set
    """
    api_key = os.getenv("GOOGLE_PLACES_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_PLACES_API_KEY not set in environment")

    # Google Places API Text Search endpoint
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {
        "query": f"{place_type} in {city}",
        "key": api_key,
    }

    # Time the API call for observability
    with timer("places_api"):
        resp = requests.get(url, params=params, timeout=20)

    resp.raise_for_status()
    data = resp.json()

    # Record latency metric
    record_latency("places_latency", get_metrics().get("places_api", 0))

    # Extract and format results
    results: List[Dict[str, Any]] = []
    for r in data.get("results", [])[:max_results]:
        results.append(
            {
                "name": r.get("name", ""),
                "address": r.get("formatted_address", ""),
                "rating": r.get("rating"),
                "location": r.get("geometry", {}).get("location", {}),
                "types": r.get("types", []),
            }
        )
    return results

@mcp.tool()
async def analyze_reel(
    video_path: str | None = None,
    max_frames: int = 8,
) -> Dict[str, Any]:
    """
    MCP Tool: Analyze a travel reel video to detect location and landmarks.
    
    This tool orchestrates:
    1. Frame extraction from video
    2. Parallel vision agent analysis
    3. Loop refinement for location accuracy
    
    Args:
        video_path: Path to the video file (defaults to data/input/reel.mp4)
        max_frames: Maximum number of frames to extract (default: 8)
        
    Returns:
        Dictionary with city, country, region, landmarks, and frame paths
    """
    try:
        # Use default video path if none provided
        if video_path is None:
            video_path = _default_video_path()

        logger.info("analyze_reel: video_path=%s", video_path)

        frames_dir = _frames_dir()

        # ----------------------------------------------------
        # 1. FRAME EXTRACTION + OBSERVABILITY
        # ----------------------------------------------------
        # Extract evenly-spaced key frames from the video
        with timer("frame_extraction"):
            frame_paths = extract_key_frames(
                video_path=video_path,
                output_dir=frames_dir,
                max_frames=max_frames,
            )

        # Track number of frames extracted
        inc("frames_extracted", len(frame_paths))


        # ----------------------------------------------------
        # 2. PARALLEL VISION AGENTS + OBSERVABILITY
        # ----------------------------------------------------
        # Run 3 vision agents in parallel for redundancy and robustness
        # Each agent analyzes all frames independently
        vision_engine = ParallelVisionEngine(num_agents=3)

        logger.info("[OBS] Starting parallel vision analysis")
        with timer("vision_parallel"):
            # All agents run concurrently on the same frames
            raw_vision = await vision_engine.analyze(frame_paths)

        # Track parallel vision calls
        inc("vision_parallel_calls", 1)

        # Record latency metric for parallel vision stage
        record_latency(
            "vision_parallel_latency",
            get_metrics()["timings"].get("vision_parallel", 0)
        )

        logger.info("[OBS] Finished parallel vision")


        # ----------------------------------------------------
        # 3. LOOP AGENT REFINEMENT + OBSERVABILITY
        # ----------------------------------------------------
        # Use loop refinement agent to iteratively improve location accuracy
        # Stops early if confidence threshold is met or confidence stops improving
        geo_agent = GeoAgent()
        refiner = RefinementLoop(threshold=0.70, max_iters=3)

        logger.info("[OBS] Starting refinement loop")
        with timer("geo_refinement"):
            # Refine location metadata (city, country, region) iteratively
            location_info, iters = refiner.refine(raw_vision, geo_agent)

        # Track number of refinement iterations used
        inc("geo_refinement_iterations", iters)

        # Record latency metric for refinement stage
        record_latency(
            "geo_refinement_latency",
            get_metrics()["timings"].get("geo_refinement", 0)
        )

        logger.info("[OBS] Refinement loop completed")


        # ----------------------------------------------------
        # 4. RETURN RESULT
        # ----------------------------------------------------
        location_info["sampled_frames"] = frame_paths  # type: ignore
        return location_info  # type: ignore


    except Exception as e:
        logger.exception("analyze_reel failed")
        return {
            "error": str(e),
            "video_path": video_path,
        }


@mcp.tool()
async def fetch_city_places(
    city: str,
    place_type: str = "tourist_attraction",
    max_results: int = 15,
) -> Dict[str, Any]:
    """
    MCP Tool: Fetch tourist attractions from Google Places API for a given city.
    
    Args:
        city: City name to search in
        place_type: Type of place to search for (default: tourist_attraction)
        max_results: Maximum number of results (default: 15)
        
    Returns:
        Dictionary with city, place_type, and list of place results
    """
    try:
        logger.info("fetch_city_places: city=%s type=%s", city, place_type)

        # Search Google Places API with timing
        with timer("places_api"):
            places = _google_places_search(city, place_type, max_results)

        # Record latency metric
        record_latency("places_latency", get_metrics().get("places_api", 0))

        return {
            "city": city,
            "place_type": place_type,
            "results": places,
        }

    except Exception as e:
        logger.exception("fetch_city_places failed")
        return {"error": str(e), "city": city}


@mcp.tool()
async def plan_itinerary_from_reel(
    video_path: str | None = None,
    days: int = 2,
) -> Dict[str, Any]:
    """
    MCP Tool: Full pipeline orchestrator - analyzes reel and generates itinerary.
    
    This is the main tool that orchestrates the entire pipeline:
    1. Analyzes the video to detect location and landmarks
    2. Fetches real places from Google Places API
    3. Generates a detailed itinerary using the itinerary agent
    
    Args:
        video_path: Path to the video file (defaults to data/input/reel.mp4)
        days: Number of days for the itinerary (default: 2)
        
    Returns:
        Dictionary with city, country, region, landmarks, places, and itinerary markdown
    """
    try:
        # Step 1: Analyze the reel to detect location and landmarks
        location_info = await analyze_reel(video_path=video_path)
        if "error" in location_info:
            return {"stage": "analyze_reel", "error": location_info["error"]}

        # Extract location details for Places API search
        city = location_info.get("city") or ""
        country = location_info.get("country") or ""
        display_city = f"{city}, {country}" if country else city

        # Step 2: Search Google Places API for attractions
        with timer("places_api"):
            places_resp = await fetch_city_places(
                city=display_city or city or " ",
                max_results=20,
            )

        record_latency("places_latency", get_metrics().get("places_api", 0))

        if "error" in places_resp:
            return {"stage": "fetch_city_places", "error": places_resp["error"]}

        places = places_resp.get("results", [])

        # Step 3: Generate itinerary using the itinerary agent
        itinerary_agent = ItineraryAgent()

        with timer("itinerary_generation"):
            # Build markdown itinerary with location info and places
            itinerary_md = itinerary_agent.build_itinerary(
                location_info=location_info,
                days=days,
                places=places,
            )

        # Record latency metric for itinerary generation
        record_latency("itinerary_latency", get_metrics().get("itinerary_generation", 0))

        # Log all observability metrics for debugging
        metrics_json = json.dumps(get_metrics(), indent=4)
        logger.info("OBSERVABILITY METRICS: \n%s", metrics_json)

        # Format observability dashboard for inclusion in output
        dashboard = format_observability_dashboard()

        # Return complete result with itinerary and observability metrics
        return {
            "city": location_info.get("city"),
            "country": location_info.get("country"),
            "region": location_info.get("region"),
            "landmarks": location_info.get("landmarks", []),
            "places": places,
            "itinerary_markdown": itinerary_md + "\n\n" + dashboard,
        }

    except Exception as e:
        logger.exception("plan_itinerary_from_reel failed")
        return {"stage": "plan_itinerary_from_reel", "error": str(e)}


# -----------------------------------------------------
# OBSERVABILITY MCP TOOL
# -----------------------------------------------------
@mcp.tool()
async def get_observability_metrics() -> Dict[str, Any]:
    """
    MCP Tool: Get observability metrics (timings, counters, latencies).
    
    Returns:
        Dictionary with all collected metrics including timings, counters, and latencies
    """
    return get_metrics()


def main() -> None:
    """
    Main entry point for the MCP server.
    Runs the server using stdio transport for communication with the ADK agent.
    """
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
