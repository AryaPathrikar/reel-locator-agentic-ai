# mcp_server/mcp_server.py

import os
import sys
import logging
from typing import Any, Dict, List
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from tools.extract_frames import extract_key_frames
from rl_agents.vision_agent import VisionAgent
from rl_agents.geo_agent import GeoAgent
from rl_agents.itinerary_agent import ItineraryAgent
from rl_agents.parallel_vision import ParallelVisionEngine
from rl_agents.refinement_agent import RefinementLoop

# Observability
from observability.obs import timer, inc, record_latency, get_metrics
from observability.dashboard import format_observability_dashboard
import requests

# Make sure logs go to STDERR, keeping STDOUT clean for MCP JSON-RPC
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("reel_locator_mcp")

load_dotenv()

mcp = FastMCP("reel_locator")


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _default_video_path() -> str:
    return os.path.join(_project_root(), "data", "input", "reel.mp4")


def _frames_dir() -> str:
    return os.path.join(_project_root(), "data", "frames")


def _google_places_search(
    city: str,
    place_type: str = "tourist_attraction",
    max_results: int = 15,
) -> List[Dict[str, Any]]:
    api_key = os.getenv("GOOGLE_PLACES_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_PLACES_API_KEY not set in environment")

    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {
        "query": f"{place_type} in {city}",
        "key": api_key,
    }

    with timer("places_api"):
        resp = requests.get(url, params=params, timeout=20)

    resp.raise_for_status()
    data = resp.json()

    record_latency("places_latency", get_metrics().get("places_api", 0))

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

    try:
        if video_path is None:
            video_path = _default_video_path()

        logger.info("analyze_reel: video_path=%s", video_path)

        frames_dir = _frames_dir()

        # ----------------------------------------------------
        # 1. FRAME EXTRACTION + OBSERVABILITY
        # ----------------------------------------------------
        with timer("frame_extraction"):
            frame_paths = extract_key_frames(
                video_path=video_path,
                output_dir=frames_dir,
                max_frames=max_frames,
            )

        inc("frames_extracted", len(frame_paths))


        # ----------------------------------------------------
        # 2. PARALLEL VISION AGENTS + OBSERVABILITY
        # ----------------------------------------------------
        vision_engine = ParallelVisionEngine(num_agents=3)

        logger.info("[OBS] Starting parallel vision analysis")
        with timer("vision_parallel"):
            raw_vision = await vision_engine.analyze(frame_paths)

        inc("vision_parallel_calls", 1)

        # correct latency reference: use timings["vision_parallel"]
        record_latency(
            "vision_parallel_latency",
            get_metrics()["timings"].get("vision_parallel", 0)
        )

        logger.info("[OBS] Finished parallel vision")


        # ----------------------------------------------------
        # 3. LOOP AGENT REFINEMENT + OBSERVABILITY
        # ----------------------------------------------------
        geo_agent = GeoAgent()
        refiner = RefinementLoop(threshold=0.70, max_iters=3)

        logger.info("[OBS] Starting refinement loop")
        with timer("geo_refinement"):
            location_info, iters = refiner.refine(raw_vision, geo_agent)

        inc("geo_refinement_iterations", iters)

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

    try:
        logger.info("fetch_city_places: city=%s type=%s", city, place_type)

        with timer("places_api"):
            places = _google_places_search(city, place_type, max_results)

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

    try:
        # 1) ANALYZE REEL
        location_info = await analyze_reel(video_path=video_path)
        if "error" in location_info:
            return {"stage": "analyze_reel", "error": location_info["error"]}

        city = location_info.get("city") or ""
        country = location_info.get("country") or ""
        display_city = f"{city}, {country}" if country else city

        # 2) PLACES SEARCH
        with timer("places_api"):
            places_resp = await fetch_city_places(
                city=display_city or city or " ",
                max_results=20,
            )

        record_latency("places_latency", get_metrics().get("places_api", 0))

        if "error" in places_resp:
            return {"stage": "fetch_city_places", "error": places_resp["error"]}

        places = places_resp.get("results", [])

        # 3) ITINERARY GENERATION
        itinerary_agent = ItineraryAgent()

        with timer("itinerary_generation"):
            itinerary_md = itinerary_agent.build_itinerary(
                location_info=location_info,
                days=days,
                places=places,
            )

        record_latency("itinerary_latency", get_metrics().get("itinerary_generation", 0))

        # Log all observability metrics
        metrics_json = json.dumps(get_metrics(), indent=4)
        logger.info("OBSERVABILITY METRICS: \n%s", metrics_json)

        dashboard = format_observability_dashboard()
        
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
# NEW OBSERVABILITY MCP TOOL
# -----------------------------------------------------
@mcp.tool()
async def get_observability_metrics() -> Dict[str, Any]:
    return get_metrics()


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
