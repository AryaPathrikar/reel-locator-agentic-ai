# rl_agents/itinerary_agent.py
"""
Itinerary Agent for generating travel itineraries

This agent creates detailed, day-by-day travel itineraries based on:
- Detected location and landmarks from vision analysis
- Real places from Google Places API
- User preferences and travel constraints
"""

from typing import Dict, Any, List
import json
import os
from google import genai
from google.genai import types as genai_types


class ItineraryAgent:
    """
    Builds a 1–2 day itinerary given location info and (optionally) Google Places
    results, using Gemini text-only completion.
    
    The agent creates a structured markdown itinerary with:
    - Day-by-day breakdown
    - Morning, afternoon, and evening activities
    - Prioritized landmarks and attractions
    - Local food recommendations
    - Walking routes and travel flow
    """

    def __init__(self, model: str = "gemini-2.0-flash") -> None:
        """
        Initialize the itinerary agent with Google GenAI client.
        
        Args:
            model: Gemini model to use (default: gemini-2.0-flash)
            
        Raises:
            RuntimeError: If GOOGLE_API_KEY is not set in environment
        """
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY must be set in environment")

        self.client = genai.Client(api_key=api_key)
        self.model = model

    def build_itinerary(
        self,
        location_info: Dict[str, Any],
        days: int = 2,
        places: List[Dict[str, Any]] | None = None,
    ) -> str:
        """
        Build a detailed travel itinerary in markdown format.
        
        Args:
            location_info: Dictionary with city, country, region, and landmarks
            days: Number of days for the itinerary (default: 2)
            places: Optional list of places from Google Places API
            
        Returns:
            Markdown-formatted itinerary string
            
        Raises:
            RuntimeError: If model returns no response
        """
        # Extract location details
        city = location_info.get("city", "the city")
        country = location_info.get("country", "")
        landmarks = location_info.get("landmarks", [])

        # Format landmarks list with confidence scores
        landmark_lines = [
            f"- {lm.get('name', '')} (conf {lm.get('confidence', 0):.2f})"
            for lm in landmarks
        ]
        landmarks_text = "\n".join(landmark_lines) or "- (none)"

        # Format Google Places results (limit to top 12)
        places_lines: List[str] = []
        if places:
            for p in places[:12]:
                name = p.get("name", "")
                rating = p.get("rating", "N/A")
                addr = p.get("address", "")
                places_lines.append(f"- {name} (rating {rating}) — {addr}")
        places_text = "\n".join(places_lines) or "- (no Google Places results found)"

        # Build comprehensive prompt with location, landmarks, and places
        prompt = (
            f"You are a travel planner.\n\n"
            f"User showed a reel inferred to be from: {city}, {country}.\n\n"
            f"Detected landmarks:\n{landmarks_text}\n\n"
            f"Real places from Google Places API:\n{places_text}\n\n"
            f"Create a realistic {days}-day itinerary with morning, afternoon, "
            f"and evening blocks for each day.\n"
            "- Prioritize detected landmarks and Google Places results.\n"
            "- Include walking order where possible and approximate travel flow.\n"
            "- Add 2–3 local food recommendations per day.\n"
            "- Return Markdown with headings 'Day 1', 'Day 2', etc.\n"
        )

        # Create content object with the prompt
        content = genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=prompt)]
        )

        # Call Gemini API to generate itinerary
        resp = self.client.models.generate_content(
            model=self.model,
            contents=content,
        )
        
        # Validate response structure
        if not resp or not resp.candidates or len(resp.candidates) == 0:
            raise RuntimeError("No response from itinerary model")
        candidate = resp.candidates[0]
        if not candidate.content or not candidate.content.parts or len(candidate.content.parts) == 0:
            raise RuntimeError("No content in response")
        
        # Extract text response
        text = candidate.content.parts[0].text
        if not text:
            raise RuntimeError("No text in response")
        return text
