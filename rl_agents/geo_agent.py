# rl_agents/geo_agent.py
"""
Geo Agent for normalizing and refining location metadata

This agent takes raw vision results and normalizes city/country names,
adds region information, and refines landmark confidence scores.
"""

from typing import Dict, Any
import json
import os
from google import genai
from google.genai import types as genai_types


class GeoAgent:
    """
    Normalizes location metadata: city, country, region.
    
    This agent refines the raw output from vision agents by:
    - Normalizing city and country names to standard formats
    - Adding region information (e.g., "Europe", "Asia", "North America")
    - Validating and refining landmark confidence scores
    """

    def __init__(self, model: str = "gemini-2.0-flash") -> None:
        """
        Initialize the geo agent with Google GenAI client.
        
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

    def refine_location(self, raw_vision_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Refine and normalize location metadata from vision results.
        
        Args:
            raw_vision_result: Dictionary with city, country, and landmarks from vision agent
            
        Returns:
            Refined dictionary with normalized names, region, and validated landmarks
            
        Raises:
            RuntimeError: If model returns no response or invalid JSON
        """
        # Prompt instructing the model to normalize location data
        prompt = (
            "You will receive a JSON blob with tentative city, country, and landmarks.\n"
            "Normalize city and country names and add a region like 'Europe', 'Asia', "
            "'North America', etc.\n\n"
            "Return STRICT JSON only:\n"
            "{\n"
            '  \"city\": \"string\",\n'
            '  \"country\": \"string\",\n'
            '  \"region\": \"string\",\n'
            '  \"landmarks\": [\n'
            '    {\"name\": \"string\", \"confidence\": 0.0-1.0}\n'
            "  ]\n"
            "}\n"
        )

        # Create content with prompt and raw vision result
        parts = [
            genai_types.Part(text=prompt),
            genai_types.Part(text=json.dumps(raw_vision_result)),
        ]

        content = genai_types.Content(role="user", parts=parts)

        # Call Gemini API with JSON response format
        resp = self.client.models.generate_content(
            model=self.model,
            contents=content,
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json"  # Request structured JSON output
            ),
        )
        
        # Validate response structure
        if not resp or not resp.candidates or len(resp.candidates) == 0:
            raise RuntimeError("No response from geo model")
        candidate = resp.candidates[0]
        if not candidate.content or not candidate.content.parts or len(candidate.content.parts) == 0:
            raise RuntimeError("No content in response")
        
        # Extract and parse JSON response
        text = candidate.content.parts[0].text
        if not text:
            raise RuntimeError("No text in response")
        return json.loads(text)
