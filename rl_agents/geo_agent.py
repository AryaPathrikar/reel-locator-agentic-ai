# rl_agents/geo_agent.py
from typing import Dict, Any
import json
import os
from google import genai
from google.genai import types as genai_types


class GeoAgent:
    """
    Normalizes location metadata: city, country, region.
    """

    def __init__(self, model: str = "gemini-2.0-flash") -> None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY must be set in environment")

        self.client = genai.Client(api_key=api_key)
        self.model = model

    def refine_location(self, raw_vision_result: Dict[str, Any]) -> Dict[str, Any]:
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

        parts = [
            genai_types.Part(text=prompt),
            genai_types.Part(text=json.dumps(raw_vision_result)),
        ]

        content = genai_types.Content(role="user", parts=parts)

        resp = self.client.models.generate_content(
            model=self.model,
            contents=content,
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json"
            ),
        )
        if not resp or not resp.candidates or len(resp.candidates) == 0:
            raise RuntimeError("No response from geo model")
        candidate = resp.candidates[0]
        if not candidate.content or not candidate.content.parts or len(candidate.content.parts) == 0:
            raise RuntimeError("No content in response")
        text = candidate.content.parts[0].text
        if not text:
            raise RuntimeError("No text in response")
        return json.loads(text)
