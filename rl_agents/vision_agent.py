# rl_agents/vision_agent.py
import json
from typing import List, Dict, Any
import os
from google import genai
from google.genai import types as genai_types
import mimetypes


class VisionAgent:
    """
    Uses Gemini multimodal vision to look at frames from a travel reel and
    propose (city, country, landmarks).
    """

    def __init__(self, model: str = "gemini-2.0-flash") -> None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY must be set in environment")

        self.client = genai.Client(api_key=api_key)
        self.model = model

    def _prompt(self) -> str:
        return (
            "You are a travel reel analyzer.\n"
            "You will see multiple frames from a short travel video.\n"
            "Infer the most likely CITY and COUNTRY, and up to 8 famous landmarks.\n\n"
            "Return STRICT JSON only. NO prose. NO markdown. NO explanation. NO surrounding ```.\n"
            "The JSON schema must be:\n"
            "{\n"
            '  \"city\": \"string\",\n'
            '  \"country\": \"string\",\n'
            '  \"landmarks\": [\n'
            '    {\"name\": \"string\", \"confidence\": 0.0-1.0, \"evidence\": \"short reason\"}\n'
            "  ]\n"
            "}\n"
        )

    def analyze_frames(self, frame_paths: List[str]) -> Dict[str, Any]:
        parts: list[genai_types.Part] = [
            genai_types.Part(text=self._prompt())
        ]

        for p in frame_paths:
            with open(p, "rb") as f:
                img_bytes = f.read()

            mime = mimetypes.guess_type(p)[0] or "image/jpeg"

            parts.append(
                genai_types.Part(
                    inline_data=genai_types.Blob(
                        mime_type=mime,
                        data=img_bytes,
                    )
                )
            )

        content = genai_types.Content(role="user", parts=parts)

        resp = self.client.models.generate_content(
            model=self.model,
            contents=content,
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json"
            ),
        )
        if not resp or not resp.candidates or len(resp.candidates) == 0:
            raise RuntimeError("No response from vision model")
        candidate = resp.candidates[0]
        if not candidate.content or not candidate.content.parts or len(candidate.content.parts) == 0:
            raise RuntimeError("No content in response")
        # Robustly get text from any part
        text = None
        for part in candidate.content.parts:
            if hasattr(part, "text") and part.text:
                text = part.text
                break

        if not text:
            raise RuntimeError("Model returned no JSON text in response")

        return json.loads(text)
