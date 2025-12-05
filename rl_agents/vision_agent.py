# rl_agents/vision_agent.py
"""
Vision Agent for analyzing travel reel frames

This agent uses Gemini's multimodal vision capabilities to analyze video frames
and detect the location (city, country) and landmarks shown in travel reels.
"""

import json
from typing import List, Dict, Any
import os
from google import genai
from google.genai import types as genai_types
import mimetypes  # For detecting image MIME types


class VisionAgent:
    """
    Uses Gemini multimodal vision to look at frames from a travel reel and
    propose (city, country, landmarks).
    
    The agent analyzes multiple frames simultaneously and returns structured JSON
    with location information and detected landmarks with confidence scores.
    """

    def __init__(self, model: str = "gemini-2.0-flash") -> None:
        """
        Initialize the vision agent with Google GenAI client.
        
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

    def _prompt(self) -> str:
        """
        Get the prompt template for vision analysis.
        
        Returns:
            Prompt string instructing the model to analyze frames and return JSON
        """
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
        """
        Analyze multiple video frames to detect location and landmarks.
        
        Args:
            frame_paths: List of file paths to image frames extracted from video
            
        Returns:
            Dictionary with city, country, and landmarks (with confidence scores)
            
        Raises:
            RuntimeError: If model returns no response or invalid JSON
        """
        # Start with the analysis prompt
        parts: list[genai_types.Part] = [
            genai_types.Part(text=self._prompt())
        ]

        # Add each frame image to the request
        for p in frame_paths:
            # Read image file as bytes
            with open(p, "rb") as f:
                img_bytes = f.read()

            # Detect MIME type (default to JPEG if unknown)
            mime = mimetypes.guess_type(p)[0] or "image/jpeg"

            # Add image as inline data part
            parts.append(
                genai_types.Part(
                    inline_data=genai_types.Blob(
                        mime_type=mime,
                        data=img_bytes,
                    )
                )
            )

        # Create content object with prompt and all frames
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
            raise RuntimeError("No response from vision model")
        candidate = resp.candidates[0]
        if not candidate.content or not candidate.content.parts or len(candidate.content.parts) == 0:
            raise RuntimeError("No content in response")
        
        # Extract text from response parts (handle different response formats)
        text = None
        for part in candidate.content.parts:
            if hasattr(part, "text") and part.text:
                text = part.text
                break

        if not text:
            raise RuntimeError("Model returned no JSON text in response")

        # Parse and return JSON response
        return json.loads(text)
