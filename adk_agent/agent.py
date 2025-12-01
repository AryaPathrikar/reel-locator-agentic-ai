# adk_agent/agent.py

import os
import asyncio
from typing import Optional

from dotenv import load_dotenv

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp.client.stdio import StdioServerParameters

from google.genai import types as genai_types

# Custom long-term memory implementation
from adk_agent.memory_bank import MemoryBank

# A2A support (now REQUIRED, not optional)
from google.adk.a2a.utils.agent_to_a2a import to_a2a
import uvicorn


load_dotenv()

DEFAULT_MODEL = "gemini-2.0-flash"


# -----------------------------------------------------------
# Helpers
# -----------------------------------------------------------
def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _mcp_server_script() -> str:
    return os.path.join(_project_root(), "mcp_server", "mcp_server.py")


# -----------------------------------------------------------
# Build Root Agent (with injected memory/context)
# -----------------------------------------------------------
def build_root_agent(session_context: str = "") -> LlmAgent:
    """
    Build the top-level ADK agent that orchestrates the entire pipeline:
    - Calls MCP tools for reel analysis + places + itinerary
    - Receives context-engineered user memory
    - Also runs in A2A mode
    """

    # Inject compacted long-term memory
    memory_block = (
        "### USER MEMORY CONTEXT ###\n"
        f"{session_context}\n"
        "### END MEMORY ###\n\n"
        if session_context else ""
    )

    instruction = (
        memory_block +
        "You are a travel assistant using tools from the 'reel_locator' MCP server.\n"
        "Pipeline:\n"
        "1. Call `plan_itinerary_from_reel` to analyze travel reels.\n"
        "2. If no video_path is provided, assume data/input/reel.mp4.\n"
        "3. Summarize city, country, landmarks.\n"
        "4. Return itinerary markdown cleanly formatted.\n"
        "If tools error, guide the user to fix the input.\n"
        "You MUST only call MCP tools that exist.\n"
        "\n"
        "CRITICAL OUTPUT FORMAT - YOU MUST FOLLOW THIS EXACTLY:\n"
        "After calling plan_itinerary_from_reel, you MUST start your response with this exact format:\n"
        "\n"
        "Okay, I have analyzed the reel and created a 2-day itinerary for [CITY]. Here's the summary:\n"
        "\n"
        "• City: [CITY]\n"
        "• Country: [COUNTRY]\n"
        "• Region: [REGION]\n"
        "• Landmarks: [LANDMARK1], [LANDMARK2], [LANDMARK3], ...\n"
        "\n"
        "Here is the detailed itinerary:\n"
        "\n"
        "[Then include the full itinerary_markdown from the tool response]\n"
        "\n"
        "MANDATORY FORMATTING RULES:\n"
        "1. You MUST put a newline character after '• City: [value]' - do NOT continue on same line\n"
        "2. You MUST put a newline character after '• Country: [value]' - do NOT continue on same line\n"
        "3. You MUST put a newline character after '• Region: [value]' - do NOT continue on same line\n"
        "4. You MUST put a newline character after '• Landmarks: [list]' - do NOT continue on same line\n"
        "5. Each bullet point MUST end with a newline - use \\n or actual line break\n"
        "6. NEVER put '• City:' and '• Country:' on the same line - they MUST be separate lines\n"
        "7. The format MUST be:\n"
        "   Line 1: • City: [value]\n"
        "   Line 2: • Country: [value]\n"
        "   Line 3: • Region: [value]\n"
        "   Line 4: • Landmarks: [comma-separated list]\n"
        "   Line 5: (blank line)\n"
        "   Line 6: Here is the detailed itinerary:\n"
        "\n"
        "Extract the city, country, region, and landmarks from the tool response. "
        "List all detected landmarks separated by commas on the same line after '• Landmarks:'. "
        "Then include the complete itinerary_markdown exactly as returned by the tool with all formatting preserved.\n"
        "\n"
        "IMPORTANT OUTPUT RULE:\n"
        "Render the full content exactly as-is, including dashboards, metrics, logs, "
        "observability output, and any additional sections. Preserve all line breaks and formatting.\n"
    )

    reel_locator_tools = McpToolset(
        connection_params=StdioConnectionParams(  # type: ignore
            server_params=StdioServerParameters(  # type: ignore
                command="python",
                args=[_mcp_server_script()],
            ),
            timeout=150,
        ),
        tool_filter=[
            "analyze_reel",
            "fetch_city_places",
            "plan_itinerary_from_reel",
        ],
    )

    return LlmAgent(
        model=DEFAULT_MODEL,
        name="reel_locator_root",
        instruction=instruction,
        description="Root agent: reel analysis + itinerary generation via MCP.",
        tools=[reel_locator_tools],
    )


# -----------------------------------------------------------
# Run Once (Sessions + MemoryBank + Context Engineering)
# -----------------------------------------------------------
async def run_once(prompt: str, session_id: Optional[str] = None) -> str:

    # persistent long-term memory
    memory_bank = MemoryBank()

    # connect MemoryBank to ADK session store
    session_service = InMemorySessionService()  # type: ignore

    app_name = "reel_locator_app"

    # create/reuse session
    session = await session_service.create_session(
        app_name=app_name,
        user_id="demo_user",
        session_id=session_id or "session_001"
    )

    # store user's request
    memory_bank.store(f"USER: {prompt}")

    # context engineering → compress memory
    session_context = memory_bank.compact()

    # build agent with personalized context
    agent = build_root_agent(session_context=session_context)

    runner = Runner(
        agent=agent,
        app_name=app_name,
        session_service=session_service
    )

    content = genai_types.Content(
        role="user",
        parts=[genai_types.Part(text=prompt)]
    )

    # run agent
    events = runner.run(
        user_id="demo_user",
        session_id=session.id,
        new_message=content  # type: ignore
    )

    # capture final output
    final_text = ""
    for event in events:
        if hasattr(event, "is_final_response") and event.is_final_response():
            if event.content:
                for part in event.content.parts:  # type: ignore
                    t = getattr(part, "text", None)
                    if t:
                        final_text += t + "\n"
    final_text = final_text.strip()
    if not final_text:
        raise RuntimeError("No response text from agent")

    # update memory with assistant response
    memory_bank.store(f"ASSISTANT: {final_text}")

    return final_text


# -----------------------------------------------------------
# A2A SERVER (always available)
# -----------------------------------------------------------
def start_a2a_server():
    """
    Starts the root agent as a full A2A microservice.
    Other agents can call it using RemoteA2aAgent.
    """
    print("⟳ Starting Reel Locator A2A Server at http://localhost:9000")
    agent = build_root_agent("")  # Stateless for A2A mode
    app = to_a2a(agent, port=9000)
    uvicorn.run(app, host="0.0.0.0", port=9000)


# -----------------------------------------------------------
# MAIN ENTRY
# -----------------------------------------------------------
if __name__ == "__main__":
    import sys
    
    # Check if user wants CLI mode (with test) or just server mode
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        # CLI mode: run test and exit
        test_prompt = (
            "I uploaded a travel reel to data/input/reel.mp4. "
            "Detect location and create a 2-day itinerary."
        )
        out = asyncio.run(run_once(test_prompt))
        print("\n\n=== FINAL OUTPUT ===\n")
        print(out)
    else:
        # Default: A2A server mode (stays running)
        print("🚀 Starting Reel Locator A2A Server...")
        print("📡 Server will run on http://localhost:9000")
        print("🛑 Press Ctrl+C to stop the server\n")
        start_a2a_server()  # This blocks and keeps server running
