# adk_agent/agent.py
"""
Root Agent for Reel Locator Multi-Agent System

This module defines the top-level orchestrator agent that:
- Coordinates the entire travel reel analysis pipeline
- Connects to MCP tools for video analysis, places search, and itinerary generation
- Manages session state and user memory
- Exposes the agent as an A2A (Agent-to-Agent) service on port 9000
- Can run in CLI test mode or persistent server mode

The agent uses Google ADK (Agent Development Kit) to create a LlmAgent that
orchestrates calls to specialized MCP tools defined in mcp_server/mcp_server.py.
"""

import os
import asyncio
from typing import Optional

# Load environment variables from .env file
from dotenv import load_dotenv

# Google ADK imports for building and running agents
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

# MCP (Model Context Protocol) tool integration
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp.client.stdio import StdioServerParameters

# Google GenAI types for content formatting
from google.genai import types as genai_types

# Custom long-term memory implementation for context engineering
from adk_agent.memory_bank import MemoryBank

# A2A (Agent-to-Agent) protocol support - enables agent communication
from google.adk.a2a.utils.agent_to_a2a import to_a2a
import uvicorn  # ASGI server for running the A2A HTTP service


load_dotenv()

# Default LLM model for the root agent
DEFAULT_MODEL = "gemini-2.0-flash"


# -----------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------
def _project_root() -> str:
    """
    Get the absolute path to the project root directory.
    Used to locate resources relative to the project structure.
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _mcp_server_script() -> str:
    """
    Get the path to the MCP server script.
    This script is launched as a subprocess to provide MCP tools to the agent.
    """
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

    # Context Engineering: Inject compacted long-term memory into agent prompt
    # This allows the agent to remember past interactions and user preferences
    memory_block = (
        "### USER MEMORY CONTEXT ###\n"
        f"{session_context}\n"
        "### END MEMORY ###\n\n"
        if session_context else ""
    )

    # Build the agent's instruction prompt with:
    # 1. Memory context (if available)
    # 2. Role definition and pipeline steps
    # 3. Strict output formatting requirements for consistent responses
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

    # Configure MCP (Model Context Protocol) toolset
    # The MCP server runs as a subprocess and provides tools via stdio
    reel_locator_tools = McpToolset(
        connection_params=StdioConnectionParams(  # type: ignore
            server_params=StdioServerParameters(  # type: ignore
                command="python",  # Command to launch MCP server
                args=[_mcp_server_script()],  # Path to MCP server script
            ),
            timeout=150,  # Timeout in seconds for tool calls
        ),
        tool_filter=[
            "analyze_reel",  # Analyze video frames and detect location
            "fetch_city_places",  # Search Google Places API
            "plan_itinerary_from_reel",  # Full pipeline orchestrator
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
    """
    Execute a single agent interaction with session management and memory.
    
    This function:
    1. Creates/retrieves a session for state management
    2. Stores user input in memory bank
    3. Compacts memory for context injection
    4. Builds agent with personalized context
    5. Runs the agent and captures response
    6. Updates memory with assistant response
    
    Args:
        prompt: User's input message/prompt
        session_id: Optional session ID for session continuity
        
    Returns:
        Agent's text response
        
    Raises:
        RuntimeError: If agent returns no response text
    """
    # Initialize persistent long-term memory for context engineering
    memory_bank = MemoryBank()

    # Connect to ADK's in-memory session service for state management
    session_service = InMemorySessionService()  # type: ignore

    app_name = "reel_locator_app"

    # Create or retrieve existing session for this user
    # Sessions enable conversation continuity and state persistence
    session = await session_service.create_session(
        app_name=app_name,
        user_id="demo_user",
        session_id=session_id or "session_001"
    )

    # Store user's request in memory bank for future context
    memory_bank.store(f"USER: {prompt}")

    # Context engineering: Compress memory into compact summary
    # This summary will be injected into the agent's instruction prompt
    session_context = memory_bank.compact()

    # Build agent with personalized context from memory
    agent = build_root_agent(session_context=session_context)

    # Create runner to execute the agent with session management
    runner = Runner(
        agent=agent,
        app_name=app_name,
        session_service=session_service
    )

    # Format user message as Content object for ADK
    content = genai_types.Content(
        role="user",
        parts=[genai_types.Part(text=prompt)]
    )

    # Execute agent and get event stream
    # Events contain tool calls, intermediate responses, and final response
    events = runner.run(
        user_id="demo_user",
        session_id=session.id,
        new_message=content  # type: ignore
    )

    # Extract final text response from event stream
    # Look for events marked as final response and collect text parts
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

    # Update memory with assistant response for future context
    memory_bank.store(f"ASSISTANT: {final_text}")

    return final_text


# -----------------------------------------------------------
# A2A SERVER (always available)
# -----------------------------------------------------------
def start_a2a_server():
    """
    Start the root agent as a full A2A (Agent-to-Agent) microservice.
    
    This exposes the agent as an HTTP service on port 9000, allowing:
    - Other agents to call it via RemoteA2aAgent
    - External systems to interact via HTTP requests
    - Integration with ADK Web mode and other clients
    
    The server runs in stateless mode (no memory context) for A2A compatibility.
    """
    print("⟳ Starting Reel Locator A2A Server at http://localhost:9000")
    # Build agent without memory context for stateless A2A operation
    agent = build_root_agent("")  # Stateless for A2A mode
    # Convert agent to A2A-compatible FastAPI app
    app = to_a2a(agent, port=9000)
    # Run uvicorn server (blocks until stopped)
    uvicorn.run(app, host="0.0.0.0", port=9000)


# -----------------------------------------------------------
# MAIN ENTRY
# -----------------------------------------------------------
if __name__ == "__main__":
    """
    Main entry point for the agent.
    
    Two modes:
    1. CLI mode (--cli): Run a one-time test and exit
    2. Server mode (default): Start A2A server and keep running
    """
    import sys
    
    # Check if user wants CLI mode (with test) or just server mode
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        # CLI mode: Run a test interaction and print results, then exit
        # Useful for testing and debugging without starting a server
        test_prompt = (
            "I uploaded a travel reel to data/input/reel.mp4. "
            "Detect location and create a 2-day itinerary."
        )
        out = asyncio.run(run_once(test_prompt))
        print("\n\n=== FINAL OUTPUT ===\n")
        print(out)
    else:
        # A2A server mode 
        print("🚀 Starting Reel Locator A2A Server...")
        print("📡 Server will run on http://localhost:9000")
        print("🛑 Press Ctrl+C to stop the server\n")
        start_a2a_server()  # This blocks and keeps server running
