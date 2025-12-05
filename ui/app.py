"""
Streamlit UI for Reel Locator Multi-Agent System

This UI provides a web interface for uploading travel reels and generating itineraries.
It can work with an A2A server OR call the agent directly (bypassing HTTP).

Features:
- Video upload interface
- Real-time processing status
- Formatted itinerary display with location and landmarks summary
- Session management
"""

import streamlit as st
import requests  # For potential A2A server communication (currently unused)
import os
import sys
import asyncio  # For running async agent functions
from pathlib import Path
import time

# Add project root to Python path to import agent modules
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Configuration
A2A_SERVER = "http://localhost:9000"  # A2A server URL (if using HTTP mode)
DATA_INPUT_DIR = Path(__file__).parent.parent / "data" / "input"  # Directory for uploaded videos
DATA_INPUT_DIR.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist

# Configure Streamlit page settings
st.set_page_config(
    page_title="Reel Locator – Travel Itinerary Generator",
    layout="wide",  # Use wide layout for better space utilization
    page_icon="🌍",
    initial_sidebar_state="expanded",  # Show sidebar by default
)

# Custom CSS for better styling and user experience
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    
    st.markdown("### 📚 How it works")
    st.markdown("""
    1. Upload a travel reel video
    2. System extracts key frames from the video
    3. AI agents detect the location (city/country) and landmarks shown
    4. System refines location accuracy through iterative analysis
    5. Google Places API fetches nearby attractions
    6. Itinerary is generated focusing on the detected landmarks
    """)
    
    st.markdown("---")

# Main content
st.markdown('<div class="main-header">🌍 Reel Locator — Travel Itinerary Generator</div>', unsafe_allow_html=True)
st.markdown("Upload a travel reel and our AI agents will automatically detect the location and landmarks shown in the video, then create a personalized itinerary focused on those specific landmarks.")

# File uploader
uploaded_video = st.file_uploader(
    "Upload Reel Video",
    type=["mp4", "mov", "avi", "mkv"],
    help="Upload a travel video (MP4, MOV, AVI, or MKV format)"
)

if uploaded_video:
    # Display video preview and file information in a two-column layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Show video preview in the main column
        st.video(uploaded_video)

    with col2:
        # Display file metadata in the sidebar column
        st.info(f"**File:** {uploaded_video.name}\n\n**Size:** {uploaded_video.size / (1024*1024):.2f} MB")
    
    # Generate Itinerary button - triggers the full pipeline
    if st.button("🚀 Generate Itinerary", type="primary", use_container_width=True):
        
        # No server check needed - we call the agent directly (bypasses HTTP)
        
        # Save video to data/input directory for processing
        video_filename = uploaded_video.name
        video_path = DATA_INPUT_DIR / video_filename
        
        try:
            # Save uploaded video file to disk
            with open(video_path, "wb") as f:
                f.write(uploaded_video.getbuffer())
            
            st.success(f"✅ Video saved to: `{video_path}`")

            # Construct user message for the agent
            # The agent will use this to call the plan_itinerary_from_reel tool
            user_message = (
                f"I uploaded a travel reel to {video_path}. "
                "Please analyze it and create a 2-day itinerary using the plan_itinerary_from_reel tool."
            )
            
            # Progress tracking UI elements
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.info("🔄 Initializing agent...")
            progress_bar.progress(10)
            
            try:
                # Import the agent's run_once function for direct execution
                from adk_agent.agent import run_once
                import uuid
                
                status_text.info("🔄 Running multi-agent pipeline...")
                progress_bar.progress(20)
                
                # Generate unique session ID for this request
                # This enables session management and memory persistence
                session_id = f"ui_{uuid.uuid4().hex[:8]}"
                
                status_text.info("🔄 Processing video with parallel agents...")
                progress_bar.progress(40)
                
                # Run async agent function in synchronous context
                # This calls the agent directly, bypassing HTTP/A2A server
                # The agent will:
                # 1. Extract frames from video
                # 2. Run parallel vision agents
                # 3. Refine location with loop agent
                # 4. Fetch places from Google Places API
                # 5. Generate itinerary
                full_text = asyncio.run(run_once(user_message, session_id=session_id))
                
                # Update progress to completion
                progress_bar.progress(100)
                status_text.empty()
                progress_bar.empty()
                
                if not full_text:
                    st.error("❌ Empty response from agent. Check logs.")
                else:
                    # Display the generated itinerary
                    st.success("✅ Itinerary generated successfully!")
                    st.subheader("🗺️ Generated Itinerary")
                    st.markdown(full_text)  # Render markdown with formatting
                    
                    # Show session information in an expandable section
                    with st.expander("ℹ️ Session Information"):
                        st.info(f"**Session ID:** {session_id}\n\n**Video Path:** {video_path}")
                
            except ImportError as e:
                # Handle import errors (e.g., if running from wrong directory)
                st.error(
                    f"❌ **Import Error:** {str(e)}\n\n"
                    "Make sure you're running from the project root directory."
                )
                st.exception(e)
            except Exception as e:
                # Handle any other errors during agent execution
                st.error(f"❌ **Error:** {str(e)}")
                st.exception(e)
                with st.expander("🔍 Debug Information"):
                    st.code(f"Video path: {video_path}\nMessage: {user_message}")
        
        except Exception as e:
            # Handle errors during video file saving
            st.error(f"❌ Error saving video: {str(e)}")
            st.exception(e)

else:
    # Show example/instructions when no video is uploaded
    st.info("👆 **Upload a travel reel video to get started!**")
    
    # Display three-step process in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🎬 Step 1")
        st.markdown("Upload your travel reel video")
    
    with col2:
        st.markdown("### 🤖 Step 2")
        st.markdown("AI agents analyze the video, detecting the location and landmarks shown")
    
    with col3:
        st.markdown("### 📋 Step 3")
        st.markdown("Get your personalized itinerary")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>Powered by <strong>Google ADK</strong> • <strong>MCP</strong> • <strong>A2A</strong> • <strong>Multi-Agent Architecture</strong></p>
        <p style='font-size: 0.9em;'>Reel Locator transforms travel inspiration into actionable plans</p>
    </div>
    """,
    unsafe_allow_html=True
)
