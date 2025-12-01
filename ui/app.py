"""
Streamlit UI for Reel Locator Multi-Agent System
Can work with A2A server OR call agent directly
"""

import streamlit as st
import requests
import os
import sys
import asyncio
from pathlib import Path
import time

# Add project root to path to import agent modules
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Configuration
A2A_SERVER = "http://localhost:9000"
DATA_INPUT_DIR = Path(__file__).parent.parent / "data" / "input"
DATA_INPUT_DIR.mkdir(parents=True, exist_ok=True)

# Page configuration
st.set_page_config(
    page_title="Reel Locator – Travel Itinerary Generator",
    layout="wide",
    page_icon="🌍",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
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
    # Display video preview
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.video(uploaded_video)
    
    with col2:
        st.info(f"**File:** {uploaded_video.name}\n\n**Size:** {uploaded_video.size / (1024*1024):.2f} MB")
    
    # Generate Itinerary button
    if st.button("🚀 Generate Itinerary", type="primary", use_container_width=True):
        
        # No server check needed - we call the agent directly
        
        # Save video to data/input directory
        video_filename = uploaded_video.name
        video_path = DATA_INPUT_DIR / video_filename
        
        try:
            # Save uploaded video
            with open(video_path, "wb") as f:
                f.write(uploaded_video.getbuffer())
            
            st.success(f"✅ Video saved to: `{video_path}`")
            
            # Call agent directly (bypasses HTTP endpoint issues)
            # This works without needing the A2A HTTP server
            user_message = (
                f"I uploaded a travel reel to {video_path}. "
                "Please analyze it and create a 2-day itinerary using the plan_itinerary_from_reel tool."
            )
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.info("🔄 Initializing agent...")
            progress_bar.progress(10)
            
            try:
                # Import and call agent directly
                from adk_agent.agent import run_once
                import uuid
                
                status_text.info("🔄 Running multi-agent pipeline...")
                progress_bar.progress(20)
                
                # Generate unique session ID for this request
                session_id = f"ui_{uuid.uuid4().hex[:8]}"
                
                status_text.info("🔄 Processing video with parallel agents...")
                progress_bar.progress(40)
                
                # Run async function in sync context
                # This calls the agent directly, bypassing HTTP
                full_text = asyncio.run(run_once(user_message, session_id=session_id))
                
                progress_bar.progress(100)
                status_text.empty()
                progress_bar.empty()
                
                if not full_text:
                    st.error("❌ Empty response from agent. Check logs.")
                else:
                    # Display results
                    st.success("✅ Itinerary generated successfully!")
                    st.subheader("🗺️ Generated Itinerary")
                    st.markdown(full_text)
                    
                    # Show session info
                    with st.expander("ℹ️ Session Information"):
                        st.info(f"**Session ID:** {session_id}\n\n**Video Path:** {video_path}")
                
            except ImportError as e:
                st.error(
                    f"❌ **Import Error:** {str(e)}\n\n"
                    "Make sure you're running from the project root directory."
                )
                st.exception(e)
            except Exception as e:
                st.error(f"❌ **Error:** {str(e)}")
                st.exception(e)
                with st.expander("🔍 Debug Information"):
                    st.code(f"Video path: {video_path}\nMessage: {user_message}")
        
        except Exception as e:
            st.error(f"❌ Error saving video: {str(e)}")
            st.exception(e)

else:
    # Show example/instructions when no video is uploaded
    st.info("👆 **Upload a travel reel video to get started!**")
    
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
