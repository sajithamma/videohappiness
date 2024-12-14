import os
import tempfile
import subprocess
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

try:
    import streamlit as st
except ModuleNotFoundError:
    raise ModuleNotFoundError("The 'streamlit' module is not installed. Please install it using 'pip install streamlit' and try again.")

# Set up Streamlit app layout
st.title("Happiness Graph Video Generator")
st.sidebar.header("Upload Video")
uploaded_video = st.sidebar.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_video:
    temp_dir = tempfile.mkdtemp()
    input_video_path = os.path.join(temp_dir, "uploaded_video.mp4")
    with open(input_video_path, "wb") as f:
        f.write(uploaded_video.read())
    st.sidebar.success("Video uploaded successfully!")

    st.sidebar.write("Processing options")
    generate_button = st.sidebar.button("Generate Happiness Graph Video")

    if generate_button:
        st.info("Analyzing video frames for happiness...")

        # Step 1: Extract frames and simulate happiness detection
        cap = cv2.VideoCapture(input_video_path)

        frame_scores = []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        timestamps = np.linspace(0, frame_count / fps, frame_count)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Simulate happiness score calculation (random or fixed values)
            happiness_score = np.random.uniform(50, 100)  # Generate random happiness score between 50 and 100
            frame_scores.append(happiness_score)

        cap.release()

        # Step 2: Generate happiness graph
        st.info("Generating happiness graph...")
        graph_path = os.path.join(temp_dir, "happiness_graph.png")
        plt.figure(figsize=(10, 2))
        plt.plot(timestamps[:len(frame_scores)], frame_scores, label="Happiness", color="green")
        plt.fill_between(timestamps[:len(frame_scores)], frame_scores, 0, alpha=0.2, color="green")
        plt.ylim(0, 100)
        plt.xlabel("Time (s)")
        plt.ylabel("Happiness (%)")
        plt.title("Happiness Graph")
        plt.grid()
        plt.savefig(graph_path, bbox_inches="tight")
        plt.close()

        st.image(graph_path, caption="Happiness Graph")

        # Step 3: Overlay the graph on the video
        st.info("Overlaying happiness graph on video...")
        output_video_path = os.path.join(temp_dir, "happiness_video.mp4")
        overlay_command = [
            "ffmpeg", "-y", "-i", input_video_path, "-i", graph_path,
            "-filter_complex", "[1:v]scale=1920:200[graph];[0:v][graph]overlay=W-w:H-h",
            "-c:a", "copy", output_video_path
        ]

        result = subprocess.run(overlay_command, capture_output=True, text=True)
        if result.returncode == 0:
            st.success("Video generated successfully!")
            with open(output_video_path, "rb") as f:
                st.download_button(
                    label="Download Happiness Video",
                    data=f,
                    file_name="happiness_video.mp4",
                    mime="video/mp4"
                )
        else:
            st.error(f"Error generating video: {result.stderr}")
