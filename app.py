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
    generate_button = st.sidebar.button("Generate Dynamic Happiness Graph Video")

    if generate_button:
        st.info("Analyzing video frames for happiness...")

        # Step 1: Extract frames and simulate happiness detection
        cap = cv2.VideoCapture(input_video_path)

        frame_scores = []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        timestamps = np.linspace(0, frame_count / fps, frame_count)

        # Streamlit progress bar
        progress_bar = st.progress(0)

        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break

            # Simulate happiness score calculation (random or fixed values)
            happiness_score = np.random.uniform(50, 100)  # Generate random happiness score between 50 and 100
            frame_scores.append(happiness_score)

            # Update progress
            progress_bar.progress((i + 1) / frame_count)

        cap.release()

        # Step 2: Generate dynamic graph video
        st.info("Generating dynamic happiness graph with transparent background...")

        # Create graph for each frame dynamically
        graph_frame_dir = os.path.join(temp_dir, "graph_frames")
        os.makedirs(graph_frame_dir, exist_ok=True)

        # Ensure frame dimensions are divisible by 2
        for i, score in enumerate(frame_scores):
            plt.figure(figsize=(10, 2))
            plt.plot(timestamps[:i + 1], frame_scores[:i + 1], color="orange")
            plt.fill_between(timestamps[:i + 1], frame_scores[:i + 1], 0, alpha=0.2, color="orange")
            plt.ylim(0, 100)
            plt.axis("off")
            plt.gcf().set_size_inches(8, 2)  # Adjust aspect ratio
            graph_frame_path = os.path.join(graph_frame_dir, f"frame_{i:04d}.png")
            plt.savefig(graph_frame_path, transparent=True, bbox_inches="tight", pad_inches=0)
            plt.close()

        # Generate video from graph frames
        graph_video_path = os.path.join(temp_dir, "happiness_graph_dynamic.mp4")
        ffmpeg_command = [
            "ffmpeg", "-y", "-framerate", str(fps), "-i",
            os.path.join(graph_frame_dir, "frame_%04d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", graph_video_path
        ]
        result = subprocess.run(ffmpeg_command, capture_output=True, text=True)

        if result.returncode != 0:
            st.error(f"Error generating graph video: {result.stderr}")

        else:
            # Step 3: Overlay the graph video on the original video
            st.info("Overlaying dynamic graph on the original video...")
            output_video_path = os.path.join(temp_dir, "final_video.mp4")
            overlay_command = [
                "ffmpeg", "-y", "-i", input_video_path, "-i", graph_video_path,
                "-filter_complex", "[1:v]scale=1920:200[graph];[0:v][graph]overlay=W-w:H-h",
                "-c:a", "copy", output_video_path
            ]
            result_overlay = subprocess.run(overlay_command, capture_output=True, text=True)

            if result_overlay.returncode == 0:
                st.success("Video generated successfully!")
                with open(output_video_path, "rb") as f:
                    st.download_button(
                        label="Download Happiness Video",
                        data=f,
                        file_name="happiness_video.mp4",
                        mime="video/mp4"
                    )
            else:
                st.error(f"Error overlaying graph on video: {result_overlay.stderr}")
