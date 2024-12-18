import os
import tempfile
import subprocess
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from deepface import DeepFace
import shutil

SAVE_SNAP = False

try:
    import streamlit as st
except ModuleNotFoundError:
    raise ModuleNotFoundError("The 'streamlit' module is not installed. Please install it using 'pip install streamlit' and try again.")

def make_dimensions_even(image_path):
    """
    Ensures that the dimensions of an image are even.
    If not, adjusts the dimensions to the nearest even value.
    """
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    height, width = img.shape[:2]
    new_width = width if width % 2 == 0 else width + 1
    new_height = height if height % 2 == 0 else height + 1

    if new_width != width or new_height != height:
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        cv2.imwrite(image_path, img)

def create_transparent_overlay_with_lines(overlay_path, width, height):
    """
    Creates a transparent PNG overlay with percentage text and horizontal grid lines,
    ensuring 100% is included at the top.
    """
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))  # Transparent background
    draw = ImageDraw.Draw(img)

    # Define font size and positions
    try:
        font = ImageFont.truetype("arial.ttf", 20)  # Adjust font size as needed
    except IOError:
        font = ImageFont.load_default()

    # Draw percentage text and horizontal lines
    for i, percent in enumerate(range(0, 101, 10)):  # Include 100%
        y_pos = int(height - (i * height / 10))

        # Draw grid lines (different color for 25%, 50%, 75%)
        line_color = (255, 255, 255, 100)  # Default white with transparency
        if percent in [20, 50, 80]:
            line_color = (255, 255, 0, 150)  # Yellow with more opacity
        draw.line([(0, y_pos), (width, y_pos)], fill=line_color, width=2)

        # Add percentage text on the left
        draw.text((10, y_pos - 10), f"{percent}%", font=font, fill=(255, 255, 255, 255))  # White text

    # Save the image
    img.save(overlay_path)

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

    snaps_dir = os.path.join(os.getcwd(), "snaps")
    os.makedirs(snaps_dir, exist_ok=True)

    graph_frame_dir = os.path.join(os.getcwd(), "processed_frames")
    os.makedirs(graph_frame_dir, exist_ok=True)

    generate_button = st.sidebar.button("Generate Happiness Video with Overlay")

    if generate_button:
        st.info("Analyzing video frames for happiness...")

        half_second_happiness = {}

        # Step 1: Extract frames and analyze happiness
        cap = cv2.VideoCapture(input_video_path)

        frame_scores = []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        timestamps = np.linspace(0, frame_count / fps, frame_count)

        progress_bar = st.progress(0)

        # Initialize OpenCV face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                print(f"Skipping frame {i} due to read error.")
                frame_scores.append(0)  # Assign a default happiness score for skipped frames
                continue

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect faces in the frame
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            happiness_score = 0
            face_count = len(faces)

            for (x, y, w, h) in faces:
                # Draw a blue bounding box around the face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue bounding box

                # Analyze emotions in the face region
                face_region = rgb_frame[y:y+h, x:x+w]
                try:
                    analysis = DeepFace.analyze(face_region, actions=["emotion"], enforce_detection=False)
                    if isinstance(analysis, list):
                        happiness_score += analysis[0]["emotion"]["happy"] if analysis else 0
                    elif isinstance(analysis, dict):
                        happiness_score += analysis["emotion"]["happy"]
                except Exception as e:
                    print(f"Error analyzing face in frame {i}: {e}")

            happiness_score = happiness_score / face_count if face_count > 0 else 0
            frame_scores.append(happiness_score)

            # Save the current frame with bounding boxes
            annotated_frame_path = os.path.join(graph_frame_dir, f"frame_{i:04d}.png")
            cv2.imwrite(annotated_frame_path, frame)
            make_dimensions_even(annotated_frame_path)

            print(f"Frame {i} - Happiness Score: {happiness_score}")
            progress_bar.progress((i + 1) / frame_count)

        cap.release()

        # Generate graph video
        graph_video_path = os.path.join(temp_dir, "happiness_graph_dynamic.mp4")
        ffmpeg_command_graph = [
            "ffmpeg", "-y", "-framerate", str(fps), "-i",
            os.path.join(graph_frame_dir, "frame_%04d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", graph_video_path
        ]
        result_graph = subprocess.run(ffmpeg_command_graph, capture_output=True, text=True)

        if result_graph.returncode != 0:
            st.error(f"Error generating graph video: {result_graph.stderr}")
        else:
            st.info("Overlaying graph and static percentage overlay...")

            # Create transparent overlay
            static_overlay_path = "static_overlay.png"
            create_transparent_overlay_with_lines(static_overlay_path, 1920, 200)

            # Overlay the graph video and static PNG on the original video
            final_video_path = os.path.join(temp_dir, "final_video.mp4")
            overlay_command = [
                "ffmpeg", "-y", "-i", graph_video_path, "-i", static_overlay_path,
                "-filter_complex", "[0:v][1:v]overlay=W-w:H-h",
                "-c:a", "copy", final_video_path
            ]

            result_overlay = subprocess.run(overlay_command, capture_output=True, text=True)

            if result_overlay.returncode == 0:
                st.success("Video generated successfully!")
                with open(final_video_path, "rb") as f:
                    st.download_button(
                        label="Download Happiness Video",
                        data=f,
                        file_name="happiness_video.mp4",
                        mime="video/mp4"
                    )
                os.remove(static_overlay_path)
                shutil.rmtree(graph_frame_dir)
            else:
                st.error(f"Error overlaying graph on video: {result_overlay.stderr}")
