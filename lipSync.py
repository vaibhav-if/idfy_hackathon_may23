import moviepy.editor as mp
import imageio
# imageio.plugins.ffmpeg.download()

def calculate_lip_sync(video_path):
    # Load the video file
    video = mp.VideoFileClip(video_path)

    # Extract the audio from the video
    audio = video.audio

    # Analyze the audio and video synchronization
    lipsync_duration = 0.0

    for frame in video.iter_frames():
        # Process each frame to detect lip movement
        # You can use a lip detection algorithm or a pre-trained model here

        # Calculate the lip sync duration based on the detected lip movements
        # You can adjust this calculation based on your lip detection approach
        lipsync_duration += 0.1

    # Calculate the lip sync ratio as a fraction of the total video duration
    total_duration = video.duration
    lip_sync_ratio = lipsync_duration / total_duration

    return lip_sync_ratio

# Provide the path to your WebM video file
video_file = "../video-agent1.webm"

# Call the function to calculate the lip sync ratio of the video
lip_sync_ratio = calculate_lip_sync(video_file)

print("Lip Sync Ratio:", lip_sync_ratio)
