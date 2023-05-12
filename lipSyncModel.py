import moviepy.editor as mp
import numpy as np

def count_lip_sync_issues(video_path, lip_sync_tolerance):
    video = mp.VideoFileClip(video_path)
    print(video)
    # Extract audio from the video
    audio = video.audio.to_soundarray()

    # Get the video frame duration (in seconds)
    frame_duration = 1 / video.fps

    # Variables to track lip sync issues
    lip_sync_issue_count = 0
    total_frames = int(video.fps * video.duration)

    # Iterate over each frame and compare audio and video
    for i in range(total_frames):
        # Calculate the current time
        current_time = i * frame_duration

        # Get the audio segment corresponding to the current frame
        start_time = int(current_time * video.fps)
        end_time = start_time + int(frame_duration * video.fps)
        audio_frame = audio[start_time:end_time]

        # Get the video frame corresponding to the current time
        video_frame = video.get_frame(current_time)

        # Check lip sync by comparing audio and video properties
        if np.abs(np.mean(audio_frame) - np.mean(video_frame)) > lip_sync_tolerance:
            lip_sync_issue_count += 1

    # Release resources
    video.close()

    return lip_sync_issue_count, total_frames

# Example usage
video_path = "../final-video.webm"
lip_sync_tolerance = 5  # Adjust the tolerance threshold as per your requirements

lip_sync_issue_count, total_frames = count_lip_sync_issues(video_path, lip_sync_tolerance)
lip_sync_percentage = (lip_sync_issue_count / total_frames) * 100

print(f"Number of frames with lip sync issues: {lip_sync_issue_count}")
print(f"Total frames analyzed: {total_frames}")
print(f"Percentage of frames with lip sync issues: {lip_sync_percentage:.2f}%")
