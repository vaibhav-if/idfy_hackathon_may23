import cv2

def calculate_flatness_score(video_path):
    cap = cv2.VideoCapture(video_path)

    # Check if the video capture is successfully opened
    if not cap.isOpened():
        print("Error opening video file")
        return 0.0

    frame_count = 0
    total_std_dev = 0.0

    while True:
        # Read the frame
        ret, frame = cap.read()

        # Check if the frame is successfully read
        if not ret:
            break

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate the standard deviation of pixel intensities in the frame
        std_dev = gray_frame.std()

        total_std_dev += std_dev
        frame_count += 1

    cap.release()

    # Calculate the average standard deviation
    avg_std_dev = total_std_dev / frame_count

    # You can adjust this threshold value based on your requirements
    threshold = 10.0

    # Calculate the flatness score as a ratio of the average standard deviation to the threshold
    flatness_score = avg_std_dev / threshold

    return flatness_score

# Provide the path to your video file
video_file = "../file__example.mp4"
# Call the function to calculate the flatness score of the video
flatness_score = calculate_flatness_score(video_file)

print("Flatness Score:", flatness_score)

