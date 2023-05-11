import cv2

def calculate_blur(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian

def calculate_flatness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    std_dev = cv2.meanStdDev(gray)[1]
    return std_dev

def calculate_noise(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean, std_dev = cv2.meanStdDev(gray)
    return std_dev[0][0]

def calculate_resolution(video_path):
    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()
    return width, height

def calculate_average_bitrate(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
    bitrate = cap.get(cv2.CAP_PROP_BITRATE)
    average_bitrate = (bitrate * duration) / frames
    cap.release()
    return average_bitrate

def calculate_average_framerate(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps
def calculate_blur(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian

def calculate_average_blur(video_path):
    cap = cv2.VideoCapture(video_path)
    total_blur = 0.0
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        blur = calculate_blur(frame)
        total_blur += blur
        frame_count += 1

    cap.release()

    if frame_count > 0:
        average_blur = total_blur / frame_count
        return average_blur
    else:
        return 0.0

# Example usage

# Calculate average blur

# Example usage
video_path = '../file__example.mp4'

# # Calculate blurryness
# image = cv2.imread('path_to_sample_image.jpg')

# blur = calculate_blur(image)
# print(f"Blurryness: {blur}")

# # Calculate flatness
# flatness = calculate_flatness(image)
# print(f"Flatness: {flatness}")

# # Calculate noise
# noise = calculate_noise(image)
# print(f"Noise: {noise}")

# Calculate resolution
width, height = calculate_resolution(video_path)
print(f"Resolution: {int(width)}x{int(height)}")

# Calculate average bit rate
average_bitrate = calculate_average_bitrate(video_path)
print(f"Average Bitrate: {average_bitrate} bps")

# Calculate average frame rate
average_framerate = calculate_average_framerate(video_path)
print(f"Average Frame Rate: {average_framerate} fps")

# Calculate packet loss (specific implementation depends on the network/protocol)
# You may need to use additional tools or libraries for this analysis.

# Calculate lip sync (specific implementation depends on the audio/video synchronization analysis)
# You may need to use additional tools or libraries for this analysis.
average_blur = calculate_average_blur(video_path)
print(f"Average Blurriness: {average_blur}")