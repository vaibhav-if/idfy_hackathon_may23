from flask import Flask, request, jsonify
import cv2
import numpy as np

app = Flask(__name__)

# Define route for processing video
@app.route('/process_video', methods=['POST'])
def process_video():
    print('skmfksmkfkjk kjkjkj')
    # Load the video file
    file = request.files['video']
    video_path = './video-ag.webm'
    file.save(video_path)

    # Initialize the face detector
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Define blockiness check function
    def blockiness(frame):
        # Split the frame into 8x8 blocks
        blocks = [frame[y:y+8, x:x+8] for y in range(0, frame.shape[0], 8) for x in range(0, frame.shape[1], 8)]
        # Calculate the standard deviation of each block
        block_stddevs = [np.std(block) for block in blocks]
        # Calculate the mean standard deviation of all blocks
        mean_stddev = np.mean(block_stddevs)
        # Normalize the mean standard deviation to a scale of 1 to 10
        blockiness_rating = int(np.interp(mean_stddev, [0, 30], [10, 1]))
        return blockiness_rating

    # Initialize variables for metrics
    frame_count = 0
    multiple_faces_count = 0
    flatness_score = 10
    blurriness_score = 10
    blockiness_score = 10

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Process each frame of the video
    while cap.isOpened():
        # Read a frame
        ret, frame = cap.read()
        if not ret:
            break

        # Increment frame count
        frame_count += 1

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Update multiple faces count
        if len(faces) > 1:
            multiple_faces_count += 1

        # Calculate flatness score
        flatness_score = min(flatness_score, np.std(gray))

        # Calculate blurriness score
        blurriness_score = min(blurriness_score, cv2.Laplacian(gray, cv2.CV_64F).var())

        # Calculate blockiness score
        blockiness_score = min(blockiness_score, blockiness(frame))

    # Calculate percentages for multiple faces
    multiple_faces_percentage = multiple_faces_count / frame_count * 100

    # Normalize scores to a scale of 1 to 10
    flatness_rating = int(np.interp(flatness_score, [0, 255], [1, 10]))
    blurriness_rating = int(np.interp(blurriness_score, [0, 1000], [10, 1]))
    blockiness_rating = blockiness_score

    # Prepare response
    response = {
        'multiple_faces_percentage': multiple_faces_percentage,
        'flatness_rating': flatness_rating,
        'blurriness_rating': blurriness_rating,
        'blockiness_rating': blockiness_rating
    }

    # Return response as JSON
    return jsonify(response)

if __name__ == '__main__':
        app.run(debug=True)
