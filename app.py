# from flask import Flask, request, jsonify
# import cv2
# import numpy as np
import subprocess
import json
import psycopg2

from flask import Flask, request, jsonify
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from vifp import vifp_mscale
from flask_cors import CORS


app = Flask(__name__)

# Connect to the database
conn = psycopg2.connect(
    host='localhost',
    database='video_solutions',
    user='vaibhavagrawal',
    password='vaibhavagrawal'
)
cursor = conn.cursor()

# Define a route to create the users table
@app.route('/create_users_table')
def create_users_table():
    # Create a table to store video stats
    cursor.execute('CREATE TABLE video_stats (id SERIAL PRIMARY KEY, blockiness_rating INTEGER, blurriness_rating INTEGER, flatness_rating INTEGER, frame_count INTEGER, multiple_faces_percentage TEXT, psnr_avg TEXT, ssim_avg TEXT, vif_avg TEXT, bit_rate BIGINT, r_frame_rate TEXT, resolution TEXT)')
    conn.commit()
    return 'Users table created successfully'


# Define a route to create the users table
@app.route('/video_stats')
def video_stats():
    # Create a table to store video stats
    cursor.execute('SELECT * FROM video_stats')
    rows = cursor.fetchall()
    results = []
    for row in rows:
        result = {
            'id': row[0],
            'blockiness_rating': row[1],
            'blurriness_rating': row[2],
            'flatness_rating': row[3],
            'frame_count': row[4],
            'multiple_faces_percentage': float(row[5]),
            'psnr_avg': float(row[6]),
            'ssim_avg': float(row[7]),
            'vif_avg': float(row[8]),
            'bit_rate': row[9],
            'r_frame_rate': row[10],
            'resolution': row[11]
        }
        results.append(result)
    return jsonify(results)

# Initialize the face detector
    # Initialize the face detector
CORS(app)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Define route for processing video
@app.route('/process_video', methods=['POST'])
def process_video():
    print("req",request.files)
    # Load the video file
    if 'video' not in request.files:
        return 'No video file found', 400
    file = request.files['video']
    video_path = './uploaded_video.mp4'
    file.save(video_path)

    # Define blockiness check function
    def blockiness(frame):
        # Split the frame into 8x8 blocks
        blocks = np.array([frame[y:y+8, x:x+8] for y in range(0, frame.shape[0], 8) for x in range(0, frame.shape[1], 8)])
        # Calculate the standard deviation of each block
        block_stddevs = np.std(blocks, axis=(1, 2))
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
    psnr_sum = 0
    ssim_sum = 0
    vif_sum = 0

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

       # Calculate PSNR, SSIM, and VIF only every n frames (n = 5 in this case)
        if frame_count % 5 == 0:
            # Load previous frame
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

            # Calculate PSNR
            psnr_sum += psnr(prev_gray, gray)

            # Calculate SSIM
            ssim_sum += ssim(prev_gray, gray, data_range=gray.max() - gray.min())

            # Calculate VIF
            vif_sum += vifp_mscale(prev_gray, gray)

        # Save current frame for next iteration
        prev_frame = frame.copy()

    # Calculate percentages for multiple faces
    multiple_faces_percentage = multiple_faces_count / frame_count * 100

    # Normalize scores to a scale of 1 to 10
    flatness_rating = int(np.interp(flatness_score, [0, 255], [1, 10]))
    blurriness_rating = int(np.interp(blurriness_score, [0, 1000], [10, 1]))
    blockiness_rating = blockiness_score

    # Calculate average metrics
    if frame_count > 1:
        psnr_avg = psnr_sum / (frame_count - 1)
        ssim_avg = ssim_sum / (frame_count - 1)
        vif_avg = vif_sum / (frame_count - 1)
    else:
        psnr_avg = ssim_avg = vif_avg = 0

    cmd = ['ffprobe', '-v', 'error', '-print_format', 'json', '-show_format', '-show_streams', video_path]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = result.stdout.decode('utf-8')
    data = json.loads(output)

    # Extract format metadata
    format_metadata = data['format']

    # Extract stream metadata
    stream_metadata = []
    for stream in data['streams']:
        stream_metadata.append(stream)

    bit_rate = format_metadata["bit_rate"]
    # hard coded, please refactor this
    r_frame_rate = "30/1"
    resolution = "1920x1080"
    blockiness_rating = 123

    cursor.execute("INSERT INTO video_stats (blockiness_rating, blurriness_rating, flatness_rating, frame_count, multiple_faces_percentage, psnr_avg, ssim_avg, vif_avg, bit_rate, r_frame_rate, resolution) VALUES (%s::integer, %s::integer, %s::integer, %s::integer, %s::decimal, %s::decimal, %s::decimal, %s::decimal, %s::bigint, %s::text, %s::text)", (blockiness_rating, blurriness_rating, flatness_rating, frame_count, multiple_faces_percentage, psnr_avg, ssim_avg, vif_avg, bit_rate, r_frame_rate, resolution))

    # Create response dictionary
    response = {
        'frame_count': frame_count,
        'multiple_faces_percentage': multiple_faces_percentage,
        'flatness_rating': flatness_rating,
        'blurriness_rating': blurriness_rating,
        'blockiness_rating': blockiness_rating,
        'psnr_avg': psnr_avg,
        'ssim_avg': ssim_avg,
        'vif_avg': vif_avg,
        'bit_rate': bit_rate,
        'r_frame_rate': r_frame_rate,
        'bit_rate': bit_rate
    }
    print(response)
    # Return the response as JSON
    return jsonify(response)



@app.route('/process', methods=['POST'])
def get_video_parameters():

    file = request.files['video']
    video_path = './video-ag.webm'
    file.save(video_path)

    cmd = ['ffprobe', '-v', 'error', '-print_format', 'json', '-show_format', '-show_streams', video_path]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = result.stdout.decode('utf-8')
    data = json.loads(output)

    # Extract format metadata
    format_metadata = data['format']

    # Extract stream metadata
    stream_metadata = []
    for stream in data['streams']:
        stream_metadata.append(stream)

    bit_rate = format_metadata["bit_rate"]
    # hard coded, please refactor this
    r_frame_rate = "30/1"
    bit_rate = "1920x1080"

    cursor.execute("INSERT INTO video_stats (bit_rate, r_frame_rate, resolution) VALUES (%s::bigint, %s::text, %s::text)", (bit_rate, r_frame_rate, resolution))

    return {'format': format_metadata, 'streams': stream_metadata}


# Close the connection to the database
# cursor.close()
# conn.close()

if __name__ == '__main__':
        app.run(debug=True)
