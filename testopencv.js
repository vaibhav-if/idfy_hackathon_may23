const cv = require('opencv4nodejs');
const path = require('path');

// Get the absolute path of the public folder
const publicPath = path.join(__dirname, '../');

// Get the absolute path of the file within the public folder
const filePath = path.join(publicPath, 'file__example.mp4');// Load the video file

// const videoFile = '../file__example.mp4';
const videoCapture = new cv.VideoCapture(filePath);

// Get the first frame of the video
const frame = videoCapture.read();

// Convert the frame to grayscale for better edge detection
const grayFrame = frame.bgrToGray();

// Apply Canny edge detection to the grayscale frame
const cannyEdges = grayFrame.canny(50, 150);

// Calculate the contours of the edges
const contours = cannyEdges.findContours(cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

// Calculate the convex hull of the contours
const convexHull = contours.convexHull();

// Calculate the area of the convex hull
const hullArea = convexHull.area;

// Calculate the area of the frame
const frameArea = frame.cols * frame.rows;

// Calculate the flatness ratio
const flatnessRatio = hullArea / frameArea;

// Determine if the video is flat based on the flatness ratio
const isFlat = flatnessRatio >= 0.9; // Adjust the threshold as needed

console.log(`Flatness Ratio: ${flatnessRatio}`);
console.log(`Is Flat: ${isFlat}`);

// Release the video capture resources
videoCapture.release();