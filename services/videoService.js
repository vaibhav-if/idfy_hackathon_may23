const fs = require('fs');

const path = require('path');

// Get the absolute path of the public folder
const publicPath = path.join(__dirname, '../public/videos');

// Get the absolute path of the file within the public folder
const filePath = path.join(publicPath, 'video.mp4');

const videoStats = async()=>{
 try {
  // Read the file from disk
  fs.readFile(filePath, (err, data) => {
   if (err) throw err;
   console.log(data)
 });
 } catch (error) {
  
 }
}

module.exports = {videoStats}