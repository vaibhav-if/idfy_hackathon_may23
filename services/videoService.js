// const fs = require('fs');

const path = require('path');

// const { exec } = require('child_process');

var ffmpeg = require('ffmpeg');

// Get the absolute path of the public folder
const publicPath = path.join(__dirname, '../public/videos');

// Get the absolute path of the file within the public folder
const filePath = path.join(publicPath, 'video.mp4');

// const command = `ffprobe -v quiet -print_format json -show_format -show_streams ${filePath}`;

const videoStats = () =>{
  return new Promise((resolve, reject) => {
    try {
      let stat_results = {}
      let video_stats = {}
      let audio_stats = {}
      var process = new ffmpeg(filePath);
      process.then(function (video) {
        // Video metadata
        let metadata = video.metadata
        video_stats.duration = metadata.duration
        video_stats.resolution = metadata.video.resolution
        video_stats.aspect = metadata.video.aspect
        video_stats.bitrate = metadata.video.bitrate
        video_stats.codec = metadata.video.codec
        video_stats.fps = metadata.video.fps
        audio_stats.codec = metadata.audio.codec
        audio_stats.bitrate = metadata.audio.bitrate
        audio_stats.channels = metadata.audio.channels
        stat_results.video_stats = video_stats
        stat_results.audio_stats = audio_stats
        console.log(stat_results);
        resolve({stat_results});
      }, function (err) {
        console.log('Error: ' + err);
        reject(err);
      });
      // return stat_results
    } catch (e) {
      console.log(e.code);
      console.log(e.msg);
    }
  })
}

module.exports = {videoStats}