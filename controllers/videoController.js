const videoService = require("../services/videoService")
const http = require('http');

const videoStats = async(req,res)=>{
 try {
//   const resp = await videoService.videoStats()
  res.send("success");
 } catch (error) {
  console.log(error)
 }
}

const options = {
  hostname: 'localhost',
  port: 5000,
  path: '/process_video',
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  }
};

const req = http.request(options, (res) => {
  console.log(`statusCode: ${res.statusCode}`);
  res.on('data', (d) => {
    process.stdout.write(d);
  });
});

req.on('error', (error) => {
  console.error(error);
});

req.write(JSON.stringify({ foo: 'bar' }));
req.end();

module.exports = {videoStats}