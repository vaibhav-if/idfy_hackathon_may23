const videoService = require("../services/videoService")

const videoStats = async(req,res)=>{
 try {
  const resp = await videoService.videoStats()
  res.send("success");
 } catch (error) {
  console.log(error)
 }
}

module.exports = {videoStats}