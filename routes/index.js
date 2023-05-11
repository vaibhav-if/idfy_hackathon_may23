const videoController = require("../controllers/videoController")

module.exports = (app) => {
  app.get("/healthcheck", (req, res) => {
    res.send("success");
  });
  app.get("/stats", videoController.videoStats)
};
