module.exports = (app) => {
  app.get("/healthcheck", (req, res) => {
    res.send("success");
  });
};
