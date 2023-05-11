const db = require("./db");

require("dotenv").config();
try {
  const app = require("./app");

  db.authenticate()
    .then(() => {
      console.log("DB Connection has been established successfully.");
      db.sync();

    })
    .catch((err) => {
      console.error("Unable to connect to the database:", err);
    });


  app.listen(app.get("port"), () => {
    //db.sync();
    console.log(
      "  App is running at http://localhost:%d in %s mode",
      app.get("port"),
      app.get("env")
    );
  });

  process
    .on("unhandledRejection", (reason, p) => {
      console.error(reason, "************Unhandled Rejection at Promise", p);
    })
    .on("uncaughtException", (err) => {
      console.error(err, "************Uncaught Exception thrown");
      process.exit(1);
    });
} catch (err) {
  console.log(err);
}

