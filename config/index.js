require("dotenv").config();

module.exports = {
  port: process.env.PORT, // 8000,
  allowHosts: process.env.ALLOW_HOST,
  dbconfig: {
    dbname: process.env.DB_NAME,
    dbuser: process.env.DB_USER,
    dbpassword: process.env.DB_PASSWORD,
    dbhost: process.env.DB_HOST,
    dbport: process.env.DB_PORT,
  },
};
