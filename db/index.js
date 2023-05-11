const { dbconfig } = require("../config");
const Sequelize = require("sequelize");

let config = dbconfig

module.exports = new Sequelize(
  config.dbname,
  config.dbuser,
  config.dbpassword,
  {
    host: config.dbhost,
    port: config.dbport,
    dialect: "postgres",
    // logging: false,
    // operatorsAliases: false,
    //insecureAuth:true,  //allow connectikon to http
    define: {
      createdAt: 'created_at',
      updatedAt: 'updated_at',
    },
    pool: {
      max: 20,
      min: 0,
      acquire: 30000,
      idle: 10000
    }
  }
);
