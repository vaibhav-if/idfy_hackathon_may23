
require('dotenv').config()
const express = require("express");
const hostValidation = require('host-validation');
const bodyParser = require("body-parser");
const helmet = require('helmet');
const cors = require("cors");
const fileUpload = require("express-fileupload"); 
const nocache = require('node-nocache');
const { xss } = require('express-xss-sanitizer');
const config = require('./config')

const {port, allowHosts} = config;

// Create Express server
const app = express();

const corsOptions = {
    origin: process.env.ALLOW_ORIGIN,
};

app.use(cors(corsOptions));

app.use(hostValidation({ hosts: allowHosts.split(",") }))


// Express configuration
app.set("port", port);
app.use(bodyParser.urlencoded({ extended: false }));
app.use(bodyParser.json());
app.use(xss());

app.use(helmet({
    cacheControl: false,
    contentSecurityPolicy: "default-src 'self';base-uri 'self';font-src 'self' https: data:;form-action 'self';frame-ancestors 'self';img-src 'self' data:;object-src 'none';script-src 'self';script-src-attr 'none';style-src 'self';upgrade-insecure-requests"
}));
app.use(
    helmet.contentSecurityPolicy({
        useDefaults: false,
        directives: {
            defaultSrc: ["'self'"],
            scriptSrc: ["'self'"],
            objectSrc: ["'none'"],
            upgradeInsecureRequests: [],
        },
    })
);

app.use(nocache);

app.use(fileUpload());

app.use(errorHandler);


function errorHandler(err, req, res, next) {
    console.log("errorHandler-res", res);
    console.log(err.stack);
    res.json({
        status: constants.SERVICE_FAILURE,
        statusCode: 500,
        message: "There is some error processing request."
    });
}

module.exports = app;
