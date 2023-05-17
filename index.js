const express = require('express')
const app = express()
require('./middlewares/gmailConnection')
require('dotenv').config()

var cors = require("cors");
const { scheduleTask } = require('./hateSpeechDetectorBots/twitterBot');
const sendMail = require('./middlewares/sendMail');
app.use(cors())

app.use(express.json());
app.use(express.urlencoded({ extended: true }));

app.use('/', require('./routes'))
  
app.listen(process.env.PORT, () => {
    console.log("App listening at port "+process.env.PORT);
});

scheduleTask('09:59', 60*1000)