const { default: axios } = require("axios")
const { runPythonModel } = require("../middlewares/runPythonModel")
const sendMail = require("../middlewares/sendMail")
require('dotenv').config()
let users = new Set()

async function getIdFromUsername(username, email) {
    let id = await axios.get(`https://api.twitter.com/2/users/by?usernames=${username}`, {
        headers: {
            'Authorization': `Bearer ${process.env.TWITTER_API_BEARER_TOKEN}`
        }
    })
    id = id.data.data[0]
    id.email = email
    users.add(id)
    console.log('Tracking started for user ', username);
}

async function fetchTweets() {
    console.log('Fetch tweets triggered.')
    try {
        for(let user of users) {
            let tweets = await axios.get(`https://api.twitter.com/2/users/${user.id}/tweets?tweet.fields=created_at`, {
                headers: {
                    'Authorization': `Bearer ${process.env.TWITTER_API_BEARER_TOKEN}`
                }
            })
            console.log(user);
            tweets = tweets.data.data
            for(let i=0;i<tweets.length;i++) {
                let {text, created_at, id} = tweets[i]
                if((Date.now()-Date.parse(created_at))/1000/60>1) continue;
                try {
                    let predictions = await runPythonModel("CrowdflowerWithOversampling", [text])
                    console.log(text, ': ', predictions);
                    if(predictions=='Error' || predictions[0]=='Normal') continue;
                    const options = {
                        to: user.email,
                        subject: "Hate speech detected",
                        html: `<html>
                        <head>
                        <style>
                        </style>
                        </head>
                        <body>
                        <blockquote class="twitter-tweet">
                        <p lang="en" dir="ltr">${text}</p>&mdash; ${user.name} (@${user.username})
                        <a href="https://twitter.com/${user.username}/status/${id}">${created_at.substring(0,10)}</a>
                    </blockquote>
                    </body>
                    <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
                    </html>`,
                        textEncoding: "base64",
                        headers: [
                        { key: "X-Application-Developer", value: "Manav Doda" },
                        { key: "X-Application-Version", value: "v1.0.0.2" },
                        ],
                    };
                    sendMail(options)
                } catch(err) {
                    console.log('Error in running python model.', err);
                }
            }
            // console.log(tweets);
        }
    } catch(err) {
        console.log(err);
    }
}

function scheduleTask(time,  gapInMs) {
    const hour = Number(time.split(':')[0]);
    const minute = Number(time.split(':')[1]);
    const startTime = new Date(); 
    startTime.setHours(hour, minute);
    const now = new Date();
    if (startTime.getTime() < now.getTime()) {
      startTime.setHours(startTime.getHours() + 24);
    }
    let firstTriggerAfterMs = startTime.getTime() - now.getTime();
    firstTriggerAfterMs = 0;
    setTimeout(function(){
      fetchTweets;
      setInterval(fetchTweets, gapInMs);
    }, firstTriggerAfterMs);
}

module.exports = {getIdFromUsername, scheduleTask}