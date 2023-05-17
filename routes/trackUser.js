const { getIdFromUsername } = require("../hateSpeechDetectorBots/twitterBot")

async function trackUser(req, res) {
    let username = req.body.username
    let email = req.body.email
    console.log('POST /track request');
    try {
        await getIdFromUsername(username, email)
        return res.json({success: true, message: "User tracking started."})
    } catch(err) {
        console.log('Error in tracking. ', err);
        return res.json({success: false, message: 'Please try again after sometime.'})
    }
} 

module.exports = trackUser