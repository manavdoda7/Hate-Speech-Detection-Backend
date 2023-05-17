const { runPythonModel } = require("../middlewares/runPythonModel")

async function predictOutputasync(req, res) {
    let {model, arr} = req.body
    console.log('POST /predict request');
    try {
        let predictions = await runPythonModel(model, arr)
        if(predictions!='Error') return res.json({success: true, predictions})
        else return res.json({success: false, message: 'Please try again'})
    } catch(err) {
        console.log('Error in running python model.', err);
        return res.json({success: false, message: 'Please try again.'})
    }
}

module.exports = predictOutputasync