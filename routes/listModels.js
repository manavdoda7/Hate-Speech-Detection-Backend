const { readContents } = require("../middlewares/listDirFolders");

async function listModels(req, res){
    console.log('GET /models request');
    let models = readContents('./models');
    return res.json({success: true, models})
}

module.exports = listModels