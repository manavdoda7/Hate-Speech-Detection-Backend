const testFolder = './tests/';
const fs = require('fs');

function readContents(dirpath) {
    let contents = []
    fs.readdirSync(dirpath).forEach(file => {
        if(file[0]!='.') contents.push(file)
    });
    return contents
}

module.exports = {readContents}