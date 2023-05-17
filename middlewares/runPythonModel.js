const { spawnSync } = require("child_process");

async function runPythonModel(model, arr) {
  const python = spawnSync("python3", [
    `./models/${model}/predict.py`,
    JSON.stringify({arr})
  ]);
  let result = python.stdout?.toString()?.trim();
  const error = python.stderr?.toString()?.trim();
  if(error) {
    console.log(error)
    return 'Error';
  } else {
    result = JSON.parse(result)
    return result.res
  }
}

module.exports = { runPythonModel };
