const { execFile } = require('child_process');
const path = require('path');

async function recommend_by_type(food, foodType) {
    return new Promise((resolve, reject) => {
  
        const recommendProcess = execFile(
          'python', 
          [path.join(__dirname, '../match.py'), food, foodType], 
          (error, stdout, stderr) => {
            if (error) {
              console.error('Recommendation Error:', stderr);
              return reject(new Error('Recommendation failed'));
            }
  
            try {
              const result = JSON.parse(stdout);
              resolve(result);
            } catch (err) {
              reject(new Error("JSON parsing failed: " + err.message));
            }
          }
        );
      
    });
  }

module.exports = { recommend_by_type };
