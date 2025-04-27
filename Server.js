const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const { spawn } = require('child_process');

const app = express();
const PORT = 3000;

app.use(cors());
app.use(bodyParser.json());

app.post('/predict', (req, res) => {
    const { testParams, filters } = req.body;
    console.log('Received request with:', { testParams, filters });

    const python = spawn('C:/Users/nimai/anaconda3/envs/myenv310/python',
        ['predict_price.py', JSON.stringify(testParams), JSON.stringify(filters)]);

    console.log('Spawned Python process with PID:', python.pid);

    let result = '';
    let errorOutput = '';

    python.stdout.on('data', (data) => {
        console.log('Python stdout:', data.toString());
        result += data.toString();
    });

    python.stderr.on('data', (data) => {
        console.error('Python stderr:', data.toString());
        errorOutput += data.toString();
    });

    python.on('close', (code) => {
        console.log(`Python process exited with code ${code}`);
        console.log('Result:', result);
        console.log('Errors:', errorOutput);

        if (code !== 0 || !result.trim()) {
            return res.status(500).json({
                error: 'Python script failed',
                exitCode: code,
                stderr: errorOutput,
                stdout: result
            });
        }

        try {
            res.json(JSON.parse(result));
        } catch (error) {
            res.status(500).json({
                error: 'Failed to parse Python output',
                rawOutput: result,
                parseError: error.message
            });
        }
    });
});

app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});
