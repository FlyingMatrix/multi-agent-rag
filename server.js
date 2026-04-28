const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
const { spawn } = require('child_process');
const path = require('path');

const app = express();
const server = http.createServer(app);
const io = new Server(server);

// Serve the frontend files
app.use(express.static('public'));

io.on('connection', (socket) => {
    console.log('User connected');

    socket.on('ask-question', (question) => {
        // 1. Start the Python process
        // 'python' might need to be 'python3' depending on your OS
        const pythonProcess = spawn('python', ['-u', 'main.py', 'query', question, '--ui']);

        // 2. Listen for output from Python
        pythonProcess.stdout.on('data', (data) => {
            socket.emit('answer-chunk', data.toString());
        });

        // 3. Listen for errors
        pythonProcess.stderr.on('data', (data) => {
            console.error(`Error: ${data}`);
        });

        pythonProcess.on('close', () => {
            socket.emit('answer-end');
        });
    });
});

server.listen(3000, () => {
    console.log('Server running at http://localhost:3000');
});

