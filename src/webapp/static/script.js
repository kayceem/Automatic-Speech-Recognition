// Recording state management
let isRecording = false;
let audioVisualizer;
let audioContext;

// DOM elements
const recordBtn = document.getElementById("recordBtn");
const resultDisplay = document.getElementById("result");
const emotionDisplay = document.getElementById("emotion");
const summarizeBtn = document.getElementById("summarizeBtn");
const summaryDisplay = document.getElementById("summary");

// Event listeners
recordBtn.addEventListener("click", toggleRecording);
summarizeBtn.addEventListener("click", getSummary);

// Audio Visualizer Class
class AudioVisualizer {
    constructor(audioContext, processFrame, processError) {
        this.audioContext = audioContext;
        this.processFrame = processFrame;
        this.stream = null;
        this.analyser = null;
        
        this.connectStream = this.connectStream.bind(this);
        this.initializeAudio();
    }

    async initializeAudio() {
        try {
            this.stream = await navigator.mediaDevices.getUserMedia({ 
                audio: true, 
                video: false 
            });
            this.connectStream(this.stream);
        } catch (error) {
            console.error("Error accessing microphone:", error);
            if (this.processError) {
                this.processError(error);
            }
        }
    }

    connectStream(stream) {
        this.analyser = this.audioContext.createAnalyser();
        const source = this.audioContext.createMediaStreamSource(stream);
        source.connect(this.analyser);
        
        this.analyser.smoothingTimeConstant = 0.5;
        this.analyser.fftSize = 256;
        
        this.initRenderLoop();
    }

    initRenderLoop() {
        const frequencyData = new Uint8Array(this.analyser.frequencyBinCount);
        
        const renderFrame = () => {
            if (this.analyser) {
                this.analyser.getByteFrequencyData(frequencyData);
                if (this.processFrame) {
                    this.processFrame(frequencyData);
                }
            }
            requestAnimationFrame(renderFrame);
        };
        
        requestAnimationFrame(renderFrame);
    }

    cleanup() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
        }
    }
}

// Visualization setup
const visualMainElement = document.querySelector('main');
const visualValueCount = 16;
let visualElements;

const colors = [
    '#ff6347', '#4682b4', '#32cd32', '#ff1493', 
    '#ffd700', '#8a2be2', '#ff4500', '#00fa9a', 
    '#a52a2a', '#5f9ea0', '#f0e68c', '#dda0dd', 
    '#0000ff', '#ff00ff', '#adff2f', '#c71585'
];

// Data mapping for visual effect
const dataMap = { 
    0: 15, 1: 10, 2: 8, 3: 9, 4: 6, 5: 5, 6: 2, 7: 1, 
    8: 0, 9: 4, 10: 3, 11: 7, 12: 11, 13: 12, 14: 13, 15: 14 
};

function createDOMElements() {
    visualMainElement.innerHTML = '';
    
    for (let i = 0; i < visualValueCount; i++) {
        const elm = document.createElement('div');
        elm.style.background = colors[i % colors.length];
        visualMainElement.appendChild(elm);
    }
    
    visualElements = document.querySelectorAll('main div');
}

function processFrame(data) {
    const values = Object.values(data);
    
    for (let i = 0; i < visualValueCount; i++) {
        const value = values[dataMap[i]] / 255;
        const elmStyles = visualElements[i].style;
        elmStyles.transform = `scaleY(${value})`;
        elmStyles.opacity = Math.max(0.25, value);
    }
}

function processError(error) {
    visualMainElement.classList.add('error');
    visualMainElement.innerText = 'Please allow access to your microphone';
    console.error("Audio visualization error:", error);
}

// Initialize visualization
function initializeVisualization() {
    try {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        createDOMElements();
        audioVisualizer = new AudioVisualizer(audioContext, processFrame, processError);
    } catch (error) {
        console.error("Failed to initialize audio visualization:", error);
        processError(error);
    }
}

// Recording functions
function toggleRecording() {
    if (!isRecording) {
        startRecording();
    } else {
        // Note: In your current setup, recording stops automatically after transcription
        // If you want manual stop functionality, uncomment the line below
        // stopRecording();
    }
}

async function startRecording() {
    try {
        setRecordingState(true);
        await transcribeAudio();
    } catch (error) {
        console.error("Error starting recording:", error);
        setRecordingState(false);
        showError("Failed to start recording. Please try again.");
    }
}

function stopRecording() {
    setRecordingState(false);
    console.log("Recording stopped.");
}

function setRecordingState(recording) {
    isRecording = recording;
    recordBtn.textContent = recording ? "Recording..." : "Start Recording";
    recordBtn.disabled = recording;
    
    if (!recording) {
        recordBtn.textContent = "Start Recording";
        recordBtn.disabled = false;
    }
}

// API calls
async function transcribeAudio() {
    try {
        const formData = new FormData();
        
        const response = await fetch("/transcribe/", {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }

        // Display transcription
        resultDisplay.textContent = data.transcription;
        
        // Stop recording and enable summarize button
        stopRecording();
        summarizeBtn.disabled = false;
        
        // Get emotion analysis
        await getEmotion();
        
    } catch (error) {
        console.error("Error during transcription:", error);
        resultDisplay.textContent = "Error during transcription: " + error.message;
        stopRecording();
    }
}

async function getEmotion() {
    try {
        const response = await fetch("/get-emotion");
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }

        emotionDisplay.textContent = data.emotion;
        
    } catch (error) {
        console.error("Error getting emotion:", error);
        emotionDisplay.textContent = "Error analyzing emotion: " + error.message;
    }
}

async function getSummary() {
    try {
        summarizeBtn.disabled = true;
        summarizeBtn.textContent = "Generating Summary...";
        
        const response = await fetch("/get-summary");
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }

        summaryDisplay.textContent = data.summary;
        
    } catch (error) {
        console.error("Error during summarization:", error);
        summaryDisplay.textContent = "Error during summarization: " + error.message;
    } finally {
        summarizeBtn.disabled = false;
        summarizeBtn.textContent = "Generate Summary";
    }
}

// Utility functions
function showError(message) {
    resultDisplay.textContent = message;
    resultDisplay.style.color = "red";
}

function clearResults() {
    resultDisplay.textContent = "";
    emotionDisplay.textContent = "";
    summaryDisplay.textContent = "";
    resultDisplay.style.color = "";
}

// Initialize everything when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    initializeVisualization();
    
    // Initialize UI state
    summarizeBtn.disabled = true;
    clearResults();
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (audioVisualizer) {
        audioVisualizer.cleanup();
    }
    if (audioContext && audioContext.state !== 'closed') {
        audioContext.close();
    }
});