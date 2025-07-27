// Background script for handling API calls and maintaining state

// Configuration
const API_BASE_URL = 'http://localhost:8080'; // Change this to your API URL

// Set recording badge
function setRecordingBadge(isRecording) {
    if (isRecording) {
        chrome.action.setBadgeText({ text: 'REC' });
        chrome.action.setBadgeBackgroundColor({ color: '#FF0000' }); // Red color
        chrome.action.setTitle({ title: 'Speech Analysis - Recording...' });
    } else {
        chrome.action.setBadgeText({ text: '' });
        chrome.action.setTitle({ title: 'Speech Analysis System' });
    }
}

// Listen for messages from popup
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.action === 'startRecording') {
        handleStartRecording();
    } else if (message.action === 'generateSummary') {
        handleGenerateSummary();
    } else if (message.action === 'setRecordingBadge') {
        setRecordingBadge(true);
    } else if (message.action === 'clearRecordingBadge') {
        setRecordingBadge(false);
    }
});

// Handle recording completion
async function handleRecordingComplete(data) {
    try {
        // Update storage with completion data
        await chrome.storage.local.set({
            isRecording: false,
            recordingStartTime: null,
            hasTranscription: true,
            transcription: data.transcription,
            emotion: data.emotion,
            lastRecordTime: new Date().toLocaleString(),
            recordingCompleted: true // Flag to indicate completion
        });
        
        // Remove recording badge
        setRecordingBadge(false);
        
        console.log('Recording completed, data stored');
        
        // Try to send message to popup if it's open
        try {
            chrome.runtime.sendMessage({
                action: 'recordingComplete',
                data: data
            });
        } catch (error) {
            // Popup might be closed, that's okay
            console.log('Popup not available, data stored for later retrieval');
        }
        
        // Show notification regardless
        chrome.notifications.create({
            type: 'basic',
            iconUrl: 'icons/icon48.png',
            title: 'Recording Complete',
            message: 'Transcription and emotion analysis ready!'
        });
        
    } catch (error) {
        console.error('Error handling recording completion:', error);
    }
}

// Handle start recording
async function handleStartRecording() {
    try {
        console.log('Starting recording process...');
        
        // Set recording badge
        setRecordingBadge(true);
        
        // Call transcribe API
        const response = await fetch(`${API_BASE_URL}/transcribe/`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({})
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }

        console.log('Transcription received:', data.transcription);
        
        // Get emotion analysis
        const emotion = await getEmotion();
        
        // Handle completion
        await handleRecordingComplete({
            transcription: data.transcription,
            emotion: emotion
        });
        
    } catch (error) {
        console.error('Error during recording:', error);
        
        // Remove recording badge on error
        setRecordingBadge(false);
        
        // Update storage with error state
        await chrome.storage.local.set({
            isRecording: false,
            recordingStartTime: null,
            recordingCompleted: false
        });
        
        // Try to send error message to popup if it's open
        try {
            chrome.runtime.sendMessage({
                action: 'recordingError',
                error: error.message
            });
        } catch (msgError) {
            console.log('Popup not available for error message');
        }
        
        // Show error notification
        chrome.notifications.create({
            type: 'basic',
            iconUrl: 'icons/icon48.png',
            title: 'Recording Failed',
            message: error.message || 'Recording failed'
        });
    }
}

// Get emotion analysis
async function getEmotion() {
    try {
        const response = await fetch(`${API_BASE_URL}/get-emotion`);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }

        return data.emotion;
        
    } catch (error) {
        console.error('Error getting emotion:', error);
        return 'Error analyzing emotion';
    }
}

// Handle generate summary
async function handleGenerateSummary() {
    try {
        console.log('Generating summary...');
        
        const response = await fetch(`${API_BASE_URL}/get-summary`);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }

        console.log('Summary generated:', data.summary);
        
        // Store summary
        await chrome.storage.local.set({
            summary: data.summary,
            hasSummary: true
        });
        
        // Send completion message to popup
        chrome.runtime.sendMessage({
            action: 'summaryComplete'
        });
        
    } catch (error) {
        console.error('Error during summary generation:', error);
        
        // Store error message
        await chrome.storage.local.set({
            summary: `Error during summarization: ${error.message}`,
            hasSummary: true
        });
        
        chrome.runtime.sendMessage({
            action: 'summaryComplete'
        });
    }
}

// Clean up on extension startup
chrome.runtime.onStartup.addListener(async () => {
    console.log('Extension started, cleaning up states...');
    
    // Check if there's an ongoing recording that should be cleaned up
    const result = await chrome.storage.local.get(['isRecording']);
    if (result.isRecording) {
        setRecordingBadge(false);
    }
    
    chrome.storage.local.set({
        isRecording: false,
        recordingStartTime: null,
        recordingCompleted: false
    });
});

// Clean up on extension install
chrome.runtime.onInstalled.addListener(() => {
    console.log('Extension installed, initializing...');
    
    // Clear any existing badge
    setRecordingBadge(false);
    
    chrome.storage.local.set({
        isRecording: false,
        recordingStartTime: null,
        hasTranscription: false,
        hasSummary: false,
        recordingCompleted: false
    });
});

// Initialize badge state on startup
chrome.runtime.onStartup.addListener(async () => {
    const result = await chrome.storage.local.get(['isRecording']);
    setRecordingBadge(result.isRecording || false);
});