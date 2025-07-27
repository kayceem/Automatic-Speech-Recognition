// DOM elements
const recordBtn = document.getElementById('recordBtn');
const viewResultsBtn = document.getElementById('viewResultsBtn');
const generateSummaryBtn = document.getElementById('generateSummaryBtn');
const recordingStatus = document.getElementById('recordingStatus');
const lastRecordTime = document.getElementById('lastRecordTime');
const progressSection = document.getElementById('progressSection');

// Initialize popup
document.addEventListener('DOMContentLoaded', async () => {
    await updateUIFromStorage();
    setupEventListeners();
});

// Event listeners
function setupEventListeners() {
    recordBtn.addEventListener('click', handleRecordClick);
    viewResultsBtn.addEventListener('click', handleViewResults);
    generateSummaryBtn.addEventListener('click', handleGenerateSummary);
}

// Update UI based on stored state
async function updateUIFromStorage() {
    try {
        const result = await chrome.storage.local.get([
            'isRecording', 
            'lastRecordTime', 
            'hasTranscription', 
            'hasSummary',
            'recordingStartTime',
            'recordingCompleted'
        ]);
        
        const isRecording = result.isRecording || false;
        const lastTime = result.lastRecordTime || 'Never';
        const hasTranscription = result.hasTranscription || false;
        const hasSummary = result.hasSummary || false;
        const recordingStartTime = result.recordingStartTime;
        const recordingCompleted = result.recordingCompleted || false;
        
        // Check if recording was completed while popup was closed
        if (isRecording && recordingCompleted) {
            console.log('Recording was completed while popup was closed, updating state...');
            await chrome.storage.local.set({
                isRecording: false,
                recordingStartTime: null,
                recordingCompleted: false
            });
            setRecordingState(false);
            
            // Tell background to clear badge
            chrome.runtime.sendMessage({
                action: 'clearRecordingBadge'
            });
            
            // Show completion notification
            chrome.notifications.create({
                type: 'basic',
                iconUrl: 'icons/icon48.png',
                title: 'Recording Complete',
                message: 'Transcription and emotion analysis ready!'
            });
        } else if (isRecording) {
            setRecordingState(true);
            // Make sure badge is set for ongoing recording
            chrome.runtime.sendMessage({
                action: 'setRecordingBadge'
            });
            // Check if recording has been going on too long (failsafe)
            if (recordingStartTime && Date.now() - recordingStartTime > 300000) { // 5 minutes
                console.warn('Recording timeout, resetting state');
                await resetRecordingState();
            }
        } else {
            setRecordingState(false);
        }
        
        // Update last record time
        lastRecordTime.textContent = lastTime;
        
        // Update button states
        viewResultsBtn.disabled = !hasTranscription;
        generateSummaryBtn.disabled = !hasTranscription;
        
    } catch (error) {
        console.error('Error updating UI from storage:', error);
    }
}

// Handle record button click
async function handleRecordClick() {
    try {
        const result = await chrome.storage.local.get(['isRecording']);
        const isCurrentlyRecording = result.isRecording || false;
        
        if (!isCurrentlyRecording) {
            await startRecording();
        }
    } catch (error) {
        console.error('Error handling record click:', error);
        showError('Failed to start recording');
    }
}

// Start recording
async function startRecording() {
    try {
        // Set recording state
        await chrome.storage.local.set({
            isRecording: true,
            recordingStartTime: Date.now(),
            hasTranscription: false,
            hasSummary: false
        });
        
        setRecordingState(true);
        
        // Send message to background script to start recording
        chrome.runtime.sendMessage({
            action: 'startRecording'
        });
        
    } catch (error) {
        console.error('Error starting recording:', error);
        await resetRecordingState();
        showError('Failed to start recording');
    }
}

// Handle view results button click
async function handleViewResults() {
    try {
        const result = await chrome.storage.local.get(['transcription', 'emotion']);
        
        if (result.transcription) {
            // Create or update results tab
            const url = chrome.runtime.getURL('results.html');
            
            // Check if results tab already exists
            const tabs = await chrome.tabs.query({ url: url });
            
            if (tabs.length > 0) {
                // Focus existing tab
                await chrome.tabs.update(tabs[0].id, { active: true });
                await chrome.windows.update(tabs[0].windowId, { focused: true });
            } else {
                // Create new tab
                await chrome.tabs.create({ url: url });
            }
        }
    } catch (error) {
        console.error('Error opening results:', error);
    }
}

// Handle generate summary button click
async function handleGenerateSummary() {
    try {
        // Disable button temporarily
        generateSummaryBtn.disabled = true;
        generateSummaryBtn.textContent = 'Generating...';
        
        // Send message to background script to generate summary
        chrome.runtime.sendMessage({
            action: 'generateSummary'
        });
        
        // Open summary tab
        setTimeout(async () => {
            const url = chrome.runtime.getURL('summary.html');
            
            // Check if summary tab already exists
            const tabs = await chrome.tabs.query({ url: url });
            
            if (tabs.length > 0) {
                // Focus existing tab
                await chrome.tabs.update(tabs[0].id, { active: true });
                await chrome.windows.update(tabs[0].windowId, { focused: true });
            } else {
                // Create new tab
                await chrome.tabs.create({ url: url });
            }
            
            // Re-enable button
            generateSummaryBtn.disabled = false;
            generateSummaryBtn.innerHTML = 'Generate Summary';
        }, 1000);
        
    } catch (error) {
        console.error('Error generating summary:', error);
        generateSummaryBtn.disabled = false;
        generateSummaryBtn.innerHTML = 'Generate Summary';
    }
}

// Set recording state UI
function setRecordingState(isRecording) {
    if (isRecording) {
        recordBtn.disabled = true;
        recordBtn.classList.add('recording');
        recordBtn.innerHTML = 'Recording...';
        recordingStatus.textContent = 'Recording';
        progressSection.style.display = 'block';
        viewResultsBtn.disabled = true;
        generateSummaryBtn.disabled = true;
    } else {
        recordBtn.disabled = false;
        recordBtn.classList.remove('recording');
        recordBtn.innerHTML = 'Record';
        recordingStatus.textContent = 'Ready';
        progressSection.style.display = 'none';
    }
}

// Reset recording state
async function resetRecordingState() {
    await chrome.storage.local.set({
        isRecording: false,
        recordingStartTime: null
    });
    setRecordingState(false);
}

// Show error message
function showError(message) {
    recordingStatus.textContent = 'Error';
    console.error(message);
    
    // Show notification
    chrome.notifications.create({
        type: 'basic',
        iconUrl: 'icons/icon48.png',
        title: 'Speech Analysis Error',
        message: message
    });
}

// Listen for messages from background script
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.action === 'recordingComplete') {
        handleRecordingComplete(message.data);
    } else if (message.action === 'recordingError') {
        handleRecordingError(message.error);
    } else if (message.action === 'summaryComplete') {
        handleSummaryComplete();
    }
});

// Handle recording completion
async function handleRecordingComplete(data) {
    try {
        // Update storage
        await chrome.storage.local.set({
            isRecording: false,
            recordingStartTime: null,
            hasTranscription: true,
            transcription: data.transcription,
            emotion: data.emotion,
            lastRecordTime: new Date().toLocaleString(),
            recordingCompleted: false // Reset the flag
        });
        
        // Update UI
        setRecordingState(false);
        lastRecordTime.textContent = new Date().toLocaleString();
        viewResultsBtn.disabled = false;
        generateSummaryBtn.disabled = false;
        
        // Show success notification only if not already shown
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

// Handle recording error
async function handleRecordingError(error) {
    await resetRecordingState();
    showError(error || 'Recording failed');
}

// Handle summary completion
function handleSummaryComplete() {
    chrome.notifications.create({
        type: 'basic',
        iconUrl: 'icons/icon48.png',
        title: 'Summary Ready',
        message: 'Text summary has been generated!'
    });
}