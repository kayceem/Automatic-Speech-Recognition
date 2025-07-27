// DOM elements
const timestampEl = document.getElementById('timestamp');
const transcriptionText = document.getElementById('transcriptionText');
const emotionResult = document.getElementById('emotionResult');
const copyTranscriptionBtn = document.getElementById('copyTranscription');
const refreshBtn = document.getElementById('refreshBtn');
const generateSummaryBtn = document.getElementById('generateSummaryBtn');
const toast = document.getElementById('toast');
const toastMessage = document.getElementById('toastMessage');

// Initialize page
document.addEventListener('DOMContentLoaded', () => {
    loadResults();
    setupEventListeners();
});

// Setup event listeners
function setupEventListeners() {
    copyTranscriptionBtn.addEventListener('click', copyTranscription);
    refreshBtn.addEventListener('click', loadResults);
    generateSummaryBtn.addEventListener('click', generateSummary);
}

// Load results from storage
async function loadResults() {
    try {
        const result = await chrome.storage.local.get([
            'transcription', 
            'emotion', 
            'lastRecordTime'
        ]);
        
        // Update timestamp
        if (result.lastRecordTime) {
            timestampEl.textContent = `Recorded: ${result.lastRecordTime}`;
        } else {
            timestampEl.textContent = 'No recording time available';
        }
        
        // Update transcription
        if (result.transcription) {
            transcriptionText.textContent = result.transcription;
            transcriptionText.classList.remove('loading', 'error');
        } else {
            transcriptionText.textContent = 'No transcription available';
            transcriptionText.classList.add('loading');
        }
        
        // Update emotion
        if (result.emotion) {
            emotionResult.innerHTML = getEmotionIcon(result.emotion);
            emotionResult.classList.remove('loading', 'error');
        } else {
            emotionResult.textContent = 'No emotion analysis available';
            emotionResult.classList.add('loading');
        }
        
    } catch (error) {
        console.error('Error loading results:', error);
        showError();
    }
}

// Update emotion icon based on emotion
function getEmotionIcon(emotion) {
    const emotionLower = emotion.toLowerCase();
    let icon = ''; // default
    
    if (emotionLower.includes('happy')) {
        icon = '<svg xmlns="http://www.w3.org/2000/svg" height="48px" viewBox="0 0 24 24" width="48px" fill="#000000"><path d="M0 0h24v24H0V0z" fill="none"/><path d="M0 0h24v24H0V0z" fill="none"/><circle cx="15.5" cy="9.5" r="1.5"/><circle cx="8.5" cy="9.5" r="1.5"/><circle cx="15.5" cy="9.5" r="1.5"/><circle cx="8.5" cy="9.5" r="1.5"/><path d="M11.99 2C6.47 2 2 6.48 2 12s4.47 10 9.99 10C17.52 22 22 17.52 22 12S17.52 2 11.99 2zM12 20c-4.42 0-8-3.58-8-8s3.58-8 8-8 8 3.58 8 8-3.58 8-8 8zm0-2.5c2.33 0 4.32-1.45 5.12-3.5h-1.67c-.69 1.19-1.97 2-3.45 2s-2.75-.81-3.45-2H6.88c.8 2.05 2.79 3.5 5.12 3.5z"/></svg>'
    } else if (emotionLower.includes('sad')) {
        icon = '<svg xmlns="http://www.w3.org/2000/svg" height="48px" viewBox="0 0 24 24" width="48px" fill="#000000"><path d="M0 0h24v24H0V0z" fill="none"/><circle cx="15.5" cy="9.5" r="1.5"/><circle cx="8.5" cy="9.5" r="1.5"/><path d="M11.99 2C6.47 2 2 6.48 2 12s4.47 10 9.99 10C17.52 22 22 17.52 22 12S17.52 2 11.99 2zM12 20c-4.42 0-8-3.58-8-8s3.58-8 8-8 8 3.58 8 8-3.58 8-8 8zm0-6c-2.33 0-4.32 1.45-5.12 3.5h1.67c.69-1.19 1.97-2 3.45-2s2.75.81 3.45 2h1.67c-.8-2.05-2.79-3.5-5.12-3.5z"/></svg>'
    } else if (emotionLower.includes('angry')) {
        icon = '<svg xmlns="http://www.w3.org/2000/svg" enable-background="new 0 0 20 20" height="48px" viewBox="0 0 20 20" width="48px" fill="#000000"><rect fill="none" height="20" width="20"/><path d="M13.89,12c-0.21-0.17-0.39-0.39-0.68-0.72c0.54-0.73,2.29-3.37,2.29-6.77C15.5,2.94,15.01,2,14,2c-1.23,0-3.17,1.65-4,4.78 C9.17,3.65,7.23,2,6,2C4.99,2,4.5,2.94,4.5,4.5c0,3.4,1.75,6.04,2.29,6.77C6.5,11.61,6.32,11.82,6.11,12 c-0.37,0.31-1.36,1.13-1.36,2.5c0,1.79,1.46,3.25,3.25,3.25c1.19,0,2-0.43,2-0.43s0.81,0.43,2,0.43c1.79,0,3.25-1.46,3.25-3.25 C15.25,13.13,14.26,12.31,13.89,12z M7.87,10.19C7.08,9.04,6,6.96,6,4.5C6,4,6.06,3.69,6.12,3.53C6.77,3.75,8.84,5.57,8.9,9.76 C8.29,9.88,7.87,10.19,7.87,10.19z M8.8,13.8c-0.22,0-0.4-0.27-0.4-0.6c0-0.33,0.18-0.6,0.4-0.6s0.4,0.27,0.4,0.6 C9.2,13.53,9.02,13.8,8.8,13.8z M10,16c-0.44,0-0.8-0.58-0.8-0.8c0-0.22,0.36-0.4,0.8-0.4s0.8,0.18,0.8,0.4 C10.8,15.42,10.44,16,10,16z M11.2,13.8c-0.22,0-0.4-0.27-0.4-0.6c0-0.33,0.18-0.6,0.4-0.6s0.4,0.27,0.4,0.6 C11.6,13.53,11.42,13.8,11.2,13.8z M11.1,9.76c0.06-4.19,2.12-6.01,2.78-6.23C13.94,3.69,14,4,14,4.5c0,2.46-1.08,4.54-1.87,5.69 C12.13,10.19,11.71,9.88,11.1,9.76z"/></svg>'
    } else if (emotionLower.includes('fear')) {
        icon = '<svg xmlns="http://www.w3.org/2000/svg" enable-background="new 0 0 24 24" height="48px" viewBox="0 0 24 24" width="48px" fill="#000000"><rect fill="none" height="24" width="24"/><path d="M21,9c-1.1,0-2-0.9-2-2c0-1.1,2-4,2-4s2,2.9,2,4C23,8.1,22.1,9,21,9z M17.5,7c0-0.73,0.41-1.71,0.92-2.66 C16.68,2.88,14.44,2,11.99,2C6.47,2,2,6.48,2,12c0,5.52,4.47,10,9.99,10C17.52,22,22,17.52,22,12c0-0.55-0.06-1.09-0.14-1.62 C21.58,10.45,21.3,10.5,21,10.5C19.07,10.5,17.5,8.93,17.5,7z M15.62,7.38l1.06,1.06L15.62,9.5l1.06,1.06l-1.06,1.06L13.5,9.5 L15.62,7.38z M7.32,8.44l1.06-1.06L10.5,9.5l-2.12,2.12l-1.06-1.06L8.38,9.5L7.32,8.44z M15.44,17c-0.69-1.19-1.97-2-3.44-2 s-2.75,0.81-3.44,2H6.88c0.3-0.76,0.76-1.43,1.34-1.99L5.24,13.3c-0.45,0.26-1.01,0.28-1.49,0c-0.72-0.41-0.96-1.33-0.55-2.05 c0.41-0.72,1.33-0.96,2.05-0.55c0.48,0.28,0.74,0.78,0.74,1.29l3.58,2.07c0.73-0.36,1.55-0.56,2.43-0.56c2.33,0,4.32,1.45,5.12,3.5 H15.44z"/></svg>'
    } else if (emotionLower.includes('surprise')) {
        icon = '<svg xmlns="http://www.w3.org/2000/svg" enable-background="new 0 0 24 24" height="48px" viewBox="0 0 24 24" width="48px" fill="#000000"><g><rect fill="none" height="24" width="24" y="0"/></g><g><path d="M20,9V7c0-1.1-0.9-2-2-2h-3c0-1.66-1.34-3-3-3S9,3.34,9,5H6C4.9,5,4,5.9,4,7v2c-1.66,0-3,1.34-3,3c0,1.66,1.34,3,3,3v4 c0,1.1,0.9,2,2,2h12c1.1,0,2-0.9,2-2v-4c1.66,0,3-1.34,3-3C23,10.34,21.66,9,20,9z M7.5,11.5C7.5,10.67,8.17,10,9,10 s1.5,0.67,1.5,1.5S9.83,13,9,13S7.5,12.33,7.5,11.5z M16,17H8v-2h8V17z M15,13c-0.83,0-1.5-0.67-1.5-1.5S14.17,10,15,10 s1.5,0.67,1.5,1.5S15.83,13,15,13z"/></g></svg>'
    } else if (emotionLower.includes('disgust')) {
        icon = '<svg xmlns="http://www.w3.org/2000/svg" enable-background="new 0 0 24 24" height="48px" viewBox="0 0 24 24" width="48px" fill="#000000"><g><rect fill="none" height="24" width="24" y="0"/></g><g><path d="M20,9V7c0-1.1-0.9-2-2-2h-3c0-1.66-1.34-3-3-3S9,3.34,9,5H6C4.9,5,4,5.9,4,7v2c-1.66,0-3,1.34-3,3c0,1.66,1.34,3,3,3v4 c0,1.1,0.9,2,2,2h12c1.1,0,2-0.9,2-2v-4c1.66,0,3-1.34,3-3C23,10.34,21.66,9,20,9z M7.5,11.5C7.5,10.67,8.17,10,9,10 s1.5,0.67,1.5,1.5S9.83,13,9,13S7.5,12.33,7.5,11.5z M16,17H8v-2h8V17z M15,13c-0.83,0-1.5-0.67-1.5-1.5S14.17,10,15,10 s1.5,0.67,1.5,1.5S15.83,13,15,13z"/></g></svg>'
    } else if (emotionLower.includes('neutral')) {
        icon = '<svg xmlns="http://www.w3.org/2000/svg" enable-background="new 0 0 20 20" height="48px" viewBox="0 0 20 20" width="48px" fill="#000000"><g><rect fill="none" height="20" width="20"/></g><g><g><path d="M10,4v4h1h1v1v0.57C12.27,9.23,12.84,9,13.5,9c0.92,0,1.67,0.45,1.67,1H12v1.65 c0.46,0.22,0.96,0.35,1.5,0.35c1.93,0,3.5-1.57,3.5-3.5V4H10z M12,7.75c-0.41,0-0.75-0.34-0.75-0.75c0-0.41,0.34-0.75,0.75-0.75 S12.75,6.59,12.75,7C12.75,7.41,12.41,7.75,12,7.75z M15,7.75c-0.41,0-0.75-0.34-0.75-0.75c0-0.41,0.34-0.75,0.75-0.75 S15.75,6.59,15.75,7C15.75,7.41,15.41,7.75,15,7.75z" fill-rule="evenodd"/><path d="M4,13.5C4,15.43,5.57,17,7.5,17s3.5-1.57,3.5-3.5V9H4V13.5z M7.5,15 c-0.92,0-1.67-0.45-1.67-1h3.33C9.17,14.55,8.42,15,7.5,15z M9,11.25c0.41,0,0.75,0.34,0.75,0.75c0,0.41-0.34,0.75-0.75,0.75 S8.25,12.41,8.25,12C8.25,11.59,8.59,11.25,9,11.25z M6,11.25c0.41,0,0.75,0.34,0.75,0.75c0,0.41-0.34,0.75-0.75,0.75 S5.25,12.41,5.25,12C5.25,11.59,5.59,11.25,6,11.25z" fill-rule="evenodd"/></g></g></svg>'
    }
    const wrap = (svg) =>
    `<div style="display: inline-flex; align-items: center; gap: 4px;">
        ${svg}
        <small style="font-size: 0.7em; opacity: 0.7;">${emotion}</small>
    </div>`;
    // Update the CSS to use this icon
    return wrap(icon);
}

// Copy transcription to clipboard
async function copyTranscription() {
    try {
        const text = transcriptionText.textContent;
        if (text && text !== 'No transcription available' && text !== 'Loading transcription...') {
            await navigator.clipboard.writeText(text);
            showToast('Transcription copied to clipboard!');
        }
    } catch (error) {
        console.error('Error copying to clipboard:', error);
        showToast('Failed to copy to clipboard', 'error');
    }
}

// Generate summary
async function generateSummary() {
    try {
        // Send message to background script
        chrome.runtime.sendMessage({
            action: 'generateSummary'
        });
        
        // Open summary tab
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
        
    } catch (error) {
        console.error('Error generating summary:', error);
        showToast('Failed to generate summary', 'error');
    }
}

// Show error state
function showError() {
    transcriptionText.textContent = 'Error loading results';
    transcriptionText.classList.add('error');
    emotionResult.textContent = 'Error loading emotion analysis';
    emotionResult.classList.add('error');
}

// Show toast message
function showToast(message, type = 'success') {
    toastMessage.textContent = message;
    toast.className = `toast ${type}`;
    
    setTimeout(() => {
        toast.classList.add('hidden');
    }, 3000);
}