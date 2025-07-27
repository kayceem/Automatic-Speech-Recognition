// DOM elements
const timestampEl = document.getElementById('timestamp');
const summaryText = document.getElementById('summaryText');
const transcriptionText = document.getElementById('transcriptionText');
const copySummaryBtn = document.getElementById('copySummary');
const copyTranscriptionBtn = document.getElementById('copyTranscription');
const refreshSummaryBtn = document.getElementById('refreshSummary');
const backToResultsBtn = document.getElementById('backToResultsBtn');
const newRecordingBtn = document.getElementById('newRecordingBtn');
const toast = document.getElementById('toast');
const toastMessage = document.getElementById('toastMessage');
const loadingOverlay = document.getElementById('loadingOverlay');

// Initialize page
document.addEventListener('DOMContentLoaded', () => {
    loadContent();
    setupEventListeners();
});

// Setup event listeners
function setupEventListeners() {
    copySummaryBtn.addEventListener('click', () => copyToClipboard(summaryText, 'Summary'));
    copyTranscriptionBtn.addEventListener('click', () => copyToClipboard(transcriptionText, 'Transcription'));
    refreshSummaryBtn.addEventListener('click', regenerateSummary);
    backToResultsBtn.addEventListener('click', goBackToResults);
    newRecordingBtn.addEventListener('click', startNewRecording);
}

// Load content from storage
async function loadContent() {
    try {
        const result = await chrome.storage.local.get([
            'summary', 
            'transcription', 
            'lastRecordTime'
        ]);
        
        // Update timestamp
        if (result.lastRecordTime) {
            timestampEl.textContent = `Generated: ${new Date().toLocaleString()}`;
        } else {
            timestampEl.textContent = 'No timestamp available';
        }
        
        // Update summary
        if (result.summary) {
            summaryText.textContent = result.summary;
            summaryText.classList.remove('loading', 'error');
        } else {
            summaryText.textContent = 'No summary available. Click refresh to generate.';
            summaryText.classList.add('loading');
        }
        
        // Update transcription
        if (result.transcription) {
            transcriptionText.textContent = result.transcription;
            transcriptionText.classList.remove('loading', 'error');
        } else {
            transcriptionText.textContent = 'No transcription available';
            transcriptionText.classList.add('loading');
        }
        
    } catch (error) {
        console.error('Error loading content:', error);
        showError();
    }
}

// Copy content to clipboard
async function copyToClipboard(element, type) {
    try {
        const text = element.textContent;
        if (text && !text.includes('No ') && !text.includes('Loading')) {
            await navigator.clipboard.writeText(text);
            showToast(`${type} copied to clipboard!`);
        } else {
            showToast(`No ${type.toLowerCase()} to copy`, 'error');
        }
    } catch (error) {
        console.error('Error copying to clipboard:', error);
        showToast('Failed to copy to clipboard', 'error');
    }
}

// Regenerate summary
async function regenerateSummary() {
    try {
        showLoading(true);
        
        // Send message to background script to regenerate summary
        chrome.runtime.sendMessage({
            action: 'generateSummary'
        });
        
        // Listen for completion
        const handleMessage = (message) => {
            if (message.action === 'summaryComplete') {
                chrome.runtime.onMessage.removeListener(handleMessage);
                setTimeout(() => {
                    loadContent();
                    showLoading(false);
                    showToast('Summary regenerated successfully!');
                }, 1000);
            }
        };
        
        chrome.runtime.onMessage.addListener(handleMessage);
        
        // Timeout after 30 seconds
        setTimeout(() => {
            chrome.runtime.onMessage.removeListener(handleMessage);
            showLoading(false);
            showToast('Summary generation timeout', 'error');
        }, 30000);
        
    } catch (error) {
        console.error('Error regenerating summary:', error);
        showLoading(false);
        showToast('Failed to regenerate summary', 'error');
    }
}

// Go back to results page
async function goBackToResults() {
    try {
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
        
        // Close current tab
        const currentTab = await chrome.tabs.getCurrent();
        if (currentTab) {
            chrome.tabs.remove(currentTab.id);
        }
        
    } catch (error) {
        console.error('Error going back to results:', error);
    }
}

// Start new recording
async function startNewRecording() {
    try {
        // Clear previous data
        await chrome.storage.local.set({
            hasTranscription: false,
            hasSummary: false,
            transcription: null,
            emotion: null,
            summary: null
        });
        
        // Send message to start recording
        chrome.runtime.sendMessage({
            action: 'startRecording'
        });
        
        showToast('New recording started!');
        
        // Close current tab after a short delay
        setTimeout(() => {
            chrome.tabs.getCurrent().then(tab => {
                if (tab) {
                    chrome.tabs.remove(tab.id);
                }
            });
        }, 1500);
        
    } catch (error) {
        console.error('Error starting new recording:', error);
        showToast('Failed to start new recording', 'error');
    }
}

// Show/hide loading overlay
function showLoading(show) {
    if (show) {
        loadingOverlay.classList.remove('hidden');
    } else {
        loadingOverlay.classList.add('hidden');
    }
}

// Show error state
function showError() {
    summaryText.textContent = 'Error loading summary';
    summaryText.classList.add('error');
    transcriptionText.textContent = 'Error loading transcription';
    transcriptionText.classList.add('error');
}

// Show toast message
function showToast(message, type = 'success') {
    toastMessage.textContent = message;
    toast.className = `toast ${type}`;
    
    setTimeout(() => {
        toast.classList.add('hidden');
    }, 3000);
}

// Listen for messages from background script
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.action === 'summaryComplete') {
        loadContent();
        showLoading(false);
    }
});