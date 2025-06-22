import librosa
import numpy as np

def getMELspectrogram(audio, sample_rate):
    """
    Compute the MEL spectrogram of an audio signal.

    Parameters:
    - audio: Audio time series
    - sample_rate: Sampling rate of the audio

    Returns:
    - mel_spec_db: MEL spectrogram in decibel scale
    """
    mel_spec = librosa.feature.melspectrogram(
        y=audio, 
        sr=sample_rate, 
        n_fft=1024, 
        win_length=512, 
        window='hamming', 
        hop_length=256, 
        n_mels=128, 
        fmax=sample_rate / 2
    )
    
    # Convert power spectrogram to decibel scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    return mel_spec_db

def splitIntoChunks(mel_spec, win_size, stride):
    """
    Split the MEL spectrogram into chunks.

    Parameters:
    - mel_spec: MEL spectrogram
    - win_size: Window size for each chunk
    - stride: Step size to move the window (overlap control)

    Returns:
    - A stack of chunks along the time axis
    """
    t = mel_spec.shape[1]
    
    # Calculate number of chunks based on stride
    num_of_chunks = int(t / stride)

    chunks = []

    # Create chunks from the spectrogram
    for i in range(num_of_chunks):
        chunk = mel_spec[:, i * stride:i * stride + win_size]
        
        # Only append chunks of the correct size
        if chunk.shape[1] == win_size:
            chunks.append(chunk)

    return np.stack(chunks, axis=0)
    # t = mel_spec.shape[1]
    # num_chunks = max(1, (t - win_size) // stride + 1)
    
    # chunks = []
    # for i in range(num_chunks):
    #     start = i * stride
    #     end = start + win_size
    #     if end <= t:
    #         chunk = mel_spec[:, start:end]
    #         chunks.append(chunk)
    
    # if not chunks:  # Fallback if no valid chunks
    #     # Pad or truncate to win_size
    #     chunk = np.zeros((mel_spec.shape[0], win_size))
    #     chunk[:, :min(win_size, mel_spec.shape[1])] = mel_spec[:, :min(win_size, mel_spec.shape[1])]
    #     chunks.append(chunk)
        
    # return np.stack(chunks, axis=0)

    
def addNoise(signal, snr_low, snr_high):
    """Add white Gaussian noise to signal"""
    noise = np.random.normal(size=len(signal))
    
    # Calculate signal and noise power
    signal_power = np.sum(signal ** 2) / len(signal)
    noise_power = np.sum(noise ** 2) / len(noise)
    
    # Random SNR
    target_snr = np.random.randint(snr_low, snr_high)
    
    # Calculate scaling factor
    k = np.sqrt((signal_power / noise_power) * 10 ** (- target_snr / 10))
    
    return signal + k * noise