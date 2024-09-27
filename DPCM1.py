import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import soundfile as sf

# Read the voice message
sample_rate, original_signal = wavfile.read('voicedc.wav')

# Print the actual sampling rate of the recorded voice
print(f"The original sampling rate of the recorded voice is {sample_rate} Hz.")

# Ask the user for a new sampling rate, provide a hint for common values
try:
    user_sampling_rate = int(input("Enter the new sampling rate for the audio (e.g., 44100, 48000) or press Enter to keep the original: ") or sample_rate)
except ValueError:
    user_sampling_rate = sample_rate  # Default to original sample rate if input is invalid
    print("Invalid input. Using the original sampling rate.")

# Convert to mono if stereo
if original_signal.ndim > 1:
    original_signal = original_signal.mean(axis=1)

# DPCM Encoding
def dpcm_encode(signal):
    encoded_signal = np.zeros(len(signal))
    prediction = signal[0]

    for i in range(len(signal)):
        encoded_signal[i] = signal[i] - prediction
        prediction = signal[i]

    return encoded_signal

# Save encoded signal to text file
def save_encoded_signal(encoded_signal, filename):
    binary_data = np.array(encoded_signal > 0, dtype=int)  # Simplistic binary encoding
    binary_string = ','.join(map(str, binary_data))
    
    with open(filename, 'w') as f:
        f.write(binary_string)

encoded_signal = dpcm_encode(original_signal)
save_encoded_signal(encoded_signal, 'encoded_signal.txt')

def dpcm_decode(encoded_signal):
    # Initialize the decoded signal and the prediction
    decoded_signal = np.zeros(len(encoded_signal))
    prediction = encoded_signal[0]  # Initial prediction (same as first sample)

    # Decode using DPCM
    for i in range(len(encoded_signal)):
        decoded_signal[i] = prediction + encoded_signal[i]
        prediction = decoded_signal[i]  # Update prediction

    return decoded_signal

# Decode the signal
decoded_signal = dpcm_decode(encoded_signal)

# Debugging prints
print("Original signal length:", len(original_signal))
print("Encoded signal length:", len(encoded_signal))
print("Decoded signal length:", len(decoded_signal))
print("Sample rate used for processing:", user_sampling_rate)

def main(input_file, output_file, user_sampling_rate):
    # Read the audio signal
    audio_signal, sample_rate = sf.read(input_file)

    # Ensure audio is mono
    if audio_signal.ndim > 1:
        audio_signal = audio_signal[:, 0]

    # DPCM Encode
    encoded_signal = dpcm_encode(audio_signal)

    # DPCM Decode
    decoded_signal = dpcm_decode(encoded_signal)

    # Write the decoded audio to a file with the user-defined or original sampling rate
    sf.write(output_file, decoded_signal, user_sampling_rate)

# Define start and end times in microseconds
start_time_us = 200000  # Start time in microseconds
end_time_us = 500000    # End time in microseconds

# Convert times to samples
start_sample = int(start_time_us * user_sampling_rate / 1_000_000)
end_sample = int(end_time_us * user_sampling_rate / 1_000_000)

# Ensure the samples are within bounds
if start_sample < 0:
    start_sample = 0
if end_sample > len(original_signal):
    end_sample = len(original_signal)

# Slice the signals
original_signal_segment = original_signal[start_sample:end_sample]
encoded_signal_segment = encoded_signal[start_sample:end_sample]
decoded_signal_segment = decoded_signal[start_sample:end_sample]

# Downsampling factor (choose a suitable factor)
downsample_factor = 100

# Downsample the segments
original_signal_segment = original_signal_segment[::downsample_factor]
encoded_signal_segment = encoded_signal_segment[::downsample_factor]
decoded_signal_segment = decoded_signal_segment[::downsample_factor]

# Debugging prints for normalization
print("Decoded signal range before normalization:", np.min(decoded_signal_segment), np.max(decoded_signal_segment))

# Normalize the decoded signal to fit in int16 range
decoded_signal_segment -= np.mean(decoded_signal_segment)  # Center around zero
max_val = np.max(np.abs(decoded_signal_segment))  # Get the max absolute value for scaling

# Scale to fit in int16 range
if max_val > 0:  # Avoid division by zero
    decoded_signal_int16 = np.int16((decoded_signal_segment / max_val) * 32767)  # Scale to int16 range
else:
    decoded_signal_int16 = np.zeros_like(decoded_signal_segment, dtype=np.int16)

# Save the normalized decoded signal to a new WAV file
# wavfile.write('decoded_v1.wav', user_sampling_rate, decoded_signal_int16)

# Create time arrays for plotting in nanoseconds
time_original = np.linspace(start_time_us, end_time_us, num=len(original_signal_segment))
time_encoded = np.linspace(start_time_us, end_time_us, num=len(encoded_signal_segment))
time_decoded = np.linspace(start_time_us, end_time_us, num=len(decoded_signal_segment))

# Plotting
plt.figure(figsize=(15, 10))

plt.subplot(3, 1, 1)
plt.title("Original Signal (Segment)")
plt.plot(time_original, original_signal_segment)
plt.xlabel("Time (nanoseconds)")
plt.ylabel("Amplitude")

plt.subplot(3, 1, 2)
plt.title("Encoded Signal (DPCM) (Segment)")
plt.plot(time_encoded, encoded_signal_segment)
plt.xlabel("Time (nanoseconds)")
plt.ylabel("Encoded Value")

plt.subplot(3, 1, 3)
plt.title("Decoded Signal (Segment)")
plt.plot(time_decoded, decoded_signal_segment)
plt.xlabel("Time (nanoseconds)")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()

if __name__ == "__main__":
    input_file = 'voicedc.wav'  # Replace with your input audio file
    output_file = r'C:\\Users\\amank\\Desktop\\dc2\\output_audio.wav'  # Desired output file name
    main(input_file, output_file, user_sampling_rate)
