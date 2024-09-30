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

# Save encoded signal in groups of two bits
def save_encoded_signal(encoded_signal, filename):
    # Convert to binary: 1 for positive, 0 for non-positive
    binary_data = np.array(encoded_signal > 0, dtype=int)

    # Group bits into pairs
    grouped_bits = []
    for i in range(0, len(binary_data) - 1, 2):
        # Concatenate two bits into a string
        bit_pair = f"{binary_data[i]}{binary_data[i+1]}"
        grouped_bits.append(bit_pair)

    # If there's an odd number of bits, append the last one as is
    if len(binary_data) % 2 != 0:
        grouped_bits.append(str(binary_data[-1]))

    # Join the grouped bits with commas
    grouped_bit_string = ','.join(grouped_bits)

    # Save the grouped bits into the file
    with open(filename, 'w') as f:
        f.write(grouped_bit_string)

# DPCM Decoding
def dpcm_decode(encoded_signal):
    # Initialize the decoded signal and the prediction
    decoded_signal = np.zeros(len(encoded_signal))
    prediction = encoded_signal[0]  # Initial prediction (same as first sample)

    # Decode using DPCM
    for i in range(len(encoded_signal)):
        decoded_signal[i] = prediction + encoded_signal[i]
        prediction = decoded_signal[i]  # Update prediction

    return decoded_signal

# Perform DPCM encoding
encoded_signal = dpcm_encode(original_signal)

# Save encoded signal to a text file in groups of two bits separated by commas
save_encoded_signal(encoded_signal, 'encoded_signal_grouped.txt')

# Decode the signal
decoded_signal = dpcm_decode(encoded_signal)

# Debugging prints
print("Original signal length:", len(original_signal))
print("Encoded signal length:", len(encoded_signal))
print("Decoded signal length:", len(decoded_signal))
print("Sample rate used for processing:", user_sampling_rate)

# Define start and end times in seconds
start_time_s = 0  # Start time in seconds
end_time_s = 0.8  # End time in seconds

# Convert times to samples
start_sample = int(start_time_s * user_sampling_rate)
end_sample = int(end_time_s * user_sampling_rate)

# Ensure the samples are within bounds
if start_sample < 0:
    start_sample = 0
if end_sample > len(original_signal):
    end_sample = len(original_signal)

# Slice the signals
original_signal_segment = original_signal[start_sample:end_sample]
encoded_signal_segment = encoded_signal[start_sample:end_sample]
decoded_signal_segment = decoded_signal[start_sample:end_sample]

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

# Create time arrays for plotting in seconds
time_original = np.linspace(start_time_s, end_time_s, num=len(original_signal_segment))
time_decoded = np.linspace(start_time_s, end_time_s, num=len(decoded_signal_segment))

# Plotting with compressed layout
plt.figure(figsize=(10, 6))  # Adjusted for a more compact layout

plt.subplot(2, 1, 1)
plt.title("Original Signal (Segment)")
plt.plot(time_original, original_signal_segment)
plt.xlabel("Time (seconds)")  # Time in seconds
plt.ylabel("Amplitude")

plt.subplot(2, 1, 2)
plt.title("Decoded Signal (Segment)")
plt.plot(time_decoded, decoded_signal_segment)
plt.xlabel("Time (seconds)")  # Time in seconds
plt.ylabel("Amplitude")

plt.tight_layout()  # Ensure no overlap between subplots
plt.show()

# Main function for processing the full audio file
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

if __name__ == "__main__":
    input_file = 'voicedc.wav'  # Replace with your input audio file
    output_file = r'C:\\Users\\amank\\Desktop\\dc2\\output_audio.wav'  # Desired output file name
    main(input_file, output_file, user_sampling_rate)
