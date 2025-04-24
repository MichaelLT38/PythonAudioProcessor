import numpy as np
import scipy.io.wavfile as wav
import tkinter as tk
from tkinter import filedialog

import warnings
warnings.simplefilter("ignore", wav.WavFileWarning)

def analyze_wav(filename, bp_low=250, bp_high=4000, hp_cutoff=650, lp_cutoff=3500, alpha=8.5, compression_factor=4.0, amplification_factor=200.0):
    # Read the WAV file
    rate, data = wav.read(filename)
    
    # If stereo, take the average of the two channels
    if len(data.shape) == 2:
        data = np.mean(data, axis=1)
    
    # Apply Fast Fourier Transform
    spectrum = np.fft.fft(data)
    frequencies = np.fft.fftfreq(len(spectrum), d=1/rate)
    
    # Bandpass filter to focus on higher frequencies of the young female voice
    spectrum[(frequencies < bp_low) | (frequencies > bp_high)] = 0
    
    # High-pass filter to attenuate lower frequencies
    attenuation_factor = 0.5
    spectrum[(frequencies < hp_cutoff)] *= attenuation_factor
    
    # Low-pass filter to attenuate very high frequencies
    attenuation_factor_high = 0.5
    spectrum[(frequencies > lp_cutoff)] *= attenuation_factor_high
    
    # Adaptive Noise Reduction using Spectral Subtraction
    noise_estimation = np.median(np.abs(spectrum))
    spectrum[np.abs(spectrum) < alpha * noise_estimation] -= noise_estimation
    
    # Perform inverse FFT to get the time-domain signal
    new_data = np.fft.ifft(spectrum).real
    
    # Dynamic range compression
    new_data = np.sign(new_data) * (np.abs(new_data) ** (1.0 / compression_factor))
    
    # Amplify the audio
    new_data = new_data * amplification_factor
    new_data = np.clip(new_data, -32768, 32767)  # Clip values to int16 range
    
    # Create a filename with the settings
    output_filename = f"reconstructed_bp_{bp_low}-{bp_high}_hp_{hp_cutoff}_lp_{lp_cutoff}_alpha_{alpha}_comp_{compression_factor}_amp_{amplification_factor}.wav"
    
    # Save the new audio data to the WAV file
    wav.write(output_filename, rate, new_data.astype(np.int16))
    print(f"New audio saved to {output_filename}")

def select_file():
    global filename  # Declare filename as global so it can be accessed elsewhere
    filename = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
    file_label.config(text=f"Selected File: {filename}")

def process_audio():
    # Get values from sliders
    bp_low = bp_low_slider.get()
    bp_high = bp_high_slider.get()
    hp_cutoff = hp_slider.get()
    lp_cutoff = lp_slider.get()
    alpha_val = alpha_slider.get()
    compression_val = compression_slider.get()
    amplification_val = amplification_slider.get()
    
    # Call your function
    analyze_wav(filename, bp_low, bp_high, hp_cutoff, lp_cutoff, alpha_val, compression_val, amplification_val)
    output_label.config(text="Processing complete!")

# Create main window
root = tk.Tk()
root.title("Audio Processor")

# Add a button to select a .wav file
select_button = tk.Button(root, text="Select WAV File", command=select_file)
select_button.pack()

# Add a label to display the selected file's name
file_label = tk.Label(root, text="No file selected")
file_label.pack()

# Add sliders for settings
bp_low_slider = tk.Scale(root, from_=0, to_=500, orient=tk.HORIZONTAL, label="BP Low")
bp_low_slider.pack()

bp_high_slider = tk.Scale(root, from_=500, to_=5000, orient=tk.HORIZONTAL, label="BP High")
bp_high_slider.pack()

hp_slider = tk.Scale(root, from_=0, to_=1000, orient=tk.HORIZONTAL, label="HP Cutoff")
hp_slider.pack()

lp_slider = tk.Scale(root, from_=1000, to_=5000, orient=tk.HORIZONTAL, label="LP Cutoff")
lp_slider.pack()

alpha_slider = tk.Scale(root, from_=0, to_=10, resolution=0.1, orient=tk.HORIZONTAL, label="Alpha")
alpha_slider.pack()

compression_slider = tk.Scale(root, from_=1, to_=10, resolution=0.1, orient=tk.HORIZONTAL, label="Compression")
compression_slider.pack()

amplification_slider = tk.Scale(root, from_=1, to_=500, resolution=1, orient=tk.HORIZONTAL, label="Amplification")
amplification_slider.pack()

# Add a button to start processing
process_button = tk.Button(root, text="Process Audio", command=process_audio)
process_button.pack()

# Add a label for output messages
output_label = tk.Label(root, text="")
output_label.pack()

# Run the GUI loop
root.mainloop()
