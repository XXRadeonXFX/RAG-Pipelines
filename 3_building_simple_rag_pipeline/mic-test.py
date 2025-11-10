import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import os

# --- Configuration ---
FS = 44100  # Sample rate (standard for audio)
DURATION = 5 # Recording duration in seconds
OUTPUT_FILENAME = "mic_test_recording.wav"

def test_microphone():
    """Records 5 seconds of audio and saves it to a WAV file."""
    print("="*40)
    print("🎤 Starting Microphone Test...")
    
    # 1. Check for available devices
    try:
        devices = sd.query_devices()
        input_devices = [d for d in devices if d['max_input_channels'] > 0]
        
        print(f"Found {len(input_devices)} input devices.")
        # Print default input device information
        default_input_index = sd.default.device[0]
        default_device_name = sd.query_devices(default_input_index)['name']
        print(f"Using default input device: {default_device_name}")

    except Exception as e:
        print(f"Error listing devices (you might need to install PortAudio via Homebrew): {e}")
        return

    # 2. Start Recording
    print(f"\n📢 Recording for {DURATION} seconds... Speak NOW!")
    try:
        recording = sd.rec(int(DURATION * FS), samplerate=FS, channels=1, dtype='int16')
        sd.wait()  # Wait until recording is finished
    except Exception as e:
        print(f"❌ Error during recording: {e}")
        print("Please check if your microphone is enabled in system settings.")
        return

    # 3. Save the Recording
    print(f"✅ Recording finished. Saving to {OUTPUT_FILENAME}...")
    try:
        wav.write(OUTPUT_FILENAME, FS, recording)
        print(f"🎉 Success! Audio saved to: {os.path.abspath(OUTPUT_FILENAME)}")
        print("\n*** Next Step: Play the file to check audio quality. ***")
    except Exception as e:
        print(f"❌ Error saving WAV file: {e}")

if __name__ == '__main__':
    test_microphone()