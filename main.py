import pvporcupine
import pyaudio
from dotenv import load_dotenv
import os
import struct
import wave
from faster_whisper import WhisperModel
import numpy as np

load_dotenv()
TOKEN = os.getenv("PORCUPINE_TOKEN")
SYSTEM = os.getenv("SYSTEM")
if SYSTEM == "mac":
    porcupine = pvporcupine.create(
        keywords=["hey lampo"], 
        access_key=TOKEN,
        keyword_paths=['mac/hey-lampo_it.ppn'],
        model_path='porcupine_params_it.pv',
    )
else:
    porcupine = pvporcupine.create(
        keywords=["blueberry"], 
        access_key=TOKEN,
    )


pa = pyaudio.PyAudio()
stream = pa.open(rate=porcupine.sample_rate,
                 channels=1,
                 format=pyaudio.paInt16,
                 input=True,
                 frames_per_buffer=porcupine.frame_length)

print("Listening for wake word...")

def transcribe_audio(filename="command.wav"):
    print("Transcribing...")
    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments, info = model.transcribe(filename, language="it", beam_size=5)
    full_text = ""

    for segment in segments:
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
        full_text += segment.text.strip() + " "

    full_text = full_text.strip()
    
    if not full_text:
        print("No transcription result found.")
        return None

    print(full_text)
    return full_text


def record_audio(filename="command.wav", record_seconds=5):
    print("Recording command...")
    frames = []

    for _ in range(0, int(porcupine.sample_rate / porcupine.frame_length * record_seconds)):
        data = stream.read(porcupine.frame_length)
        frames.append(data)

    # Save to a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
    wf.setframerate(porcupine.sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()
    print(f"Saved to {filename}")



def record_until_silence(filename="command.wav", max_record_seconds=10, silence_threshold=500, silence_duration=1.0):
    print("Recording until silence...")

    frames = []
    silent_chunks = 0
    silent_chunk_limit = int((silence_duration * porcupine.sample_rate) / porcupine.frame_length)
    max_chunks = int((max_record_seconds * porcupine.sample_rate) / porcupine.frame_length)

    for i in range(max_chunks):
        data = stream.read(porcupine.frame_length, exception_on_overflow=False)
        frames.append(data)

        pcm = np.frombuffer(data, dtype=np.int16)
        volume = np.abs(pcm).mean()

        if volume < silence_threshold:
            silent_chunks += 1
        else:
            silent_chunks = 0

        if silent_chunks > silent_chunk_limit:
            print("Silence detected, stopping...")
            break

    # Save to WAV
    wf = wave.open(filename, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
    wf.setframerate(porcupine.sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()
    print(f"Saved to {filename}")

try:
    while True:
        data = stream.read(porcupine.frame_length, exception_on_overflow=False)
        pcm = struct.unpack_from("h" * porcupine.frame_length, data)

        if porcupine.process(pcm) >= 0:
            print("Wake word detected!")
            record_until_silence("command.wav")
            command_text = transcribe_audio("command.wav")
            print(command_text)

finally:
    stream.stop_stream()
    stream.close()
    pa.terminate()
    porcupine.delete()
