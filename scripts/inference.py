import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Load processor and model
processor = WhisperProcessor.from_pretrained("models/whisper-hindi-finetuned")
model = WhisperForConditionalGeneration.from_pretrained("models/whisper-hindi-finetuned")

# Load and preprocess audio
audio_path = "path_to_your_audio.wav"
speech_array, sampling_rate = torchaudio.load(audio_path)
inputs = processor(speech_array[0], sampling_rate=sampling_rate, return_tensors="pt")

# Generate transcription
with torch.no_grad():
    generated_ids = model.generate(inputs["input_features"])

# Decode transcription
transcription = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("Transcription:", transcription)
