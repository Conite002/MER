import torch
import librosa
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model


def load_wav2vec_model(model_name="facebook/wav2vec2-base"):
    """
    Load the Wav2Vec model and processor.
    Args:
        model_name (str): Hugging Face model name for Wav2Vec.

    Returns:
        tuple: (processor, model)
    """
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name)
    model.eval()
    return processor, model


def preprocess_audio_for_model(audio_path, processor, model, target_sample_rate=16000, target_duration=8.0):
    """
    Preprocess audio and extract embeddings using a specified model.

    Args:
        audio_path (str): Path to the audio file.
        processor: Hugging Face processor for the model.
        model: Hugging Face model.
        target_sample_rate (int): Sampling rate for the audio.
        target_duration (float): Target audio duration in seconds.

    Returns:
        np.ndarray: Extracted audio embeddings.
    """
    try:
        waveform, _ = librosa.load(audio_path, sr=target_sample_rate)
        max_length = int(target_sample_rate * target_duration)
        if len(waveform) > max_length:
            waveform = waveform[:max_length]
        else:
            waveform = np.pad(waveform, (0, max_length - len(waveform)), mode='constant')

        inputs = processor(waveform, sampling_rate=target_sample_rate, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

        return embeddings
    except Exception as e:
        print(f"Error processing audio: {e}")
        return None
