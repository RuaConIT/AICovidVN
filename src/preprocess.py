from src.dataclass import AudioData, Sample
import numpy


def normalize_audio(audio_data: AudioData) -> AudioData:
    wav = audio_data.features.wav
    max_amp = numpy.max(wav)
    wav = 0.9 * (wav / abs(max_amp if max_amp else 1.0))
    audio_data.features.wav = wav
    return audio_data
