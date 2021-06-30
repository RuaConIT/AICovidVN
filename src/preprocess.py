from src.dataclass import AudioData, Sample
import numpy


def normalize_audio(audio_data: AudioData) -> AudioData:
    wav = audio_data.features.wav
    thres = 0.005
    rv_result = []
    wav = numpy.array([x for x in wav if str(x) != 'nan'])
    for i in wav:
        if numpy.abs(i) >= thres:
            rv_result.append(i)
    if len(rv_result) > 0:
        wav = numpy.array(rv_result)
    if len(rv_result) > 0:
        max_amp = numpy.max(wav)
        wav = 0.9 * (wav / abs(max_amp))
    audio_data.features.wav = wav
    return audio_data
