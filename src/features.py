from src.dataclass import AudioData
import librosa
import numpy
import librosa
from typing import List
from scipy.stats import kurtosis


def extract_mfccs(data_list: List[AudioData],
                  pad_mfccs=True,
                  pad_mode='constant',
                  pad_constant_values=0,
                  max_mfccs_length=None,
                  **librosa_mfcc_kwargs) -> List[AudioData]:
    for data in data_list:
        amp = data.features.wav
        sr = data.sampling_rate
        mfccs = librosa.feature.mfcc(y=amp, sr=sr, **librosa_mfcc_kwargs)
        data.features.mfccs = mfccs

    if not pad_mfccs:
        return data_list
    else:
        return padding_mfccs(data_list,
                            pad_mode=pad_mode,
                            pad_constant_values=pad_constant_values,
                            max_mfccs_length=max_mfccs_length)


def padding_mfccs(data_list: List[AudioData],
                  pad_mode='constant',
                  pad_constant_values=0,
                  max_mfccs_length=None) -> List[AudioData]:
    longest_shape = -1
    if not max_mfccs_length:
        for data in data_list:
            mfccs = data.features.mfccs
            if mfccs.shape[1] > longest_shape:
                longest_shape = mfccs.shape[1]

    for data in data_list:
        mfccs = data.features.mfccs
        if mfccs.shape[1] < longest_shape:
            mfccs = numpy.pad(mfccs, [(0, 0),
                                      (0, longest_shape - mfccs.shape[1])],
                              mode=pad_mode,
                              constant_values=pad_constant_values)
        elif mfccs.shape[1] > longest_shape:
            mfccs = mfccs[:, :longest_shape]
        data.features.mfccs = mfccs

    return data_list


def kurtosisOfMfccs(data_list: List[AudioData]) -> numpy.ndarray:
    kurtosis_of_Mfccs = []
    for audio_data in data_list:
        kurtosis_of_Mfccs.append(kurtosis(audio_data.features.mfccs.reshape(-1)))
    return numpy.array(kurtosis_of_Mfccs)

