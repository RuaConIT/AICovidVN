from src.dataclass import AudioData
import librosa
import numpy
import librosa
from typing import List
from src.preprocess import normalize_audio
import numpy as np
import warnings


def extract_deltas_of_mfccs(data_list: List[AudioData]) -> List[AudioData]:
    """
    Extract velocity (delta) and acceleration (delta-delta) of MFCC
    """
    result = []
    for audio_data in data_list:
        mfccs = audio_data.features.mfccs
        if mfccs is None:
            warnings.warn("MFCCS is None. Ignoring.", RuntimeWarning)
        else:
            first_col = numpy.reshape(numpy.zeros_like(mfccs[:, 0]), (-1, 1))
            velocity = numpy.diff(mfccs, prepend=first_col)
            acceleration = numpy.diff(velocity, prepend=first_col)
            audio_data.features.velocity_of_mfccs = velocity
            audio_data.features.acceleration_of_mfccs = acceleration
            result.append(audio_data)

    return result


def calculator(data_list: List[AudioData],
               **librosa_mfcc_kwargs) -> List[AudioData]:
    result = []
    for data in data_list:
        sr = data.sampling_rate
        y = data.features.wav
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        spec_cent = librosa.feature.spectral_centroid(y=y,
                                                      sr=sr,
                                                      **librosa_mfcc_kwargs)
        # rmse = librosa.feature.rmse(y=y)
        spec_bw = librosa.feature.spectral_bandwidth(y=y,
                                                     sr=sr,
                                                     **librosa_mfcc_kwargs)
        rolloff = librosa.feature.spectral_rolloff(y=y,
                                                   sr=sr,
                                                   **librosa_mfcc_kwargs)
        zcr = librosa.feature.zero_crossing_rate(y, **librosa_mfcc_kwargs)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, **librosa_mfcc_kwargs)
        to_append = f'{np.mean(chroma_stft)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
        for i in mfcc:
            to_append += f' {np.mean(i)}'
        data.features.allfeat = np.array(str(to_append).split(' ')).astype(
            np.float)
    return data_list


def kurtosisOfMfccs(data_list: List[AudioData]) -> numpy.ndarray:
    kurtosis_of_Mfccs = []
    for audio_data in data_list:
        kurtosis_of_Mfccs.append(
            kurtosis(audio_data.features.mfccs.reshape(-1)))
    return numpy.array(kurtosis_of_Mfccs)


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
    else:
        longest_shape = max_mfccs_length

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