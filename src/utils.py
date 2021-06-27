import os
import seaborn as sns
import numpy
from typing import List, Union
from glob import glob
import librosa
import librosa.display
import matplotlib.pyplot as plt
from src.dataclass import AudioData, AudioFeatures, DataSet


def plot_wav(audio_data: AudioData) -> None:
    amp = audio_data.features.wav
    sr = audio_data.sampling_rate
    time_stamp = numpy.linspace(0, len(amp) / sr, num=len(amp))
    ax = sns.lineplot(x=time_stamp, y=amp)
    ax.set(xlabel='time', ylabel='amplitude')
    plt.show()


def get_path(*args: str) -> str:
    return os.path.join(*args)


def load_audio_data(path: str,
                    load_from_npy=False,
                    limit=None) -> List[AudioData]:
    if load_from_npy:
        return numpy.load(path, allow_pickle=True).tolist()

    data = []
    for idx, file in enumerate(glob(os.path.join(path, '*.wav'))):
        data_dict = {}
        file_name = file.split(os.path.sep)[-1]
        (amp, sr) = librosa.load(file)
        data_dict['name'] = file_name
        data_dict['features'] = AudioFeatures(wav=amp)
        data_dict['sampling_rate'] = sr
        audio_data = AudioData(**data_dict)
        data.append(audio_data)
        if limit and idx + 1 == limit:
            break

    return data


def display_mfccs(data: AudioData) -> None:
    fig, ax = plt.subplots(nrows=1, ncols=1)
    img = librosa.display.specshow(data.features.mfccs,
                                   y_axis='log',
                                   sr=data.sampling_rate,
                                   x_axis='time')
    fig.colorbar(img, ax=ax, format='%+2.f dB')


def save_data(output_file: str, data_list: List[Union[AudioData,
                                                      DataSet]]) -> None:
    numpy.save(output_file, numpy.array(data_list, dtype='object'))


def load_dataset(path) -> DataSet:
    return numpy.load(path, allow_pickle=True)[()]