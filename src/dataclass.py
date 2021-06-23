from typing import List, Optional
import numpy
from dataclasses import dataclass, field


@dataclass
class AudioFeatures:
    wav: numpy.ndarray = field(default=None)
    mfccs: numpy.ndarray = field(default=None)

    def __repr__(self) -> str:
        wav_shape = self.wav.shape
        wav_dtype = self.wav.dtype
        wav_str = f'array(shape={wav_shape}, dtype={wav_dtype})'

        mfccs_str = None
        if self.mfccs is not None:
            mfccs_shape = self.mfccs.shape
            mfccs_dtype = self.mfccs.dtype
            mfccs_str = f'array(shape={mfccs_shape}, dtype={mfccs_dtype})'

        return f'AudioFeatures(wav={wav_str}, mfccs={mfccs_str})'


@dataclass
class AudioData:
    name: str = field(default_factory=str)
    features: AudioFeatures = field(default_factory=AudioFeatures)
    sampling_rate: int = field(default=None)


@dataclass
class Sample:
    uuid: str = field(default=None)
    subject_gender: str = field(default=None)
    subject_age: int = field(default=None)
    assessment_result: Optional[int] = field(default=None)
    audio_data: AudioData = field(default_factory=AudioData)

    def __post_init__(self):
        self.uuid = None if self.uuid is numpy.nan else self.uuid
        self.subject_gender = None if self.subject_gender is numpy.nan else self.subject_gender
        self.subject_age = None if self.subject_age is numpy.nan else self.subject_age
        self.assessment_result = None if self.assessment_result is numpy.nan else self.assessment_result


@dataclass
class DataSet:
    samples: List[Sample] = field(default_factory=list)
    is_train: bool = field(default=True)

    def __repr__(self) -> str:
        return f'DataSet(Sample(len={len(self.samples)}), is_train={self.is_train})'