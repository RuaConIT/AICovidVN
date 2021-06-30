from typing import List, Optional
import numpy
from dataclasses import dataclass, field


@dataclass
class AudioFeatures:
    wav: numpy.ndarray = field(default=None)
    mfccs: numpy.ndarray = field(default=None)
    velocity_of_mfccs: numpy.ndarray = field(default=None)
    acceleration_of_mfccs: numpy.ndarray = field(default=None)

    def __repr__(self) -> str:
        repr_strs = []
        for attr_name, attr_value in vars(self).items():
            repr_str = None
            if attr_value is not None:
                if isinstance(attr_value, numpy.ndarray):
                    shape = attr_value.shape
                    dtype = attr_value.dtype
                    repr_str = f'{attr_name}=array(shape={shape}, dtype={dtype})'
                repr_strs.append(repr_str)

        return f'AudioFeatures({", ".join(repr_strs)})'


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