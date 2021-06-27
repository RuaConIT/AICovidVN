import pandas
from src.dataclass import AudioData, DataSet, Sample
from typing import Dict, List


def adapt_to_dataset(data_frame: pandas.DataFrame,
                     audio_data_list: List[AudioData],
                     is_train=True) -> DataSet:
    audio_data_dict = _get_audio_data_dict(audio_data_list)
    samples = []
    for _, row in data_frame.iterrows():
        sample = _adapt_row_to_sample(row, is_train)
        sample.audio_data = audio_data_dict[row.file_path]
        samples.append(sample)

    return DataSet(samples, is_train)


def _adapt_row_to_sample(row: pandas.Series, is_train) -> Sample:
    return Sample(
        uuid=row.uuid,
        subject_gender=row.subject_gender,
        subject_age=row.subject_age,
        assessment_result=row.assessment_result if is_train else None)


def _get_audio_data_dict(
        audio_data_list: List[AudioData]) -> Dict[str, AudioData]:
    return {audio_data.name: audio_data for audio_data in audio_data_list}