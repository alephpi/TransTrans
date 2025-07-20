import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from funasr import AutoModel
from numpy.typing import NDArray


class Transcript:
    def __init__(self, chars: list[str], timestamps: list[tuple[int, int]]):
        self.chars: NDArray[np.str_] = np.array(chars)
        self.timestamps: NDArray[np.int32] = np.array(timestamps)
        self.original_text_len: int = len(self.chars)
        self.original_duration: int = self.timestamps[-1][-1]
        self.char_durations: NDArray[np.int32] = self.timestamps[:,1] - self.timestamps[:,0]
        self.avg_char_duration: int = self.char_durations.sum() // self.original_text_len
        self.qikou: int = self.avg_char_duration # 初始化气口长度与平均字符时长相同
        self.char_intervals: NDArray[np.int32] = np.append(self.timestamps[1:,0] - self.timestamps[:-1,1], 10*self.qikou)
        self.punc_list: NDArray[np.str_] = np.array(['', '', '，', '。', '？', '、']) # align with funasr
        self.is_hanzi: NDArray[np.bool_] = ~np.vectorize(lambda x: x.isascii())(self.chars)
        self.mask: NDArray[np.bool_] = np.ones(len(self.chars), dtype=np.bool_) # 用于标记需要保留的字

    @classmethod
    def from_char_timestamp(cls, chars: list[str], timestamps: list[tuple[int, int]]):
        assert len(chars) == len(timestamps), "length of chars and timestamps mismatch"
        return cls(chars, timestamps)

    @classmethod
    def from_json(cls, file_path: Path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            chars = []
            timestamps = []
            for d in data:
                chars.append(d[0])
                timestamps.append(tuple(d[1]))
        return cls(chars, timestamps)
    
    def chinese_only(self):
        self.mask = self.is_hanzi
        self.update()

    def remove(self, indices: list[int]):
        indices = list(set(indices))
        self.mask[indices] = False
        self.update()

    def set_qikou(self, ratio=1, ms:Optional[int]=None):
        if ms is not None:
            self.qikou = ms
        else:
            self.qikou = int(ratio * self.avg_char_duration)
    
    def update(self):
        self.chars = self.chars[self.mask]
        self.timestamps = self.timestamps[self.mask]
        self.char_durations = self.char_durations[self.mask]
        self.char_intervals = self.char_intervals[self.mask]
        self.is_hanzi = self.is_hanzi[self.mask]
        self.mask = np.ones(len(self.chars), dtype=np.bool_)
    
    @property
    def punc_array(self) -> NDArray[np.uint8]:
        return np.where(self.char_intervals > self.qikou, 2, 1) # 2 for comma, 1 for no punc, align with funasr output

    @property
    def text(self):
        return "".join(self.chars_format)
    
    @property
    def text_with_punc(self):
        puncs = self.punc_list[self.punc_array]
        # puncs = (self.punc_list[punc_id] for punc_id in self.punc_array)
        chars_with_punc = np.char.add(self.chars_format, puncs)
        return "".join(chars_with_punc)

    @property
    def chars_format(self):
        # 若为英文单词则在前面加空格
        is_word = ~self.is_hanzi
        left_is_word = np.roll(is_word, 1)
        left_is_word[0] = False  # 左边界补0
        add_space: NDArray[np.str_] = np.where(is_word & left_is_word, ' ', '')

        return np.char.add(add_space, self.chars)

    @property
    def text_len(self):
        return len(self.chars)

    @property
    def time_len(self):
        active_duration: int = self.char_durations.sum()
        pause_duration: int = np.clip(self.char_intervals, a_min=None, a_max=self.qikou).sum()

        total_duration = active_duration + pause_duration
        return total_duration

    def stats(self):
        print(f"文本原长 {self.original_text_len} 字，现长 {self.text_len} 字")
        print(f"音频原长 {convert_time(self.original_duration)}，现长 {convert_time(self.time_len)}")
        text_rate =  1 - self.text_len / self.original_text_len
        time_rate = 1 - self.time_len / self.original_duration
        print(f"压缩率：文本 {text_rate*100:.2f}%，音频 {time_rate*100:.2f}%")

    def to_json(self, file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            data = list(zip(self.chars, self.timestamps))
            json.dump(data, f, ensure_ascii=False)

    def to_txt(self, file_path, with_punc=False):
        with open(file_path, "w", encoding="utf-8") as f:
            if with_punc:
                f.write(self.text_with_punc)
            else:
                f.write(self.text)

    @classmethod
    def load(cls, file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)

    def save(self, file_path):
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

def load_asr_model():
    # paraformer-zh is a multi-functional asr model
    # use vad, punc, spk or not as you need
    model = AutoModel(
        model="paraformer-zh",
        model_revision="v2.0.4",
        vad_model="fsmn-vad",
        vad_model_revision="v2.0.4",
        # punc_model="ct-punc-c",
        # punc_model_revision="v2.0.4",
        # spk_model="cam++", spk_model_revision="v2.0.2",
        hub="ms",
    )
    return model

def load_punc_model():
    # ct-punc-c is a punc model for chinese, don't confuse it with ct-punc
    model = AutoModel(
        model="ct-punc-c",
        model_revision="v2.0.4",
        hub="ms",
    )
    return model

def load_all_hotwords(dir_path: Path):
    all_hotwords = []
    for file in dir_path.iterdir():
        hotwords = load_hotwords(file)
        all_hotwords.extend(hotwords)
    logging.info(f"{all_hotwords=}")
    return all_hotwords


def load_hotwords(file_path: Path):
    with open(file_path, "r", encoding="utf-8") as f:
        hotwords = [line.strip() for line in f.readlines()]
    return hotwords

def asr(model: AutoModel, audio_file: Path, hotwords=[]):
    """转录文本+字符级时间戳"""
    transcript = model.generate(
        input=str(audio_file),
        batch_size_s=300,
        hotword=" ".join(hotwords),
        sentence_timestamp = False
    )
    chars = transcript[0]['text'].split(' ')
    timestamps = transcript[0]['timestamp']
    timestamps = [tuple(t) for t in timestamps]
    transcript = Transcript.from_char_timestamp(chars, timestamps)
    return transcript

def punctuate(model: AutoModel, transcript: Transcript):
    """标点符号识别"""
    res = model.generate(transcript.text)
    punc_array_infered_by_model = res[0]['punc_array'].tolist()
    punc_array_infered_from_interval = transcript.punc_array
    assert len(punc_array_infered_by_model) == len(punc_array_infered_from_interval), f"length of puncs mismatch, get {len(punc_array_infered_by_model)=} and {len(punc_array_infered_from_interval)=}"
    punc_array_merged = [max(p1, p2) for p1, p2 in zip(punc_array_infered_by_model, punc_array_infered_from_interval)]
    transcript.punc_array = punc_array_merged
    return transcript

def convert_time(timestamp):
    """convert milleseconds to hh:mm:ss:ms format
    """
    milliseconds = timestamp % 1000
    seconds = timestamp / 1000
    minutes = seconds // 60
    seconds = seconds % 60
    hours = minutes // 60
    minutes = minutes % 60
    return "%d:%02d:%02d.%03d" % (hours, minutes, seconds, milliseconds)

def export(transcript, audio: Path):
    """ export subtitle file
    """
    output_dir = audio.parent
    srt = output_dir / "subtitle.srt"
    txt = output_dir / "plain.txt"
    with open(srt, "w", encoding="utf-8") as f:
        for s in transcript[0]["sentence_info"]:
            start = convert_time(s["start"])
            end = convert_time(s["end"])
            print(f'{start}->{end}, {s["text"]}', file=f)
    with open(txt, "w", encoding="utf-8") as f:
        print(transcript[0]['text'], file=f)