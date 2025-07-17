import json
from pathlib import Path

from funasr import AutoModel


class Transcript:
    def __init__(self, data: list[tuple[str, tuple[int, int]]]):
        self.data = data
        self.original_text_len = self.text_len
        self.original_duration = self.data[-1][1][-1]
        self.time_per_char: int = int(self.original_duration / self.original_text_len)
        self.qikou: int = self.time_per_char # 初始化气口长度与平均字符时长相同

    @classmethod
    def from_char_timestamp(cls, chars: list[str], timestamps: list[tuple[int, int]]):
        assert len(chars) == len(timestamps), "length of chars and timestamps mismatch"
        data = list(zip(chars, timestamps))
        return cls(data)

    @classmethod
    def from_json(cls, file_path: Path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            data = [(d[0], tuple(d[1])) for d in data]
        return cls(data)

    def remove(self, indices: list[int]):
        indices = list(set(indices))
        indices = sorted(indices, reverse=True)
        for i in indices:
            del self.data[i]

    def set_qikou(self, ratio=1, ms:int =0):
        if ms:
            self.qikou = ms
        else:
            self.qikou = int(ratio * self.time_per_char)

    @property
    def text(self):
        return "".join(d[0] for d in self.data)

    @property
    def text_list(self):
        return [d[0] for d in self.data]

    @property
    def timestamps(self):
        return [d[1] for d in self.data]

    @property
    def text_len(self):
        return len(self.text)

    @property
    def time_len(self):
        active = 0
        pause = 0
        prev_t_end = 0
        for t_begin, t_end in self.timestamps:
            active += t_end - t_begin
            pause += min(t_begin - prev_t_end, self.qikou) # 若前后两个字之间有停顿，停顿最长不超过气口，以此进一步压缩时长
            prev_t_end = t_end

        total = active + pause
        return total
    
    def stats(self):
        print(f"文本原长 {self.original_text_len} 字，现长 {self.text_len} 字")
        print(f"音频原长 {convert_time(self.original_duration)}，现长 {convert_time(self.time_len)}")
        text_rate =  1 - self.text_len / self.original_text_len
        time_rate = 1 - self.time_len / self.original_duration
        print(f"压缩率：文本 {text_rate*100:.2f}%，音频 {time_rate*100:.2f}%")

    def to_json(self, file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False)

    def to_txt(self, file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(self.text)

def load_asr_model():
    # paraformer-zh is a multi-functional asr model
    # use vad, punc, spk or not as you need
    model = AutoModel(
        model="paraformer-zh",
        model_revision="v2.0.4",
        vad_model="fsmn-vad",
        vad_model_revision="v2.0.4",
        # punc_model="ct-punc",
        # punc_model_revision="v2.0.4",
        # spk_model="cam++", spk_model_revision="v2.0.2",
        hub="ms",
    )
    return model

def load_all_hotwords(dir_path: Path):
    all_hotwords = []
    for file in dir_path.iterdir():
        hotwords = load_hotwords(file)
        all_hotwords.extend(hotwords)
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