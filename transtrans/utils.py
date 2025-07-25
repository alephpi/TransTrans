import os
from pathlib import Path

DICT_DIR = "dicts"

def download_audio(bvid):
    os.system(f"yutto {bvid} --config yutto.toml")

def convert_audio(bvid):
    DIR = Path(f"./data/{bvid}")
    audio_in = DIR / "audio.m4a" 
    audio_out = DIR / "audio.wav" 
    os.system(f"ffmpeg -i {audio_in} -n -ac 1 -ar 16000 {audio_out}")
    return audio_out

def load_dict(file_paths: str|list[str]):
    l: list[str] = []
    d: dict[str,str] = {}
    if isinstance(file_paths, str):
        file_paths = [file_paths]
    for file_path in file_paths:
        with open(Path(DICT_DIR)/file_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = line.strip()
                words = line.split()
                if len(words) == 1:
                    l.append(words[0])
                elif len(words) == 2:
                    d.update({words[0]:words[1]})
    return l,d

def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

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
