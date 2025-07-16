import os
from argparse import ArgumentParser
from pathlib import Path

from funasr import AutoModel


def download_audio(bvid):
    os.system(f"yutto {bvid} --config yutto.toml")

def convert_audio(bvid):
    DIR = Path(f"./data/{bvid}")
    audio_in = DIR / "audio.m4a" 
    audio_out = DIR / "audio.wav" 
    os.system(f"ffmpeg -i {audio_in} -n -ac 1 -ar 16000 {audio_out}")
    return audio_out

def load_asr_model():
    # paraformer-zh is a multi-functional asr model
    # use vad, punc, spk or not as you need
    model = AutoModel(
        model="paraformer-zh",
        model_revision="v2.0.4",
        vad_model="fsmn-vad",
        vad_model_revision="v2.0.4",
        punc_model="ct-punc-c",
        punc_model_revision="v2.0.4",
        # spk_model="cam++", spk_model_revision="v2.0.2",
        hub="ms",
    )
    return model

def load_hotwords(file):
    with open(file, "r", encoding="utf-8") as f:
        hotwords = [line.strip() for line in f.readlines()]
    return hotwords

def asr(model: AutoModel, audio_file: Path, hotwords=[]):
    transcript = model.generate(
        input=str(audio_file),
        batch_size_s=300,
        hotword=" ".join(hotwords),
        sentence_timestamp=True,  # return sentence level information when spk_model is not given
    )
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

def main(args):
    download_audio(args.bvid)
    audio = convert_audio(args.bvid)
    model = load_asr_model()
    hotwords = load_hotwords(args.hotwords) if args.hotwords else []
    transcript = asr(model, audio, hotwords)
    export(transcript, audio)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("bvid", type=str, help="bv number")
    # parser.add_argument("-i,--input_file", required=True, type=str, help="input audio file")
    parser.add_argument("-w","--hotwords", required=False, default=None, type=str, help="file of hotword list")
    parser.add_argument("-d", "--debug", action="store_true", help="debug mode")
    args = parser.parse_args()
    if args.debug:
        import debugpy
        try:
            debugpy.listen(('localhost', 9501))
            print('Waiting for debugger attach')
            debugpy.wait_for_client()
        except Exception as e:
            pass
    main(args)