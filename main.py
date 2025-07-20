import os
from argparse import ArgumentParser
from pathlib import Path

from transcript import (
    asr,
    load_all_hotwords,
    load_asr_model,
    load_punc_model,
    punctuate,
)


def download_audio(bvid):
    os.system(f"yutto {bvid} --config yutto.toml")

def convert_audio(bvid):
    DIR = Path(f"./data/{bvid}")
    audio_in = DIR / "audio.m4a" 
    audio_out = DIR / "audio.wav" 
    os.system(f"ffmpeg -i {audio_in} -n -ac 1 -ar 16000 {audio_out}")
    return audio_out

def main(args):
    download_audio(args.bvid)
    audio = convert_audio(args.bvid)
    asr_model = load_asr_model()
    hotwords = load_all_hotwords(Path(args.hotwords)) if args.hotwords else None
    transcript = asr(asr_model, audio, hotwords)
    punc_model = load_punc_model()
    transcript = punctuate(punc_model, transcript)
    transcript.to_txt(audio.parent / "transcript.txt", with_punc=True)
    transcript.save(audio.parent / "transcript.pkl")


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