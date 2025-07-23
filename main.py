from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path

from deoral import main as deoral
from transcript import main as transcript
from utils import convert_audio, download_audio


def pipeline(args):
    download_audio(args.bvid)
    audio = convert_audio(args.bvid)
    args.input_file = audio
    transcript(args)
    args.text = audio.parent / "transcript.txt"
    deoral(args)

if __name__ == "__main__":
    parser = ArgumentParser(
        description='Pipeline',
        formatter_class=RawTextHelpFormatter
    )

    # 全局选项：控制Pipeline流程
    parser.add_argument("--bvid", type=str, help="bv number")
    parser.add_argument("--debug", action='store_true', help="debug mode")
    args = parser.parse_args()
    if args.debug:
        import debugpy
        try:
            debugpy.listen(('localhost', 9501))
            print('Waiting for debugger attach')
            debugpy.wait_for_client()
        except Exception as e:
            pass
    pipeline(args)
