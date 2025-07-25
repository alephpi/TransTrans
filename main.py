from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path

from transtrans import annotator, deoralor, transcriptor
from transtrans.utils import convert_audio, download_audio


def pipeline(args):
    download_audio(args.bvid)
    audio = convert_audio(args.bvid)
    args.input_file = audio
    transcriptor(args)
    args.transcript = audio.parent / "transcript.json"
    annotator(args)
    args.annotation = audio.parent / "annotation.json"
    deoralor(args)

if __name__ == "__main__":
    parser = ArgumentParser(
        description='Pipeline',
        formatter_class=RawTextHelpFormatter
    )

    # 全局选项：控制Pipeline流程
    parser.add_argument("--bvid", type=str, help="bv number")
    parser.add_argument("--hotwords", type=str, help="hotwords dict")
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
