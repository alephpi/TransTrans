import re
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List

from transcript import Transcript

DIR = Path("./orals/")
def load_all_fillers():
    all_fillers = []
    all_replaces = {}
    for file in DIR.iterdir():
        fillers, replaces = load_fillers(file)
        all_fillers.extend(fillers)
        all_replaces.update(replaces)
    return all_fillers, all_replaces


def load_fillers(file_path):
    fillers = []
    replaces = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split(' ')
            fillers.append(line[0])
            if len(line) > 1:
                replaces[line[0]] = line[1]
    return fillers, replaces

def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

def remove_english(text: list[str]):
    """
    去除英文
    """
    pattern = r'[a-zA-Z]+'
    remove_indices: list[int] = []
    for i, t in enumerate(text):
        if re.match(pattern, t):
            remove_indices.append(i)
    return remove_indices


def remove_fillers(text:str, fillers:List[str], replaces: Dict[str, str], *, aggressive=False):
    """
    去除或替换填充词
    """
    remove_indices: list[int] = []
    fillers.sort(key=lambda x: len(x), reverse=True)
    if aggressive:
        pattern_filler = re.compile('|'.join(fillers))
        for m in re.finditer(pattern_filler, text):
            start, end = m.span()
            remove_indices.extend(range(start, end))
    else:
        replace_fillers = list(replaces.keys())
        pattern_replace_filler = re.compile('|'.join(replace_fillers))
        for m in re.finditer(pattern_replace_filler, text):
            start, end = m.span()
            remove_indices.extend(range(start+1, end)) # 加1是因为保留第一个字符，例如这个->这
        rest_fillers = [f for f in fillers if f not in replaces]
        pattern_rest_filler = re.compile('|'.join(rest_fillers))
        for m in re.finditer(pattern_rest_filler, text):
            start, end = m.span()
            remove_indices.extend(range(start, end))
    return remove_indices

def remove_repetitions(text:str, ngram=1):
    win_len = ngram
    remove_indices: list[int] = []
    for win_shift in range(win_len):
        for i in range(0, len(text) - win_len + 1, win_len):
                if text[i+win_shift:i+win_shift+win_len] == text[i+win_shift+win_len:i+win_shift+2*win_len]:
                    # 若 n-gram 重复，则将删去前面的 n-gram
                    remove_indices.extend(range(i, i+win_len))
    return remove_indices

# def remove_short_phrases(text:str, n=5):
#     punc = "，。？！；：“”‘’《》、,.?!;:-——"
#     punc_idx = []
#     for i in range(len(text)-n):
#         if text[i] in punc:
#             punc_idx.append(i)
#     diff = [punc_idx[i+1] - punc_idx[i] for i in range(len(punc_idx)-1)]
#     remove_indices: list[int] = []
#     for i, d in enumerate(diff):
#         if d <= 5:
#             remove_indices.extend(range(punc_idx[i], punc_idx[i+1]))
#     return remove_indices

def save_text(text, args):
    output_path = Path(args.text).with_suffix('.deoral')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)

def main(args):
    fillers, replaces = load_all_fillers()
    print("load transcript")
    transcript = Transcript.from_json(Path(args.text))

    transcript.set_qikou(ms=1000)
    transcript.stats()

    print("remove english")
    remove_indices = remove_english(transcript.text_list)
    transcript.remove(remove_indices)
    transcript.stats()

    print("remove fillers")
    remove_indices = remove_fillers(transcript.text, fillers, replaces, aggressive=args.aggressive)
    transcript.remove(remove_indices)
    transcript.stats()

    print("remove repetitions")
    Ngram = 10
    for n in range(1, Ngram+1):
        remove_indices = remove_repetitions(transcript.text, n)
        transcript.remove(remove_indices)
        transcript.stats()

    # text = remove_short_phrases(text)
    transcript.to_json(Path(args.text).with_name("transcript_deoral.json"))
    transcript.to_txt(Path(args.text).with_name("transcript_deoral.txt"))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("text", type=str, help="text to process")
    parser.add_argument("-a", "--aggressive", action="store_true", default=False, help="if aggressive, remove all fillers rather than replacing")
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
