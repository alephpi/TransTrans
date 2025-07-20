import re
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Literal

import hanlp
import numpy as np
from hanlp.pretrained.pos import (
    C863_POS_ELECTRA_SMALL,
    CTB9_POS_ELECTRA_SMALL,
    PKU_POS_ELECTRA_SMALL,
)
from hanlp.pretrained.tok import COARSE_ELECTRA_SMALL_ZH, FINE_ELECTRA_SMALL_ZH
from numpy.typing import NDArray

from transcript import Transcript

DIR = Path("./orals/")

def load_tok_model(fine=True):
    if fine:
        model = hanlp.load(FINE_ELECTRA_SMALL_ZH)
    else:
        model = hanlp.load(COARSE_ELECTRA_SMALL_ZH)
    model.config.output_spans = True
    return model

def load_pos_model(standard:Literal["ctb9", "c863", "pku"]="pku"):
    model = hanlp.load(f"{standard.upper()}_POS_ELECTRA_SMALL")
    return model

def tok_tag(tok_model, pos_model, text):
    res_tok = tok_model(text)
    tokens: list[str] = [i[0] for i in res_tok]
    spans: NDArray[np.uint32] = np.array([i[1:] for i in res_tok])
    tags: list[str] = pos_model(tokens)
    return tokens, spans, tags

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

# def remove_english(text: list[str]):
#     """
#     去除英文
#     """
#     pattern = r'[a-zA-Z\']+' # 匹配英文字母和缩写引号
#     remove_indices: list[int] = []
#     for i, t in enumerate(text):
#         if re.match(pattern, t):
#             remove_indices.append(i)
#     return remove_indices

def match_fillers(tokens: list[str], fillers: list[str]):
    # 用分词的结果而非纯字符串正则匹配，避免在错误位置断词导致误删，例如“曹操他妈”被误匹配“操他妈”。
    matched_indices: list[int] = []
    max_len_filler = max(len(f) for f in fillers)
    prefix_set = set()
    for filler in fillers:
        for k in range(1, len(filler)+1):
            prefix_set.add(filler[:k])

    for i in range(len(tokens)):
        current_str = ""
        current_str += tokens[i]
        if current_str not in prefix_set:
            continue
        if current_str in fillers:
            matched_indices.append(i)
        for j in range(i+1, i+max_len_filler):
            current_str += tokens[j]
            if len(current_str) > max_len_filler:
                break
            if current_str not in prefix_set:
                break
            if current_str in fillers:
                matched_indices.extend(range(i,j+1))
    
    matched_indices = sorted(set(matched_indices))  # 去重
    return matched_indices

def match_repetitions(tokens: list[str], ngram=1):
    pass


# def remove_fillers(text:str, fillers:List[str], replaces: Dict[str, str], *, aggressive=False):
#     """
#     去除或替换填充词
#     """
#     remove_indices: list[int] = []
#     fillers.sort(key=lambda x: len(x), reverse=True)
#     if aggressive:
#         pattern_filler = re.compile('|'.join(fillers))
#         for m in re.finditer(pattern_filler, text):
#             start, end = m.span()
#             remove_indices.extend(range(start, end))
#     else:
#         replace_fillers = list(replaces.keys())
#         pattern_replace_filler = re.compile('|'.join(replace_fillers))
#         for m in re.finditer(pattern_replace_filler, text):
#             start, end = m.span()
#             remove_indices.extend(range(start+1, end)) # 加1是因为保留第一个字符，例如这个->这
#         rest_fillers = [f for f in fillers if f not in replaces]
#         pattern_rest_filler = re.compile('|'.join(rest_fillers))
#         for m in re.finditer(pattern_rest_filler, text):
#             start, end = m.span()
#             remove_indices.extend(range(start, end))
#     return remove_indices

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
