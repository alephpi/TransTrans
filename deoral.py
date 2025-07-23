from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
from typing import Dict, Literal, Sequence

import hanlp
import numpy as np
from hanlp.components.tokenizers.transformer import (
    TransformerTagger,
    TransformerTaggingTokenizer,
)
from hanlp.pretrained.pos import (
    C863_POS_ELECTRA_SMALL,
    CTB9_POS_ELECTRA_SMALL,
    PKU_POS_ELECTRA_SMALL,
)
from hanlp.pretrained.tok import COARSE_ELECTRA_SMALL_ZH, FINE_ELECTRA_SMALL_ZH
from numpy.typing import NDArray

from transcript import Transcript, load_punc_model, punctuate
from utils import load_dict


class Annotation:
    def __init__(self, tokens: list[str], tags: list[str], spans: list[tuple[int, int]]):
        self.tokens: NDArray[np.str_] = np.array(tokens)
        self.tags: NDArray[np.str_] = np.array(tags)
        self.spans: NDArray[np.int32] = np.array(spans)
        self.mask: NDArray[np.bool_] = np.ones(len(tokens), dtype=bool)
        self.char_indices: list[int] = []

    def remove(self, indices: list[int]):
        indices = list(set(indices))
        self.mask[indices] = False
        self.update()

    def update(self):
        # update char-level mask, after all tokens are cleaned, sync the cleaned-version back to char level
        for span in self.spans[~self.mask]:
            self.char_indices.extend(range(*span))

        self.tokens = self.tokens[self.mask]
        self.tags = self.tags[self.mask]
        self.spans = self.spans[self.mask]
        self.mask = np.ones(len(self.tokens), dtype=bool)
    
    def stats(self):
        count = Counter(self.tokens)
        freq_pairs = sorted(count.items(), key=lambda x: x[1], reverse=True)
        return freq_pairs

    def save(self, file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            for token, tag in zip(self.tokens, self.tags):
                f.write(f"{token, tag}\n")
    
    def __repr__(self):
        return str([(token, tag) for token, tag in zip(self.tokens, self.tags)])

def load_tok_model(fine=True, dictionary: set[str]=set()):
    model: TransformerTaggingTokenizer
    if fine:
        model = hanlp.load(FINE_ELECTRA_SMALL_ZH)
    else:
        model = hanlp.load(COARSE_ELECTRA_SMALL_ZH)
    model.config.output_spans = True
    model.dict_combine = dictionary
    return model

def load_pos_model(standard:Literal["ctb9", "c863", "pku"]="pku", dictionary:Dict[str|Sequence[str],str|Sequence[str]]={}):
    model: TransformerTagger
    if standard == "ctb9":
        model = hanlp.load(CTB9_POS_ELECTRA_SMALL)
    elif standard == "c863":
        model = hanlp.load(C863_POS_ELECTRA_SMALL)
    else:
        model = hanlp.load(PKU_POS_ELECTRA_SMALL)
    model.dict_tags = dictionary
    return model

def tok_tag(tok_model, pos_model, text):
    res_tok = tok_model(text)
    tokens = [i[0] for i in res_tok]
    spans = [i[1:] for i in res_tok]
    tags = pos_model(tokens)
    return Annotation(tokens, tags, spans)



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

def match_fillers(tokens: list[str], tags: list[str]):
    # # 用分词的结果而非纯字符串正则匹配，避免在错误位置断词导致误删，例如“曹操他妈”被误匹配“操他妈”。
    # matched_indices: list[int] = []
    # max_len_filler = max(len(f) for f in fillers)
    # prefix_set = set()
    # for filler in fillers:
    #     for k in range(1, len(filler)+1):
    #         prefix_set.add(filler[:k])

    # for i in range(len(tokens)):
    #     current_str = ""
    #     current_str += tokens[i]
    #     if current_str not in prefix_set:
    #         continue
    #     if current_str in fillers:
    #         matched_indices.append(i)
    #     for j in range(i+1, i+max_len_filler):
    #         current_str += tokens[j]
    #         if len(current_str) > max_len_filler:
    #             break
    #         if current_str not in prefix_set:
    #             break
    #         if current_str in fillers:
    #             matched_indices.extend(range(i,j+1))

    # instead we use hanlp built-in dict merge and tagging to match fillers
    matched_indices = []
    for i, tag in enumerate(tags):
        if tag in ['query','curse','pet']:
            matched_indices.append(i)
        elif tokens[i] == '的话':
            matched_indices.append(i)
    return matched_indices

def match_ambiguous_fillers(tokens:list[str], tags:list[str]):
    # ambiguous filler that needs to be matched with tag and context info
    matched_indices: list[int] = []
    for i in range(len(tokens)):
        if tokens[i] in ["就是","就"] and tokens[i+1] == "不是":
            matched_indices.append(i)
        # elif tokens[i] == "这个" and tags[i-1] == tags[i+1] == 'r' and tokens[i-1] == tokens[i+1]:
            # 代词 + 这个 + 代词，如“你这个你”
            # matched_indices.extend((i-1,i))
        elif tokens[i] == "这个" and tags[i-1] != 'c': # 前面不是连词，即“这个”不作句首主语
            matched_indices.append(i)
        elif tags[i] == "common":
            # if tag[i+1] == 
            ...
    return matched_indices

def match_exclamations(tokens: list[str], tags: list[str]):
    matched_indices: list[int] = []
    for i in range(len(tokens)):
        if tags[i] == 'e' or tokens[i] == '啊':
            matched_indices.append(i)
        elif tags[i:i+2] == ['的','啊']:
            matched_indices.extend((i,i+1))
    return matched_indices

def match_breaks():
    ...

def match_repetitions(tokens: list[str], tags:list[str], ngram=5):
    win_len = ngram
    matched_indices: list[int] = []
    for i in range(0, len(tokens) - 1):
        prev_string = "".join(tokens[i:i+win_len])
        next_string = "".join(tokens[i+win_len:i+2*win_len])
        # 若 n-gram 重复，则将删去前面的 n-gram，这里的重复可以是前缀重复
        # 例如'不','不会'
        if next_string.startswith(prev_string):
            matched_indices.extend(range(i, i+win_len))
        # 因为hanlp的分词较细，我们只需要对1-gram处理即可。
    return matched_indices


def match_pieces(tokens: list[str], tags: list[str], pad=5):
    matched_indices: list[int] = []
    # padding to avoid for-loop index out of range
    tokens.extend(["[PAD]"] * pad)
    tags.extend(["[PAD]"] * pad)

    # pron_clusters (consecutive 'r's), e.g. 我这个我
    # match all pron_clusters, remove all except the last pron in the cluster
    pron_clusters = []
    for i in range(len(tags)):
        if tags[i:i+2] == ['r','r']:
            pron_clusters.append(i)
        elif tags[i] == 'r' and tags[i+2] == 'r':
            if tags[i+1] == 'u' or tokens[i+1] in ["是","觉得"]:
                pron_clusters.extend((i,i+1))
            elif tokens[i] == tokens[i+2] and tags[i+1] == 'c':
                pron_clusters.extend((i,i+1))
        #TODO vdv structure
        # elif tags

    matched_indices.extend(pron_clusters)
    return matched_indices

def match_orphan_phrases():
    # after removing all above, split the sentence by punctuation and remove short phrases
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

# def remove_repetitions(text:str, ngram=1):
#     win_len = ngram
#     remove_indices: list[int] = []
#     for win_shift in range(win_len):
#         for i in range(0, len(text) - win_len + 1, win_len):
#                 if text[i+win_shift:i+win_shift+win_len] == text[i+win_shift+win_len:i+win_shift+2*win_len]:
#                     # 若 n-gram 重复，则将删去前面的 n-gram
#                     remove_indices.extend(range(i, i+win_len))
#     return remove_indices

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
    filler_dict_paths = ["common", "curse", "pet", "query"]
    fillers_l, fillers_d = load_dict(filler_dict_paths)
    print("load transcript")
    transcript = Transcript.from_json(Path(args.text))
    punc_model = load_punc_model()
    transcript = punctuate(punc_model, transcript)

    # transcript.set_qikou(ms=1000)
    transcript.stats()

    print("remove english")
    transcript.chinese_only()
    transcript.stats()

    tok_model = load_tok_model(fine=True, dictionary=set(fillers_l).union(set(fillers_d.keys())))
    pos_model = load_pos_model(dictionary=fillers_d) # type: ignore
    annotation = tok_tag(tok_model, pos_model, transcript.text)
    annotation.save(Path(args.text).with_name("transcript.annotation"))
    freq_stats =annotation.stats()
    with open(Path(args.text).with_name("freq_stats.txt"), 'w', encoding='utf-8') as f:
        for token, freq in freq_stats:
            f.write(f"{token} {freq}\n")

    # matched_indices = match_fillers(annotation.tokens.tolist(), fillers)
    matched_indices = match_fillers(annotation.tokens.tolist(), tags=annotation.tags.tolist())
    annotation.remove(matched_indices)
    print(f"removing fillers: {len(matched_indices)} tokens removed")

    matched_indices = match_ambiguous_fillers(annotation.tokens.tolist(), tags=annotation.tags.tolist())
    annotation.remove(matched_indices)
    print(f"removing ambiguous fillers: {len(matched_indices)} tokens removed")

    matched_indices = match_exclamations(annotation.tokens.tolist(), annotation.tags.tolist())
    annotation.remove(matched_indices)
    print(f"removing exclamations: {len(matched_indices)} tokens removed")

    print("remove breaks")

    print("remove repetitions")
    Ngram = 10
    ngram = 1
    while ngram <= Ngram:
        matched_indices = match_repetitions(annotation.tokens.tolist(), annotation.tags.tolist(), ngram)
        if (l:=len(matched_indices)) > 0:
            annotation.remove(matched_indices)
            print(f"removing {ngram}-gram repetitions: {l} tokens removed")
            if ngram > 1:
                ngram = 1
        else:
            ngram += 1

    # matched_indices = match_pieces(annotation.tokens.tolist(), annotation.tags.tolist())
    # annotation.remove(matched_indices)
    # print(f"removing pieces: {len(matched_indices)} tokens removed")

    # sync the cleaned mask from annotation
    transcript.remove(annotation.char_indices)
    transcript.stats()

    # text = remove_short_phrases(text)
    annotation.save(Path(args.text).with_name("transcript_deoral.annotation"))
    # transcript.to_txt(Path(args.text).with_name("transcript_deoral.txt"))
    transcript.to_txt(Path(args.text).with_name("transcript_deoral.txt"), with_punc=True)
    transcript.to_json(Path(args.text).with_name("transcript_deoral.json"))

def init_parser():
    parser = ArgumentParser()
    parser.add_argument("text", type=str, help="text to process")
    # parser.add_argument("-a", "--aggressive", action="store_true", default=False, help="if aggressive, remove all fillers rather than replacing")
    return parser

if __name__ == '__main__':
    parser = init_parser()
    parser.add_argument("--debug", action="store_true", help="debug mode")
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
