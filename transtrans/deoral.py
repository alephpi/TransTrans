from argparse import ArgumentParser
from pathlib import Path

from .annotate import Annotation
from .constants import IGNORE_TAGS

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

# # 用分词的结果而非纯字符串正则匹配，避免在错误位置断词导致误删，例如“曹操他妈”被误匹配“操他妈”。
# def match_fillers(tokens: list[str], tags: list[str]):
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

    # return matched_indices


# instead we use hanlp built-in dict merge and tagging to match fillers
def match_fillers(tokens: list[str], tags: list[str]):
    matched_indices = []
    for i, tag in enumerate(tags):
        if tag in ['query','curse','pet','filler']:
            matched_indices.append(i)
    return matched_indices

def match_ambiguous_fillers(tokens:list[str], tags:list[str]):
    # ambiguous filler that needs to be matched with tag and context info
    matched_indices: list[int] = []
    return matched_indices

def match_interjection(tokens: list[str], tags: list[str]):
    matched_indices: list[int] = []
    for i in range(len(tokens)):
        if tags[i] in ['interj']:
            matched_indices.append(i)
    return matched_indices

def match_breaks():
    ...

def match_repetitions(tokens: list[str], tags:list[str], ngram=5):
    # 匹配 n-gram 重复
    win_len = ngram
    matched_indices: list[int] = []
    for i in range(len(tokens)):
        prev_string = "".join(tokens[i:i+win_len])
        next_string = "".join(tokens[i+win_len:i+2*win_len])
        # 若 n-gram 重复，则将删去前面的 n-gram，这里的重复可以是前缀重复
        # 例如'不','不会'
        if next_string.startswith(prev_string):
            matched_indices.extend(range(i, i+win_len))
    return matched_indices

def match_repetitions_robust(tokens: list[str], tags:list[str], ngram: int, ignore_tags:list[str], match_limit: int=20):
    # 匹配忽略特定词性的 n-gram 重复，例如“我感到快乐”和“我感到啊快乐”重复
    win_len = ngram
    matched_indices: list[int] = []
    token_len = len(tokens)
    i = 0
    while i < token_len:
        # print(f"{i=}")
        if tags[i] in ignore_tags:
            i += 1
            continue

        prev_tokens, prev_indices, len_prev_tokens, j = [], [], 0, i
        while len_prev_tokens < win_len and j < token_len:
            # print(f"{j=}")
            if tags[j] not in ignore_tags:
                prev_tokens.append(tokens[j])
                prev_indices.append(j)
                len_prev_tokens += 1
            j += 1
        if len_prev_tokens < win_len:
            i += 1
            continue
        prev_string = "".join(prev_tokens)

        next_tokens, next_indices, len_next_tokens, k = [], [], 0, j
        # 这里有个问题是，如果从prev_tokens到next_tokens中间的忽略词过长，
        # 就会导致prev_tokens与很远处的next_tokens匹配，
        # 这种重复可能并不是我们想匹配的
        # 因此有必要给next_tokens的匹配距离加一个限制
        while len_next_tokens < win_len and k < token_len and k-j < match_limit:
            # print(f"{k=}")
            if tags[k] not in ignore_tags:
                next_tokens.append(tokens[k])
                next_indices.append(k)
                len_next_tokens += 1
            k += 1
        if len_next_tokens < win_len:
            i += 1
            continue
        next_string = "".join(next_tokens)

        if next_string.startswith(prev_string):
            # 返回prev tokens的索引（包含中间跳过的忽略词）
            # matched_indices.extend(range(prev_indices[0], prev_indices[-1]+1))

            # 返回从prev tokens起始到next tokens起始的索引（包含中间跳过的忽略词）
            matched_indices.extend(range(prev_indices[0], next_indices[0]))

            # 返回next tokens的索引（包含中间跳过的忽略词）
            # matched_indices.extend(range(next_indices[0], next_indices[-1]+1))
            i = next_indices[0]

        i = i + 1

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
    # matched_indices = match_fillers(annotation.tokens.tolist(), fillers)
    annotation = Annotation()
    matched_indices = match_fillers(annotation.tokens.tolist(), tags=annotation.tags.tolist())
    annotation.remove(matched_indices)
    print(f"removing fillers: {len(matched_indices)} tokens removed")

    matched_indices = match_ambiguous_fillers(annotation.tokens.tolist(), tags=annotation.tags.tolist())
    annotation.remove(matched_indices)
    print(f"removing ambiguous fillers: {len(matched_indices)} tokens removed")

    matched_indices = match_interjection(annotation.tokens.tolist(), annotation.tags.tolist())
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
