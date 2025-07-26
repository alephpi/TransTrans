from collections import defaultdict

import numpy as np

from .annotate import Annotation
from .transcript import Transcript

PUNC_LIST = ['', '', '，', '。', '？', '、'] # align with funasr


def punctuate_by_pause(transcript: Transcript):
    punctuation = None
    return punctuation

def punctuate_by_tag(annotation: Annotation):
    # record after which indices a punc should be added
    punc_dict: dict[int, int] = defaultdict(int)
    for i, (token, tag, span) in enumerate(annotation):
        if tag == 'interj':
                punc_dict[i] = max(2, punc_dict[i])
        if token in ["吗","呢","吧"]:
            punc_dict[i] = max(4, punc_dict[i])
        # 连词
        elif tag == 'c' and i > 0:
            punc_dict[i-1] = max(3, punc_dict[i-1])
            
    return punc_dict

def annotate_by_punctuation(annotation: Annotation, punc_dict: dict[int,int]):
    tokens, tags, spans = annotation.tokens, annotation.tags, annotation.spans
    sorted_indices = sorted(punc_dict.keys())
    original_len = len(annotation)
    insert_len = len(sorted_indices)
    insert_indices = [min(idx + 1 + i, original_len+insert_len) for i, idx in enumerate(sorted_indices)]
    insert_tokens = [PUNC_LIST[punc_dict[idx]] for idx in sorted_indices]
    insert_tags = ["w"] * insert_len
    insert_spans = [[-1,-1]] * insert_len

    mask = np.zeros(original_len + insert_len, dtype=bool)
    mask[insert_indices] = True

    new_tokens = np.empty_like(mask, dtype=object)
    new_tags = np.empty_like(mask, dtype=object)
    new_spans = np.empty((len(mask),2), dtype=np.int32)

    new_tokens[~mask] = tokens
    new_tokens[mask] = insert_tokens

    new_tags[~mask] = tags
    new_tags[mask] = insert_tags

    new_spans[~mask] = spans
    new_spans[mask] = insert_spans

    return Annotation(new_tokens.tolist(), new_tags.tolist(), new_spans.tolist())