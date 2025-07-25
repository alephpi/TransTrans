import json
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

from .transcript import Transcript
from .utils import load_dict


class Annotation:
    def __init__(self, tokens: list[str], tags: list[str], spans: list[tuple[int, int]]):
        self.tokens: NDArray[np.str_] = np.array(tokens)
        self.tags: NDArray[np.str_] = np.array(tags)
        self.spans: NDArray[np.int32] = np.array(spans)
        self.mask: NDArray[np.bool_] = np.ones(len(tokens), dtype=bool)
        self.char_indices: list[int] = []

    def remove(self, indices: list[int]):
        """remove by indices

        Args:
            indices (list[int]): indices to remove, make sure the `indices` are unique
        """
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

    def to_txt(self, file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            for token, tag in zip(self.tokens, self.tags):
                f.write(f"{token} {tag}\n")
        print(f"annotation saved to {file_path}")
    
    @classmethod
    def from_json(cls, file_path: Path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        tokens, tags, spans = [],[],[]
        for (token, tag, span) in data:
            tokens.append(token)
            tags.append(tag)
            spans.append(span)
        return cls(tokens, tags, spans)

    def to_json(self, file_path):
        data = list(zip(self.tokens, self.tags, self.spans.tolist()))
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
 
    def __repr__(self):
        return str([(token, tag) for token, tag in zip(self.tokens, self.tags)])
    
    def __eq__(self, other):
        return np.array_equal(self.tokens,other.tokens) and np.array_equal(self.tags,other.tags) and np.array_equal(self.spans,other.spans)
    
    def __len__(self):
        return len(self.tokens)

def load_tok_model(fine=True, dictionary: set[str]=set()):
    model: TransformerTaggingTokenizer
    if fine:
        model = hanlp.load(FINE_ELECTRA_SMALL_ZH)
    else:
        model = hanlp.load(COARSE_ELECTRA_SMALL_ZH)
    model.config.output_spans = True
    model.dict_combine = dictionary
    def tok_model_wrapper(text):
        res_tok = model(text)
        tokens = [i[0] for i in res_tok]
        spans = [i[1:] for i in res_tok]
        return tokens, spans
    return tok_model_wrapper

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


def init_parser():
    parser = ArgumentParser()
    parser.add_argument("-t","--transcript", type=str, help="transcript json file")
    return parser

def main(args):
    filler_dict_paths = ["指示代词", "不定代词", "填充词", "语气词", "詈语", "口头禅", "询问语"]
    fillers_l, fillers_d = load_dict(filler_dict_paths)
    print("load transcript")
    transcript = Transcript.from_json(Path(args.transcript))

    # temporarily ignore english
    # TODO: support English deoralization
    print("remove english")
    transcript.chinese_only()
    transcript.stats()

    tok_model = load_tok_model(fine=True, dictionary=set(fillers_l).union(set(fillers_d.keys())))
    pos_model = load_pos_model(dictionary=fillers_d) # type: ignore
    pipeline = hanlp.pipeline()\
        .append(tok_model, output_key=('tokens','spans'))\
        .append(pos_model,input_key='tokens', output_key='tags')
    annotation = Annotation(**pipeline(transcript.text))
    annotation.to_json(Path(args.transcript).parent/"annotation.json")
    annotation.to_txt(Path(args.transcript).parent/"annotation.txt")

    # stats
    freq_stats =annotation.stats()
    with open(Path(args.transcript).with_name("freq_stats.txt"), 'w', encoding='utf-8') as f:
        for token, freq in freq_stats:
            f.write(f"{token} {freq}\n")

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