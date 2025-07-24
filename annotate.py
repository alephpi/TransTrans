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
        print(f"annotation saved to {file_path}")
    
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
    parser.add_argument("transcript", type=str, help="transcript pickle file")
    # parser.add_argument("-a", "--aggressive", action="store_true", default=False, help="if aggressive, remove all fillers rather than replacing")
    return parser

def main(args):
    from transcript import Transcript, load_punc_model, punctuate
    filler_dict_paths = ["common", "curse", "pet", "query"]
    fillers_l, fillers_d = load_dict(filler_dict_paths)
    print("load transcript")
    transcript = Transcript.load(Path(args.transcript))
    if getattr(transcript, "annotation", None) is not None:
        print("transcript already annotated, skip annotation")
        return

    # punc_model = load_punc_model()
    # transcript = punctuate(punc_model, transcript)
    # transcript.set_qikou(ms=1000)

    print("remove english")
    transcript.chinese_only()
    transcript.stats()

    tok_model = load_tok_model(fine=True, dictionary=set(fillers_l).union(set(fillers_d.keys())))
    pos_model = load_pos_model(dictionary=fillers_d) # type: ignore
    pipeline = hanlp.pipeline()\
        .append(tok_model, output_key=('tokens','spans'))\
        .append(pos_model,input_key='tokens', output_key='tags')
    annotation = Annotation(**pipeline(transcript.text))
    transcript.annotation = annotation

    annotation.save(Path(args.transcript).parent/"transcript.annotation")
    freq_stats =annotation.stats()
    with open(Path(args.transcript).with_name("freq_stats.txt"), 'w', encoding='utf-8') as f:
        for token, freq in freq_stats:
            f.write(f"{token} {freq}\n")

    transcript.save(Path(args.transcript))


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