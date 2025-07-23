from itertools import product
from pathlib import Path


class Phrase:
    tag = ''
    @classmethod
    def filter(cls, phrases):
        return [p for p in phrases if len(p) > 1] # 过滤掉空字符串和单字

    @classmethod
    def generate(cls):
        raise NotImplementedError
    
    @classmethod
    def export(cls):
        DIR = Path("./dicts")
        DIR.mkdir(exist_ok=True)
        phrases = cls.generate()
        with open(DIR / cls.__name__.lower(), "w", encoding="utf-8") as f:
            for phrase in phrases:
                f.write(f"{phrase} {cls.tag}\n")

class Common(Phrase):
    tag = ''

    prefix1 = ["这", "那", "一"]
    suffix1 = ["个","种","些","样"]

    prefix2 = ["就","不","而"]
    suffix2 = ["是"]

    @classmethod
    def generate(cls):
        cas1 = ["".join(p) for p in product(
            cls.prefix1,
            cls.suffix1,
            )] 

        cas2 = ["".join(p) for p in product(
            cls.prefix2,
            cls.suffix2,
            )] 
        
        cas3 = ["的话"]
        return cls.filter(cas1+cas2+cas3)

class Query(Phrase):
    tag = 'query'

    subject = ["你", ""]
    verb = ["懂", "明白", "知道", "了解", "理解", "记得"]
    perfect_tense = ["了", ""]
    question = ["吗", "吧"]
    predicate = ["是", "对", "好", "行", "可以", "OK"]

    @classmethod
    def _generate_negations(cls, words):
        return [f"{word[0]}不{word}" for word in words]

    @classmethod
    def generate(cls):
        verb_negate_verb = cls._generate_negations(cls.verb)
        pred_negate_pred = cls._generate_negations(cls.predicate)
        
        case1 = ["".join(p) for p in product(
            cls.subject, 
            cls.verb, 
            cls.perfect_tense, 
            cls.question
        )]
        
        case2 = ["".join(p) for p in product(
            cls.subject, 
            verb_negate_verb
        )]
        
        case3 = ["".join(p) for p in product(
            cls.predicate, 
            cls.perfect_tense, 
            cls.question
        )]
        
        case4 = pred_negate_pred
        
        return cls.filter(case1 + case2 + case3 + case4)

class Curse(Phrase):
    tag = 'curse'

    w = ["我"]
    c = ["操","干","日"]
    # complement = ["烂","翻","死","爆","碎","完"]
    n = ["你","他"] # + ["她","它"]
    # pronoun_plural = ["们",""]
    m = ["妈"]
    b = ["逼","巴子","个逼","了个逼","了巴子","个巴子","了个巴子",""]
    d = ["的",""]

    @classmethod
    def generate(cls):

        cas1 = ["".join(p) for p in product(
            cls.w,
            cls.c
            )]

        cas2 = ["".join(p) for p in product(
            cls.n,
            cls.m,
            cls.b,
            cls.d
            )]

        cas3 = ["".join(p) for p in product(
            cls.m,
            cls.b,
            cls.d
            )]

        cas4 = ["".join(p) for p in product(
            cls.c,
            cls.n,
            cls.m,
            cls.b,
            cls.d
            )]

        return cls.filter(cas1+cas2+cas3+cas4)


# 使用示例
if __name__ == "__main__":
    Common.export()
    Query.export()
    Curse.export()
