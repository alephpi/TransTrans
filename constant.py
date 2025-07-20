import abc
from itertools import product
from pathlib import Path


class Phrase:
    @classmethod
    def filter(cls, phrases):
        return [p for p in phrases if len(p) > 1] # 过滤掉空字符串和单字

    @classmethod
    def generate(cls):
        raise NotImplementedError
    
    @classmethod
    def export(cls):
        DIR = Path("./orals")
        DIR.mkdir(exist_ok=True)
        phrases = cls.generate()
        with open(DIR / f"{(cls.__name__).lower()}.oral", "w", encoding="utf-8") as f:
            f.write("\n".join(phrases))

class Query(Phrase):

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

    subject = ["我", ""]
    verb = ["操","干","日"]
    complement = ["烂","翻","死","爆","碎","完"]
    # perfect_tense = ["了", ""]
    pronoun = ["你","他"] # + ["她","它"]
    # pronoun_plural = ["们",""]
    m = ["妈"]
    lg = ["个","了个",""]
    b = ["逼"]
    # d = ["的",""]

    @classmethod
    def generate(cls):
        # case1 = ["".join(p) for p in product(
        #     cls.verb,
        #     cls.complement,
        #     # cls.perfect_tense,
        # )]
        # case1 更接近表程度，而非填充词

        case2 = ["".join(p) for p in product(
            cls.subject,
            cls.verb,
            # cls.perfect_tense,
        )]

        case3 = ["".join(p) for p in product(
            cls.pronoun,
            # cls.pronoun_plural,
            cls.m,
            # cls.d,
            )]

        case4 = ["".join(p) for p in product(
            cls.m,
            cls.lg,
            cls.b,
            # cls.d,
        )]

        return cls.filter(case2 + case3 + case4)


# 使用示例
if __name__ == "__main__":
    Query.export()
    Curse.export()
