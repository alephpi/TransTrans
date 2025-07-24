from itertools import product
from pathlib import Path


def filter(phrases):
    return [p for p in phrases if len(p) > 1] # 过滤掉空字符串和单字
class Phrase:
    tag = 'phrase'

    @classmethod
    def generate(cls):
        raise NotImplementedError
    
    @classmethod
    def export(cls):
        DIR = Path("./dicts")
        DIR.mkdir(exist_ok=True)
        phrases = cls.generate()
        with open(DIR / cls.__name__, "w", encoding="utf-8") as f:
            for phrase in phrases:
                f.write(f"{phrase} {cls.tag}\n")

class 指示代词(Phrase):
    # 指示代词词常用于填充，例如“这个”，但本身不完全等同于填充，部分情况需保留
    # 指示代词往往相当于定冠词 the，而定冠词并不存在于中文，中文的特指性用指示代词表示。
    tag = 'rd' #pronoun demonstrative

    prefix = ["这","那"]
    suffix = ["个","种","些","样"]

    @classmethod
    def generate(cls):
        cas1 = ["".join(p) for p in product(
            cls.prefix,
            cls.suffix,
            )] 

        return filter(cas1)


class 不定代词(Phrase):
    # 不定代词常用于填充，例如“一个、什么”，但本身不完全等同于填充，部分情况需保留
    # 不定代词往往相当于不定冠词 a，但中文里没有不定冠词，中文的泛指性用单位数词表示
    tag = 'ri' #pronoun indefinite

    prefix = ["一","某"]
    suffix = ["个","种","些","样"]

    @classmethod
    def generate(cls):

        cas = ["".join(p) for p in product(
            cls.prefix,
            cls.suffix,
            )] 
        
        cas2 = ["什么"]

        return cas + cas2

class 填充词(Phrase):
    # 填充词是必定可以去除而不改变原意的
    tag = 'filler'

    @classmethod
    def generate(cls):
        # “的话”显然是，“就是，就”有时用于强调语气，但我们激进一点，将其视为填充词。
        cas = ["的话","就是","就"]

        return cas 

class 询问语(Phrase):
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
        
        return filter(case1 + case2 + case3 + case4)

class 詈语(Phrase):
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

        return filter(cas1+cas2+cas3+cas4)

class 语气词(Phrase):
    tag = 'interj'

    @classmethod
    def generate(cls):
        s = '了 吗 吧 呀 呃 呗 呢 呦 呵 呵呵 哈 哈哈 哇 哎 哎呀 哎呦 哦 唉 啊 啦 嗯 嘛 嘿 噢 哼 嗨 喽 呗 哩'
        s = set(s.split())
        s = s.difference({'了','吗','呢'})
        return sorted(list(s))

class 口头禅(Phrase):
    tag = 'pet'
    @classmethod
    def generate(cls):
        l = [
        '我真的实事求是',
        '我实事求是',
        '我跟你们讲',
        '我跟你讲',
        '我跟你们说',
        '我跟你说',
        '简单来讲',
        '简单讲',
        '实事求是',
        '实事求是讲',
        '实话实说',
        '很好理解',
        '乱七八遭',
        '就是说',
        '这就是说',
        '那就是说',
        '也就是说',
        '或者说',
        ]
        return l



# 使用示例
if __name__ == "__main__":
    指示代词.export()
    不定代词.export()
    填充词.export()
    询问语.export()
    詈语.export()
    语气词.export()
    口头禅.export()
