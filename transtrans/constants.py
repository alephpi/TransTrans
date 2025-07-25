NOUN = {'n', 'nr', 'ns', 'nt', 'nx', 'nz', 'Ng'}
VERB = {'v', 'vn', 'vd', 'Vg'}
ADJ = {'a', 'ad', 'an', 'Ag', 'z'}
NUM = ['m','q','Mg']
PRON = {'r','Rg'}
ADV = {'d','Dg'}
SPACE = {'f','s'}
TIME = {'t','Tg'}
E = {'e'}
IL = {'i','l'}

INTERJ = {'interj'}

CONTENT_TAGS = set.union(set(), NOUN, VERB, ADJ, NUM, PRON, ADV, SPACE, TIME, E)
IGNORE_TAGS = set.union(set(), PRON, {'c'}, {'u'})

PKU_DICT = {
    "Ag": "形语素（形容词性语素）",
    "a": "形容词",
    "ad": "副形词（直接作状语的形容词）",
    "an": "名形词（具有名词功能的形容词）",
    "Bg": "区别语素（区别词性语素）",
    "b": "区别词",
    "c": "连词",
    "Dg": "副语素（副词性语素）",
    "d": "副词",
    "e": "叹词",
    "f": "方位词",
    "h": "前接成分",
    "i": "成语",
    "j": "简称略语",
    "k": "后接成分",
    "l": "习用语",
    "Mg": "数语素（数词性语素）",
    "m": "数词",
    "Ng": "名语素（名词性语素）",
    "n": "名词",
    "nr": "人名",
    "ns": "地名",
    "nt": "机构团体",
    "nx": "外文字符",
    "nz": "其他专名",
    "o": "拟声词",
    "p": "介词",
    "q": "量词",
    "Rg": "代语素",
    "r": "代词",
    "s": "处所词",
    "Tg": "时语素",
    "t": "时间词",
    "u": "助词",
    "Vg": "动语素",
    "v": "动词",
    "vd": "副动词",
    "vn": "名动词",
    "w": "标点符号",
    "x": "非语素字",
    "Yg": "语气语素",
    "y": "语气词",
    "z": "状态词"
}

CUSTOM_DICT = {
    "curse": "詈语",
    "filler": "填充词",
    "interj": "语气词",
    "rd": "指示代词",
    "ri": "不定代词",
    "pet": "口头禅",
    "query": "询问语",
}

DICT = {}
DICT.update(PKU_DICT)
DICT.update(CUSTOM_DICT)