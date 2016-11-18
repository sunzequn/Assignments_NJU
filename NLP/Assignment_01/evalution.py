# -*- coding: utf-8 -*-
import pypinyin
import random
from model import *

n = 10000

phrases = [phrase for phrase, frequency in load_phrase()]
phrase_sample = random.sample(phrases, n)

total = 0
t = time.time()
for ph in phrase_sample:
    pinyins = pypinyin.pinyin(ph, style=pypinyin.NORMAL)
    pinyin_line = ''
    total += len(pinyins)
    for pinyin in pinyins:
        pinyin_line += pinyin[0]
        # print(pinyin_line)
        split_pinyin = pinyin_seperator(pinyin_line)
        candidates = cal_candidates(split_pinyin, if_print=False)
print("短语数量: %d" % n)
print("平均长度: %f" % (total / n))
print("耗时: %f s" % (time.time() - t))
