# -*- coding: utf-8 -*-

import math
import time
from config import *


def load_phrase():
    """
    加载词表，返回迭代器
    """
    f = open(phrase_file, 'r', encoding='utf8')
    lines = f.readlines()
    for line in lines:
        params = line.strip().split(split)
        yield params[0], int(params[1])


def sigmod(x):
    return 1 / (1 + math.exp(-x))


def cal_prob_sigmod(frequency, base_number):
    return round(sigmod(frequency / base_number), 8)


def cal_prob_e(frequency, base_number):
    """
    计算概率，保留8位小数，概率计算公式为p = exp(frequency / base_number)
    :param frequency:频数
    :param base_number:基数
    :return:保留8位后的概率
    """
    return round(math.exp(frequency / base_number), 8)


def cal_prob_log(frequency, base_number):
    """
    计算概率，保留8位小数，概率计算公式为p = log(frequency / base_number)
    :param frequency:频数
    :param base_number:基数
    :return:保留8位后的概率
    """
    # if frequency == base_number:
    #     return -0.00001
    return round(start_prop_punish_weight * math.log(frequency / base_number), 8)


def states_list():
    t = time.time()
    statuses = []
    f = open(states_file, 'r', encoding='utf8')
    lines = f.readlines()
    for line in lines:
        statuses.append(line.strip())
    print("加载状态列表，耗时: %f s" % (time.time() - t))
    return statuses


def start_prop_dict():
    t = time.time()
    prop_dict = {}
    f = open(start_probability_file, 'r', encoding='utf8')
    lines = f.readlines()
    for line in lines:
        params = line.strip().split(split)
        prop_dict[params[0]] = params[1]
    print("加载初始概率，耗时: %f s" % (time.time() - t))
    return prop_dict


def emission_prop_dict():
    t = time.time()
    prop_dict = {}
    f1 = open(emission_probability_file, 'r', encoding='utf8')
    lines1 = f1.readlines()
    f2 = open(emission_probability_abbr_file, 'r', encoding='utf8')
    lines2 = f2.readlines()
    lines1.extend(lines2)
    for line in lines1:
        params = line.strip().split(split)
        # 拼音作为key
        if params[1] in prop_dict:
            prop_dict[params[1]][params[0]] = params[2]
        else:
            prop_dict[params[1]] = {params[0]: params[2]}
    print("加载放射概率，耗时: %f s" % (time.time() - t))
    return prop_dict


def transition_prop_dict(file):
    t = time.time()
    prop_dict = {}
    f = open(file, 'r', encoding='utf8')
    lines = f.readlines()
    for line in lines:
        params = line.strip().split(split)
        # 先前的词作为key
        if params[0] in prop_dict:
            prop_dict[params[0]][params[1]] = params[2]
        else:
            prop_dict[params[0]] = {params[1]: params[2]}
    print("加载转移概率，耗时: %f s" % (time.time() - t))
    return prop_dict


def break_punisher(p):
    p = abs(p)
    return p * break_punish_weight


def p_punisher(p):
    p = abs(p)
    return p * prop_punish_weight


shengmu = ['b', 'p', 'm', 'f', 'd', 't', 'n', 'l', 'g', 'k', 'h',
          'j', 'q', 'x', 'zh', 'ch', 'sh', 'r', 'z', 'c', 's', 'y', 'w']
yunmu = ['a', 'o', 'e', 'i', 'u', 'v', 'ai', 'ei', 'ui', 'ao', 'ou',
        'iu', 'ie', 've', 'er', 'an', 'en', 'in', 'un', 'vn', 'ang', 'eng', 'ing', 'ong',
        'ia', 'ua', 'ue', 'uo', 'uai', 'uan', 'ian', 'iao', 'iang', 'uang', 'iong']
duliyunmu = ['a', 'o', 'e', 'ai', 'ei', 'ui', 'ao', 'ou',
        'er', 'an', 'en', 'ang', 'eng']


def get_abbr(pinyin):
    """
    给定拼音，得到其声母部分或者首字符
    :param pinyin:
    :return:
    """
    for sm in shengmu:
        if pinyin.startswith(sm):
            return sm
    return pinyin[0]


def generate_pinyin_rules():
    rules = []
    rules += shengmu
    rules += yunmu
    for sm in shengmu:
        rules += map(lambda s:s[0]+s[1], [(sm,ym) for ym in yunmu])
    rules += duliyunmu
    return set(rules)


def isduli(s):
    if s in shengmu or s in duliyunmu:
        return True
    else:
        return False


def check(i, txt):
    length = len(txt)
    if i > length-1:
        return True
    if i == length-1 and isduli(txt[i]):
        return True
    if i < length-1 and (isduli(txt[i]) or isduli(txt[i:i+2])):
        return True
    return False


def pinyin_seperator(txt):
    rules = generate_pinyin_rules()
    results = []
    while(txt):
        now_word = ''
        tag, btag = 0, False
        length = len(txt)
        for i in range(1, min(7,length+1)):
            if txt[:i][-1] == "'":
                now_word = txt[:i-1][:]
                btag = True
                tag = i
                break
            if txt[:i] in rules:
                now_word = txt[:i][:]
                tag = i
        if not btag and now_word[-1] in shengmu and not check(tag, txt):
            tag -= 1
            now_word = now_word[:-1]
        txt= txt[tag:]
        results.append(now_word)
    return results


if __name__ == '__main__':
    for i in range(1, 1):
        print(i)