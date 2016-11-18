# -*- coding: utf-8 -*-

from nlp_utils import *
import copy

states = states_list()
start_dict = start_prop_dict()
emission_dict = emission_prop_dict()
transition_dict = transition_prop_dict(transition_probability_file)
transition_pp_dict = transition_prop_dict(transition_probability_pp_file)


def get_start_prop(word):
    """
    查询某个汉字的初始化概率
    :param word:
    :return:
    """
    if word in start_dict:
        return float(copy.deepcopy(start_dict[word]))
    else:
        return None


def get_words_emit_pinyin(pinyin):
    """
    查询放射矩阵，得到可以放射到某个拼音的汉字
    :param pinyin:
    :return:
    """
    words_dict = emission_dict.get(pinyin, None)
    if words_dict is None:
        return None
    return copy.deepcopy(words_dict)


def get_transition_prob(pre_word, word):
    """
    得到两个汉字之间的转移概率
    :param word:
    :return:
    """
    if pre_word in transition_dict:
        transition = transition_dict.get(pre_word)
        if word in transition:
            return float(copy.deepcopy(transition[word]))
    return None


def get_transition_pp_prob(prepre_word, word):
    """
    得到两个汉字之间的转移概率
    :param post_word:
    :return:
    """
    if prepre_word in transition_pp_dict:
        transition = transition_pp_dict.get(prepre_word)
        if transition is not None and word in transition:
            return float(copy.deepcopy(transition[word]))
    return None


def get_max_prob(prob_dict):
    """
    得到概率最大的词组
    :param prob_dict:
    :return:
    """
    probs_list = sorted(prob_dict.items(), key=lambda d: d[1], reverse=True)
    return probs_list[0]


def order_dict(dic):
    """
    给词典按照value值由大到小排序
    :param dic:
    :return:
    """
    return sorted(dic.items(), key=lambda d: d[1], reverse=True)


def merge_path(path_list):
    """
    合并路径列表为词序列，便于输出
    :param path_list:
    :return:
    """
    line = ''
    for path in path_list:
        line += path
    return line



def viterbi(obs, words_list, limit=8):
    """
    维特比算法的实现
    :param obs:
    :param limit:
    :return:
    """

    # 初始情况，t=0
    V = [{}]
    path = {}
    for word, emit_prob in words_list:
        V[0][word] = emit_prob
        path[word] = [word]

    # print(order_dict(V[0]))

    # 当t>0
    for t in range(1, len(obs)):
        V.append({})
        new_path = {}
        words_dict_t = get_words_emit_pinyin(obs[t])
        if words_dict_t is None:
            return None
        # 标记当前层和前一层是否有连接
        is_connected = False
        for word_t, emit_prob_t in words_dict_t.items():
            # 当前层的某个字到前一层的最大概率的连接
            word_max = None
            p_max = 0
            for word, prob in V[t - 1].items():
                trans_p = get_transition_prob(word, word_t)
                if trans_p is not None:
                    p_t = float(prob) * trans_p  # * float(emit_prob_t)

                    if t > 1: # 3个字及以上
                        # print(type(path[word]), path[word])
                        temp_path = path[word]
                        pre_word = temp_path[len(temp_path) - 1]
                        prepre_word = temp_path[len(temp_path) - 2]
                        pre = prepre_word + pre_word
                        # print(pre, word_t)
                        trans_pp_prob = get_transition_pp_prob(pre, word_t)
                        if trans_pp_prob is not None:
                            # print(pre, word_t, trans_pp_prob)
                            p_t = p_t * trans_pp_prob * pp_weight
                    p_t = p_punisher(p_t)

                    if p_t > p_max:
                        p_max = p_t
                        word_max = word
            if word_max is not None:
                is_connected = True
                V[t][word_t] = p_max
                new_path[word_t] = path[word_max] + [word_t]
        if is_connected is False:
            # print("false")
            # 重新开始，选择当前起始概率最大的汉字，与前一层概率最大的汉字连接
            new_start_words = {}
            for w in word_t:
                new_start_words[w] = get_start_prop(w)
            new_start_word_max, new_start_prob = get_max_prob(new_start_words)
            V[t][new_start_word_max] = break_punisher(p_punisher(new_start_prob * float(words_dict_t[new_start_word_max])))
            word_max = get_max_prob(V[t - 1])[0]
            new_path[new_start_word_max] = path[word_max] + [new_start_word_max]
        path = new_path

    last = V[len(obs) - 1]
    last = sorted(last.items(), key=lambda d: d[1], reverse=True)
    if len(last) > limit:
        last = last[0: limit]
    return last, path


def cal_candidates(pinyins, if_print = True, page = 1, limit=18):
    """
    根据已经分好的拼音序列，计算对应的汉字序列
    :param pinyins:
    :return:
    """
    if len(pinyins) == 1:
        limit = 10000
    # 初始情况，t=0
    words_dict = get_words_emit_pinyin(pinyins[0])
    if words_dict is None:
        return None
    for word, emit_prob in words_dict.items():
        start_prop = get_start_prop(word)
        # 计算初始概率
        p0 = start_prop # * float(emit_prob)
        words_dict[word] = p_punisher(p0)
    # 截取起始的搜索词，以初始概率从大到小
    words_dict = order_dict(words_dict)
    # print(words_dict)
    size = limit_size * limit
    start = size * (page - 1)
    if len(words_dict) > start:
        words_dict = words_dict[start: size * page]

    candidates = []
    result = viterbi(pinyins, words_dict, limit)
    if result is not None:
        for l, p in result[0]:
            phrase = merge_path(result[1][l])
            if if_print:
                print(phrase, p)
            candidates.append(phrase)
    return candidates


if __name__ == '__main__':
    while True:
        raw_pinyins = input('input:').strip()
        if raw_pinyins.strip() == '':
            continue
        if raw_pinyins.startswith('end'):
            print("Bye!")
            exit()
        if ' ' in raw_pinyins.strip():
            pinyins = raw_pinyins.split(' ')
        else:
            pinyins = pinyin_seperator(raw_pinyins)
        print(pinyins)
        cal_candidates(pinyins, page=1, limit=20)


