# -*- coding: utf-8 -*-

from nlp_utils import *
import pypinyin
import time


def start_probability():
    """
    计算初始概率，写入文件
    顺便把汉字列表写入文件
    """
    t = time.time()
    num = 0
    probs_dict = {}
    for phrase, frequency in load_phrase():
        num += 1
        for word in phrase:
            probs_dict[word] = probs_dict.get(word, 0) + frequency
    min_value = 1
    print("计算初始概率耗时: %f s" % (time.time() - t))
    f = open(start_probability_file, 'w', encoding='utf8')
    f1 = open(states_file, 'w', encoding='utf8')
    probs_items = probs_dict.items()
    # 将概率
    # for word, prob in probs_items:
    #     p = cal_prob_log(prob, num)
    #     if p < min_value:
    #         min_value = p
    # print(min_value)
    min_value = abs(min_value) + 1
    for word, prob in probs_items:
        p = cal_prob_e(prob, num) #+ min_value
        f.write(word + split + str(p) + end)
        f1.write(word + end)
    f.close()
    f1.close()


def emission_probability():
    """
    计算发射概率，写入文件
    计算时候保存两种概率，一种是汉字到拼音，另一种是汉字到首字母
    """
    t = time.time()
    # 每个汉字对应的拼音和概率
    word_pinyins_dict = {}
    word_pinyins_dict_abbr = {}
    for phrase, frequency in load_phrase():
        # 可能有多音字
        phrase_pinyins = pypinyin.pinyin(phrase, style=pypinyin.NORMAL)
        # 每个汉字和它对应的拼音列表
        for word, pinyins in zip(phrase, phrase_pinyins):
            if word not in word_pinyins_dict:
                word_pinyins_dict[word] = {pinyin: frequency / len(pinyins) for pinyin in pinyins}
                word_pinyins_dict_abbr[word] = {get_abbr(pinyin): frequency / len(pinyins) for pinyin in pinyins}
            else:
                word_pinyins = word_pinyins_dict[word]
                word_pinyins_abbr = word_pinyins_dict_abbr[word]
                for pinyin in pinyins:
                    word_pinyins[pinyin] = word_pinyins.get(pinyin, 0) + frequency / len(pinyins)
                    word_pinyins_abbr[get_abbr(pinyin)] = word_pinyins.get(get_abbr(pinyin), 0) + frequency / len(pinyins)
    print("计算放射概率耗时: %f s" % (time.time() - t))
    f = open(emission_probability_file, 'w', encoding='utf8')
    for word, pinyins in word_pinyins_dict.items():
        total = sum(pinyins.values())
        for pinyin, prob in pinyins.items():
            f.write(word + split + pinyin + split + str(cal_prob_e(prob, total)) + end)
    f.close()

    f = open(emission_probability_abbr_file, 'w', encoding='utf8')
    for word, pinyins in word_pinyins_dict_abbr.items():
        total = sum(pinyins.values())
        for pinyin, prob in pinyins.items():
            f.write(word + split + pinyin + split + str(cal_prob_e(prob, total)) + end)
    f.close()


def transition_probability():
    """
    计算转移概率，写入文件
    """
    transition_dict = {}
    transition_dict_pp = {}
    t = time.time()

    for phrase, frequency in load_phrase():
        #只考虑前一个汉字，当前字转移到后一个字的概率
        for i in range(len(phrase) - 1):
            word = phrase[i]
            post_word = phrase[i + 1]
            if word in transition_dict:
                transition_dict[word][post_word] = transition_dict[word].get(post_word, 0) + frequency
            else:
                transition_dict[word] = {post_word: frequency}

    # 再考虑前前一个：当前字到后后一个字的转移概率
    for phrase, frequency in load_phrase():
        if (len(phrase) > 2):

            for i in range(len(phrase) - 2):
                word = phrase[i]
                mid_word = phrase[i + 1]
                word = word + mid_word
                pp_word = phrase[i + 2]
                if word in transition_dict_pp:
                    transition_dict_pp[word][pp_word] = transition_dict_pp[word].get(pp_word, 0) + frequency
                else:
                    transition_dict_pp[word] = {pp_word: frequency}

    print("计算转移概率耗时: %f s" % (time.time() - t))

    f = open(transition_probability_file, 'w', encoding='utf8')
    for word, post_words in transition_dict.items():
        total = sum(post_words.values())
        for post_word, prob in post_words.items():
            f.write(word + split + post_word + split + str(cal_prob_e(prob, total)) + end)
    f.close()

    f = open(transition_probability_pp_file, 'w', encoding='utf8')
    for word, pp_words in transition_dict_pp.items():
        total = sum(pp_words.values())
        for pp_word, prob in pp_words.items():
            f.write(word + split + pp_word + split + str(cal_prob_e(prob, total)) + end)
    f.close()


if __name__ == '__main__':
    start_probability()
    emission_probability()
    transition_probability()
