# -*- coding: utf-8 -*-

split = '\t'
end = '\n'

phrase_file = 'data/corpus/corpus.txt'

states_file = 'data/hmm/states.txt'
start_probability_file = 'data/hmm/start_probability.txt'
transition_probability_file = 'data/hmm/transition_probability.txt'
transition_probability_pp_file = 'data/hmm/transition_probability_pp.txt'
emission_probability_file = 'data/hmm/emission_probability.txt'
emission_probability_abbr_file = 'data/hmm/emission_probability_abbr.txt'

pp_weight = 1.4
break_punish_weight = 0.0000001
start_prop_punish_weight = 0.5
prop_punish_weight = 0.5
limit_size = 2
