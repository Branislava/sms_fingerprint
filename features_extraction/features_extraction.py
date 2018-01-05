#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from .language_resources import LanguageResources
from .emoji import Emoji

class FeaturesExtraction:

    @staticmethod
    def sum_emojis_of_type(messages_body, emoji_type):
        sum_list = list()

        for body in messages_body:
            sum_list.append(sum(len(re.findall(Emoji.table[emoji][0], body)) for emoji in Emoji.table if Emoji.table[emoji][1] == emoji_type))

        return sum_list

    @staticmethod
    def count_feature(messages_body, pattern=None):
        counts_list = list()

        for body in messages_body:
            counts_list.append(len(re.findall(pattern, body)))

        return counts_list

    @staticmethod
    def add_features_ratio(messages_body, pattern1, pattern2):
        counts_list = list()

        for body in messages_body:
            num = len(re.findall(pattern1, body))
            denum = len(re.findall(pattern2, body))
            counts_list.append(0 if not num or not denum else min(num/denum, denum/num))

        return counts_list

    @staticmethod
    def add_tfidf(messages_body, n_low=1, n_up=3):
        v = TfidfVectorizer(ngram_range=(n_low, n_up), stop_words=LanguageResources.stopwords)
        x = v.fit_transform(messages_body).todense()
        return pd.DataFrame(x, columns=v.get_feature_names())
