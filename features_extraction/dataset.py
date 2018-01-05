#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from .emoji import Emoji
from .features_extraction import FeaturesExtraction
from .language_resources import LanguageResources
from .regex_features import RegexFeatures


class Dataset:

    # object constructor
    def __init__(self, filename, verbose=False):

        # reading messages dataframe
        print('Reading data...') if verbose else False
        self.data, self.target = self.dataframe(filename)

        # add emoji counts
        print('Counting emojis...') if verbose else False
        self.add_emoji_count('all')

        # add emoji type counts
        print('Summarizing emoji information...') if verbose else False
        self.add_emoji_type_count('all')

        # add features defined as regex count
        print('Adding linguistic features...') if verbose else False
        self.add_feature_count('all')

        # count abbreviations usage
        print('Adding abbreviations usage...') if verbose else False
        self.add_abbrev_count()

        # adding features ratio
        print('Adding certain linguistic features ratio...') if verbose else False
        self.add_features_ratio([
            ('exclamation_mark', 'len'),
            ('question_mark', 'len'),
            ('dot', 'len'),
            ('comma', 'len'),
            ('punctuation', 'len'),
            ('alpha', 'len'),
            ('diacritics', 'len'),
            ('umlauts', 'len'),
            ('uppercase', 'len'),
            ('lowercase', 'len'),
            ('number', 'len'),
            ('uppercase', 'lowercase'),
            ('punctuation', 'alpha'),
            ('number', 'alpha'),
            ('cyrillic', 'len'),
            ('cyrillic', 'alpha'),
        ])

        # preprocessing: aurora, reducing similar words
        #print('Preprocessing data...') if verbose else False
        #self.clean_data()

        # adding tf-idf for bag-of-words
        #print('Adding tf-idf features for BoW...') if verbose else False
        #self.add_tfidf()

        # discard features with constant value
        print('Discarding constant columns...') if verbose else False
        self.discard_constant_columns()
        
        # dropping str features
        self.data.drop('body', axis=1, inplace=True)

        print('Dataset ready!') if verbose else False

    # read messages from xml file
    def read_messages(self, filename):
        # reading and parsing XML file
        soup = BeautifulSoup(open(filename).read().lower(), "lxml")

        # reading file and storing it
        X = np.array([sms['body'] for sms in soup.findAll('sms')])
        y = np.array([sms['type'] for sms in soup.findAll('sms')])
        
        return X, np.array(list(map(int, y)))

    # messages to dataframe
    def dataframe(self, filename):

        # reading corpora
        X, y = self.read_messages(filename)

        # panda frame
        return pd.DataFrame(data=X, index=range(0, len(X)), columns=['body']), y

    # compose preprocessing functions
    def compose(self, g, h, i, j, k):
        # WARNING: LanguageResources.recognize_foreign
        return lambda body: g((h(i(j(k(body))))))

    # preprocessing
    def clean_data(self):

        # WARNING: LanguageResources.recognize_foreign
        composition = self.compose(LanguageResources.stem, LanguageResources.replace_abbrevs, LanguageResources.encode_aurora, LanguageResources.encode_umlauts, LanguageResources.encode_constants)

        self.data['body'] = self.data.apply(lambda row: composition(row['body']), axis=1)

    # add emoji count features
    def add_emoji_count(self, type='all'):

        if type == 'all':
            for emoji in Emoji.table:
                self.data[emoji] = FeaturesExtraction.count_feature(self.data['body'], pattern=Emoji.table[emoji][0])
        else:
            self.data[type] = FeaturesExtraction.count_feature(self.data['body'], pattern=Emoji.table[type][0])

    # emoji group type
    def add_emoji_type_count(self, type='all'):

        # different classes of emoji
        emoji_types = set(t[1] for t in Emoji.table.values())

        if type == 'all':
            for emoji_type in emoji_types:
                self.data[emoji_type] = FeaturesExtraction.sum_emojis_of_type(self.data['body'], emoji_type)
        else:
            self.data[type] = FeaturesExtraction.sum_emojis_of_type(self.data['body'], type)

    # add feature count
    def add_feature_count(self, feature_name):

        if feature_name == 'all':
            for feature in RegexFeatures.table:
                self.data[feature] = FeaturesExtraction.count_feature(self.data['body'], pattern=RegexFeatures.table[feature])
        else:
            self.data[feature_name] = FeaturesExtraction.count_feature(self.data['body'], pattern=RegexFeatures.table[feature_name])

    # discard constant columns
    def discard_constant_columns(self):
        self.data = self.data.loc[:, (self.data != self.data.iloc[0]).any()] 

    # retrieve feature position in sentence (beginning - -1, middle - 0, end - 1)
    def add_feature_position(self, feature_name):

        if feature_name == 'all':
            # emojis
            for emoji in Emoji.table:
                self.data[emoji + '_pos'] = FeaturesExtraction.retrieve_feature_position(self.data['body'], pattern=Emoji.table[emoji][0])
        else:
            if feature_name in Emoji.table:
                self.data[feature_name + '_pos'] = FeaturesExtraction.retrieve_feature_position(self.data['body'], pattern=Emoji.table[feature_name][0])

    # adding abbreviations usage
    def add_abbrev_count(self):
        
        for feature in LanguageResources.abbreviations:
            self.data[feature] = FeaturesExtraction.count_feature(self.data['body'], pattern=feature)

    # we want ratio of some features
    def add_features_ratio(self, values=None):

        for feature_name1, feature_name2 in values:
            self.data['%s_%s_ratio' % (feature_name1, feature_name2)] = FeaturesExtraction.add_features_ratio(self.data['body'], RegexFeatures.table[feature_name1], RegexFeatures.table[feature_name2])

    # adding tf-idf for bag-of-words for 'body' column
    def add_tfidf(self):

        df1 = FeaturesExtraction.add_tfidf(self.data['body'])
        self.data = pd.concat([self.data, df1], axis=1)
