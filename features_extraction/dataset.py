import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from features_extraction.features_extraction import FeaturesExtraction
from features_extraction.emoji import Emoji
from features_extraction.regex_features import RegexFeatures

class Dataset:

    # object constructor
    def __init__(self, filename):
        # reading messages dataframe
        self.data = self.dataframe(filename)

        # add emoji counts
        # self.add_emoji_count('all')

        # add emoji type counts
        self.add_emoji_type_count('all')

        # add features defined as regex count
        self.add_feature_count('all')

        # adding regex relative position in sentence
        # self.add_feature_position('all')

        # counting occurence of same consecutive chars
        self.add_feature_count('consecutive_characters')

        # discard features with constant value
        self.discard_constant_columns()

    # read messages from xml file
    def read_messages(self, filename):
        # reading and parsing XML file
        soup = BeautifulSoup(open(filename).read(), "lxml")

        # reading file and storing it
        return ['address', 'type', 'body'], np.array([[sms['address'], sms['type'], sms['body']] for sms in soup.findAll('sms')])

    # messages to dataframe
    def dataframe(self, filename):

        # reading corpora
        cols, messages = self.read_messages(filename)

        # panda frame
        return pd.DataFrame(data=messages, index=range(0, len(messages)), columns=cols)

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
        self.data = self.data.loc[:, self.data.apply(pd.Series.nunique) != 1]

    # retrieve feature position in sentence (beginning - -1, middle - 0, end - 1)
    def add_feature_position(self, feature_name):

        if feature_name == 'all':
            # emojis
            for emoji in Emoji.table:
                self.data[emoji + '_pos'] = FeaturesExtraction.retrieve_feature_position(self.data['body'], pattern=Emoji.table[emoji][0])
        else:
            if feature_name in Emoji.table:
                self.data[feature_name + '_pos'] = FeaturesExtraction.retrieve_feature_position(self.data['body'], pattern=Emoji.table[feature_name][0])