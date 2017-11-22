import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from features_extraction.features_extraction import FeaturesExtraction
from features_extraction.emoji import Emoji

class Dataset:

    # object constructor
    def __init__(self, filename):
        # reading messages dataframe
        self.data = self.dataframe(filename)

    # read messages from xml file
    def read_messages(self, filename):
        # reading and parsing XML file
        soup = BeautifulSoup(open(filename).read(), "lxml")

        # reading file and storing it
        return ['address', 'type', 'body'], \
               np.array([[sms['address'], sms['type'], sms['body']] for sms in soup.findAll('sms')])

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
                self.data[emoji] = FeaturesExtraction.count_emoji(self.data['body'], pattern=emoji)

    # emoji group type
    def add_emoji_type_count(self, type='all'):

        # different classes of emoji
        emoji_types = set(t[1] for t in Emoji.table.values())

        if type == 'all':
            for emoji_type in emoji_types:
                self.data[emoji_type] = self.data.apply(lambda row: FeaturesExtraction.sum_emojis_of_type(row, emoji_type), axis=1)