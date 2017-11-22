import re
from features_extraction.emoji import Emoji

class FeaturesExtraction:

    @staticmethod
    def count_emoji(messages_body, pattern=None):
        counts_list = list()

        for body in messages_body:
            counts_list.append(len(re.findall(Emoji.table[pattern][0], body)))

        return counts_list

    @staticmethod
    def sum_emojis_of_type(row, emoji_type):
        return sum(row[column] for column in Emoji.table if Emoji.table[column][1] == emoji_type)
