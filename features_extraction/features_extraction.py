import re
from features_extraction.emoji import Emoji

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
    def retrieve_feature_position(messages_body, pattern=None):

        position_list = list()
        # 0: 0 - 33%
        # 1: 34% - 66%
        # 2: 67 - 100%
        for body in messages_body:
            N = len(body)
            found_regex = re.search(pattern, body)

            if not found_regex:
                position_list.append(-1)
            else:
                relative_position = 100.0 * found_regex.start() / N

                if relative_position < 33:
                    position_list.append(0)
                elif relative_position < 66:
                    position_list.append(1)
                else:
                    position_list.append(2)

        return position_list
