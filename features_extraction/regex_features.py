#!/usr/bin/env python
# -*- coding: utf-8 -*-

import string
import re

class RegexFeatures:

    table = {
        'len': r'.',
        'exclamation_mark': r'!',
        'question_mark': r'\?',
        'dot': r'[^\.]*\.[^\.]*',
        'comma': r',',
        'punctuation': r'[{0}]'.format(re.escape(string.punctuation)),
        'consecutive_characters': r'(.)\1+',
        'alpha': r'[А-ШA-ZÜÖÄßа-шa-züäöß]',
        'diacritics': r'[šđžćčŠĐŽĆČ]',
        'cyrillic': r'[а-шА-Ш]',
        'umlauts': r'[üäößÜÖÄß]',
        'uppercase': r'[А-ШA-ZÜÖÄß]',
        'lowercase': r'[а-шa-züäöß]',
        'spaces_after_punctuation': r'[{0}] +'.format(re.escape(string.punctuation)),
        'glued_sentences': r'[^{0}]{0}[^{0}]'.format(re.escape(string.punctuation)),
        'number': r'[0-9]',
        'ne_joined_verb': r'ne[a-zA-Z]+',
        'non_capital_sent': r'\. ?[a-z]',
        'bad_dot': r'\.{2}|\.{4}',
        'bad_question': r'\?{2}|\?{4}',
    }