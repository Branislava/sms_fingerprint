import string

class RegexFeatures:

    table = {
        'exclamation_mark': r'!',
        'question_mark': r'\?',
        'dot': r'[^\.]*\.[^\.]*',
        'comma': r',',
        'consecutive_characters': r'(.)\1+',
        'diacritics': r'[šđžćčŠĐŽĆČ]',
        'cyrilic:': r'[а-шА-Ш]',
        'umlauts': r'[üäöß]',
        'uppercase': r'[А-ШA-ZÜÖÄß]',
        'lowercase': r'[а-шa-züäöß]',
        'spaces_after_punctuation': r'[%s] +' % string.punctuation,
        'glued_sents': r'[^\.]\.[^\.]',
        'number': r'\d',
    }