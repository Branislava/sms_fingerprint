#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import string

from nltk import wordpunct_tokenize

class LanguageResources:
    
    # stopwords
    stopwords = ['a', 'ako', 'ali', 'bi', 'bih', 'bila', 'bili', 'bilo', 'bio', 'bismo', 'biste', 'biti', 'bumo', 'da', 'do', 'duzx', 'ga', 'hocxe', 'hocxemo', 'hocxete', 'hocxesx', 'hocxu', 'i', 'iako', 'ih', 'ili', 'iz', 'ja', 'je', 'jedna', 'jedne', 'jedno', 'jer', 'jesam', 'jesi', 'jesmo', 'jest', 'jeste', 'jesu', 'jim', 'joj', 'josx', 'ju', 'kada', 'kako', 'kao', 'koja', 'koje', 'koji', 'kojima', 'koju', 'kroz', 'li', 'me', 'mene', 'meni', 'mi', 'mimo', 'moj', 'moja', 'moje', 'mu', 'na', 'nad', 'nakon', 'nam', 'nama', 'nas', 'nasx', 'nasxa', 'nasxe', 'nasxeg', 'ne', 'nego', 'neka', 'neki', 'nekog', 'neku', 'nema', 'netko', 'necxe', 'necxemo', 'necxete', 'necxesx', 'necxu', 'nesxto', 'ni', 'nije', 'nikoga', 'nikoje', 'nikoju', 'nisam', 'nisi', 'nismo', 'niste', 'nisu', 'nxega', 'nxegov', 'nxegova', 'nxegovo', 'nxemu', 'nxezin', 'nxezina', 'nxezino', 'nxih', 'nxihov', 'nxihova', 'nxihovo', 'nxim', 'nxima', 'nxoj', 'nxu', 'no', 'o', 'od', 'odmah', 'on', 'ona', 'oni', 'ono', 'ova', 'pa', 'pak', 'po', 'pod', 'pored', 'prije', 's', 'sa', 'sam', 'samo', 'se', 'sebe', 'sebi', 'si', 'smo', 'ste', 'su', 'sve', 'svi', 'svog', 'svoj', 'svoja', 'svoje', 'svom', 'ta', 'tada', 'taj', 'tako', 'te', 'tebe', 'tebi', 'ti', 'to', 'toj', 'tome', 'tu', 'tvoj', 'tvoja', 'tvoje', 'u', 'uz', 'vam', 'vama', 'vas', 'vasx', 'vasxa', 'vasxe', 'vecx', 'vi', 'vrlo', 'za', 'zar', 'zbog', 'sxta', 'cxe', 'cxemo', 'cxete', 'cxesx', 'cxu', 'sxto', 'а', 'ако', 'али', 'би', 'бих', 'била', 'били', 'било', 'био', 'бисмо', 'бисте', 'бити', 'бумо', 'да', 'до', 'дуж', 'га', 'хоће', 'хоћемо', 'хоћете', 'хоћеш', 'хоћу', 'и', 'иако', 'их', 'или', 'из', 'ја', 'је', 'једна', 'једне', 'једно', 'јер', 'јесам', 'јеси', 'јесмо', 'јест', 'јесте', 'јесу', 'јим', 'јој', 'још', 'ју', 'када', 'како', 'као', 'која', 'које', 'који', 'којима', 'коју', 'кроз', 'ли', 'ме', 'мене', 'мени', 'ми', 'мимо', 'мој', 'моја', 'моје', 'му', 'на', 'над', 'након', 'нам', 'нама', 'нас', 'наш', 'наша', 'наше', 'нашег', 'не', 'него', 'нека', 'неки', 'неког', 'неку', 'нема', 'нетко', 'неће', 'нећемо', 'нећете', 'нећеш', 'нећу', 'нешто', 'ни', 'није', 'никога', 'никоје', 'никоју', 'нисам', 'ниси', 'нисмо', 'нисте', 'нису', 'њега', 'његов', 'његова', 'његово', 'њему', 'њезин', 'њезина', 'њезино', 'њих', 'њихов', 'њихова', 'њихово', 'њим', 'њима', 'њој', 'њу', 'но', 'о', 'од', 'одмах', 'он', 'она', 'они', 'оно', 'ова', 'па', 'пак', 'по', 'под', 'поред', 'прије', 'с', 'са', 'сам', 'само', 'се', 'себе', 'себи', 'си', 'смо', 'сте', 'су', 'све', 'сви', 'свог', 'свој', 'своја', 'своје', 'свом', 'та', 'тада', 'тај', 'тако', 'те', 'тебе', 'теби', 'ти', 'то', 'тој', 'томе', 'ту', 'твој', 'твоја', 'твоје', 'у', 'уз', 'вам', 'вама', 'вас', 'ваш', 'ваша', 'ваше', 'већ', 'ви', 'врло', 'за', 'зар', 'ће', 'ћемо', 'ћете', 'ћеш', 'ћу', 'што', 'због', 'шта']

    # abbreviations
    abbreviations = {"jel": ["je li"], "god": ["godina", "godisxte"], "bg": ["Beograd"], "fb": ["Facebook"], "jbg": ["jebi ga"], "zvrc": ["nazovi me"], "pozz": ["pozdrav"], "odg": ["odgovori", "odgovor"], "di": ["gde"], "vcs": ["vecyeras"], "dr": ["doktor"], "fejsu": ["Facebook-u"], "poz": ["pozdrav"], "jes": ["da"], "pls": ["molim te"], "npr": ["na primer"], "jbt": ["jebo te"], "rodj": ["rodxendan"], "tnx": ["hvala"], "k": ["u redu"], "faxu": ["Fakultetu"], "fax": ["Fakultet"], "cet": ["chat"], "disi": ["gde si"], "fala": ["hvala"], "kul": ["cool"], "msm": ["mislim"], "pon": ["ponedelxak"], "tj": ["to jest"],  "vrv": ["verovatno"], "bga": ["Beograda"], "bgu": ["Beogradu"], "faceu": ["Facebook-u"], "faxa": ["Fakulteta"], "itd": ["i tako dalxe"], "pozzz": ["pozdrav"], "zad": ["zadatak"], "wca": ["toaleta"], "wcu": ["toaletu"], "alg": ["algoritam"], "bzvz": ["bezveze"], "cek": ["cyekaj"], "cetv": ["cyetvrtak"], "dadada": ["da"], "dejt": ["sastanak"], "deste": ["gde ste"], "dipl": ["diploma"], "diste": ["gde ste"], "doca": ["doktor"], "dop": ["dopisivanxe", "dopuna"], "drz": ["drzxi"], "dudarimo": ["da udarimo"], "dja": ["svidxa"], "esi": ["jesi li"], "fak": ["Fakultet"], "fbu": ["Facebook-u"], "fejsa": ["Facebook-a"], "gdi": ["gde"], "gud": ["dobro"], "isklj": ["isklxucyiti"], "izvolte": ["izvolite"], "jelde": ["je li"], "koj": ["koji"], "kolok": ["kolokvijum"], "mda": ["ma da"], "mjok": ["ma ne"], "narafski": ["naravno"], "ned": ["nedelxa"], "nnc": ["nema na cyemu"], "ofkors": ["naravno"], "pliz": ["molim te"], "por": ["poruka"], "pozd": ["pozdrav"], "pvo": ["Pancyevo"], "sept": ["septembar"], "simpa": ["simpaticyno"], "smthn": ["nesxto"], "takm": ["takmicyenxe"], "ustv": ["u stvari"], "zab": ["zaboraviti"], "nmvz": ["nema veze"], "wtf": ["sxta koji moj"], "ftw": ["napred"], "najvrv": ["najverovatnije"], "nzm": ["ne znam"], "nmg": ["ne mogu"], "tel": ["telefon"], "min": ["minumum", "minut"], "max": ["maksimalno"], "rlab": ["racynarska laboratorija"], "komp": ["racynar"], "kred": ["kredit"], "vib": ["viber"], "mob": ["mobilni"], "juhu": ["pozdrav"], "hej": ["pozdrav"], "zvrcni": ["pozovi"], "desi": ["gde si"], "ej": ["pozdrav"], "aj": ["hajde"], "vamo": ["ovamo"], "bani": ["navrati"], "cimni": ["pozovi na kratko"],"ae": ["hajde"],"al": ["ali"],"dz": ["dyabe"],"eo": ["evo"],"gl": ["glavno"],"il": ["ili"],"ng": ["Nova godina"],"np": ["nema problema"],"ok": ["u redu"],"vs": ["vidimo se"],"zv": ["zovi"],"wow": ["odusxevlxenxe"],"cao": ["pozdrav"],"des": ["gde si"],"dja": ["svidxa"],"dje": ["gde"],"dog": ["dogovoriti", "dogovor"],"fix": ["fiksni telefon"],"hey": ["pozdrav"],"ing": ["inzxenxer"],"inz": ["inzxenxer"],"jej": ["radost"],"jem": ["jedem"],"koa": ["koja"],"mix": ["kombinacija"],"mos": ["mozxesx"],"net": ["Internet"],"sec": ["sekund"],"aham": ["da"],"ajoj": ["uzdah"], 'npm': ['nemam pojma'], 'mzd': ['mozxda'], 'nzm': ['ne znam'], 'stv': ['stvarno']}

    @staticmethod
    # function that replaces abbreviations with their meaning, according to lookup dictionary
    def replace_abbrevs(s):
        tokens = LanguageResources.tokenize(s)
        new_tokens = list()
        for token in tokens:
            if token in LanguageResources.abbreviations:
                # TODO: find the most adequate abbreviation from the list, not just the first
                new_tokens.append(LanguageResources.abbreviations[token][0])
            else:
                new_tokens.append(token)
        return ', '.join(new_tokens)

    # function that splits message into tokens
    @staticmethod
    def tokenize(s):
        return wordpunct_tokenize(s)

    # function that convert s to lowercase
    @staticmethod
    def lowercase(s):
        return s.lower()

    # function that returns text coded as aurora
    @staticmethod
    def encode_aurora(s):
        map_aurora = [
            ('š', 'sx'),
            ('ć', 'cx'),
            ('č', 'cy'),
            ('dž', 'zy'),
            ('dj', 'dx'),
            ('ž', 'zx'),
            ('nj', 'nx'),
            ('lj', 'lx')
        ]
        return LanguageResources.str2str(s, map_aurora)

    # aurora code umlauts
    @staticmethod
    def encode_umlauts(s):
        map_umlauts = [
            ('ü', 'ue'),
            ('ö', 'oe'),
            ('ä', 'ae'),
            ('ß', 'ss')
        ]
        return LanguageResources.str2str(s, map_umlauts)

    '''
    # recognize foreign words first using WordNet
    @staticmethod
    def recognize_foreign(s):
        new_tokens = list()
        tokens = LanguageResources.tokenize(s)

        d_en = enchant.Dict("en_US")
        d_de = enchant.Dict("de_DE")
        for token in tokens:
            if token and (wn.synsets(token) or d_en.check(token) or d_de.check(token)):
                new_tokens.append('const_strana_recy')
            else:
                new_tokens.append(LanguageResources.encode_umlauts(token))

        return ', '.join(new_tokens)
    '''

    # function that replaces smileys with certain tokens
    @staticmethod
    def encode_constants(s):
        regexps = [
            (r'\ba+h+a*\b', 'const_yes'),
            (r'\ba*([xh][ai]|[hx]|[ai])+[hx]*\b', r'const_laugh'),
            (r'([:;x]-?[\)d]+)+', r'const_smile'),
            (r'([:;x]-?p+)+', r'const_tongue'),
            (r'([:;x]-?[\/s]+)+', r'const_confused'),
            (r'([:;x]-?\*+)+', r'const_kiss'),
            (r'([:;x]-?\(+)+', r'const_sad'),
            (r'<+3+', r'const_heart'),
            (r'[0-9]+', r'const_number'),
            (r'(.)\1+', r'\1')
        ]
        return LanguageResources.regexp2token(s, regexps)

    # mimic "stem"
    @staticmethod
    def stem(s):
        return s[:5] if len(s) >= 5 else s

    # function that maps specific regex to specific token
    # within a string
    @classmethod
    def regexp2token(cls, s, regexps):
        for (regexp, replace_token) in regexps:
            s = re.sub(regexp, replace_token, s)
        return s

    # function that replaces specific substring with another string
    # within a string
    @classmethod
    def str2str(cls, s, mappings):
        for (orig_substr, replace_substr) in mappings:
            s = s.replace(orig_substr, replace_substr)
        return s

    # function that extracts only alphabetical tokens
    @classmethod
    def sanitize(cls, tokens):
        result_pair_tokens = list()
        regex_interpunc = re.compile(r'[%s]' % '\\-\!"#$%&\'()*+,./:;<=>?@[\]^`{|}~')
        regex_multichars = re.compile(r'([a-z])\1+')
        for token in tokens:
            new_token = ''.join(c for c in regex_interpunc.sub('', token) if c in (string.printable + "абвгдђежзијклљмнњопрстуфхцчћџшüäöß"))
            new_token = re.sub(regex_multichars, r'\1', new_token)
            if new_token in LanguageResources.stopwords:
                new_token = ''
            result_pair_tokens.append((token, new_token))
        return result_pair_tokens

    # function that prepares string for analysis (i.e. extracts words only)
    @staticmethod
    def prepare(s):
        return LanguageResources.sanitize(LanguageResources.tokenize(s))