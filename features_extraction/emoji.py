#!/usr/bin/env python
# -*- coding: utf-8 -*-

class Emoji:

    table = {
        ### smiley ###
        'smiley_wo_nose': (r':\)', 'smiley'),
        'smiley_w_nose': (r':-\)', 'smiley'),

        'smileys_wo_nose_reverse': (r'\({2,}:', 'smiley'),
        #'smileys_w_nose_reverse': (r'\({2,}-:', 'smiley'),
        'smiley_wo_nose_reverse': (r'\(:', 'smiley'),
        #'smiley_w_nose_reverse': (r'\(-:', 'smiley'),

        #'smileys_cry_wo_nose': (r':\'\){2,}', 'smiley'),
        #'smileys_cry_w_nose': (r':-\'\){2,}', 'smiley'),
        #'smiley_cry_wo_nose': (r':\'\)', 'smiley'),
        #'smiley_cry_w_nose': (r':-\'\)', 'smiley'),

        #'smileyes_closed_wo_nose': (r'x\){2,}', 'smiley'),
        #'smileyes_closed_w_nose': (r'x-\){2,}', 'smiley'),
        #'smiley_closed_wo_nose': (r'x\)', 'smiley'),
        #'smiley_closed_w_nose': (r'x-\)', 'smiley'),

        ### happy ###
        'happies_wo_nose': (r':D{2,}', 'happy'),
        'happies_w_nose': (r':-D{2,}', 'happy'),
        'happy_wo_nose': (r':D', 'happy'),
        #'happy_w_nose': (r':-D', 'happy'),

        'happies_closed_wo_nose': (r'xD{2,}', 'happy'),
        #'happies_closed_w_nose': (r'x-D{2,}', 'happy'),
        'happy_closed_wo_nose': (r'xD', 'happy'),
        #'happy_closed_w_nose': (r'x-D', 'happy'),

        #'relax_happies': (r'=D{2,}', 'happy'),

        #'happy_angel_face_wo_nose': (r'O:D', 'happy'),
        #'happy_angel_face_w_nose': (r'O:D', 'happy'),
        #'happy_angel_face_wo_nose_0': (r'0:D', 'happy'),
        #'happy_angel_face_w_nose_0': (r'0:D', 'happy'),

        ### sad ###
        'sad_wo_nose': (r':\(', 'sad'),
        'sad_w_nose': (r':-\(', 'sad'),
        #'sad_oblique_wo_nose': (r':\[', 'sad'),
        #'sad_oblique_w_nose': (r':-\[', 'sad'),

        'sads_w_nose': (r':-\({2,}', 'sad'),
        'sads_wo_nose': (r':\({2,}', 'sad'),
        #'sads_oblique_wo_nose': (r':\[{2,}', 'sad'),
        #'sads_oblique_w_nose': (r':-\[{2,}', 'sad'),
        #'crys_wo_nose': (r":'\({2,}", 'sad'),
        'crys_w_nose': (r":-'\({2,}", 'sad'),
        #'cry_wo_nose': (r":'\(", 'sad'),
        'cry_w_nose': (r":-'\(", 'sad'),

        ### suprise ###
        'suprised_wo_nose_small': (r':o', 'suprised'),
        #'suprised_wo_nose': (r':O', 'suprised'),
        'suprised_w_nose_small': (r':-o', 'suprised'),
        #'suprised_w_nose': (r':-O', 'suprised'),

        ### kiss ###
        'kisses_wo_nose': (r':\*{2,}', 'kiss'),
        'kisses_w_nose': (r':-\*{2,}', 'kiss'),
        #'kiss_wo_nose_closed': (r'x\*', 'kiss'),
        #'kiss_w_nose_closed': (r'x-\*', 'kiss'),
        #'kisses_wo_nose_closed': (r'x\*{2,}', 'kiss'),
        #'kisses_w_nose_closed': (r'x-\*{2,}', 'kiss'),
        'kiss_wo_nose': (r':\*', 'kiss'),
        'kiss_w_nose': (r':-\*', 'kiss'),

        ### wink ###
        'winks_wo_nose': (r';\){2,}', 'wink'),
        'winks_w_nose': (r';-\){2,}', 'wink'),
        #'winks_happy_wo_nose': (r';D{2,}', 'wink'),
        #'winks_happy_w_nose': (r';-D{2,}', 'wink'),
        'wink_wo_nose': (r';\)', 'wink'),
        'wink_w_nose': (r';-\)', 'wink'),
        #'wink_happy_wo_nose': (r';D', 'wink'),
        #'wink_happy_w_nose': (r';-D', 'wink'),

        ### tongue ###
        'tongue_w_nose': (r':-P{2,}', 'tongue'),
        'tongue_w_nose_small': (r':-p{2,}', 'tongue'),
        'tongues_wo_nose_small': (r':p{2,}', 'tongue'),
        'tongues_wo_nose': (r':-P{2,}', 'tongue'),
        #'tongue_w_nose_closed': (r'x-P{2,}', 'tongue'),
        #'tongue_w_nose_small_closed': (r'x-p{2,}', 'tongue'),
        #'tongues_wo_nose_small_closed': (r'xp{2,}', 'tongue'),
        #'tongues_w_nose_small_closed': (r'x-p{2,}', 'tongue'),
        #'tongue_wo_nose': (r':P', 'tongue'),
        #'tongue_wo_nose_small': (r':p', 'tongue'),
        #'tongues_wo_nose': (r':P', 'tongue'),
        #'tongues_w_nose': (r':-P', 'tongue'),
        #'tongue_wo_nose_closed': (r'xP', 'tongue'),
        #'tongue_wo_nose_small_closed': (r'xp', 'tongue'),
        #'tongues_wo_nose_closed': (r'xP', 'tongue'),
        #'tongues_w_nose_closed': (r'x-P', 'tongue'),

        ### skeptic ###
        'skeptics_wo_nose': (r':/{2,}', 'skeptic'),
        #'skeptics_w_nose': (r':-/{2,}', 'skeptic'),
        'skeptic_wo_nose': (r':/', 'skeptic'),
        #'skeptic_w_nose': (r':-/', 'skeptic'),
        #'straight_wo_nose': (r':\|', 'skeptic'),
        #'straight_w_nose': (r':-\|', 'skeptic'),

        ### piggy, relax, curly, shy, glasses, angels etc. ###
        #'curly_wo_nose': (r':\}', 'misc'),
        #'curly_w_nose': (r':-\}', 'misc'),
        #'piggy_nose': (r':o\)', 'misc'),
        'relax_angle': (r'=\]', 'misc'),

        #'relax_obliques': (r'=\){2,}', 'misc'),
        #'relax_oblique': (r'=\)', 'misc'),

        'relax_happy': (r'=D', 'misc'),
        #'shy_wo_nose': (r':>', 'misc'),
        #'shy_w_nose': (r':->', 'misc'),

        #'glasses_wo_nose': (r'8\){2,}', 'misc'),
        #'glasses_w_nose': (r'8-\){2,}', 'misc'),
        #'glass_wo_nose': (r'8\)', 'misc'),
        'glass_w_nose': (r'8-\)', 'misc'),

        #'smiley_angel_face_wo_nose': (r'O:\)', 'misc'),
        'smiley_angel_face_w_nose': (r'O:-\)', 'misc'),
        #'smiley_angel_face_wo_nose_0': (r'0:\)', 'misc'),
        #'smiley_angel_face_w_nose_0': (r'0:-\)', 'misc')
    }