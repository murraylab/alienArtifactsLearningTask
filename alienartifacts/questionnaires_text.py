"""
Check to see if subjects are paying attention

"""

answers = {
    'Very rarely': 0,
    'Rarely': 0,
    'Occasionally': 0,
    'Somewhat often': 0,
    'Often': 0,
    'Very often': 1
}

QUESTIONNAIRE_ATTENTION_CHECK = {
    'I  pay attention during online experiments, choose very often as your answer.': {
        'subscale': 'NA',
        'answers': answers,
        'question_number': 1
    },
    'I am consistent in paying attention, choose the same answer as the last question.': {
        'subscale': 'NA',
        'answers': answers,
        'question_number': 2
    },
}
