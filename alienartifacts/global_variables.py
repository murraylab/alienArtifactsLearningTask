import os
from .questionnaires_text import *


#Choose task
TASK = 'context-generalization_v1' # 'example-generalization', 'context-generalization_v1', 'context-generalization_v2', 'diagnostic'

# Specify if deploying
DEPLOYMENT = True
PAYMENT_TOKEN = 'replace_token'
ATTENTION_FAILURE_TOKEN = 'replace_token'
ATTENTION_CHECK = True

# Specific  text
INSTRUCTIONS = "You're a space pirate roving the galaxy, trying to get rich. On this quest for fame \
            and fortune, you come across a store of alien artifacts. They hold incredible power - if you can figure out \
            how to activate them. First though, we'll travel through space in a tutorial on task structure."
EXIT_TEXT = "Good work Space Pirate! You accumulated %d units of energy from the artifacts."

# Questions
if DEPLOYMENT:
    SUBJECT_SOURCES = [("prolific", "Prolific"),('amazon', "Amazon Mechanical Turk"), ("pavlovia", "Pavlovia")]
else:
    SUBJECT_SOURCES = [('internal', 'Internal')]

GENDERS = [('','Please Select Response'),('female', 'Female'), ('male', 'Male'), ('nonbinary', 'Nonbinary'), ('none', 'Prefer not to say')]
AGES = [('','Please Select Response'),('<10', "Under 10"), ('10-20', '11-20'), ('21-30', '21-30'), ('31-45', '31-45'),
        ('46-65', '46-65'), ('>65', 'Over 65')]
EDUCATION = [('','Please Select Response'),('<highschool', "Some Highschool"), ('highschool', 'Highschool Graduate'), ('<college', 'Some College'),
             ('college', 'College Graduate'), ('postgrad', 'Postgraduate')]
DIFFICULTY = [('easy', 'Easy'), ('medium', 'Medium'), ('hard', 'Hard')]
SUBSTANCES = [('caffeine','Caffeine'),('adhd_stimulants','ADHD medication (e.g. Ritalin)'),('alcohol','Alcohol'),
              ('tobacco','Tobacco'),('marijuana','Marijuana'),('opioids','Opioids'),('illicit_stimulants','Cocaine or meth')]
MH_HISTORY = [('asd','Autism spectrum disorder'),('adhd','Attention deficit hyperactivity disorder'),
              ('ocd','Obsessive compulsive disorder'),('depression','Depression'),('schizophrenia','Schizophrenia'),
              ('schizotypy','Schizotypal personality'),('addiction','Substance use disorder')]
ATTENTION_CHECK_HISTORY = [('fail_attention_check','Femur dissolution'),('fail_attention_check','Malconforsethia'),
              ('pass_attention_check','Prosochiphelia'),('fail_attention_check','Retinal dermatitis')]


QUESTIONNAIRES = {
        'att_check': QUESTIONNAIRE_ATTENTION_CHECK,
    }


# Task Structure
SINGLE_PAGE_BLOCK_LENGTH = 10 # 10
DEBUG = os.environ.get('DJANGO_DEBUG', '') != 'False'
STRUCTURED = True
PAYMENT_TOKEN_LENGTH = 16

DIAGNOSTIC_COUNTER_BLOCK_LEN = 10
DIAGNOSTIC_COUNTER_N_BLOCKS = 1000
DIAGNOSTIC_COUNTER_CURRENT_BLOCK = 0