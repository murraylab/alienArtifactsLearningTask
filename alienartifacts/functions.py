from django.shortcuts import redirect, render
from django.http import HttpResponseRedirect
from django.urls import reverse
from django.views import generic
from django.conf import settings
from django.contrib import messages

import json
import urllib
import random
import base64
from django.http import JsonResponse
from django import forms
from django.forms import ModelForm, modelformset_factory, BaseModelFormSet
from .models import Stimulus, Session, Subject, Trial, TutorialStimulus, RewardStimulus, QuestionnaireQ
from azuresite.settings import GOOGLE_RECAPTCHA_SITE_KEY
from datetime import datetime
from django.core.files import File
import numpy as np
from glob import glob
import os, logging
import copy
from .global_variables import *
from secrets import token_urlsafe

from secrets import token_urlsafe

## FUNCTIONS

def stimulusCombinations(stimuli,reward_rules=None):
    if 'colors' in stimuli.keys():
        try:
            stimulus_combinations_initial = [[i, j, k, l] for i in stimuli['colors']
                                     for j in stimuli['shapes']
                                     for k in stimuli['textures']
                                     for l in stimuli['sizes']]
        except:
            stimulus_combinations_initial = [[i, j, k] for i in stimuli['colors']
                                       for j in stimuli['shapes']
                                       for k in stimuli['textures']]
    else:
        stimulus_combinations_initial = []
        for stim in stimuli:
            stim_combo = [stimuli[stim]['color'], stimuli[stim]['shape'], stimuli[stim]['texture'],
                          stimuli[stim]['size']]
            stimulus_combinations_initial.append(stim_combo)
    if reward_rules is not None: #Make sure the stimuli are included in the reward probabilities
        stimulus_combinations = []
        for combo in stimulus_combinations_initial:
            if getRewardProbabilities(shape=combo[1], color=combo[0], texture=combo[2], size=combo[3],
                                      reward_rules=reward_rules) is not None:
                stimulus_combinations.append(combo)
    else:
        stimulus_combinations = stimulus_combinations_initial
    return stimulus_combinations


def getRewardProbabilities(shape,color,texture,size=None,reward_rules=None):
    # Check rules and make sure there's no overlapping membership
    if reward_rules is None:
        raise ValueError('Need Reward Rules')
    rules = reward_rules.keys()
    probs_set = False
    reward_probabilities = None
    for rule in rules:
        if size is None:
            if (shape in reward_rules[rule]['attributes']['shape']) and \
                    (color in reward_rules[rule]['attributes']['color']) and \
                    (texture in reward_rules[rule]['attributes']['texture']):
                if probs_set:
                    raise ValueError("Two rules have overlapping membership")
                else:
                    reward_probabilities = reward_rules[rule]['reward_probabilities']
                    probs_set = True
        else:
            if (shape in reward_rules[rule]['attributes']['shape']) and \
                    (color in reward_rules[rule]['attributes']['color']) and \
                    (texture in reward_rules[rule]['attributes']['texture']) and \
                    (size in reward_rules[rule]['attributes']['size']):
                if probs_set:
                    raise ValueError("Two rules have overlapping membership")
                else:
                    reward_probabilities = reward_rules[rule]['reward_probabilities']
                    probs_set = True
        if (rule == 'default') & (not probs_set):
            reward_probabilities = reward_rules[rule]['reward_probabilities']
    return reward_probabilities


def assignKeys(reward_rules_in, possible_keys_in, block_categories_in):
    # Check to make sure the inputs all add up
    categories = []
    possible_keys = copy.deepcopy(possible_keys_in)
    reward_rules = copy.deepcopy(reward_rules_in)
    block_categories = copy.deepcopy(block_categories_in)
    rules = reward_rules.keys()
    for rule in rules:
        categories += list(reward_rules[rule]['reward_probabilities'].keys())
    for block in block_categories:
        categories += block
    categories = np.unique(categories)
    if len(categories) != len(possible_keys):
        raise ValueError('The number of categories and key do not match')
    # Randomize key order
    np.random.shuffle(possible_keys)
    # Go back through and replace responses
    for rule in rules:
        keys = list(reward_rules[rule]['reward_probabilities'].keys())
        for k in range(len(keys)):
            reward_rules[rule]['reward_probabilities'][possible_keys[k]] = \
                reward_rules[rule]['reward_probabilities'].pop(keys[k])
    valid_keys = copy.deepcopy(block_categories)
    for i in range(len(block_categories)):
        for j in range(len(block_categories[i])):
            valid_keys[i][j] = possible_keys[list(categories).index(block_categories[i][j])]
    # Create a conversion dictionary
    conversion = {
        'category': list(categories),
        'key': possible_keys
    }

    return reward_rules, valid_keys, conversion


def getStimulusOrder(stimulus_combinations, n_trials=None, trials_per_stim=None, structured=False):
    if structured:
        order = []
        stim_indx = np.arange(0,len(stimulus_combinations))
        for t in range(trials_per_stim):
            np.random.shuffle(stim_indx)
            order += stim_indx.tolist()
    else:
        order = np.random.choice(len(stimulus_combinations),
                         n_trials).tolist()
    return order


def sessionStimulisRewardProbs(valid_keys=None,stimulus_combinations=None,reward_rules=None):
    if valid_keys is None:
        raise ValueError('valid_keys is now required.')
    if stimulus_combinations is None:
        raise ValueError('stimulus_combinations is now required.')
    stimulus_urls = []
    reward_probabilities = np.zeros((len(stimulus_combinations),len(valid_keys)))
    for s in range(len(stimulus_combinations)):
        stimulus_key = stimulus_combinations[s]
        stimulus = Stimulus.objects.filter(color=stimulus_key[0], shape=stimulus_key[1], texture=stimulus_key[2],
                                           size=stimulus_key[3])[0]
        # Build variables
        stimulus_urls.append(stimulus.image.url)
        reward_prob_dict = getRewardProbabilities(color=stimulus_key[0], shape=stimulus_key[1], \
                                                  texture=stimulus_key[2],size=stimulus_key[3], \
                                                  reward_rules=reward_rules)
        for k in range(len(valid_keys)):
            reward_probabilities[s, k] = reward_prob_dict[valid_keys[k]]

    return stimulus_urls, reward_probabilities


def orderIndx(lst, ref):
     indx = []
     for i in range(len(lst)):
             indx.append(ref.index(lst[i]))
     return indx

def createPlanetIntros(valid_keys,key_actions,task='context-generalization'):
    if 'context-generalization' in task:
        planet_intros = [
            f'Welcome to Planet Waz-up, home to the long-deceased Waz civilization. Here you will find artifacts that ' +
            f"are activated with either a {key_actions[0][0]} (press '{valid_keys[0][0]}') or a  {key_actions[0][1]} " +
            f"(press '{valid_keys[0][1]}'). You'll have to figure out what works!",

            f"Your work on Planet Waz-up is complete!\n\n"
            f"After hopping in your spaceship, you traveled to Planet Oh-Kay. " +
            f"Here once lived the proud species Oh. Their artifacts operate completely differently, and " +
            f"are activated with either a " +
            f"{key_actions[1][0]} (press '{valid_keys[1][0]}') or a  {key_actions[1][1]} (press '{valid_keys[1][1]}'). " +
            f"Forget what you learned on Planet Waz-up. Planet Oh-Kay's artifacts have their own rules!",

            f"Good work space pirate!\n\n" +
            f"You learned of planet Blabla, the only place in the galaxy where both the Waz and Oh once lived! Here you " +
            f"discover artifacts from both civilizations. That means you can {key_actions[0][0]} (press " +
            f"'{valid_keys[0][0]}'), {key_actions[0][1]} (press '{valid_keys[0][1]}'), {key_actions[1][0]} (press " +
            f"'{valid_keys[1][0]}') or {key_actions[1][1]} (press '{valid_keys[1][1]}'). Go collect some energy!"
        ]
    elif task == 'example-generalization':
        planet_intro = 'You stumbled upon a treasure trove of alien artifacts! To activate the alien artifacts, you might'
        for k in range(len(valid_keys)):
            if k == (len(valid_keys)-1):
                planet_intro += f' or'
            planet_intro += f' {key_actions[k].lower()} (press "{valid_keys[k]}"),'
        planet_intro = planet_intro[:-1] + '. It could be that all the actions are useful, or it could be that just ' +\
        'a few are useful. There also might be patterns. You have to figure it out by trial and error. Good luck!'
        planet_intros = [planet_intro]
    else:
        raise ValueError('Invalid task')

    return planet_intros

def buildTutorialStimulusDB(reward_rules,
        file_dir='stimuli/individual_png/'):
    #Loop through each image file
    f_names = glob(file_dir + '*' + '.png')
    for f in range(len(f_names)):
        print(f'Prosessing tutorial stimulis {f} of {len(f_names)-1}')
        try:
            spaceship, setting = os.path.basename(f_names[f]).split('.')[0].split('_')
        except:
            continue
        #Check if it exists
        if TutorialStimulus.objects.filter(spaceship=spaceship,setting=setting).exists():
            stimulus = TutorialStimulus.objects.filter(spaceship=spaceship,setting=setting)[0]
            print('Existing stimulus found')
        else: # Create new entry
            stimulus = TutorialStimulus(spaceship=spaceship,setting=setting)
        stimulus.image.save(os.path.basename(f_names[f]), File(open(f_names[f], 'rb')))
        #Check rules and make sure there's no overlapping membership
        rules = reward_rules.keys()
        probs_set = False
        for rule in rules:
            if (spaceship in reward_rules[rule]['attributes']['spaceship']) and \
               (setting in reward_rules[rule]['attributes']['setting']):
                if probs_set:
                    raise ValueError("Two rules have overlapping membership")
                else:
                    stimulus.reward_probabilities = reward_rules[rule]['reward_probabilities']
                    print('Reward probabilities updated')
                    probs_set = True
        #If they don't match a rule, use the generic settings.
        if not probs_set:
            raise ValueError(f"Reward probabilities needed for {spaceship}, {setting}")
        # Create the new entry
        stimulus.save()


def buildRewardStimulusDB(
        file_dir='stimuli/individual_png/'):
    #Loop through each image file
    f_names = glob(file_dir + '*' + '.png')
    for f in range(len(f_names)):
        print(f'Prosessing tutorial stimulis {f} of {len(f_names)-1}')
        try:
            outcome, typ, _, _ = os.path.basename(f_names[f]).split('.')[0].split('_')
        except:
            continue
        #Check if it exists
        if RewardStimulus.objects.filter(outcome=outcome,type=typ).exists():
            stimulus = RewardStimulus.objects.filter(outcome=outcome,type=typ)[0]
            print('Existing stimulus found')
        else: # Create new entry
            stimulus = RewardStimulus(outcome=outcome,type=typ)
        stimulus.image.save(os.path.basename(f_names[f]), File(open(f_names[f], 'rb')))
        stimulus.save()


def buildStimulusDB(reward_rules=None,
        file_dir='stimuli/individual_png/'):
    #Loop through each image file
    f_names = glob(file_dir + '*' + '.png')
    for f in range(len(f_names)):
        print(f'Prosessing stimulis {f} of {len(f_names)-1}')
        try:
            shape, color, texture, size = os.path.basename(f_names[f]).split('.')[0].split('_')
        except:
            continue
        #Check if it exists
        if Stimulus.objects.filter(shape=shape,color=color,texture=texture,size=size).exists():
            stimulus = Stimulus.objects.filter(shape=shape, color=color, texture=texture,size=size)[0]
            print('Existing stimulus found')
        else: # Create new entry
            stimulus = Stimulus(shape=shape,color=color,texture=texture,size=size)
        stimulus.image.save(os.path.basename(f_names[f]), File(open(f_names[f], 'rb')))
        if reward_rules is not None:
            #Check rules and make sure there's no overlapping membership
            rules = reward_rules.keys()
            probs_set = False
            for rule in rules:
                if (shape in reward_rules[rule]['attributes']['shape']) and \
                   (color in reward_rules[rule]['attributes']['color']) and \
                   (texture in reward_rules[rule]['attributes']['texture']) and \
                   (size in reward_rules[rule]['attributes']['size'])     :
                    if probs_set:
                        raise ValueError("Two rules have overlapping membership")
                    else:
                        stimulus.reward_probabilities = reward_rules[rule]['reward_probabilities']
                        print('Reward probabilities updated')
                        probs_set = True
            #If they don't match a rule, use the generic settings.
            if not probs_set:
                try:
                    stimulus.reward_probabilities = reward_rules['default']['reward_probabilities']
                    print('Reward probabilities updated')
                except:
                    raise ValueError("Generic reward probabilities needed")
        # Create the new entry
        stimulus.save()


def checkSessionAtts(task='example-generalization',STIMULI_BLOCK_0=None,STIMULI_BLOCK_1=None,
                     attributes=['colors','shapes','textures']):
    if task == 'example-generalization':
        if 'colors' not in STIMULI_BLOCK_0.keys():
            stimuli_cond_stim = list(STIMULI_BLOCK_0.keys())
            stimuli_gen_stim = np.setdiff1d(list(STIMULI_BLOCK_1.keys()), stimuli_cond_stim)
            atts = ['color', 'shape', 'texture']
            stimuli_block_0, stimuli_block_1 = dict(), dict()
            for att in atts:
                stimuli_block_0[f'{att}s'], stimuli_block_1[f'{att}s'] = [], []
            for stim in stimuli_cond_stim:
                for key in STIMULI_BLOCK_0[stim].keys():
                    if key not in atts:
                        continue
                    if STIMULI_BLOCK_0[stim][key] not in stimuli_block_0[f'{key}s']:
                        stimuli_block_0[f'{key}s'].append(STIMULI_BLOCK_0[stim][key])
            for stim in stimuli_gen_stim:
                for key in STIMULI_BLOCK_1[stim].keys():
                    if key not in atts:
                        continue
                    if STIMULI_BLOCK_1[stim][key] not in stimuli_block_1[f'{key}s']:
                        stimuli_block_1[f'{key}s'].append(STIMULI_BLOCK_1[stim][key])
            STIMULI_BLOCK_0, STIMULI_BLOCK_1 = stimuli_block_0, stimuli_block_1
        conditioning, generalization = [], []
        for att in attributes:
            if len(STIMULI_BLOCK_0[att]) > 1:
                conditioning.append(att)
            if STIMULI_BLOCK_0[att] != STIMULI_BLOCK_1[att]:
                generalization.append(att)
        return conditioning, generalization
    elif ('context-generalization' in task) or (task == 'diagnostic'):
        if 'colors' in STIMULI_BLOCK_0.keys():
            set_1, set_2 = [], []
            for att in attributes:
                if len(STIMULI_BLOCK_0[att]) == 1:
                    set_1.append(att)
                if len(STIMULI_BLOCK_1[att]) == 1:
                    set_2.append(att)
        else:
            set_1 = checkSessionAttsHelper(STIMULI_BLOCK_0, atts=['color','shape','texture','size'],task='context-generalization')
            set_2 = checkSessionAttsHelper(STIMULI_BLOCK_1, atts=['color', 'shape', 'texture', 'size'],task='context-generalization')
        return set_1, set_2


def getPaymentToken():
    if PAYMENT_TOKEN is None:
        payment_token = token_urlsafe(PAYMENT_TOKEN_LENGTH)
    else:
        payment_token = copy.copy(PAYMENT_TOKEN)
    return payment_token


def checkSessionAttsHelper(stimuli,atts=['color','shape','texture','size'],task='example-generalization'):
    if 'context-generalization' in task:
        att_dict = dict()
        for att in atts:
            att_dict[att] = []

        for stim in stimuli.keys():
            for att in atts:
                att_dict[att].append(stimuli[stim][att])

        att_bool = np.zeros(len(atts))

        for a in range(len(atts)):
            if len(np.unique(np.array(att_dict[atts[a]]))) == 1:
                att_bool[a] = 1

        set_atts = list(np.array(atts)[att_bool>0])
    elif task == 'example-generalization':
        set_atts = dict()
        for att in atts:
            set_atts[att] = []
        for stim in stimuli.keys():
            for keys in stimuli[stim].keys():
                set_atts[att].append(stimuli[stim][keys])

    return set_atts


# Form functions
class substancesModelForm(ModelForm):
    def __init__(self, *args, **kwargs):
        super(substancesModelForm, self).__init__(*args, **kwargs)
        self.fields['substances'].required = False

    class Meta:
        model = Session
        fields = ('substances',)
        widgets = {
            'substances': forms.CheckboxSelectMultiple(choices=SUBSTANCES,attrs={"required": False,'initial':['None']})
        }


class mentalHealthModelForm(ModelForm):
    def __init__(self, *args, **kwargs):
        super(mentalHealthModelForm,self).__init__(*args, **kwargs)
        self.fields['psych_history'].required = False

    class Meta:
        model = Subject
        fields = ('psych_history',)
        widgets = {
            'psych_history': forms.CheckboxSelectMultiple(
                choices=MH_HISTORY,attrs={"required": False,'initial':['None']})
        }


def makeMentalHealthModelForm(
        label='Have you recieved a formal diagnosis for any of the following?',
                                instance=None):
    form = mentalHealthModelForm(instance=instance)
    form.fields['psych_history'].label = label
    form.fields['psych_history'].initial = ['None']
    return form


def makeSubstancesModelForm(label='In the last two hours, have you used any of the following?',
                                instance=None):
    form = substancesModelForm(instance=instance)
    form.fields['substances'].label = label
    form.fields['substances'].initial = ['None']
    return form


class BaseQuestionnaireFormSet(BaseModelFormSet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.queryset = QuestionnaireQ.objects.none()


def makeQuestionnaireFormSet(questionnaires):
    initial = []
    for questionnaire_name, questionnaire in zip(questionnaires.keys(), questionnaires.values()):
        for question in list(questionnaire.keys()):
            initial_form = {
                'questionnaire_name': questionnaire_name,
                'answer': forms.IntegerField(label=question, widget=forms.RadioSelect(choices=[(v,k) for k,v in questionnaire[question]['answers'].items()])),
                'possible_answers': questionnaire[question]['answers'],
                'question': question,
                'questionnaire_question_number': questionnaire[question]['question_number'],
                'subscale': questionnaire[question]['subscale']
            }
            initial.append(initial_form)

    # initial = [initial[0], initial[1]]

    QuestionnaireFormSet = modelformset_factory(QuestionnaireQ,
                      exclude=('session',),
                      widgets={'questionnaire_name'
                               '': forms.HiddenInput(),
                                   'possible_answers': forms.HiddenInput(),
                                   'subscale': forms.HiddenInput(),
                                   'question': forms.HiddenInput(),
                                   'questionnaire_question_number': forms.HiddenInput()
                                   },
                      formset=BaseQuestionnaireFormSet,
                      extra=len(initial))

    formset = QuestionnaireFormSet()

    for f in range(len(formset.forms)):
        formset.forms[f].fields['question'].initial = initial[f]['question']
        formset.forms[f].fields['possible_answers'].initial = initial[f]['possible_answers']
        formset.forms[f].fields['questionnaire_name'].initial = initial[f]['questionnaire_name']
        formset.forms[f].fields['subscale'].initial = initial[f]['subscale']
        formset.forms[f].fields['questionnaire_question_number'].initial = initial[f]['questionnaire_question_number']
        formset.forms[f].fields['answer'].widget = forms.RadioSelect(
            choices=[(v, k) for k, v in initial[f]['possible_answers'].items()],
            attrs={"required": True})
        formset.forms[f].fields['answer'].label = initial[f]['question']

    return formset


class RegistrationForm(forms.Form):
    user_ID = forms.CharField(label="Your ID Number")
    subject_source = forms.CharField(label="Who Sent You", required=True, widget=forms.Select(choices=SUBJECT_SOURCES))
    age = forms.CharField(label="Age", required=True, widget=forms.Select(choices=AGES))
    gender = forms.CharField(label="Gender", required=True, widget=forms.Select(choices=GENDERS))
    education = forms.CharField(label="Education Level", required=True, widget=forms.Select(choices=EDUCATION))
    start_time = forms.DateTimeField(label='start_time',required=True, widget=forms.HiddenInput())


class TaskForm(forms.Form):
    key_pressed = forms.CharField(label='key_pressed', max_length=1, widget=forms.HiddenInput())
    start_time = forms.DateTimeField(label='start_time', widget=forms.HiddenInput())


class OnePageTaskForm(forms.Form):
    key_pressed = forms.CharField(label='key_pressed', max_length=1, widget=forms.HiddenInput())
    start_time = forms.DateTimeField(label='start_time', widget=forms.HiddenInput())


class strategyForm(forms.Form):
    strategy_free = forms.CharField(min_length=0, max_length=1000, widget=forms.Textarea, label=False)


class difficultyForm(forms.Form):
    perceived_difficulty = forms.ChoiceField(choices=DIFFICULTY, widget=forms.RadioSelect(attrs={'class': "custom-radio-list"}), label=False)

class attentionCheckList(forms.Form):
    label = 'Have you been reading closely? If so, choose prosochiphelia from the following fake conditions.'
    attention_checkbox = forms.CharField(label=label, required=False, widget=\
    forms.CheckboxSelectMultiple(choices=ATTENTION_CHECK_HISTORY,attrs={"required": False,'initial':['None']}))


# Helper and analysis functions
def chooseLargest(response, reward_probabilities):
    vals = np.array(list(reward_probabilities.values()))
    keys = np.array(list(reward_probabilities.keys()))
    indx = vals >= max(vals)
    largest_chosen = response in keys[indx]
    largest_key = ''.join(keys[indx])
    return largest_chosen, largest_key

def getTrialStimulus(request):
    trial_n = request.session['trial_number']
    stimulus_key = STIMULUS_COMBINATIONS_BLOCK_0[request.session['stimulus_order'][trial_n]]
    stimulus = Stimulus.objects.filter(color=stimulus_key[0],shape=stimulus_key[1],texture=stimulus_key[2])[0]
    return stimulus


def calcSessionPerformance(session):
    chose_largest = np.zeros(session.trials.count())
    block = chose_largest.copy()
    for t in range(session.trials.count()):
        #Determine if they chose the largest option
        response = session.trials.all()[t].response
        resp_prob = session.trials.all()[t].reward_probs_record[response]
        chose_largest[t] = int(resp_prob == max(session.trials.all()[t].reward_probs_record.values()))
        #Determine what block the trial was in
        block[t] = BLOCK_NAMES.index(session.trials.all()[t].block)
    return chose_largest, block


def getStimulusKey(request,trial_n):
    current_block = request.session['block'][trial_n]
    stimulus_key = STIMULUS_COMBINATIONS[current_block][request.session['stimulus_order'][trial_n]]
    return stimulus_key


def getStimuliOutcomes(request):
    n_trials = N_TRIALS_BLOCK_0 + N_TRIALS_BLOCK_1
    stimulus_urls = []
    valid_keys = []
    for t in range(n_trials):
        #Obtain the stimulus
        stimulus_key = getStimulusKey(request, t)
        stimulus = Stimulus.objects.filter(color=stimulus_key[0], shape=stimulus_key[1], texture=stimulus_key[2],
                                           size=stimulus_key[3])[0]
        #Build variables
        stimulus_urls.append(stimulus.image.url)
        valid_keys.append(list(stimulus.reward_probabilities.keys()))
        if t == 0:
            outcomes = np.zeros((n_trials, len(valid_keys[-1])))
        reward_probabilities = np.array(list(stimulus.reward_probabilities.values()))
        outcomes[t,:] = np.random.rand(len(reward_probabilities)) < reward_probabilities
    return stimulus_urls, outcomes, valid_keys


def obscureOutcomes(outcomes,method='simple'):
    if method == 'simple':
        step1 = outcomes.flatten()
        step2 = step1 + np.arange(len(step1))
        step3 = step2**2
        step4 = []
        for v in range(len(step3)):
            val = str(base64.b64encode(str(step3[v]).encode("utf-8")))[2:-1]
            step4.append(val)
    return step4


## GLOBAL VARIABLES REQUIRING FUNCTIONS

# QUESTIONNAIRE_FORMSET = makeQuestionnaireFormSet(QUESTIONNAIRES)

# Construct task-specific global variables
all_colors = ['blue', 'magenta', 'orange', 'purple', 'yellow']
all_shapes = ['circle', 'hexagon', 'square', 'circle', 'star', 'x']
all_textures = ['capsules','checker','diagonal','dots', 'solid']

tutorial_reward_rules = {
    'rule_1':
    {   'attributes': { 'spaceship': ['red-spaceship', 'green-spaceship'],
                        'setting': ['space']
                        },
        'reward_probabilities': {   'v': 0.90,
                                    'b': 0.00
                                }
    },
    'rule_2':
    {
        'attributes': { 'spaceship': ['blue-spaceship','purple-spaceship'],
                        'setting': ['tan-planet', 'green-planet']
                        },
        'reward_probabilities': {   'v': 0.00,
                                    'b': 0.90
                                }
    },
    'rule_3':
    {
        'attributes': { 'spaceship': ['no-spaceship'],
                        'setting': ['blue-planet','gold-planet']
                        },
        'reward_probabilities': {   'v': 0.80,
                                    'b': 0.20
                                }
    }
}
if TASK == 'example-generalization':
    BLOCK_NAMES = ["Initial-Learning", "Generalization"]

    STIMULI_BLOCK_0 = {
        'stim1': {
            'color': 'orange',
            'shape': 'circle',
            'texture': 'dots',
            'size': 'large',
            'category': 'A'
        },
        'stim2': {
            'color': 'orange',
            'shape': 'square',
            'texture': 'dots',
            'size': 'large',
            'category': 'A'
        },
        'stim3': {
            'color': 'magenta',
            'shape': 'star',
            'texture': 'dots',
            'size': 'large',
            'category': 'B'
        },
        'stim4': {
            'color': 'blue',
            'shape': 'star',
            'texture': 'dots',
            'size': 'large',
            'category': 'B'
        },
        'stim5': {
            'color': 'blue',
            'shape': 'circle',
            'texture': 'dots',
            'size': 'large',
            'category': 'C'
        },
        'stim6': {
            'color': 'blue',
            'shape': 'square',
            'texture': 'dots',
            'size': 'large',
            'category': 'C'
        },
        'stim7': {
            'color': 'magenta',
            'shape': 'circle',
            'texture': 'dots',
            'size': 'large',
            'category': 'C'
        },
        'stim8': {
            'color': 'magenta',
            'shape': 'square',
            'texture': 'dots',
            'size': 'large',
            'category': 'C'
        }
    }

    STIMULI_BLOCK_1 = {** STIMULI_BLOCK_0, **{
        'stim9': {
            'color': 'orange',
            'shape': 'circle',
            'texture': 'diagonal',
            'size': 'large',
            'category': 'A'
        },
        'stim10': {
            'color': 'orange',
            'shape': 'square',
            'texture': 'diagonal',
            'size': 'large',
            'category': 'A'
        },
        'stim11': {
            'color': 'magenta',
            'shape': 'star',
            'texture': 'diagonal',
            'size': 'large',
            'category': 'B'
        },
        'stim12': {
            'color': 'blue',
            'shape': 'star',
            'texture': 'diagonal',
            'size': 'large',
            'category': 'B'
        },
        'stim13': {
            'color': 'blue',
            'shape': 'circle',
            'texture': 'diagonal',
            'size': 'large',
            'category': 'C'
        },
        'stim14': {
            'color': 'blue',
            'shape': 'square',
            'texture': 'diagonal',
            'size': 'large',
            'category': 'C'
        },
        'stim15': {
            'color': 'magenta',
            'shape': 'circle',
            'texture': 'diagonal',
            'size': 'large',
            'category': 'C'
        },
        'stim16': {
            'color': 'magenta',
            'shape': 'square',
            'texture': 'diagonal',
            'size': 'large',
            'category': 'C'
        }
    }}

    REWARD_RULES = {
        'rule_1':
            {'attributes': {'color': ['orange'],
                            'shape': ['circle', 'hexagon', 'square', 'circle', 'x'],
                            'texture': all_textures,
                            'size': 'large'},
             'reward_probabilities': {'A': 1.00,
                                      'B': 0.00,
                                      'C': 0.00,
                                      'D': 0.00}
             },
        'rule_2':
            {
                'attributes': {'color': ['blue', 'magenta', 'purple', 'yellow'],
                               'shape': ['star'],
                               'texture': all_textures,
                            'size': 'large'},
                'reward_probabilities': {'A': 0.00,
                                         'B': 1.00,
                                         'C': 0.00,
                                         'D': 0.00}
            },
        'rule_3':
            {
                'attributes': {'color': ['orange'],
                               'shape': ['star'],
                               'texture': all_textures,
                                'size': 'large'},
                'reward_probabilities': {'A': 1.00,
                                         'B': 1.00,
                                         'C': 0.00,
                                         'D': 0.00}
            },
        'default':
            {
                'attributes': {'color': 'default',
                               'shape': 'default',
                               'texture': 'default',
                               'size': 'default'},
                'reward_probabilities': {'A': 0.00,
                                         'B': 0.00,
                                         'C': 1.00,
                                         'D': 0.00}
            }
    }

    TRIALS_PER_STIM_BLOCK_0 = 16 # 16  # 10 #If trial order is structured
    TRIALS_PER_STIM_BLOCK_1 = 5 #5  # 5 #If trial order is structured
    STIMULUS_COMBINATIONS_BLOCK_0 = stimulusCombinations(STIMULI_BLOCK_0,reward_rules=REWARD_RULES)
    STIMULUS_COMBINATIONS_BLOCK_1 = stimulusCombinations(STIMULI_BLOCK_1,reward_rules=REWARD_RULES)
    STIMULUS_COMBINATIONS = [STIMULUS_COMBINATIONS_BLOCK_0 + STIMULUS_COMBINATIONS_BLOCK_1] * 2
    if STRUCTURED:
        N_TRIALS_BLOCK_0 = len(STIMULUS_COMBINATIONS_BLOCK_0) * TRIALS_PER_STIM_BLOCK_0
        N_TRIALS_BLOCK_1 = len(STIMULUS_COMBINATIONS_BLOCK_1) * TRIALS_PER_STIM_BLOCK_1
    else:
        N_TRIALS_BLOCK_0 = 10  # If trial order is unstructured
        N_TRIALS_BLOCK_1 = 20  # If trial order is unstructured

    N_TRIALS_PER_BLOCK = np.array([N_TRIALS_BLOCK_0, N_TRIALS_BLOCK_1])
    POSSIBLE_KEYS = ['c','v', 'b', 'n']
    BLOCK_CATEGORIES = [['A', 'B', 'C','D'], ['A', 'B', 'C','D']]
    KEY_ACTIONS = [['Sing', 'Dance', 'Wave', 'Bite']] * 2

    STRATEGIES = [('none', 'I DID NOT HAVE a strategy'), ('memorized', 'I MEMORIZED EACH individually'),
                  ('shape', 'I used SHAPE to group them'),('color', 'I used COLOR to group them'),
                  ('texture', 'I used TEXTURE to group them'),('shape-color', 'I used SHAPE and COLOR to group them'),
                  ('shape-texture', 'I used SHAPE and TEXTURE to group them'),
                  ('color-texture', 'I used COLOR and TEXTURE to group them')]

elif (TASK == 'context-generalization') or (TASK == 'context-generalization_v1') or \
        (TASK == 'context-generalization_v2')or (TASK == 'diagnostic'):
    BLOCK_NAMES = ["Context_1", "Context_2", "Generalization-Context"]

    if TASK == 'context-generalization_v1':
        ## TASK VERSION 1
        STIMULI_BLOCK_0 = {
            'colors': ['orange', 'magenta'],
            'shapes': ['circle', 'square'],
            'textures': ['dots'],
            'sizes': ['large', 'small']
        }
        REWARD_RULES = {
            'rule_1':
                {'attributes': {'color': ['magenta'],
                                'shape': ['circle'],
                                'texture': ['dots'],
                                'size': ['large', 'small']},
                 'reward_probabilities': {'A': 1.00,
                                          'B': 0.00,
                                          'C': 0.00,
                                          'D': 0.00}
                 },
            'rule_2':
                {
                    'attributes': {'color': ['orange'],
                                   'shape': ['square'],
                                   'texture': ['dots'],
                                   'size': ['large', 'small']},
                    'reward_probabilities': { 'A': 0.00,
                                              'B': 1.00,
                                              'C': 0.00,
                                              'D': 0.00}
                },
            'rule_3':
                {'attributes': {'color': ['magenta', 'blue'],
                                'shape': ['circle'],
                                'texture': ['diagonal'],
                                'size': ['large']},
                 'reward_probabilities': {'A': 0.00,
                                          'B': 0.00,
                                          'C': 1.00,
                                          'D': 0.00}
                 },
            'rule_4':
                {
                    'attributes': {'color': ['magenta', 'blue'],
                                   'shape': ['square'],
                                   'texture': ['diagonal'],
                                   'size': ['large']},
                    'reward_probabilities': { 'A': 0.00,
                                              'B': 0.00,
                                              'C': 0.00,
                                              'D': 1.00}
                },
        }
        STIMULI_BLOCK_1 = {
            'colors': ['blue', 'magenta'],
            'shapes': ['circle', 'square'],
            'textures': ['diagonal'],
            'sizes': ['large', 'small']
        }
        STIMULI_BLOCK_2 = {
            'colors': ['orange', 'magenta','blue'],
            'shapes': ['circle','square'],
            'textures': ['dots', 'diagonal'],
            'sizes': ['large', 'small']
        }

    else:
        ## VERSION 2
        STIMULI_BLOCK_0 = {
            'stim1': {
                'color': 'orange',
                'shape': 'circle',
                'texture': 'dots',
                'size': 'large',
                'category': 'A'
            },
            'stim2': {
                'color': 'magenta',
                'shape': 'circle',
                'texture': 'dots',
                'size': 'small',
                'category': 'A'
            },
            'stim3': {
                'color': 'magenta',
                'shape': 'square',
                'texture': 'dots',
                'size': 'large',
                'category': 'B'
            },
            'stim4': {
                'color': 'orange',
                'shape': 'square',
                'texture': 'dots',
                'size': 'large',
                'category': 'B'
            }
        }

        STIMULI_BLOCK_1 = {
            'stim5': {
                'color': 'magenta',
                'shape': 'circle',
                'texture': 'diagonal',
                'size': 'large',
                'category': 'C'
            },
            'stim6': {
                'color': 'blue',
                'shape': 'circle',
                'texture': 'diagonal',
                'size': 'large',
                'category': 'C'
            },
            'stim7': {
                'color': 'magenta',
                'shape': 'square',
                'texture': 'diagonal',
                'size': 'small',
                'category': 'D'
            },
            'stim8': {
                'color': 'blue',
                'shape': 'square',
                'texture': 'diagonal',
                'size': 'large',
                'category': 'D'
            }
        }

        STIMULI_BLOCK_2 = {**STIMULI_BLOCK_0, **STIMULI_BLOCK_1}

        REWARD_RULES = {
            'rule_1':
                {'attributes': {'color': ['orange', 'magenta'],
                                'shape': ['circle'],
                                'texture': ['dots'],
                                'size': ['large', 'small']},
                 'reward_probabilities': {'A': 1.00,
                                          'B': 0.00,
                                          'C': 0.00,
                                          'D': 0.00}
                 },
            'rule_2':
                {
                    'attributes': {'color': ['magenta', 'orange'],
                                   'shape': ['square'],
                                   'texture': ['dots'],
                                   'size': ['large']},
                    'reward_probabilities': {'A': 0.00,
                                             'B': 1.00,
                                             'C': 0.00,
                                             'D': 0.00}
                },
            'rule_3':
                {'attributes': {'color': ['magenta', 'blue'],
                                'shape': ['circle'],
                                'texture': ['diagonal'],
                                'size': ['large']},
                 'reward_probabilities': {'A': 0.00,
                                          'B': 0.00,
                                          'C': 1.00,
                                          'D': 0.00}
                 },
            'rule_4':
                {
                    'attributes': {'color': ['magenta', 'blue'],
                                   'shape': ['square'],
                                   'texture': ['diagonal'],
                                   'size': ['small', 'large']},
                    'reward_probabilities': {'A': 0.00,
                                             'B': 0.00,
                                             'C': 0.00,
                                             'D': 1.00}
                },
        }

    TRIALS_PER_STIM_BLOCK_0 = 12 #12  # 10 #If trial order is structured
    TRIALS_PER_STIM_BLOCK_1 = 12 #12  # 5 #If trial order is structured
    TRIALS_PER_STIM_BLOCK_2 = 10 #10  # 5 #If trial order is structured
    STIMULUS_COMBINATIONS_BLOCK_0 = stimulusCombinations(STIMULI_BLOCK_0,reward_rules=REWARD_RULES)
    STIMULUS_COMBINATIONS_BLOCK_1 = stimulusCombinations(STIMULI_BLOCK_1,reward_rules=REWARD_RULES)
    STIMULUS_COMBINATIONS_BLOCK_2 = stimulusCombinations(STIMULI_BLOCK_2,reward_rules=REWARD_RULES)
    STIMULUS_COMBINATIONS = [STIMULUS_COMBINATIONS_BLOCK_0, STIMULUS_COMBINATIONS_BLOCK_1, \
                            STIMULUS_COMBINATIONS_BLOCK_2]

    if STRUCTURED:
        N_TRIALS_BLOCK_0 = len(STIMULUS_COMBINATIONS_BLOCK_0) * TRIALS_PER_STIM_BLOCK_0
        N_TRIALS_BLOCK_1 = len(STIMULUS_COMBINATIONS_BLOCK_1) * TRIALS_PER_STIM_BLOCK_1
        N_TRIALS_BLOCK_2 = len(STIMULUS_COMBINATIONS_BLOCK_2) * TRIALS_PER_STIM_BLOCK_2
    else:
        N_TRIALS_BLOCK_0 = 2  # If trial order is unstructured
        N_TRIALS_BLOCK_1 = 2
        N_TRIALS_BLOCK_2 = 3

    N_TRIALS_PER_BLOCK = np.array([N_TRIALS_BLOCK_0, N_TRIALS_BLOCK_1, N_TRIALS_BLOCK_2])
    POSSIBLE_KEYS = ['z', 'x', 'c', 'v']
    BLOCK_CATEGORIES = [['A', 'B'], ['C', 'D'], ['A', 'B', 'C', 'D']]
    KEY_ACTIONS = [['Shake', 'Sing'],['Dance','Wave'],['Shake', 'Sing', 'Dance','Wave']]

    STRATEGIES = [('none', 'I DID NOT HAVE a strategy'), ('memorized', 'I MEMORIZED each artifact individually'),
                  ('resemble', "I learned artifacts by what they RESEMBELED, such as 'square waffles,' or 'round tennis rackets.'"),
                  ('given-attributes', 'I GROUPED them by color, shape, texture or size.')]
else:
    raise ValueError(f'{TASK} is invalid for TASK parameter')


## FUNCTIONS REQUIRED THE VARIABLES JUST DEFINED

class strategyRadioForm(forms.Form):
    strategy = forms.ChoiceField(choices=STRATEGIES, widget=forms.RadioSelect(attrs={'class': "custom-radio-list"}), label=False)