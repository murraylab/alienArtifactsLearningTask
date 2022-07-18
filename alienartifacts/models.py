import datetime

from django.db import models
from django.utils import timezone


class Stimulus(models.Model):
    #Contains stimuli used in the task, organized according to attributes
    color = models.CharField(max_length=64)
    shape = models.CharField(max_length=64)
    texture = models.CharField(max_length=64)
    size = models.CharField(max_length=64,default='')
    reward_probabilities = models.JSONField(default=dict)
    image = models.ImageField(upload_to='images/')


class TutorialStimulus(models.Model):
    spaceship = models.CharField(max_length=64)
    setting = models.CharField(max_length=64)
    reward_probabilities = models.JSONField(default=dict)
    image = models.ImageField(upload_to='images/')


class RewardStimulus(models.Model):
    outcome = models.CharField(max_length=64)
    type = models.CharField(max_length=64)
    image = models.ImageField(upload_to='images/')


class Subject(models.Model):
    #Contains user index, external identification number, where they came from
    external_ID = models.CharField(max_length=64)
    external_source = models.CharField(max_length=64)
    age = models.CharField(max_length=20)
    gender = models.CharField(max_length=20)
    education = models.CharField(max_length=20)
    is_bot = models.BooleanField(default=False)
    psych_history = models.JSONField(default=list,blank=True,null=True)


class Session(models.Model):
    #Summary details, such as length, number of stimuli seen, average performance, date, etc.
    n_trials = models.IntegerField(default=0)
    start_time = models.DateTimeField()
    end_time = models.DateTimeField()
    session_completed = models.BooleanField(default=False)
    tutorial_completed = models.BooleanField(default=False)
    conditioning_completed = models.BooleanField(default=False)
    total_reward = models.IntegerField(default=0)
    total_payment = models.DecimalField(max_digits=5, decimal_places=2, default=0)
    payment_issued = models.BooleanField(default=False)
    final_performance = models.DecimalField(max_digits=6, decimal_places=5, null=True)
    payment_token = models.CharField(max_length=40)
    subject = models.ForeignKey(Subject, on_delete=models.CASCADE, related_name="sessions")
    strategy = models.CharField(max_length=1000, default="")
    strategy_radio = models.CharField(max_length=30, default="")
    perceived_difficulty = models.CharField(max_length=10, default="")
    conditioning_attributes = models.JSONField(default=list)
    generalization_attributes = models.JSONField(default=list)
    set_1_attribute = models.JSONField(default=list)
    set_2_attribute = models.JSONField(default=list)
    task = models.CharField(max_length=100, default="")
    key_conversion = models.JSONField(default=dict)
    substances = models.JSONField(default=list,blank=True,null=True)
    passed_attention_check = models.BooleanField(default=True)


class Trial(models.Model):
    #All data pertinent for an individual trial
    stimulus = models.ForeignKey(Stimulus, on_delete=models.CASCADE, related_name='trials')
    session = models.ForeignKey(Session, on_delete=models.CASCADE, related_name='trials')
    response = models.CharField(max_length=2)
    reward = models.BooleanField(null=True)
    start_time = models.DateTimeField()
    end_time = models.DateTimeField()
    block = models.CharField(max_length=20)
    reward_probs_record = models.JSONField(default=dict)


class QuestionnaireQ(models.Model):
    session = models.ForeignKey(Session, on_delete=models.CASCADE, related_name='questionnaire_q')
    questionnaire_name = models.CharField(max_length=100)
    subscale = models.CharField(max_length=40,blank=True,null=True)
    possible_answers = models.JSONField(default=dict)
    question = models.CharField(max_length=1000)
    answer = models.IntegerField()
    questionnaire_question_number = models.IntegerField()