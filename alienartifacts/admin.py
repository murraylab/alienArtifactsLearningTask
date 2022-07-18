from django.contrib import admin

from .models import Stimulus, Session, Subject, Trial, TutorialStimulus, RewardStimulus

#Register models
admin.site.register(Stimulus)
admin.site.register(TutorialStimulus)
admin.site.register(RewardStimulus)
admin.site.register(Session)
admin.site.register(Subject)
admin.site.register(Trial)