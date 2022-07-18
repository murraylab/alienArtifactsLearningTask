from .settings import *
import os
import sys

# Configure the domain name using the environment variable
# that Azure automatically creates for us.
ALLOWED_HOSTS = [os.environ['WEBSITE_HOSTNAME']] if 'WEBSITE_HOSTNAME' in os.environ else []

# WhiteNoise configuration
MIDDLEWARE = [                                                                   
    'django.middleware.security.SecurityMiddleware',
# Add whitenoise middleware after the security middleware                             
    'whitenoise.middleware.WhiteNoiseMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',                      
    'django.middleware.common.CommonMiddleware',                                 
    'django.middleware.csrf.CsrfViewMiddleware',                                 
    'django.contrib.auth.middleware.AuthenticationMiddleware',                   
    'django.contrib.messages.middleware.MessageMiddleware',                      
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

INSTALLED_APPS = INSTALLED_APPS + [
    'storages'
]

# STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')

# Added during deployment check for improved security
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
# SECURE_SSL_REDIRECT = True

# DBHOST is only the server name, not the full URL
hostname = os.environ['DBHOST']


# Configure Postgres database; the full username is username@servername,
# which we construct using the DBHOST value.
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': os.environ['DBNAME'],
        'HOST': hostname + ".postgres.database.azure.com",
        'USER': os.environ['DBUSER'] + "@" + hostname,
        'PASSWORD': os.environ['DBPASS'] 
    }
}

# For hosting of media files
DEFAULT_FILE_STORAGE = 'backend.custom_azure.AzureMediaStorage'
STATICFILES_STORAGE = 'backend.custom_azure.AzureStaticStorage'

STATIC_LOCATION = "static"
MEDIA_LOCATION = "media"

AZURE_ACCOUNT_NAME = "alienartifactsstorage"
AZURE_CUSTOM_DOMAIN = f'{AZURE_ACCOUNT_NAME}.blob.core.windows.net'
STATIC_URL = f'https://{AZURE_CUSTOM_DOMAIN}/{STATIC_LOCATION}/'
MEDIA_URL = f'https://{AZURE_CUSTOM_DOMAIN}/{MEDIA_LOCATION}/'


# APPLICATION_INSIGHTS = {
#     # Your Application Insights instrumentation key
#     'ikey': os.environ.get('INSIGHTS_KEY', "4hkj5kfcc6npsa8i2pb6uyqkpxgdy3tjfwscnbrr"),
#
#     # (optional) By default, request names are logged as the request method
#     # and relative path of the URL.  To log the fully-qualified view names
#     # instead, set this to True.  Defaults to False.
#     'use_view_name': True,
#
#     # (optional) To log arguments passed into the views as custom properties,
#     # set this to True.  Defaults to False.
#     'record_view_arguments': True,
# }


#Logging
if not DEBUG:
    LOGGING = {
        'version': 1,
        "handlers": {
            "azure": {
                "level": "DEBUG",
                "class": "opencensus.ext.azure.log_exporter.AzureLogHandler",
                "instrumentation_key": "ec561b4c-4e50-4796-a936-256a16e0e77f",
            },
            "console": {
                "level": "DEBUG",
                "class": "logging.StreamHandler",
                "stream": sys.stdout,
            },
            'file': {
                'level': 'INFO',
                'class': 'logging.FileHandler',
                'filename': '.logs/debug_server.log',
                'formatter': 'simple',
            },
        },
        "loggers": {
            "django": {"handlers": ["azure", "console",'file']},
        },
        'formatters': {
            'simple': {
                'format': '{levelname} {message}',
                'style': '{',
            }
        }
    }
else:
    LOGGING = {
        'version': 1,
        'loggers': {
            'django': {
                'handlers': ['file','console'],
                'level': 'DEBUG'
            },
        },
        'handlers': {
            'file': {
                'level': 'INFO',
                'class': 'logging.FileHandler',
                'filename': '.logs/debug.log',
                'formatter': 'simple',
            },
            'console': {
                'level': 'INFO',
                'class': 'logging.StreamHandler',
                'formatter': 'simple'
            },
        },
        'formatters': {
            'simple': {
                'format': '{levelname} {message}',
                'style': '{',
            }
        }
    }