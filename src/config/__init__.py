# -*- coding: utf-8 -*-
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
''' load default variables '''
import os
from pathlib import Path  # Python 3.6+ only
from dotenv import load_dotenv
import redis
from src.config.logger import LoggerClass
logging = LoggerClass.instance()
# OR, the same with increased verbosity
basedir = os.path.abspath(os.path.dirname(__file__))
# OR, explicitly providing path to '.env'
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)
logging.logger.debug("Loading variables")


# CONFIGURATION GENERIC AND DATABASE
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgres://docker:docker@localhost:5432/postgres").replace("://", "ql://", 1)
DATABASE_RETENTION_DAYS = os.getenv("DATABASE_RETENTION_DAYS", "30")
CREATE_ENGINE_DEBUG = os.getenv("CREATE_ENGINE_DEBUG", '')
DATABASE_HEROKU_CONNECT = os.getenv("DATABASE_HEROKU_CONNECT", 'False')
WORKER_TYPE = os.getenv("WORKER_TYPE", '1')
SECRET_KEY = os.getenv("SECRET_KEY", "change-this")
APP_SETTINGS = os.getenv("APP_SETTINGS", "src.config.DevelopmentConfig")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
# if we want to retry data from salesforce or working with the local data
RETRY_DATA_FROM_SF = os.getenv("RETRY_DATA_FROM_SF", False)

# CONFIGURATION SALESFORCE
CONSUMER_KEY = os.getenv("CONSUMER_KEY", '')
CONSUMER_SECRET = os.getenv("CONSUMER_SECRET", '')
SFDC_MYDOMAIN = os.getenv("SFDC_MYDOMAIN", '')
SFDC_AUTH = os.getenv("SFDC_AUTH", "login")
SFDC_USERNAME = os.getenv("SFDC_USERNAME", "SOMEUSER@USERNAME.COM")
SFDC_PASSWORD = os.getenv("SFDC_PASSWORD", "SOMEPASSWORD")
SFDC_SECURITY_TOKEN = os.getenv("SFDC_SECURITY_TOKEN", "SOMETOKEN")
SFDC_SANDBOX = os.getenv("SFDC_SANDBOX", 'False')

# CONFIGURATION MERLION
MERLION_PLOT_SHOW = os.getenv("MERLION_PLOT_SHOW", 'False')
MERLION_TRAINING_SAMPLE = os.getenv("MERLION_TRAINING_SAMPLE", 0.9)
MERLION_PREDICT_MODEL = os.getenv("MERLION_PREDICT_MODEL", 'selector')
MERLION_LIMIT_EVENTS = os.getenv("MERLION_LIMIT_EVENTS", 12)
MERLION_MAX_FORECAST_STEPS = os.getenv("MERLION_MAX_FORECAST_STEPS", 100)

# CONFIGURATION AWS S3
AWS_S3_ENABLE = os.getenv("AWS_S3_ENABLE")
AWS_S3_BUCKET_NAME = os.getenv("AWS_S3_BUCKET_NAME", "mlimit")+"/"
AWS_S3_ACCESS_KEY_ID = os.getenv("AWS_S3_ACCESS_KEY_ID")
AWS_S3_SECRET_ACCESS_KEY = os.getenv("AWS_S3_SECRET_ACCESS_KEY")


class Config(object):
    """ Config """
    DEBUG = False
    TESTING = False
    CSRF_ENABLED = True
    SECRET_KEY = SECRET_KEY
    SQLALCHEMY_DATABASE_URI = DATABASE_URL
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SESSION_REDIS = redis.Redis("redis")
    REDIS_URL = REDIS_URL


class ProductionConfig(Config):
    """ ProductionConfig """
    DEBUG = False
    PROPAGATE_EXCEPTIONS = True


class StagingConfig(Config):
    """ StagingConfig """
    DEVELOPMENT = True
    DEBUG = True


class DevelopmentConfig(Config):
    """ DevelopmentConfig """
    DEVELOPMENT = True
    DEBUG = True


class TestingConfig(Config):
    """ TestingConfig """
    TESTING = True


# DEBUGGING
# CONFIGURATION GENERIC AND DATABASE
logging.logger.debug('REDIS_URL %s', REDIS_URL)
logging.logger.debug('WORKER_TYPE %s', WORKER_TYPE)
logging.logger.debug('CREATE_ENGINE_DEBUG %s', CREATE_ENGINE_DEBUG)
logging.logger.debug('DATABASE_RETENTION_DAYS %s', DATABASE_RETENTION_DAYS)
logging.logger.debug('DATABASE_URL %s', DATABASE_URL)
logging.logger.debug('DATABASE_HEROKU_CONNECT %s', DATABASE_HEROKU_CONNECT)
logging.logger.debug('APP_SETTINGS %s', APP_SETTINGS)
logging.logger.debug('RETRY_DATA_FROM_SF %s', RETRY_DATA_FROM_SF)
# CONFIGURATION SALESFORCE
logging.logger.debug('CONSUMER_KEY %s', CONSUMER_KEY)
logging.logger.debug('SFDC_MYDOMAIN %s', SFDC_MYDOMAIN)
logging.logger.debug('SFDC_AUTH %s', SFDC_AUTH)
logging.logger.debug('SFDC_SANDBOX %s', SFDC_SANDBOX)
logging.logger.debug('SFDC_USERNAME %s', SFDC_USERNAME)
# CONFIGURATION MERLION
logging.logger.debug('MERLION_PLOT_SHOW %s', MERLION_PLOT_SHOW)
logging.logger.debug('MERLION_TRAINING_SAMPLE %s', MERLION_TRAINING_SAMPLE)
logging.logger.debug('MERLION_PREDICT_MODEL %s', MERLION_PREDICT_MODEL)
logging.logger.debug('MERLION_LIMIT_EVENTS %s', MERLION_LIMIT_EVENTS)
logging.logger.debug('MERLION_MAX_FORECAST_STEPS %s', MERLION_MAX_FORECAST_STEPS)
# CONFIGURATION AWS S3
logging.logger.debug('AWS_S3_ENABLE %s', AWS_S3_ENABLE)
logging.logger.debug('AWS_S3_BUCKET_NAME %s', AWS_S3_BUCKET_NAME)
