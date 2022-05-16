# -*- coding: utf-8 -*-
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
''' Class for logging text messages '''
import os
from pathlib import Path  # Python 3.6+ only
import logging
import json
import requests
from dotenv import load_dotenv
from src.config.singleton import SingletonClass
load_dotenv()
# OR, explicitly providing path to '.env'
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)
LOGLEVEL = os.getenv(
    "LOGLEVEL", 'INFO')
WEBHOOK_URL = os.getenv("WEBHOOK_URL")
from requests.adapters import HTTPAdapter


class CustomHttpHandler(logging.Handler):
    # code retrieved from here https://stackoverflow.com/a/64360700
    def __init__(self, url: str, silent: bool = True):
        '''
        Initializes the custom http handler
        Parameters:
            url (str): The URL that the logs will be sent to
            token (str): The Authorization token being used
            silent (bool): If False the http response and logs will be sent 
                           to STDOUT for debug
        '''
        self.url = url
        self.silent = silent
        # sets up a session with the server
        self.MAX_POOLSIZE = 100
        self.session = session = requests.Session()
        session.headers.update({
            'Content-Type': 'application/json'
        })
        self.session.mount('https://', HTTPAdapter(
            pool_connections=self.MAX_POOLSIZE,
            pool_maxsize=self.MAX_POOLSIZE
        ))

        super().__init__()

    def emit(self, record):
        '''
        This function gets called when a log event gets emitted. It recieves a
        record, formats it and sends it to the url
        Parameters:
            record: a log record
        '''
        logEntry = self.format(record)
        myobj = {"text":logEntry}
        response = self.session.post(self.url, json=myobj)

        if not self.silent:
            print(logEntry)
            print(response.content)




@SingletonClass
class LoggerClass:
    """ SingletonClass to establish a connection to Salesforce. """
    logger = None

    def __init__(self):
        """ __login
        Parameters
        ----------

        Returns
        -------

        """
        self.__load()

    def __load(self):
        logger = logging.getLogger(__name__)
        level = logging.getLevelName(LOGLEVEL)
        logger.setLevel(level)
        logger.propagate = False
        # create console handler with a higher log level
        console_handler = logging.StreamHandler()
        console_handler.setLevel(LOGLEVEL)
        formatter = logging.Formatter(
            "%(asctime)s [%(filename)-20.20s] [%(lineno)-4.4d] [%(levelname)-8.8s] %(message)s"
        )
        console_handler.setFormatter(formatter)
        # add the handlers to the logger
        if (logger.hasHandlers()):
            logger.handlers.clear()
        logger.addHandler(console_handler)
        
        if WEBHOOK_URL is not None:
            # create formatter - this formats the log messages accordingly
            http_formatter = logging.Formatter(json.dumps({
                'app':'mlimit-merlion',
                'time': '%(asctime)s',
                'pathname': '%(pathname)s',
                'line': '%(lineno)d',
                'logLevel': '%(levelname)s',
                'message': '%(message)s'
            }))
            # create a custom http logger handler
            httpHandler = CustomHttpHandler(
                url=WEBHOOK_URL,
                silent=False
            )
            httpHandler.setLevel(logging.ERROR)
            # add formatter to custom http handler
            httpHandler.setFormatter(http_formatter)
            logger.addHandler(httpHandler)
            # add handler to logger
        
        
        self.logger = logger
