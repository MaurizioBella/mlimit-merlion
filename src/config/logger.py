# -*- coding: utf-8 -*-
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
''' Class for logging text messages '''
import os
from pathlib import Path  # Python 3.6+ only
import logging
from dotenv import load_dotenv
from src.config.singleton import SingletonClass
load_dotenv()
# OR, explicitly providing path to '.env'
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)
LOGLEVEL = os.getenv(
    "LOGLEVEL", 'INFO')


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
        self.logger = logger
