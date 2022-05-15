# -*- coding: utf-8 -*-
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""init file """
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
import src.config as config


def enable():
    return True if config.CREATE_ENGINE_DEBUG == "True" else False


engine = create_engine(
    config.DATABASE_URL,
    echo=enable()
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def db_session():
    """ fuction db_session
        Parameters
        ----------
        Returns
        -------
        """
    return scoped_session(sessionmaker(
        autocommit=False, autoflush=False, bind=engine))
