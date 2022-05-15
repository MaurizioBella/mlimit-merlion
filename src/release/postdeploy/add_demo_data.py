# -*- coding: utf-8 -*-
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""post deploy"""
import os
import pandas as pd
from src.models.measure_config import MeasureConfig
from src.models.measure import Measure
from src.models.sfdc_auth import SfdcAuth
from src.models.holiday import Holiday
import src.config as config
from src.config.logger import LoggerClass
from src.models import db_session
import warnings
from sqlalchemy import create_engine, exc
Session = db_session()

logging = LoggerClass.instance()
warnings.filterwarnings("ignore")
engine = create_engine(
    config.DATABASE_URL, echo=True
)

__location__ = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../..', 'database/data'))


def add_config_auth():
    """
    Add demo configuration authentication
    """
    sfdcAuth = SfdcAuth(
        active=True
    )
    try:
        Session.add(sfdcAuth)
        Session.commit()
        logging.logger.debug('added Authentication record')
    except exc.SQLAlchemyError as exc_error:
        logging.logger.error(exc_error)
        raise
    finally:
        Session.close()


def add_holiday():
    '''
    Add demo record to Holiday
    '''
    source = 'restapi'
    df_limits = Session.query(
        MeasureConfig.sfid).filter(MeasureConfig.active == True, MeasureConfig.source == source).first()
    if len(df_limits) > 0:
        holiday = Holiday(
            ds='2021-01-29',
            holiday='demo',
            limitname=df_limits[0])
        try:
            Session.add(holiday)
            Session.commit()
            logging.logger.debug('added Holiday record')
        except exc.SQLAlchemyError as exc_error:
            logging.logger.error(exc_error)
            raise
        finally:
            Session.close()
    else:
        logging.logger.error(
            'Prophet Holiday: nothing to add. Check the Config Limits')


def add_resource_limit():
    '''
    Add demo record to Measure
    '''
    df_resource = pd.read_csv(os.path.join(__location__, 'measure.csv'))

    source = 'restapi'
    df_limits = pd.read_sql(Session.query(
        MeasureConfig).filter(MeasureConfig.active == True, MeasureConfig.source == source).statement, Session.bind)

    if len(df_limits.index) > 0:
        df_resource['limitname'] = df_resource['limitname'].map(
            df_limits.set_index("name")["sfid"])
        df = df_resource.dropna()
        try:
            Session.bulk_insert_mappings(
                Measure, df.to_dict(orient="records"))
            Session.commit()
            logging.logger.debug('added Measure records %s', df.shape)
        except exc.SQLAlchemyError as exc_error:
            logging.logger.error(exc_error)
            raise
        finally:
            Session.close()
    else:
        logging.logger.error(
            'Resource Limit: nothing to add. Check the Config Limits')


def add_config_limit():
    '''
    Add demo record to MeasureConfig
    '''
    df_config = pd.read_csv(os.path.join(__location__, 'measureconfig.csv'))

    try:
        Session.bulk_insert_mappings(
            MeasureConfig, df_config.to_dict(orient="records"))
        Session.commit()
        logging.logger.debug('Added Config Limit records %s', df_config.shape)
    except exc.SQLAlchemyError as exc_error:
        logging.logger.error(exc_error)
        raise
    finally:
        Session.close()


def main():
    ''' Add demo records '''
    if config.DATABASE_HEROKU_CONNECT == 'False':
        add_config_limit()
        add_config_auth()
    add_resource_limit()
    add_holiday()


if __name__ == '__main__':
    main()
