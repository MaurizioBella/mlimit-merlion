# -*- coding: utf-8 -*-
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""class in charge of forecasting"""
import time
from src.models.measure_config_type import MeasureConfigType
from src.models.sfdc_auth import SfdcAuth
from src.models.measure import Measure
from src.config.logger import LoggerClass
from src.models import db_session
from src.models.forecaster import Forecaster
import src.config as config
from sqlalchemy import create_engine, exc
from src.models.measure_config import MeasureConfig
from dotenv import load_dotenv
import numpy as np
import warnings
import pandas as pd
load_dotenv()
logging = LoggerClass.instance()
Session = db_session()
warnings.filterwarnings("ignore")
engine = create_engine(
    config.DATABASE_URL
)


def get_measure_config_by_limitname(prediction):
    return Session.query(MeasureConfig).filter(
        MeasureConfig.sfid == prediction.measure).all()


def add_sfdc_auth_config(**kwargs):
    response = kwargs['response'] if "response" in kwargs else None
    heroku_app_name = kwargs['heroku_app_name'] if "heroku_app_name" in kwargs else None
    refresh_token = kwargs['refresh_token'] if "refresh_token" in kwargs else kwargs['response']['refresh_token']
    if response is None:
        logging.logger.error('ConnectionError')
        raise ConnectionError

    responseid = response['id']
    r = responseid.split('/')
    orgid_18 = r[4]
    orgid = orgid_18[:-3]  # remove last 3 characters from 18 to 15 digit

    u1 = SfdcAuth(
        heroku_app_name=heroku_app_name,
        access_token=response['access_token'],
        refresh_token=refresh_token,
        oauth_signature=response['signature'],
        scope=response['scope'],
        instance_url=response['instance_url'],
        userid=response['id'],
        orgid=orgid,
        token_type=response['token_type'],
        issued_at=time.strftime(
            '%Y-%m-%d %H:%M:%S',
            time.gmtime(int(response['issued_at'])/1000)),
        active=True
    )
    try:
        Session.add(u1)
        Session.commit()
        logging.logger.debug(
            'Refresh token - added salesforce authentication')
    except exc.SQLAlchemyError as exc_error:
        Session.rollback()
        logging.logger.error(exc_error)
        raise
    finally:
        Session.close()


def delete_sfdc_auth_config(INSTANCE_URL):
    """ Utils method to delete OAuth configurations with Salesforce
    """
    try:
        Session.query(SfdcAuth).filter(
            SfdcAuth.instance_url == INSTANCE_URL).delete()
        Session.commit()
        logging.logger.debug(
            'Deleted SFDC Auth record')
    except exc.SQLAlchemyError as exc_error:
        Session.rollback()
        logging.logger.error(exc_error)
        raise
    finally:
        Session.close()


def get_sfdc_auth_config():
    """ Utils method to return the active OAuth configuration with Salesforce

    Returns:
        list: Org Id with the configuration
    """
    try:
        return Session.query(SfdcAuth).filter(
            SfdcAuth.active == True).all()
    except exc.SQLAlchemyError as exc_error:
        logging.logger.error(exc_error)
        raise
    finally:
        Session.close()


def get_active_measure_config():
    """ Retry all the governor limits configured as measures in the object Measure Config

    Returns:
        text: Id of governor limits configured in Measure Config
    """
    config_limits = Session.query(
        MeasureConfig.sfid).filter(MeasureConfig.active == True).all()
    return config_limits


# def get_measure_config_by_limitname(measure):
#     return pd.DataFrame(Session.query(
#         MeasureConfig).filter(MeasureConfig.sfid == measure).all())


def get_df_measure_config_by_limitname(measure):
    return pd.DataFrame(Session.query(
        MeasureConfig).filter(MeasureConfig.sfid == measure).all())


def get_measures_by_limitname(measure):
    """_summary_

    Args:
        measure (DataFrame): _description_
    """
    return pd.DataFrame(Session.query(
        Measure)
        .filter(Measure.limitname == measure).all())


def update_measure_config(prediction):
    # smape, rmse, last_train_time, model_name, model_path, measure = prediction
    try:
        Session.query(MeasureConfig).filter(
            MeasureConfig.sfid == prediction.measure).update(
            {MeasureConfig.model_name: prediction.model_name,
                MeasureConfig.model_path: prediction.model_path,
                MeasureConfig.smape: prediction.smape,
                MeasureConfig.last_train_time: prediction.last_train_time,
                MeasureConfig.rmse: prediction.rmse}
        )
        Session.commit()
        logging.logger.debug(
            'MeasureConfig updated sfid %s' % (prediction.measure))
    except exc.SQLAlchemyError as exc_error:
        Session.rollback()
        logging.logger.error(exc_error)
        raise
    finally:
        Session.close()


def insert_forecaster(measure, df):
    """ Inserts DataFrame to the Database
    Parameters
    ----------
    df: Pandas DataFrame

    Returns
    -------
    None
    """
    df['limitname'] = measure
    df_limits = pd.read_sql(Session.query(
        MeasureConfig).filter(MeasureConfig.sfid == measure).statement, Session.bind)
    df['name'] = df['limitname'].map(
        df_limits.set_index("sfid")["name"])
    df = df.replace({np.nan: None})
    try:
        Session.query(Forecaster).filter(
            Forecaster.limitname == measure).delete()
        Session.commit()
        Session.bulk_insert_mappings(
            Forecaster, df.to_dict(orient="records"))
        Session.commit()
        logging.logger.info('Added %s records to the database.', len(df))
    except exc.SQLAlchemyError as exc_error:
        Session.rollback()
        logging.logger.error(exc_error)
        raise
    finally:
        Session.close()


def update_measure_config_type(source, df):
    try:
        Session.query(MeasureConfigType).filter(
            MeasureConfigType.name == source).delete()
        Session.query(MeasureConfig).filter(
            MeasureConfig.source == source).delete()
        Session.commit()
        measureConfigType = MeasureConfigType(
            name=source,
            active=True
        )
        Session.add(measureConfigType)
        Session.bulk_insert_mappings(
            MeasureConfig, df.to_dict(orient="records"))
        Session.commit()
    except exc.SQLAlchemyError as exc_error:
        Session.rollback()
        logging.logger.error(exc_error)
    finally:
        Session.close()
