# -*- coding: utf-8 -*-
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
''' retry governor limits from Salesforce Organization'''
from sqlalchemy import exc
import pandas as pd
from src.models.measure_config import MeasureConfig
from src.models.measure import Measure
from src.models import db_session
from src.config.logger import LoggerClass
from src.config.salesforce import SalesforceConnection


def retry_rest_api_resource_limits():
    """ main function to retry data from Salesforce Org using REST API Limits endpoint
        Parameters
        ----------
        Returns
        -------
        """
    # Instantiate classes
    sf = SalesforceConnection.instance()
    Session = db_session()
    logging = LoggerClass.instance()
    logging.logger.info('process started')
    limit = sf.get_limits()  # return limits from the ORG
    if limit is not None:
        df = pd.DataFrame.from_records(limit.json())
        df = df.T
        df.drop(df.columns.difference(
            ['Max', 'Remaining']), 1, inplace=True)
        df.reset_index(inplace=True)
        df = df.rename(columns={
            'index': 'name'})
        df.rename(columns={"Max": "max_value",
                           "Remaining": "remaining_value"}, inplace=True)
        source = 'restapi'
        df_limits = pd.read_sql(Session.query(
            MeasureConfig).filter(MeasureConfig.active == True, MeasureConfig.source == source).statement, Session.bind)

        if len(df_limits.index) > 0:
            df['name'] = df['name'].map(df_limits.set_index("name")["sfid"])
            df = df.dropna()
            df.rename(columns={"name": "limitname"}, inplace=True)

            try:
                Session.bulk_insert_mappings(
                    Measure, df.to_dict(orient="records"))
                Session.commit()
                logging.logger.debug('added records %s', df.shape)
            except exc.SQLAlchemyError as exc_error:
                logging.logger.error(exc_error)
                raise
            finally:
                Session.close()
        else:
            logging.logger.debug(
                'Rest API limits are not configured in Config Limit')
    else:
        logging.logger.error(
            'No Rest API limits found')
