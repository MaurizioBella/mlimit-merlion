# -*- coding: utf-8 -*-
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Data Retention
import datetime
from src.models.measure import Measure
import src.config as config
from src.config.logger import LoggerClass
from src.models import db_session
import warnings
from sqlalchemy import exc
import os
import psycopg2

Session = db_session()

logging = LoggerClass.instance()
warnings.filterwarnings("ignore")


def main():
    '''
    delete records older than DATABASE_RETENTION_DAYS
    '''
    if config.DATABASE_RETENTION_DAYS:
        time_retention_days = datetime.datetime.now(
        ) - datetime.timedelta(days=int(config.DATABASE_RETENTION_DAYS))
        try:
            result = Session.query(
                Measure).filter(Measure.eventdate < time_retention_days).delete()
            Session.commit()
            logging.logger.debug('delete %s records', result)
        except exc.SQLAlchemyError as exc_error:
            logging.logger.error(exc_error)
            raise
        finally:
            Session.close()


def delete_trigger_logs():
    """ Populate tables
        Parameters
        ----------

        Returns
        -------

        """

    __location__ = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '../..', 'database'))

    try:
        # connect to the PostgreSQL server
        logging.logger.debug('Connecting to the PostgreSQL database')
        conn = psycopg2.connect(config.DATABASE_URL)
        # conn = psycopg2.connect(**params)

        if config.DATABASE_HEROKU_CONNECT == 'True':
            with conn.cursor() as cursor:
                cursor.execute(
                    open(os.path.join(__location__, 'cleaner.sql'), "r").read())
                # cursor.execute(open("dataload.sql", "r").read())
                # commit the changes
            logging.logger.debug('Executed cleaner.sql')
            conn.commit()

        # close communication with the PostgreSQL database server
        conn.close()

    except psycopg2.DatabaseError as db_error:
        logging.logger.error(db_error)
    finally:
        if conn is not None:
            conn.close()


if __name__ == '__main__':
    main()
    delete_trigger_logs()
