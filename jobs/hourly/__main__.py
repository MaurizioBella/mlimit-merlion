# -*- coding: utf-8 -*-
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
''' init file '''
import traceback
from src.config.logger import LoggerClass
import src.config as config
from worker import conn
import time
import sys
from memory_profiler import profile
from rq import Queue
from jobs.hourly.predict_hourly_limits import predict_hourly_limits
from src.utils.retry_rest_api_resource_limits import retry_rest_api_resource_limits
from src.utils.retry_generic_limits import main as retry_generic_limits
from src.config.error import NoSalesforceConnection
import src.utils.database as utils_db
logging = LoggerClass.instance()


@profile
def calculate_memory(measure, train_only):
    """Calculate Memory
    profile is a python module for monitoring memory consumption of a process
    Args:
        measure ([List]): list of event rest limits
    """
    predict_hourly_limits(measure, train_only)


if __name__ == '__main__':
    """ Schedule this job to feed the model
    """
    train_only = False
    if "--train-only" in sys.argv:
        train_only = True
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    start_time = time.time()
    logging.logger.info('Start jobs process at %s', current_time)
    logging.logger.debug('WORKER_TYPE %s', config.WORKER_TYPE)
    if (config.RETRY_DATA_FROM_SF == 'True' and train_only == False):
        logging.logger.debug('Start retrieving Salesforce Org limits...')
        try:
            retry_rest_api_resource_limits()
            retry_generic_limits()
        except NoSalesforceConnection as no_salesforce_connection:
            logging.logger.error(no_salesforce_connection)
    for variate in utils_db.get_active_measure_config():
        logging.logger.debug(
            'Start worker variate %s train %s' % (variate[0], train_only))
        if int(config.WORKER_TYPE) == 1:
            calculate_memory(variate[0], train_only)
            end_time = round(time.time() - start_time, 2)
            logging.logger.info('End jobs process took %s secs', end_time)
        elif int(config.WORKER_TYPE) == 2:
            q = Queue('low', connection=conn)
            try:
                dar = q.enqueue(predict_hourly_limits,
                                # depends_on=gl,
                                ttl=3600,
                                failure_ttl=3600,
                                job_timeout=3600,
                                kwargs={
                                    'measure': variate[0],
                                    'train_only': train_only
                                })
            except Exception as e:
                traceback.print_exc()
                logging.logger.error(e)
                raise
        else:
            logging.logger.error('Worker Type not supported %s', config.WORKER_TYPE)
