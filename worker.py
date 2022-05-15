# -*- coding: utf-8 -*-
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""run worker"""
from rq import Worker, Queue, Connection
import redis
import src.config as config
from src.config.logger import LoggerClass
logging = LoggerClass.instance()

listen = ['high', 'default', 'low']

conn = redis.from_url(config.REDIS_URL)

if __name__ == '__main__':
    with Connection(conn):
        logging.logger.info('created a new Queue connection')
        worker = Worker(map(Queue, listen))
        worker.work()
