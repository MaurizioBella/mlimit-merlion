# -*- coding: utf-8 -*-
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from sqlalchemy import create_engine
import warnings
import shutil
import s3fs
import os
import src.config as config
from src.config.logger import LoggerClass
logging = LoggerClass.instance()
warnings.filterwarnings("ignore")


def clean_directory(dir_path):
    """ Delete all the files within the path and from S3 if enabled. 
    Local root directory is /models/ 

    Args:
        dir_path (_type_): directory name
    """
    try:
        shutil.rmtree(dir_path)
        logging.logger.debug("Deleted local folder %s" % (dir_path))
    except OSError as e:
        logging.logger.warning("OSError: %s : %s" % (dir_path, e.strerror))
    if (config.AWS_S3_ENABLE == 'True'):
        fs = s3fs.S3FileSystem(
            key=config.AWS_S3_ACCESS_KEY_ID,
            secret=config.AWS_S3_SECRET_ACCESS_KEY)
        if fs.exists(os.path.join(config.AWS_S3_BUCKET_NAME, dir_path)):
            fs.rm(config.AWS_S3_BUCKET_NAME+dir_path,
                  recursive=True)  # FileNotFoundError
        logging.logger.debug(
            "Deleted AWS S3 folder %s from bucket %s" % (dir_path, config.AWS_S3_BUCKET_NAME))


def save_to_s3(path):
    """ save() method creates a new directory at the specified path, where it saves a json file representing the model's config, as well as a binary file for the model's state.
    this method copies those files in S3 within the specified path: e.g. /model_arima/

    Args:
        path (_type_): path name e.g. model_arima
    """
    if (config.AWS_S3_ENABLE == 'True'):
        fs = s3fs.S3FileSystem(
            key=config.AWS_S3_ACCESS_KEY_ID,
            secret=config.AWS_S3_SECRET_ACCESS_KEY)
        local_path = path
        s3_path = config.AWS_S3_BUCKET_NAME+path
        fs.put(local_path, s3_path, recursive=True)
        logging.logger.debug('Saved file to S3: %s' % path)


def read_from_s3(path):
    """ load() methods from local path or from S3 if enabled

    Args:
        path (string): path name e.g. model_arima
    """
    if (config.AWS_S3_ENABLE == 'True'):        
        fs = s3fs.S3FileSystem(
            key=config.AWS_S3_ACCESS_KEY_ID,
            secret=config.AWS_S3_SECRET_ACCESS_KEY)
        check_exists = fs.exists(config.AWS_S3_BUCKET_NAME+path)
        fs.download(config.AWS_S3_BUCKET_NAME+path, path, recursive=True)
        logging.logger.debug('Downloaded folder from S3 %s and it exists: ' % (path,check_exists))
