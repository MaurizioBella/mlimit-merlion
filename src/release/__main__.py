# -*- coding: utf-8 -*-
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
''' run this only postdeploy'''
from src.config.logger import LoggerClass
logging = LoggerClass.instance()


def release():
    """ function for release
        Parameters
        ----------
        Returns
        -------
        """
    logging.logger.info("start process")


if __name__ == '__main__':
    release()
