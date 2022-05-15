# -*- coding: utf-8 -*-
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from src.config.logger import LoggerClass
logging = LoggerClass.instance()
    
def check_request(request):
    """Utility function to check the request

    Args:
        request (_type_): _description_

    Raises:
        TypeError: _description_
        TypeError: _description_
        Exception: _description_
        Exception: _description_
    """
    logging.logger.debug('request.headers: \n%s', request.headers)
    logging.logger.debug('request.body: \n%s', request.json)
    if 'X-Fields' not in request.headers:
        raise TypeError('Header wrong or missing')
    if 'Authorization' not in request.headers:
        raise TypeError('Header wrong or missing')
    if 'status' not in request.json:
        raise TypeError('Body wrong or missing')
    auth = request.headers['Authorization'].split()
    if len(auth) != 2 and auth[0].lower() != "basic":
        raise TypeError('Wrong authentication')