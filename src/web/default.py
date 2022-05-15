# -*- coding: utf-8 -*-
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree
# Default blueprint used for configure the OAuth2 web server flow with a Salesforce Organization
import os
import src.utils.database as utils_db
from urllib.parse import urlparse
import warnings
from flask import Blueprint, redirect, request, send_from_directory, current_app
import src.config as config
from src.config.logger import LoggerClass
from sqlalchemy import exc
import requests
logging = LoggerClass.instance()
warnings.filterwarnings("ignore")
default_app = Blueprint('default', __name__, url_prefix='/')


def _get_redirect_uri(o):
    """ Return the redirect URI required by the OAuth2 authorization

    Args:
        o (base_url): domain

    Returns:
        String: the callback url
    """
    REDIRECT_URI = None
    if o.hostname == 'localhost':
        REDIRECT_URI = 'http://localhost:8000/callback'
    else:
        REDIRECT_URI = f'https://{o.hostname}/callback'
    return REDIRECT_URI


@default_app.route('/', methods=['GET'])
def home():
    return 'mlimit-merlion is running', 200


@default_app.route('/login', methods=['GET', 'POST'])
def _login():
    """Login is required for the OAuth2 authentication with Salesforce

    Returns:
        redirect: redirect to callback page once authenticated
    """
    heroku_app_name = urlparse(request.base_url)
    # The connected app redirect URI, which you can find on the connected app Manage Connected Apps page or from the connected app definition.
    REDIRECT_URI = _get_redirect_uri(heroku_app_name)
    logging.logger.debug('REDIRECT_URI: %s', REDIRECT_URI)
    logging.logger.debug('SFDC_MYDOMAIN: %s', config.SFDC_MYDOMAIN)
    if config.SFDC_MYDOMAIN is not None and len(config.SFDC_MYDOMAIN) != 0:
        AUTHORIZE_URL = f'https://{config.SFDC_MYDOMAIN}/services/oauth2/authorize'
        logging.logger.debug('AUTHORIZE_URL: %s', AUTHORIZE_URL)
        url = "%s?response_type=code&client_id=%s&redirect_uri=%s" % (
            AUTHORIZE_URL, config.CONSUMER_KEY, REDIRECT_URI)
        return redirect(url)
    else:
        return "something goes wrong", 500


@default_app.route('/callback')
def _callback():
    """Callback endpoint

    Returns:
        String: 201 for success and token saved to the database, 500 in case of error
    """
    heroku_app_name = urlparse(request.base_url)
    # The connected app redirect URI, which you can find on the connected app Manage Connected Apps page or from the connected app definition.
    REDIRECT_URI = _get_redirect_uri(heroku_app_name)
    code = request.args.get('code')
    data = {
        'grant_type': 'authorization_code',
        'redirect_uri': REDIRECT_URI,
        'code': code,
        'client_id': config.CONSUMER_KEY,
        'client_secret': config.CONSUMER_SECRET
    }
    headers = {
        'content-type': 'application/x-www-form-urlencoded'
    }
    INSTANCE_URL = f'https://{config.SFDC_MYDOMAIN}'
    ACCESS_TOKEN_URL = f'{INSTANCE_URL}/services/oauth2/token'
    logging.logger.debug('ACCESS_TOKEN_URL: %s', ACCESS_TOKEN_URL)
    req = requests.post(ACCESS_TOKEN_URL, data=data, headers=headers)
    response = req.json()
    logging.logger.debug('response: %s', response)

    if int(req.status_code) < 400 and response is not None:
        try:
            # Clean up the session token based on the Organization ID
            utils_db.delete_sfdc_auth_config(INSTANCE_URL)

            # Create new Salesforce Authentication
            utils_db.add_sfdc_auth_config(
                response=response, heroku_app_name=heroku_app_name.hostname)

            return "ok, connected to Salesforce", 201
        except ConnectionError:
            return "something goes wrong on the connection", 500
        except exc.SQLAlchemyError:
            return "something goes wrong on the db", 500
    else:
        return "something goes wrong", 500


@default_app.route('/favicon.ico')
def _favicon():
    """favicon

    Returns:
        send_from_directory: icon
    """
    return send_from_directory(os.path.join(current_app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')
