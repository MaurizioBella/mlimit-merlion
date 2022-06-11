# -*- coding: utf-8 -*-
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
''' class used to connect to Salesforce Organization'''
import json
import requests
from simple_salesforce import Salesforce
from simple_salesforce.exceptions import SalesforceGeneralError, SalesforceExpiredSession, SalesforceResourceNotFound, SalesforceRefusedRequest, SalesforceAuthenticationFailed, SalesforceMalformedRequest
from src.config.error import NoSalesforceConnection
import src.utils.database as utils_db
import src.config as config
from src.config.logger import LoggerClass
from src.config.singleton import SingletonClass
logging = LoggerClass.instance()


@SingletonClass
class SalesforceConnection:
    """ SingletonClass to establish a connection to Salesforce. """
    session = None
    __sfdc_auth = None

    def __init__(self):
        """ __login
        Parameters
        ----------

        Returns
        -------

        """
        logging.logger.debug('SalesforceConnection init')
        self.__login()

    def __set_refresh_token(self):
        """ run the refresh token to get the access token and update the SFDC Auth entity
        """
        url_service = f'{self.__sfdc_auth[0].instance_url}/services/oauth2/token?grant_type=refresh_token&client_id={config.CONSUMER_KEY}&client_secret={config.CONSUMER_SECRET}&refresh_token={self.__sfdc_auth[0].refresh_token}'
        req = requests.post(url_service)
        response = json.loads(req.text)
        # Clean up the session token based on the Organization ID
        INSTANCE_URL = f'https://{config.SFDC_MYDOMAIN}'
        utils_db.delete_sfdc_auth_config(INSTANCE_URL)

        # Create new Salesforce Authentication
        utils_db.add_sfdc_auth_config(
            response=response, refresh_token=self.__sfdc_auth[0].refresh_token)

    def __login(self,orgid=None):
        """ Method to login to a Salesforce Organization

        Raises:
            NoSalesforceConnection: _description_
        """
        sfdc_auth = utils_db.get_sfdc_auth_config()
        try:
            if sfdc_auth is not None and len(sfdc_auth) > 0 \
                    and config.SFDC_AUTH.upper() == 'OAUTH2':
                sf = Salesforce(
                    instance_url=sfdc_auth[0].instance_url,
                    session_id=sfdc_auth[0].access_token
                )
                self.__sfdc_auth = sfdc_auth
                # just to check if the session is active
                sf.query_all("SELECT Id FROM Contact limit 1")
                self.session = sf
                logging.logger.debug('SalesforceConnection login')

            elif config.SFDC_AUTH.upper() == 'LOGIN':
                SFDC_MYDOMAIN = f'https://{config.SFDC_MYDOMAIN}'
                if config.SFDC_SANDBOX == 'True':
                    self.session = Salesforce(client_id='MLimit', domain='test',
                                              username=config.SFDC_USERNAME,
                                              password=config.SFDC_PASSWORD,
                                              security_token=config.SFDC_SECURITY_TOKEN)
                else:
                    self.session = Salesforce(instance_url=SFDC_MYDOMAIN,
                                              client_id='MLimit',
                                              username=config.SFDC_USERNAME,
                                              password=config.SFDC_PASSWORD,
                                              security_token=config.SFDC_SECURITY_TOKEN)
            else:
                raise NoSalesforceConnection()
        except SalesforceGeneralError as sge:
            logging.logger.error(sge)
            raise
        except SalesforceAuthenticationFailed as saf:
            logging.logger.error(saf)
            raise
        except SalesforceExpiredSession as ses:
            # [{'message': 'Session expired or invalid', 'errorCode': 'INVALID_SESSION_ID'}]
            logging.logger.error(ses)
            self.__set_refresh_token() # only set the new access_token without retrieving downloading the data
            raise
        except SalesforceResourceNotFound as srn:
            logging.logger.error(srn)
            raise
        except SalesforceRefusedRequest as srr:
            logging.logger.error(srr)
            raise
        except SalesforceMalformedRequest as smr:
            logging.logger.error(smr)
            raise

    def get_limits(self):
        """ returns lists information about limits in your Organization
        refer to online documentation https://developer.salesforce.com/docs/atlas.en-us.api_rest.meta/api_rest/resources_limits.htm
        Parameters ----------

        Returns
        -------

        """

        limit = None
        if self.session is not None:
            url = 'https://'+self.session.sf_instance+'/services/data/v51.0/limits/'
            headers = {'Authorization': 'Bearer '+self.session.session_id}
            logging.logger.debug('retrieving limits from %s' % (url))
            limit = requests.get(url, headers=headers)
        return limit

    def get_info(self):
        """ returns list the resources available for the specified API version
        This provides the name and URI of each additional resource.
        Parameters
        ----------

        Returns
        -------

        """
        data_service = None
        if self.session is not None:
            url_service = 'https://'+self.session.sf_instance+'/services/data/v51.0/'
            headers = {'Authorization': 'Bearer '+self.session.session_id}
            data_service = requests.get(url_service, headers=headers)
        return data_service
