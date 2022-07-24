# -*- coding: utf-8 -*-
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
''' blueprint for API '''
import traceback
from werkzeug.exceptions import BadRequest
import binascii
from flask_restx import Resource, Api, fields
import base64
import src.config as config
from sqlalchemy import exc
from flask import Blueprint, request
from src.config.salesforce import SalesforceConnection
from src.config.error import NoSalesforceConnection
from src.models.sfdc_auth import SfdcAuth
import src.utils.api as utils_api
import pandas as pd
from src.config.logger import LoggerClass
from src.models.measure_config import MeasureConfig
from src.models.measure_config_type import MeasureConfigType

from src.models import db_session
logging = LoggerClass.instance()
Session = db_session()
config_route = Blueprint('config', __name__)

api = Api(config_route,
          title='MLimit Measure Configuration',
          version='1.0',
          description='Measure Configuration Type',
          # All API metadatas
          )

ns = api.namespace('configs', description='Measure Config Types')

m_config_type = api.model('MeasureConfigType', {
    'name': fields.String,
    'status': fields.Boolean,
})

api_response_response = api.model('ApiResponseCode', {
    'message': fields.String(default='ok'),
    'code': fields.Integer(default=400),
})


@ns.route('/', methods=['GET', 'POST'],
          strict_slashes=False)
class ConfigRestApi(Resource):

    def __limits_to_df(self):
        """ Create a Pandas DataFrame from the Salesforce Organization governor limits

        Returns:
            DataFrame: list of governor limits
        """
        sf = SalesforceConnection.instance()
        limit = sf.get_limits()  # return limits from the ORG
        if limit is not None:
            df = pd.DataFrame.from_records(limit.json())
            df = df.T
            df.drop(df.columns.difference(
                ['Max']), 1, inplace=True)
            df.reset_index(inplace=True)
            df = df.rename(columns={
                'index': 'name'})
            df.rename(columns={"Max": "max_value"}, inplace=True)
            df['source'] = 'restapi'
            df['frequency'] = df.name.apply(lambda x:
                                            'H' if x.startswith('Daily') else
                                            'D' if x.startswith('Monthly') else
                                            'M' if x.startswith('Hourly') else
                                            'D')
            logging.logger.info('retrieved %s limits \n %s' %
                                (len(df), df.head(3).T))
            return df

    @ns.doc('mlimit_config_type_get', params={'type': 'Measure Configuration Type name'})
    @ns.marshal_with(api_response_response, code=200, description='Success')
    def get(self):
        """ Retry the status of a specific Measure Configuration Type configured.
        """
        logging.logger.debug('request.headers: \n%s', request.headers)
        logging.logger.debug('request.body: \n%s', request.json)
        data = {}
        data['message'] = 'ok'
        data['code'] = 201
        logging.logger.debug(data)
        return data, 201

    @ns.doc('mlimit_config_type_post')
    @ns.expect(m_config_type)
    # @ns.header('X-Header', 'Some class header')
    @ns.marshal_with(api_response_response, code=201, description='Success')
    @ns.marshal_with(api_response_response, code=400, description='Error Message')
    def post(self, **kwargs):
        """ Enable or Disable a Measure Configuration Type.
        """
        data = {}
        df = None
        try:
            utils_api.check_request(request)
            orgid = request.headers.get('X-Fields')
            logging.logger.debug('orgid %s', orgid)
            sfdc_auth = Session.query(SfdcAuth).filter(
                SfdcAuth.active == True).filter(
                SfdcAuth.orgid == orgid).all()
            if sfdc_auth is None and len(sfdc_auth) == 0:
                raise Exception('Missing SfdcAuth')
            auth = request.headers['Authorization'].split()
            username, password = base64.b64decode(
                auth[1]).decode('utf8').split(':')
            if password != config.SECRET_KEY:
                raise Exception('Wrong secret key')
            logging.logger.debug(
                'Access granted username %s', username)
            df = self.__limits_to_df()
            source = 'restapi'
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
            Session.close()
            data['message'] = 'ok'
            data['code'] = 201
            logging.logger.debug(data)
            return data, 201
                    
        except NoSalesforceConnection as no_salesforce_connection:
            traceback.print_exc()
            logging.logger.error(
                no_salesforce_connection)
            e = BadRequest(str(no_salesforce_connection))
            e.data = {'message': e.description, 'code': 1000}
            logging.logger.error(e)
            raise e
        except TypeError as tp:
            traceback.print_exc()
            logging.logger.error(tp)
            e = BadRequest(str(tp))
            e.data = {'message': e.description, 'code': 400}
            logging.logger.error(e)
            raise e
        except exc.SQLAlchemyError as exc_error:
            traceback.print_exc()
            Session.rollback()
            logging.logger.error(exc_error)
            e = BadRequest('Something went wrong on the database')
            e.data = {'message': e.description, 'code': 400}
            logging.logger.error(e)
            raise e
        except binascii.Error as err:
            traceback.print_exc()
            logging.logger.error(err)
            e = BadRequest(str(err))
            e.data = {'message': e.description, 'code': 400}
            logging.logger.error(e)
            raise e
        except Exception as ex:
            logging.logger.error(ex)
            traceback.print_exc()
            e = BadRequest('Something went wrong')
            e.data = {'message': e.description, 'code': 400}
            logging.logger.error(e)
            raise e