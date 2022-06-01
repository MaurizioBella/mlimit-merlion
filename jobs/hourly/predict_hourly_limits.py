# -*- coding: utf-8 -*-
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
''' predict governor limits per hour '''
import traceback
import warnings
from src.config.error import ValueTooSmallError
import src.utils.database as utils_db
from src.prediction import Prediction
from src.config.logger import LoggerClass
logging = LoggerClass.instance()
warnings.filterwarnings("ignore")


def predict_hourly_limits(measure, train_only=False):
    """ Start training and prediction for a hourly frequency
        Parameters
        ----------
        Returns
        -------
        """
    df = utils_db.get_measures_by_limitname(measure)
    if len(df.index) > 0:
        logging.logger.debug('df head(3) \n %s', df.head(3).T)
        logging.logger.debug('df tail(3) \n %s', df.tail(3).T)
        logging.logger.debug('df info() \n %s', df.info())
        df_config = utils_db.get_df_measure_config_by_limitname(measure)
        y_max_value = df_config['max_value']
        try:
            prediction = Prediction(
                measure=measure,
                y_max_value=y_max_value)
            prediction.preprocessing(
                df=df,
                ds='eventdate',
                y='remaining_value',
                freq='H')
            if (train_only == True):
                # Train the model
                prediction.train()
                # Update the Measure Config with some attributes such as sMAPE and RMSE
                utils_db.update_measure_config(prediction)
            else:
                prediction.predict()
                if prediction.forecast is not None:
                    prediction.postprocessing()

                    logging.logger.debug('prediction \n%s',
                                         prediction.df_postprocessing)

                    prediction.detect_anomalies()
                    logging.logger.debug('detect anomaly tail(3) \n %s',
                                         prediction.df_anomalies.loc[prediction.df_anomalies['anomaly'] == True,
                                                                     ['ds', 'anomaly', 'anomaly_weight', 'fact',
                                                                      'yhat_upper']].head(3).T)

                    utils_db.insert_forecaster(
                        measure, df=prediction.df_anomalies)

        except ValueTooSmallError as value_too_small:
            logging.logger.warning(value_too_small)
        except ValueError as value_error:
            traceback.print_exc()
            logging.logger.critical(value_error)
        except FileNotFoundError as file_not_found:
            traceback.print_exc()
            logging.logger.warning(file_not_found)
        except Exception as e:
            traceback.print_exc()
            logging.logger.error(e)
    else:
        logging.logger.debug('Nothing to predict')
