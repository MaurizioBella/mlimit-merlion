# -*- coding: utf-8 -*-
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from merlion.models.ensemble.combine import Mean, ModelSelector
from merlion.models.ensemble.forecast import ForecasterEnsemble, ForecasterEnsembleConfig
from merlion.transform.resample import TemporalResample
from merlion.transform.base import Identity
from merlion.models.forecast.smoother import MSES, MSESConfig
from merlion.models.forecast.prophet import Prophet, ProphetConfig
from merlion.models.forecast.arima import Arima, ArimaConfig
from merlion.models.forecast.trees import LGBMForecaster, LGBMForecasterConfig
from merlion.models.automl.autosarima import AutoSarima, AutoSarimaConfig
import matplotlib.pyplot as plt
import src.utils.os as utils_os
import src.utils.database as utils_db
import matplotlib.pyplot as plt
from merlion.evaluate.forecast import ForecastMetric
import os
import src.config as config
from src.config.logger import LoggerClass
logging = LoggerClass.instance()
warnings.filterwarnings("ignore")


def model_evaluation(test_data):
    """ Util method to return the configuration for each Model
        It uses three different forecasting models:
            ARIMA (a classic stochastic process model)
            Prophet (Facebook's popular time series forecasting model)
            MSES (the Multi-Scale Exponential Smoothing model, developed in-house)
            AutoSARIMA (Train a full AutoSarima model with approximation)

    Returns:
        ensemble, selector: return the 3 models plus ensemble and the selector
    """
    config_arima = ArimaConfig(max_forecast_steps=len(test_data), order=(20, 1, 5),
                          transform=TemporalResample(granularity="1h"))
    model_arima = Arima(config_arima)
    
    config_prophet = ProphetConfig(max_forecast_steps=None, transform=Identity())
    model_prophet = Prophet(config_prophet)
    
    config_mses = MSESConfig(max_forecast_steps=len(test_data), max_backstep=60,
                         transform=TemporalResample(granularity="1h"))
    model_mses = MSES(config_mses)
    
    config_sarima = AutoSarimaConfig(auto_pqPQ=True, auto_d=True, auto_D=True, auto_seasonality=True,
                            approximation=True, maxiter=5)
    model_sarima = AutoSarima(config_sarima)
    
    # The combiner here will simply take the mean prediction of the ensembles here
    ensemble_config = ForecasterEnsembleConfig(
        combiner=Mean(), models=[model_arima, model_prophet, model_mses, model_sarima])
    
    ensemble = ForecasterEnsemble(config=ensemble_config)
    
    # selects the model with the lowest sMAPE
    selector_config = ForecasterEnsembleConfig(
        combiner=ModelSelector(metric=ForecastMetric.sMAPE))
    selector = ForecasterEnsemble(
        config=selector_config, models=[model_arima, model_prophet, model_mses,model_sarima])
    
    return ensemble, selector


class Selector:
    """ Selector selects the best individual model based on its sMAPE (symmetric Mean Average Precision Error).
    It uses three different forecasting models:
        ARIMA (a classic stochastic process model)
        Prophet (Facebook's popular time series forecasting model)
        MSES (the Multi-Scale Exponential Smoothing model, developed in-house)
    """

    def __set__(self, prediction, val):
        train_data, test_data, measure = val
        lowest_smape_list = []
        # model_arima, model_prophet, model_mses, model_ensemble, model_selector = model_evaluation(test_data)
        ensemble, selector = model_evaluation(test_data)
        
        logging.logger.info("Training %s ..." %
                            (type(ensemble).__name__))
        forecast_e, stderr_e = ensemble.train(train_data)

        logging.logger.info("Training %s ..." %
                            (type(selector).__name__))
        forecast_s, stderr_s = selector.train(train_data)

        # Obtain the time stamps corresponding to the test data
        time_stamps = test_data.time_stamps
        sub_test_data = test_data
        
        # The same options are available for ensembles as well, though the stderr is None
        forecast_e, stderr_e = ensemble.forecast(time_stamps=time_stamps)
        forecast_s, stderr_s = selector.forecast(
            time_stamps=time_stamps, time_series_prev=train_data)

        # Compute the sMAPE of the ensemble's forecast (scale is 0 to 100)
        smape_e = ForecastMetric.sMAPE.value(sub_test_data, forecast_e)
        lowest_smape_list.append({
            'model_name': type(ensemble).__name__,
            'model_path': 'ensemble',
            'sMAPE': smape_e
        })
        logging.logger.info("Model {0} sMAPE is {1:.3f}".format(
            type(ensemble).__name__, smape_e))
        # print(f"Ensemble sMAPE is {smape_e:.3f}")
        path = os.path.join("models", "ensemble", measure)
        utils_os.clean_directory(path)
        ensemble.save(path)
        # Visualize the forecast.
        fig, ax = ensemble.plot_forecast(time_series=test_data,
                                               plot_time_series_prev=True)
        plt.savefig(path+'/graph_training.png')
        utils_os.save_to_s3(path)
        if (config.MERLION_PLOT_SHOW == 'True'):
            plt.show()

        # Compute the sMAPE of the selector's forecast (scale is 0 to 100)
        smape_s = ForecastMetric.sMAPE.value(sub_test_data, forecast_s)
        lowest_smape_list.append({
            'model_name': type(selector).__name__,
            'model_path': 'selector',
            'sMAPE': smape_s
        })
        logging.logger.info("Model {0} sMAPE is {1:.3f}".format(
            type(selector).__name__, smape_s))
        # print(f"Selector sMAPE is {smape_s:.3f}")

        # Save the selector
        path = os.path.join("models", "selector", measure)
        utils_os.clean_directory(path)
        selector.save(path)
        # Visualize the forecast.
        fig, ax = selector.plot_forecast(time_series=test_data,
                                            time_series_prev=train_data,
                                            plot_time_series_prev=True,
                                            plot_forecast_uncertainty=True)
        plt.savefig(path+'/graph_training.png')
        utils_os.save_to_s3(path)
        if (config.MERLION_PLOT_SHOW == 'True'):
            plt.show()


    def __get__(self, prediction, objtype=None):
        ''' identify which model to use for prediction return path and model name
        factory names here https://github.com/salesforce/Merlion/blob/7af892c57401ebd1883febcba2de5d8d5422cb56/merlion/models/factory.py#L1'''
        # objtype = <class 'src.prediction.forecaster_merlion.Prediction'>
        predict = config.MERLION_PREDICT_MODEL
        model_path = None
        model_name = None
        logging.logger.debug(
            'Predict {0} for measure {1}'.format(predict, prediction.measure))
        if (predict == 'ensemble'):
            model_path = 'ensemble'
            model_name = 'ForecasterEnsemble'
        elif (predict == 'selector'):
            model_path = 'selector'
            model_name = 'ForecasterEnsemble'
        else:
            raise Exception("There is no model available")
        logging.logger.debug('Used model name: %s from path: %s',
                             model_name, model_path)
        prediction.model_path = model_path
        prediction.model_name = model_name
        return model_path, model_name
