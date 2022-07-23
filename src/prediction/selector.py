# -*- coding: utf-8 -*-
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from merlion.models.ensemble.combine import Mean, ModelSelector
from merlion.models.ensemble.forecast import ForecasterEnsemble, ForecasterEnsembleConfig
from merlion.transform.base import Identity
from merlion.transform.resample import TemporalResample
from merlion.models.forecast.smoother import MSES, MSESConfig
from merlion.models.forecast.prophet import Prophet, ProphetConfig
from merlion.models.forecast.arima import Arima, ArimaConfig
from merlion.models.forecast.ets import ETS, ETSConfig
from merlion.models.forecast.trees import LGBMForecaster, LGBMForecasterConfig
from merlion.models.automl.autosarima import AutoSarima, AutoSarimaConfig
from merlion.models.automl.autoets import AutoETS, AutoETSConfig
from merlion.models.automl.autoprophet import AutoProphet, AutoProphetConfig
import matplotlib.pyplot as plt
import src.utils.os as utils_os
import matplotlib.pyplot as plt
from merlion.evaluate.forecast import ForecastMetric
import os
import src.config as config
from src.config.logger import LoggerClass
logging = LoggerClass.instance()
warnings.filterwarnings("ignore")


def model_evaluation(test_data):
    """ Util method to return the configuration for each Model
        It uses different forecasting models:
            # ARIMA (a classic stochastic process model)
            # Prophet (Facebook's popular time series forecasting model)
            # MSES (the Multi-Scale Exponential Smoothing model, developed in-house)
            # AutoSARIMA (Train a full AutoSarima model with approximation)
            AutoProphet (Automatic (multi)-seasonality detection for Facebook's Prophet)
            AutoETP (Automatic seasonality detection for ETS.)

    Returns:
        ensemble, selector: return the 3 models plus ensemble and the selector
    """
    models = []
    # config_arima = ArimaConfig(max_forecast_steps=len(test_data), order=(20, 1, 5),
    #                       transform=TemporalResample(granularity="1h"))
    # model_arima = Arima(config_arima)
    # models.append(model_arima)
    
    # config_prophet = ProphetConfig(max_forecast_steps=None, transform=Identity())
    # model_prophet = Prophet(config_prophet)
    # models.append(model_prophet)
    
    # config_mses = MSESConfig(max_forecast_steps=len(test_data), max_backstep=60,
    #                      transform=TemporalResample(granularity="1h"))
    # model_mses = MSES(config_mses)
    # models.append(model_mses)
    
    # config_sarima = AutoSarimaConfig(auto_pqPQ=True, auto_d=True, auto_D=True, auto_seasonality=True,
    #                         approximation=True, maxiter=5)
    # model_sarima = AutoSarima(config_sarima)
    # models.append(model_sarima)
    
    # AutoProphet (Automatic (multi)-seasonality detection for Facebook's Prophet)
    config_autoprophet = AutoProphetConfig(target_seq_index=0, transform=Identity(), periodicity_strategy="Max")
    model_autoprophet = AutoProphet(config=config_autoprophet)
    models.append(model_autoprophet)
    
    # AutoETP (Automatic seasonality detection for ETS.)
    # config_autoets = AutoETSConfig(pval=0.1, error="add", trend="add", seasonal="add", damped_trend=True)
    # model_autoets = AutoETS(config=config_autoets)    
    # models.append(model_autoets)
    
    # The combiner here will simply take the mean prediction of the ensembles here
    ensemble_config = ForecasterEnsembleConfig(
        combiner=Mean(), models=models)
    
    ensemble = ForecasterEnsemble(config=ensemble_config)
    
    # selects the model with the lowest sMAPE
    selector_config = ForecasterEnsembleConfig(
        combiner=ModelSelector(metric=ForecastMetric.sMAPE))
    selector = ForecasterEnsemble(
        config=selector_config, models=models)
    
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
        
        # Obtain the time stamps corresponding to the test data
        sub_test_data = test_data[:len(test_data)-1]
        # sub_test_data = test_data[:-1]
        # Obtain the time stamps corresponding to the test data
        time_stamps = sub_test_data.univariates[sub_test_data.names[0]].time_stamps
        
        ensemble, selector = model_evaluation(test_data)
        
        logging.logger.info("Training %s ..." %
                            (type(ensemble).__name__))
        forecast_e, stderr_e = ensemble.train(train_data)

        logging.logger.info("Training %s ..." %
                            (type(selector).__name__))
        forecast_s, stderr_s = selector.train(train_data)

        
        
        # The same options are available for ensembles as well, though the stderr is None
        forecast_e, stderr_e = ensemble.forecast(time_stamps=time_stamps)
        forecast_s, stderr_s = selector.forecast(
            time_stamps=time_stamps, time_series_prev=train_data)

        # Compute the sMAPE of the ensemble's forecast (scale is 0 to 100)
        smape_e = ForecastMetric.sMAPE.value(sub_test_data, forecast_e)        
        logging.logger.info("Model {0} sMAPE is {1:.3f}".format(
            type(ensemble).__name__, smape_e))
        
        path = os.path.join("models", "ensemble", measure)
        utils_os.clean_directory(path)
        ensemble.save(path)
        # Visualize the forecast.
        fig, ax = ensemble.plot_forecast(time_series=sub_test_data,
                                               plot_time_series_prev=True)
        plt.savefig(path+'/graph_training.png')
        utils_os.save_to_s3(path)
        if (config.MERLION_PLOT_SHOW == 'True'):
            plt.show()

        # Compute the sMAPE of the selector's forecast (scale is 0 to 100)
        smape_s = ForecastMetric.sMAPE.value(sub_test_data, forecast_s)
        logging.logger.info("Model {0} sMAPE is {1:.3f}".format(
            type(selector).__name__, smape_s))
        
        # Save the selector
        path = os.path.join("models", "selector", measure)
        utils_os.clean_directory(path)
        selector.save(path)
        # Visualize the forecast.
        fig, ax = selector.plot_forecast(time_series=sub_test_data,
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
