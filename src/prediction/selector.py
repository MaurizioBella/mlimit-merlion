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


def model_configs():
    """ Util method to return the configuration for each Model
        It uses three different forecasting models:
            ARIMA (a classic stochastic process model)
            Prophet (Facebook's popular time series forecasting model)
            MSES (the Multi-Scale Exponential Smoothing model, developed in-house)

    Returns:
        model_arima, model_prophet, model_mses, model_ensemble, model_selector: return the 3 models plus ensemble and the selector
    """
    models = []
    config1 = ArimaConfig(max_forecast_steps=int(config.MERLION_MAX_FORECAST_STEPS), order=(20, 1, 5),
                          transform=TemporalResample(granularity="1h"))
    model_arima = Arima(config1)
    models.append(model_arima)
    config2 = ProphetConfig(max_forecast_steps=None, transform=Identity())
    model_prophet = Prophet(config2)
    models.append(model_prophet)
    config3 = MSESConfig(max_forecast_steps=int(config.MERLION_MAX_FORECAST_STEPS), max_backstep=60,
                         transform=TemporalResample(granularity="1h"))
    model_mses = MSES(config3)
    models.append(model_mses)
    ensemble_config = ForecasterEnsembleConfig(
        combiner=Mean(), models=[model_arima, model_prophet, model_mses])
    model_ensemble = ForecasterEnsemble(config=ensemble_config)
    models.append(model_ensemble)
    selector_config = ForecasterEnsembleConfig(
        combiner=ModelSelector(metric=ForecastMetric.sMAPE))
    model_selector = ForecasterEnsemble(
        config=selector_config, models=[model_arima, model_prophet, model_mses])
    models.append(model_selector)
    return model_arima, model_prophet, model_mses, model_ensemble, model_selector


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
        model_arima, model_prophet, model_mses, model_ensemble, model_selector = model_configs()
        logging.logger.info("Training %s ..." % (type(model_arima).__name__))
        forecast1, stderr1 = model_arima.train(train_data)

        logging.logger.info("Training %s ..." % (type(model_prophet).__name__))
        forecast2, stderr2 = model_prophet.train(train_data)

        logging.logger.info("Training %s ..." % (type(model_mses).__name__))
        forecast3, stderr3 = model_mses.train(train_data)

        logging.logger.info("Training %s ..." %
                            (type(model_ensemble).__name__))
        forecast_e, stderr_e = model_ensemble.train(train_data)

        logging.logger.info("Training %s ..." %
                            (type(model_selector).__name__))
        forecast_s, stderr_s = model_selector.train(train_data)

        # Truncate the test data to ensure that we are within each model's maximum
        # forecast horizon.
        sub_test_data = test_data[:50]

        # Obtain the time stamps corresponding to the test data
        time_stamps = sub_test_data.univariates[sub_test_data.names[0]].time_stamps

        # Get the forecast & standard error of each model. These are both
        # merlion.utils.TimeSeries objects. Note that the standard error is None for
        # models which don't support uncertainty estimation (like MSES and all
        # ensembles).
        forecast1, stderr1 = model_arima.forecast(time_stamps=time_stamps)
        forecast2, stderr2 = model_prophet.forecast(time_stamps=time_stamps)

        # You may optionally specify a time series prefix as context. If one isn't
        # specified, the prefix is assumed to be the training data. Here, we just make
        # this dependence explicit. More generally, this feature is useful if you want
        # to use a pre-trained model to make predictions on data further in the future
        # from the last time it was trained.
        forecast3, stderr3 = model_mses.forecast(
            time_stamps=time_stamps, time_series_prev=train_data)

        # The same options are available for ensembles as well, though the stderr is None
        forecast_e, stderr_e = model_ensemble.forecast(time_stamps=time_stamps)
        forecast_s, stderr_s = model_selector.forecast(
            time_stamps=time_stamps, time_series_prev=train_data)

        # We begin by computing the sMAPE of ARIMA's forecast (scale is 0 to 100)
        smape1 = ForecastMetric.sMAPE.value(ground_truth=sub_test_data,
                                            predict=forecast1)
        lowest_smape_list.append({
            'model_name': type(model_arima).__name__,
            'model_path': 'model_arima',
            'sMAPE': smape1
        })
        logging.logger.info("Model {0} sMAPE is {1:.3f}".format(
            type(model_arima).__name__, smape1))
        path = os.path.join("models", "model_arima", measure)
        utils_os.clean_directory(path)
        # save() method creates a new directory at the specified path, where it saves a json file representing the model's config, as well as a binary file for the model's state.
        model_arima.save(path)
        # Next, we can visualize the actual forecast, and understand why it
        # attains this particular sMAPE. Since ARIMA supports uncertainty
        # estimation, we plot its error bars too.
        fig, ax = model_arima.plot_forecast(time_series=sub_test_data,
                                            time_series_prev=train_data,
                                            plot_time_series_prev=True,
                                            plot_forecast_uncertainty=True)
        plt.savefig(path+'/graph_training.png')
        utils_os.save_to_s3(path)
        if (config.MERLION_PLOT_SHOW == 'True'):
            plt.show()

        # We begin by computing the sMAPE of Prophet's forecast (scale is 0 to 100)
        smape2 = ForecastMetric.sMAPE.value(sub_test_data, forecast2)
        lowest_smape_list.append({
            'model_name': type(model_prophet).__name__,
            'model_path': 'model_prophet',
            'sMAPE': smape2
        })
        logging.logger.info("Model {0} sMAPE is {1:.3f}".format(
            type(model_prophet).__name__, smape2))
        # print(f"{type(model_prophet).__name__} sMAPE is {smape2:.3f}")

        path = os.path.join("models", "model_prophet", measure)
        utils_os.clean_directory(path)
        model_prophet.save(path)
        # Next, we can visualize the actual forecast, and understand why it
        # attains this particular sMAPE. Since Prophet supports uncertainty
        # estimation, we plot its error bars too.
        # Note that we can specify time_series_prev here as well, though it
        # will not be visualized unless we also supply the keyword argument
        # plot_time_series_prev=True.

        fig, ax = model_prophet.plot_forecast(time_series=sub_test_data,
                                              time_series_prev=train_data,
                                              plot_time_series_prev=True,
                                              plot_forecast_uncertainty=True)
        plt.savefig(path+'/graph_training.png')
        utils_os.save_to_s3(path)
        if (config.MERLION_PLOT_SHOW == 'True'):
            plt.show()

        # We begin by computing the sMAPE of MSES's forecast (scale is 0 to 100)
        smape3 = ForecastMetric.sMAPE.value(sub_test_data, forecast3)
        lowest_smape_list.append({
            'model_name': type(model_mses).__name__,
            'model_path': 'model_mses',
            'sMAPE': smape3
        })
        logging.logger.info("Model {0} sMAPE is {1:.3f}".format(
            type(model_mses).__name__, smape3))
        # print(f"{type(model_mses).__name__} sMAPE is {smape3:.3f}")

        path = os.path.join("models", "model_mses", measure)
        utils_os.clean_directory(path)
        model_mses.save(path)
        # Next, we visualize the actual forecast, and understand why it
        # attains this particular sMAPE.

        fig, ax = model_mses.plot_forecast(time_series=sub_test_data,
                                           plot_forecast_uncertainty=True)
        plt.savefig(path+'/graph_training.png')
        utils_os.save_to_s3(path)
        if (config.MERLION_PLOT_SHOW == 'True'):
            plt.show()

        # Compute the sMAPE of the ensemble's forecast (scale is 0 to 100)
        smape_e = ForecastMetric.sMAPE.value(sub_test_data, forecast_e)
        lowest_smape_list.append({
            'model_name': type(model_ensemble).__name__,
            'model_path': 'model_ensemble',
            'sMAPE': smape_e
        })
        logging.logger.info("Model {0} sMAPE is {1:.3f}".format(
            type(model_ensemble).__name__, smape_e))
        # print(f"Ensemble sMAPE is {smape_e:.3f}")
        path = os.path.join("models", "model_ensemble", measure)
        utils_os.clean_directory(path)
        model_ensemble.save(path)
        # Visualize the forecast.
        fig, ax = model_ensemble.plot_forecast(time_series=sub_test_data,
                                               plot_forecast_uncertainty=True)
        plt.savefig(path+'/graph_training.png')
        utils_os.save_to_s3(path)
        if (config.MERLION_PLOT_SHOW == 'True'):
            plt.show()

        # Compute the sMAPE of the selector's forecast (scale is 0 to 100)
        smape_s = ForecastMetric.sMAPE.value(sub_test_data, forecast_s)
        lowest_smape_list.append({
            'model_name': type(model_selector).__name__,
            'model_path': 'model_selector',
            'sMAPE': smape_s
        })
        logging.logger.info("Model {0} sMAPE is {1:.3f}".format(
            type(model_selector).__name__, smape_s))
        # print(f"Selector sMAPE is {smape_s:.3f}")

        # Save the selector
        path = os.path.join("models", "model_selector", measure)
        utils_os.clean_directory(path)
        model_selector.save(path)
        # Visualize the forecast.
        fig, ax = model_selector.plot_forecast(time_series=sub_test_data,
                                               plot_time_series_prev=True,
                                               plot_forecast_uncertainty=True)
        plt.savefig(path+'/graph_training.png')
        utils_os.save_to_s3(path)
        if (config.MERLION_PLOT_SHOW == 'True'):
            plt.show()
        lowest_smape = sorted(lowest_smape_list, key=lambda d: d['sMAPE'])
        prediction.model_path = lowest_smape[0].get('model_path')
        prediction.model_name = lowest_smape[0].get('model_name')
        # prediction.lowest_smape = lowest_smape[0]
        # return lowest_smape

    def __get__(self, prediction, objtype=None):
        ''' identify which model to use for prediction return path and model name
        factory names here https://github.com/salesforce/Merlion/blob/7af892c57401ebd1883febcba2de5d8d5422cb56/merlion/models/factory.py#L1'''
        # objtype = <class 'src.prediction.forecaster_merlion.Prediction'>
        predict = int(config.MERLION_PREDICT_MODEL)
        model_path = None
        model_name = None
        logging.logger.debug(
            'Predict {0} for measure {1}'.format(predict, prediction.measure))
        if (predict == 0):
            measure_config = utils_db.get_measure_config_by_limitname(
                prediction)
            print(measure_config)
            model_path = measure_config[0].model_path
            model_name = measure_config[0].model_name
            logging.logger.debug(
                "Loaded from MeasureConfig model name {0} with path {1}".format(model_name, model_path))
        elif (predict == 1):
            model_path = 'model_arima'
            model_name = 'Arima'
        elif (predict == 2):
            model_path = 'model_prophet'
            model_name = 'Prophet'
        elif (predict == 3):
            model_path = 'model_mses'
            model_name = 'MSES'
        elif (predict == 4):
            model_path = 'model_ensemble'
            model_name = 'ForecasterEnsemble'
        elif (predict == 5):
            model_path = 'model_selector'
            model_name = 'ForecasterEnsemble'
        else:
            raise Exception("There is no model available")
        logging.logger.debug('Used model name: %s from path: %s',
                             model_name, model_path)
        prediction.model_path = model_path
        prediction.model_name = model_name
        # return model_path, model_name
