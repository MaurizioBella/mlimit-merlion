# -*- coding: utf-8 -*-
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""class in charge of forecasting"""
from merlion.evaluate.forecast import ForecastEvaluator, ForecastEvaluatorConfig, ForecastMetric
from merlion.models.ensemble.forecast import ForecasterEnsemble
from src.config.logger import LoggerClass
from src.config.error import ValueTooSmallError
import src.utils.dataprep as dataprep
import src.utils.os as utils_os
from .selector import Selector
import src.config as config
from psutil import virtual_memory
import psutil
import time
from dotenv import load_dotenv
# from merlion.models.factory import ModelFactory
import os
import warnings
import pandas as pd
load_dotenv()
logging = LoggerClass.instance()
warnings.filterwarnings("ignore")
ROUND_DIGIT = 2
mem = virtual_memory()
process = psutil.Process(os.getpid())


def create_evaluator(model):
    # Re-initialize the model, so we can re-train it from scratch
    model.reset()

    # Create an evaluation pipeline for the model, where we
    # -- get the model's forecast every hour
    # -- have the model forecast for a horizon of 6 hours
    # -- re-train the model every 12 hours
    # -- when we re-train the model, retrain it on only the past 2 weeks of data
    evaluator = ForecastEvaluator(
        model=model, config=ForecastEvaluatorConfig(
            cadence="1h", horizon="6h", retrain_freq="12h")
    )
    return evaluator


class Prediction:
    """Prophet forecaster.
    Parameters
    ----------
    df: Pandas DataFrame
    measure: name of the time-dependent variable
    y_max_value: governor limits usage (e.g. 15,000 batches rolling 24hs)
    """
    selector = Selector()

    def __init__(
            self,
            measure,
            y_max_value):

        self.__start_time = time.time()
        self.__test_data = None
        self.__train_data = None
        self.__df = None  # Pandas Dataframes
        self.__df_data = None
        self.__process_memory = None  # memory info
        self.__y_max_value = y_max_value
        self.__smape = None
        self.__rmse = None
        self.__last_train_time = None
        self.__measure = measure
        self.__path_name = None
        self.__model_name = None
        self.__forecast = None
        self.__df_postprocessing = None
        self.__df_anomalies = None

        logging.logger.info(
            "Init Prediction class measure: %s", measure)

    def _start_time(self):
        process_memory = process.memory_info().rss
        logging.logger.info("Memory used in GB before train: %0.2f" %
                            float(process_memory/(10**9)))  # Check memory usage before loading the model
        time_local = time.localtime()
        current_time = time.strftime("%H:%M:%S", time_local)
        self.start_time = time.time()
        self.process_memory = process_memory
        logging.logger.info('Training process starts at %s', current_time)

    def _end_time(self):
        end_time = round(time.time() - self.start_time, 2)
        logging.logger.info('End jobs process took %s secs', end_time)
        post = process.memory_info().rss
        ttl = mem.total  # Total memory available
        logging.logger.debug("Percentage increase in memory usage: {:.2f}% -\
             total memory available {:.2f}".format(
            float((post/self.process_memory)*100), ttl))  # Percentage increase in memory after loading the model
        logging.logger.debug("Memory used in GB after training the model: {:.2f}".format(
            float(post/(10**9))))  # Calculate the memory used after loading the model
    
    # forecast_ub
    @property
    def forecast_ub(self):
        return self.__forecast_ub

    @forecast_ub.setter
    def forecast_ub(self, forecast_ub):
        self.__forecast_ub = forecast_ub

    # forecast
    @property
    def forecast(self):
        return self.__forecast

    @forecast.setter
    def forecast(self, forecast):
        self.__forecast = forecast

    # forecast_lb
    @property
    def forecast_lb(self):
        return self.__forecast_lb

    @forecast_lb.setter
    def forecast_lb(self, forecast_lb):
        self.__forecast_lb = forecast_lb

    # forecast
    @property
    def forecast(self):
        return self.__forecast

    @forecast.setter
    def forecast(self, forecast):
        self.__forecast = forecast

    # start_time
    @property
    def start_time(self):
        return self.__start_time

    @start_time.setter
    def start_time(self, start_time):
        self.__start_time = start_time

    # start_time
    @property
    def start_time(self):
        return self.__start_time

    @start_time.setter
    def start_time(self, start_time):
        self.__start_time = start_time

    # start_time
    @property
    def start_time(self):
        return self.__start_time

    @start_time.setter
    def start_time(self, start_time):
        self.__start_time = start_time

    # df
    @property
    def df(self):
        return self.__df

    @df.setter
    def df(self, df):
        self.__df = df

    # train_data
    @property
    def train_data(self):
        return self.__train_data

    @train_data.setter
    def train_data(self, train_data):
        self.__train_data = train_data

    # test_data

    @property
    def test_data(self):
        return self.__test_data

    @test_data.setter
    def test_data(self, test_data):
        self.__test_data = test_data

    # process_memory
    @property
    def process_memory(self):
        return self.__process_memory

    @process_memory.setter
    def process_memory(self, process_memory):
        self.__process_memory = process_memory

    # y_max_value
    @property
    def y_max_value(self):
        return self.__y_max_value

    @y_max_value.setter
    def y_max_value(self, y_max_value):
        self.__y_max_value = y_max_value

    # smape
    @property
    def smape(self):
        return self.__smape

    @smape.setter
    def smape(self, smape):
        self.__smape = smape

    # rmse
    @property
    def rmse(self):
        return self.__rmse

    @rmse.setter
    def rmse(self, rmse):
        self.__rmse = rmse

    # last_train_time
    @property
    def last_train_time(self):
        return self.__last_train_time

    @last_train_time.setter
    def last_train_time(self, last_train_time):
        self.__last_train_time = last_train_time

    # measure
    @property
    def measure(self):
        return self.__measure

    @measure.setter
    def measure(self, measure):
        self.__measure = measure

    # model_path
    @property
    def model_path(self):
        return self.__path_name

    @model_path.setter
    def model_path(self, model_path):
        self.__path_name = model_path

    # model_name
    @property
    def model_name(self):
        return self.__model_name

    @model_name.setter
    def model_name(self, model_name):
        self.__model_name = model_name

    def preprocessing(self, df, ds, y, freq):
        if len(df) < int(config.MERLION_LIMIT_EVENTS):
            raise ValueTooSmallError()
        val = (df, ds, y, freq)
        self.df_preprocessed = val

    @property
    def df_preprocessed(self):
        """ getter
        Returns:
            DataFrame: return DataFrame preprocessed ready to be trained with Merlion
        """
        return self.__df_data

    # df_anomalies
    @property
    def df_anomalies(self):
        return self.__df_anomalies

    @df_anomalies.setter
    def df_anomalies(self, df_anomalies):
        self.__df_anomalies = df_anomalies

    # df_postprocessing
    @property
    def df_postprocessing(self):
        return self.__df_postprocessing

    @df_postprocessing.setter
    def df_postprocessing(self, df_postprocessing):
        self.__df_postprocessing = df_postprocessing

    def postprocessing(self):
        self.df_postprocessing = dataprep.postprocessing(self)

    @df_preprocessed.setter
    def df_preprocessed(
            self, val):
        """ data preparation of the DataFrame
        - fillna for NaN data
        - rename columns to ds and y
        - define frequency
        Parameters
        ----------
        ds: column name of DataFrame containing timestamps
        y: column name of DataFrame with features.
        freq: Any valid frequency for pd.date_range, such as 'D' or 'M'.

        Returns
        -------
        pd.DataFrame
        """
        df, ds, y, freq = val
        df, df_data, train_data, test_data = dataprep.preprocessing(
            df, ds, y, freq)
        self.train_data = train_data
        self.test_data = test_data
        self.df = df
        self.__df_data = df_data

    def train(self):
        """ Training the model
        Parameters
        -----------
        holiday: Will check on public.prophetholiday if True

        Returns
        -------
        self
        """
        logging.logger.debug('Training the model with %s records',
                             len(self.df))
        if self.df is not None:
            logging.logger.debug('df head(3) \n %s', self.df.head(3).T)
            logging.logger.debug('df tail(3) \n %s', self.df.tail(3).T)
            logging.logger.debug('df info() \n %s', self.df.info())
            self._start_time()
            
            val = self.train_data, self.test_data, self.measure
            self.selector = val

            model_path, model_name = self.selector
            path = os.path.join("models", model_path, self.measure)
            selector_factory_loaded = ForecasterEnsemble.load(
                dirname=path)
            # selector_factory_loaded = ModelFactory.load(
            #     name=model_name, model_path=path)
            selector_evaluator = create_evaluator(selector_factory_loaded)
            self.last_train_time = selector_factory_loaded._last_train_time
            selector_train_result, selector_test_result = selector_evaluator.get_predict(
                train_vals=self.train_data, test_vals=self.test_data)
            # Evaluate the selector's sMAPE and RMSE
            self.smape = selector_evaluator.evaluate(
                ground_truth=self.test_data,
                predict=selector_test_result,
                metric=ForecastMetric.sMAPE)
            self.rmse = selector_evaluator.evaluate(
                ground_truth=self.test_data,
                predict=selector_test_result,
                metric=ForecastMetric.RMSE)
            logging.logger.debug("Evaluation pipeline for model: {0} sMAPE: {1:.3f} RMSE: {2:.3f} last training time {3} path {4}".format(
                model_name, self.smape, self.rmse, self.last_train_time, model_path))

    def predict(self):
        """ output of the Prophet algorithm after it has been trained and applied to new
        data when forecasting the likelihood of a particular outcome
        Parameters
        ----------
        None

        Returns
        -------
        DataFrame: forecasting the likelihood of a particular outcome
        Mape: mean absolute percentage error
        """
        logging.logger.debug(f'Started measure: {self.measure}')

        result = self.selector
        path = os.path.join("models", self.model_path, self.measure)
        is_dir = os.path.isdir(path)
        if not is_dir:
            utils_os.read_from_s3(path)
        is_dir = os.path.isdir(path)
        if not is_dir:
            raise Exception('there is no folder %s retrain the model' % (path))
        # Load the selector using the ModelFactory
        logging.logger.debug('load predict model from %s', path)

        selector_factory_loaded = ForecasterEnsemble.load(
                dirname=path)
        # selector_factory_loaded = ModelFactory.load(
        #     name=model_name, model_path=path)
        
        sub_test_data = self.test_data[:len(self.test_data)-1]

        # Obtain the time stamps corresponding to the test data
        # time_stamps = sub_test_data.univariates[sub_test_data.names[0]].time_stamps
        time_stamps = [max(self.df_preprocessed)[
                0] + x * 60 * 60 for x in range(int(config.MERLION_MAX_FORECAST_STEPS))]
        logging.logger.debug('time stamps: %s', time_stamps)
        self.forecast, self.forecast_lb, self.forecast_ub = selector_factory_loaded.forecast(
            time_stamps=time_stamps,
            time_series_prev=self.train_data,
            return_iqr=True,
            return_prev=True
        )

    def detect_anomalies(self):
        """ Detect anomalies
        Parameters
        ----------
        df: Pandas DataFrame

        Returns
        -------
        Pandas.DataFrame with the anomalies
        """
        df = self.df_postprocessing
        df['anomaly'] = (((df['fact'] < df['yhat_lower']) | (
            df['fact'] > df['yhat_upper'])) & pd.notna(df['fact'])).astype(int)
        df['anomaly_weight'] = 0.0
        df['max_limit'] = int(self.y_max_value)
        df.loc[df['fact'] >
               df['yhat_upper'], 'anomaly_weight'] = (df['fact'] - df['yhat_upper'])/(df['max_limit'])
        df.loc[df['fact'] <
               df['yhat_lower'], 'anomaly_weight'] = (df['yhat_lower'] - df['fact'])/(df['max_limit'])
        self.df_anomalies = df
