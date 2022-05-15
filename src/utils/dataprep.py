import pandas as pd
import numpy as np
import src.config as config
from src.config.logger import LoggerClass
from merlion.utils import TimeSeries
logging = LoggerClass.instance()


def preprocessing(
        df,
        ds,
        y,
        freq):
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
    if isinstance(df, pd.DataFrame):
        df = df.rename(
            columns={ds: "ds", y: "y"})
        df.drop(df.columns.difference(
            ['ds', 'y']), 1, inplace=True)
        df.dropna(subset=['ds', 'y'])
        df['ds'] = pd.to_datetime(df.ds)

        df.index = pd.DatetimeIndex(df.index).to_period(freq)
        df = df.set_index('ds')
        timedeltas = np.diff(df.index)
        if timedeltas.size != 0:
            logging.logger.warning('Has missing data: %s',
                                any(timedeltas != timedeltas[0]))
        # print(f"Has missing data: {any(timedeltas != timedeltas[0])}")
        df = df.resample(freq).sum()
        column_0 = df['y']
        count_0 = column_0[column_0 == 0].count()
        logging.logger.warning(
            '%s records after resample (y = 0)', count_0)
        # replace missing values as 0 with previous values
        df['y'].replace(to_replace=0, method='ffill', inplace=True)
        df['y'].fillna(method='ffill', inplace=True)
        # df.reset_index(inplace=True)
        df_data = TimeSeries.from_pd(df)
        # self.df_data = df_data
        # self.df = df
        # self.df = df_data
        # select the first 90% of data to train
        n = int(config.MERLION_TRAINING_SAMPLE)
        train = df.head(int(len(df)*(n/100)))
        test = df.tail(int(len(df)*((100-n)/100)))
        train_data = TimeSeries.from_pd(train)
        test_data = TimeSeries.from_pd(test)
        return df, df_data, train_data, test_data

    else:
        logging.logger.error(
            "Trying to load Prediction but the DataFrame was missing")
        raise NameError(
            'Trying to load Prediction but the DataFrame was missing')


def postprocessing(prediction):
    # df_fact = df_fact.rename(columns={
    #     'y': 'fact'})

    df_forecast = prediction.forecast.to_pd()
    df_lb = prediction.forecast_lb.to_pd()
    df_ub = prediction.forecast_ub.to_pd()
    df = pd.concat([df_forecast, df_lb, df_ub],
                   axis=1, join="inner")

    df.reset_index(level=0, inplace=True)
    df = df.rename(columns={
        'time': 'ds',
        'y': 'yhat',
        'y_upper': 'yhat_upper',
        'y_lower': 'yhat_lower'})

    df_prev = prediction.df_preprocessed.to_pd()
    df_prev.reset_index(level=0, inplace=True)
    df_prev = df_prev.rename(columns={
        'time': 'ds',
        'y': 'fact'})
    df_result = df.merge(
        left_on="ds", right_on="ds", how='left', right=df_prev)
    return df_result
