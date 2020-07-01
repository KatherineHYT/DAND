# DAND
The Detecting Anomalies in Non-stationary Data method

The DAND method combines ideas from Munir et al. (2018) and Choi and Lee (2018). The DAND method consists of two modules: a time-series predictor module and an anomaly detector module, as illustrated in Figure below. The time-series predictor module predicts a value or a set of value for the next timestamp, and the anomaly detector module determines the given time series data points as normal or abnormal. The time-series predictor has an ensemble structure to enhance the prediction accuracy (Adhikari, 2015). Because using multiple sequence lengths allows for creating multiple LSTM models in the ensemble, it is likely to enhance the predictions on multiple time scales. As a result, this ensemble method may tackle nonlinear, non-stationary time series data better and mitigate the influence of inappropriate sliding window size. 

![DAND figure](https://user-images.githubusercontent.com/43935090/86256042-22c60200-bbb8-11ea-8ceb-c5c5787913cb.jpg)

Reference
Munir, M., Siddiqui, S.A., Dengel, A., and Ahmed, S. (2018). DeepAnt: A deep learning approach for unsupervised anomaly detection in time series. IEEE Access, 2018, 7, 1991–2005. Retrieved from https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8581424

Choi J.Y. and Lee B. (2018). Combining LSTM network ensemble via adaptive weighting for improved time series forecasting. Mathematical Problems in Engineering, 2018. Retrieved from https://www.hindawi.com/journals/mpe/2018/2470171/

Ensemble codes modified from https://machinelearningmastery.com/weighted-average-ensemble-for-deep-learning-neural-networks/
