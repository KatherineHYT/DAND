# St Bernard (STB) environmental monitoring systems
Column definitions:

1. Station ID
2. Year
3. Month
4. Day
5. Hour
6. Minute
7. Second
8. Time since the epoch [s]
9. Ambient Temperature [°C]
10. Surface Temperature [°C]
11. Solar Radiation [W/m^2]
12. Relative Humidity [%]
13. Soil Moisture [%]
14. Watermark [kPa]
15. Rain Meter [mm]
16. Wind Speed [m/s]
17. Wind Direction [°]

Source: EPFL repository

*DAND.py: the code for DAND method\
*LSTM_or_ensemble.py: the code for vanilla LSTM or ensemble with fixed size of sliding window. The model depends on the parameter "n_members".
