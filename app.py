import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import plotly
import plotly.graph_objects as go
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from pmdarima.arima import auto_arima
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly

st.title('$\curlyeqsucc$ Time Series Analysis $\curlyeqprec$')
st.markdown('This is a web app to analyze :blue[Chicago Taxi Trip] time series data')

# Load data
data = pd.read_csv('chicag daily.csv')

# Renaming the column name
data.rename(columns={'f0_':'Trips'},inplace=True)
data.rename(columns={'daily':'Date'},inplace=True)

data.Date = pd.to_datetime(data.Date, errors = 'coerce')

tab1, tab2, tab3 = st.tabs(["📈 Chart", "🗃 Data", "📅Yearly"])

with tab1:
   tab1.subheader("Raw data plot")
   fig, ax = plt.subplots(figsize=(12, 6))
   ax.plot(data['Date'], data['Trips'])
   ax.set_xlabel('Date')
   ax.set_ylabel('Number of trips')
   ax.set_title('Chicago Taxi Trips')
   ax.grid(True)
   st.pyplot(fig)

with tab2:
   tab2.subheader("Top 5 rows of the data")
   tab2.write(data.head())


# Data preprocessing
yearly = pd.pivot_table(data,
                         values='Date',
                         index = pd.Grouper(freq = 'A',key='Date'),aggfunc='sum')

# freq = 'A' - Annual, 'M' - Monthly, 'Q' - Quarterly, 'D' - Daily
# aggfunc = "mean", "sum", "median", "sd"

# st.write(yearly.head())

# Plotting yearly data
with tab3:
   tab3.subheader('Yearly data plot')
   fig, ax = plt.subplots(figsize=(12, 6))
   ax.plot(yearly,marker='o',linestyle='--',color='black',linewidth=1,markersize=7,markerfacecolor='red',markeredgecolor='black',markeredgewidth=2,alpha=0.7, label='Yearly',zorder=2,clip_on=True,clip_box=None,fillstyle='full',markevery=None)
   ax.set_xlabel('Year')
   ax.set_ylabel('Number of trips')
   ax.set_title('Chicago Taxi Trips')
   ax.grid(True)
   st.pyplot(fig)

yearly = yearly.reset_index()

# Quarterly data
yearly_quarter = pd.pivot_table(data,
                         values='Date',
                         index = pd.Grouper(freq = 'Q',key='Date'),aggfunc='sum')

yearly_quarter = yearly_quarter.reset_index()

data.index = data.Date
data = data.drop('Date',axis = 1)

# Diff()

data_diff = data.diff()
data_diff = data_diff.dropna()
st.divider()
# Plotting diff data
col3,col4 = st.columns(2)
# adfuller test
with col3:
    st.subheader('Augmented Dickey-Fuller test')
    result = adfuller(data_diff.Trips)
    st.write('ADF Statistic: %f' % result[0])
    st.write('p-value: %f' % result[1])
    st.write('Critical Values:')
    for key, value in result[4].items():
     st.write('\t%s: %.3f' % (key, value))

# kpss test
with col4:
    st.subheader('Kwiatkowski-Phillips-Schmidt-Shin test')
    result = kpss(data_diff.Trips)
    st.write('ADF Statistic: %f' % result[0])
    st.write('p-value: %f' % result[1])
    st.write('Critical Values:')
    for key, value in result[3].items():
        st.write('\t%s: %.3f' % (key, value))


with st.expander("See explanation"):
   st.write("""
:blue[Augmented Dickey Fuller Test of Stationarity] - Identifies the whether the stationary data is Stationary or Not.
Null - Unit Root Present or Data is not Stationary.
Alt  - No Unit Root or Data is Stationary.

A unit root process is a data-generating process whose first difference is stationary. In other words, 
a unit root process yt has the form.
yt = yt-1 + stationary process.

:red[Interpretation is baesd on p-value]
p-value less than 0.05, Reject Null
p-value greater than 0.05, Fail to Reject Null

:blue[Kwiatkowaski-Phillips-Schmidt-Schmidt-Shin(KPSS)] test for stationaryity.
null hypothesis that x is level or trend stationary.
Alt hypothessis that x is not level or not trend stationary.""")


tab4,tab5 = st.tabs(["ACF & PACF","Decomposition"])

# Plotting ACF and PACF
with tab4:
   st.subheader('ACF and PACF plots')
   fig, ax = plt.subplots(2,1,figsize=(12, 6))
   plot_acf(data_diff, ax=ax[0])
   plot_pacf(data_diff, ax=ax[1])
   st.pyplot(fig)

# Decomposition
with tab5:
   st.subheader('Decomposition')
   decomposition = seasonal_decompose(data, model='additive', period=365)
   fig = decomposition.plot()
   st.pyplot(fig)

with st.expander("See explanation"):
   st.write(""" :green[Autocorrelation Function (ACF)] and :orange[Partial Autocorrelation Function (PACF)] plots are graphical tools in time series 
   analysis. :green[ACF] shows the correlation between a time series and its lagged values, while :orange[PACF] reveals the direct relationship while 
   controlling for intermediate lags, helping identify patterns and optimal lag orders for time series modeling.
   :blue[Decomposition] splits a time series into trend, seasonality, and residual components, helping to identify underlying patterns and fluctuations. """)

st.divider()

# Arima model
arima_model = auto_arima(data,max_p=10,max_q=10 , stationary=False)
# st.write(arima_model.summary())

# Predictions
taxipredict = pd.DataFrame(arima_model.predict(n_periods=120))
index_of_fc = pd.date_range(data.index[-1],periods=120,freq="M")
taxipredict = pd.DataFrame(taxipredict)
taxipredict.index = index_of_fc

# Plotting predictions
st.subheader('Predictions')
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(data, label='Original', color='blue')
ax.plot(taxipredict, label='Predicted', color='red')
ax.set_xlabel('Date')
ax.set_ylabel('Number of trips')
ax.set_title('Chicago Taxi Trips')
ax.grid(True)
ax.legend(loc='upper left')
st.pyplot(fig)
st.divider()
# Prophet model
data_prophet = data.reset_index()
data_prophet.columns = ['ds','y']
m = Prophet()
m.fit(data_prophet)
future = m.make_future_dataframe(periods=24,freq='M')
forecast = m.predict(future)
fig1 =  plot_plotly(m,forecast)
st.plotly_chart(fig1)
st.divider()
fig2 = plot_components_plotly(m,forecast)
st.plotly_chart(fig2)