import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing # SES
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing # 
import statsmodels.graphics.tsaplots as tsa_plots
import statsmodels.tsa.statespace as tm_models
from datetime import datetime,time
#from sm.tsa.statespace import sa
Airlines = pd.read_excel("C:/Training/Analytics/Forecasting/Airlines/Airlines+Data.xlsx")

#Airlines.rename(columns={"Passengers ('000)":"Passengers"},inplace=True)   
Airlines.rename(columns={"Passengers ": 'Passengers'}, inplace=True)
# Converting the normal index of Airlines to time stamp 
Airlines.index = pd.to_datetime(Airlines.Month,format="%b-%y")

Airlines.Passengers.plot() # time series plot 
# Creating a Date column to store the actual Date format for the given Month column
Airlines["Date"] = pd.to_datetime(Airlines.Month,format="%b-%y")

# Extracting Day, weekday name, month name, year from the Date column using 
# Date functions from pandas 

Airlines["month"] = Airlines.Date.dt.strftime("%b") # month extraction
#Airlines["Day"] = Airlines.Date.dt.strftime("%d") # Day extraction
#Airlines["wkday"] = Airlines.Date.dt.strftime("%A") # weekday extraction
Airlines["year"] = Airlines.Date.dt.strftime("%Y") # year extraction

# Some EDA on Time series data 
# Heat map visualization 
heatmap_y_month = pd.pivot_table(data=Airlines,values="Passengers",index="year",columns="month",aggfunc="mean",fill_value=0)
sns.heatmap(heatmap_y_month,annot=True,fmt="g")

# Boxplot for ever
sns.boxplot(x="month",y="Passengers",data=Airlines)
sns.boxplot(x="year",y="Passengers",data=Airlines)
# sns.factorplot("month","Passengers",data=Airlines,kind="box")

# Line plot for Passengers based on year  and for each month
sns.lineplot(x="year",y="Passengers",hue="month",data=Airlines)


# moving average for the time series to understand better about the trend character in Airlines
Airlines.Passengers.plot(label="org")
for i in range(2,24,6):
    Airlines["Passengers"].rolling(i).mean().plot(label=str(i))
plt.legend(loc=3)
    
# Time series decomposition plot 
decompose_ts_add = seasonal_decompose(Airlines.Passengers,model="additive")
decompose_ts_add.plot()
decompose_ts_mul = seasonal_decompose(Airlines.Passengers,model="multiplicative")
decompose_ts_mul.plot()

# ACF plots and PACF plots on Original data sets 
tsa_plots.plot_acf(Airlines.Passengers,lags=10)
tsa_plots.plot_pacf(Airlines.Passengers)

# Airlines.index.freq = "MS" 
# splitting the data into Train and Test data and considering the last 12 months data as 
# Test data and left over data as train data 

Train = Airlines.head(79)
Test = Airlines.tail(17)
# to change the index value in pandas data frame 
# Test.set_index(np.arange(1,13),inplace=True)

# Creating a function to calculate the MAPE value for test data 
def MAPE(pred,org):
    temp = np.abs((pred-org))*100/org
    return np.mean(temp)

#MAPE Mean Absolute Percentage Error (Lower is better)
    #Subtract MAPE from 100 to provide accuracy in terms of percentage
    
# Simple Exponential Method
ses_model = SimpleExpSmoothing(Train["Passengers"]).fit()
pred_ses = ses_model.predict(start = Test.index[0],end = Test.index[-1])
se_MAPE = MAPE(pred_ses,Test.Passengers) # 7.846321

# Holt method 
hw_model = Holt(Train["Passengers"]).fit()
pred_hw = hw_model.predict(start = Test.index[0],end = Test.index[-1])
hw_MAPE = MAPE(pred_hw,Test.Passengers) # 7.261176729658341



# Holts winter exponential smoothing with additive seasonality and additive trend
hwe_model_add_add = ExponentialSmoothing(Train["Passengers"],seasonal="add",trend="add",seasonal_periods=12,damped=True).fit()
pred_hwe_add_add = hwe_model_add_add.predict(start = Test.index[0],end = Test.index[-1])
hwe_MAPE = MAPE(pred_hwe_add_add,Test.Passengers) # 4.500954



# Holts winter exponential smoothing with multiplicative seasonality and additive trend
hwe_model_mul_add = ExponentialSmoothing(Train["Passengers"],seasonal="mul",trend="add",seasonal_periods=12).fit()
pred_hwe_mul_add = hwe_model_mul_add.predict(start = Test.index[0],end = Test.index[-1])
hwe_ex_MAPE = MAPE(pred_hwe_mul_add,Test.Passengers) # 4.109309


# Lets us use auto_arima from p


import pmdarima as pm
#from pmdarima.model_selection import train_test_split
import matplotlib.pyplot as plt

## Load/split your data
#y = pm.datasets.load_wineind()
#train, test = train_test_split(y, train_size=79)

# Fit your model
model = pm.auto_arima(Train['Passengers'], seasonal=True, m=12)

model.summary()

# make your forecasts
pred_arima = model.predict(Test.shape[0])  # predict N steps into the future

## Visualize the forecasts (blue=train, green=forecasts)
#x = np.arange(Airlines.shape[0])
#plt.plot(x[:150], Train['Passengers'], c='blue')
#plt.plot(x[150:], pred_arima, c='green')
#plt.show()

arima_MAPE = MAPE(pred_arima,Test.Passengers)  # 5.43


# Visualization of Forecasted values for Test data set using different methods 
plt.plot(Train.index, Train["Passengers"], label='Train',color="black")
plt.plot(Test.index, Test["Passengers"], label='Test',color="blue")
plt.plot(pred_ses.index, pred_ses, label='SimpleExponential',color="green")
plt.plot(pred_hw.index, pred_hw, label='Holts_winter',color="red")
plt.plot(pred_hwe_add_add.index,pred_hwe_add_add,label="HoltsWinterExponential_1",color="brown")
plt.plot(pred_hwe_mul_add.index,pred_hwe_mul_add,label="HoltsWinterExponential_2",color="yellow")
plt.plot(pred_hwe_mul_add.index,pred_hwe_mul_add,label="Auto_Arima",color="grey")
plt.plot(pred_hwe_mul_add.index,srma_pred,label="Auto_Sarima",color="purple")
plt.legend(loc='best')

# Models and their MAPE values
model_mapes = pd.DataFrame(columns=["model_name","mape"])
model_mapes["model_name"] = [""]

data = {"MODEL":pd.Series(["Simple Exponential Method","Holt method","Holts winter exponential smoothing_additive","Holts winter exponential smoothing_multiplicative","ARIMA"]),"MAPE_Values":pd.Series([se_MAPE,hw_MAPE,hwe_MAPE,hwe_ex_MAPE,arima_MAPE])}
table_MAPE=pd.DataFrame(data)
table_MAPE

# Visualizing the ACF and PACF plots for errors 