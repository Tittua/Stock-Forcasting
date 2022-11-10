from turtle import title
from unicodedata import name
import pandas as pd
import nsepy 
from datetime import date
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from plotly import graph_objs as go
from PIL import Image
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import altair as alt
import seaborn as sns

#date 
start=date(2005,1,1)
end_date=date.today()

#front end
st.title('Stock Prediction')
st.sidebar.title('Input ')
selected_stock=st.sidebar.selectbox('Select Stock',('SBIN','ASHOK LEYLAND LIMITED','AXIS BANK LIMITED','MARUTI SUZUKI INDIA LIMITED'))


#data extraction
@st.cache   #saving the data
def load_data(selected_stock):
    data=nsepy.get_history(symbol=selected_stock,start=start,end=end_date)
    data.reset_index(inplace=True)
    data.drop(['Trades','Turnover','Deliverable Volume','%Deliverble'],axis=1,inplace=True)
    no_of_data=len(data)
    return data,no_of_data

#Data downloading status
data_load_state=st.text('Loading data...')
data,no_of_data=load_data(selected_stock)
data_load_state.text('Loading data... successful!')



#Gives a summary about the company
summary=('Ashok Leyland is an Indian multinational automotive manufacturer, headquartered in Chennai. It is owned by the Hinduja Group. It was founded in 1948 as Ashok Motors and became Ashok Leyland in the year 1955','State Bank of India is an Indian multinational public sector bank and financial services statutory body headquartered in Mumbai, Maharashtra.','Axis Bank Limited, formerly known as UTI Bank, is an Indian banking and financial services company headquartered in Mumbai, Maharashtra. It sells financial services to large and mid-size companies, SMEs and retail businesses. As of 30 June 2016, 30.81% shares are owned by the promoters and the promoter group.','Maruti Suzuki India Limited, formerly known as Maruti Udyog Limited, is an Indian automobile manufacturer, based in New Delhi. It was founded in 1981 and owned by the Government of India until 2003, when it was sold to the Japanese automaker Suzuki Motor Corporation.')
def print_summary(selected_stock):
    if selected_stock=='SBIN':
        sum_out=summary[1]
        img=Image.open('sbi.jpg')
        order=(1,1,2)
    elif selected_stock=='ASHOK LEYLAND LIMITED':
        sum_out=summary[0]
        img=Image.open('leyland.jpg')
        order=(1,1,2)
    elif selected_stock=='AXIS BANK LIMITED':
        sum_out=summary[2]
        img=Image.open('axis.jpg')
        order=(1,0,0)
    else:
        sum_out=summary[3]
        img=Image.open('maruti.jpg')
        order=(2,1,1)
    return sum_out,img,order


#display summary of the selected company

disp,img,order=print_summary(selected_stock)
st.subheader(selected_stock)
st.image(img)
st.write(disp)

#display raw data
st.subheader('Raw data')
st.write(data.head())
st.subheader('No of data points :-')
st.write(no_of_data)

#model building 
def model_building(data):
    df=pd.DataFrame()
    df['Date']=data['Date']
    df['Close']=data['Close']
    df['Date']=pd.to_datetime(df['Date'])
    df=df.set_index(df['Date'])
    df.drop(['Date'],axis=1,inplace=True)
    df=df.resample(rule='w').mean()
    model=ARIMA(df['Close'],order=order)
    model_fit=model.fit()
    forcast=model_fit.forecast(steps=n_weeks)
    return forcast




#forcast slider
st.subheader('Forcast')
n_weeks=st.slider('Forcast weeks',1,12)
forcast=model_building(data)
st.subheader('Forcast for the next {} weeks'.format(n_weeks))
st.write(forcast)


#using pyplot
st.subheader('Line Plot')
fig=plt.figure()
fig=plt.figure(figsize=(10,5))
sns.lineplot(data=data,x='Date',y='Close',label=selected_stock)
plt.plot(forcast,label='Forcast')
plt.xlabel('Date')
plt.ylabel('Close price')
plt.legend()
st.pyplot(fig)
