import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

#open the excel file of stock data
df= pd.read_csv('BTC-USD.csv')

#function for SMA
def SMA(data,period=30,coloum='close'):
    return data[coloum].rolling(window=period).mean()

#function for EMA
def EMA (data,period=30,coloum='close'):
    return data[coloum].ewm(span=period,adjust=False).mean()

#function for MACD
def MACD (data,period_long=26,period_short=12,period_signal=9,coloum='close'):
    ShortEMA=EMA(data,period=period_short,coloum=coloum)
    longEMA=EMA(data,period=period_long,coloum=coloum)
    data['MACD']=ShortEMA-longEMA
    data['signal_line']=EMA(data,period=period_signal,coloum='MACD')
    return data

#function for RSI
def RSI(data,period=14,coloum='close'):
    delta=data[coloum].diff(1)
    delta=delta.dropna()
    up=delta.copy()
    down=delta.copy()
    up[up<0]=0
    down[down>0]=0
    data['up']=up
    data['down']=down
    AVG_gain=SMA(data,period,coloum='up')
    AVG_loss=abs(SMA(data,period,coloum='down'))
    RS=AVG_gain/AVG_loss
    RSI=100.0-(100.0/(1.0+RS))

    data['RSI']=RSI
    return data

#compute teh MACD, EMA ,RSI and SMA
MACD(df)
RSI(df)
df['SMA']=SMA(df)
df['EMA']=EMA(df)

#display the target column
df['target']=np.where(df['close'].shift(-1)>df['close'],1,0)
pd.set_option("display.max_rows",None,"display.max_columns",None)

df=df[29:]

#filter the file to the following coloumns
keep_columns=['close','MACD','signal_line','RSI','SMA','EMA']
print(df)
x=df[keep_columns].values
print(x)
y=df['target'].values


#train the x and y values of the model with test values of 10% of data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=472)

#fit the model using the DecisionTree classifier
tree = DecisionTreeClassifier().fit(x_train, y_train)


#show the perdiction result and accuracy
print(tree.score(x_test, y_test))
print(tree.predict(df[keep_columns].tail(1).values))