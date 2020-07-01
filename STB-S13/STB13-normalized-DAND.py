from datetime import datetime, timedelta
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import matplotlib.pyplot as plt
from numpy import hstack
from numpy import mean
from numpy import std
from numpy import array
from numpy import argmax, argmin
from numpy import tensordot
from numpy.linalg import norm
import seaborn as sns
from scipy.optimize import differential_evolution
from sklearn.metrics import mean_absolute_error
import gc
from sklearn.metrics import roc_curve,auc

import glob
import os
folder_name = 'db'
file_type = 'txt'
seperator =' '

dataframe = pd.concat([pd.read_csv(f, sep=seperator, header=None) for f in glob.glob(folder_name + "/*."+file_type)],ignore_index=True)
dataframe.columns=['stationID','year','month','day','hour','min','sec','epoch','AmbientTemp','SurfaceTemp','Radiaiton','RH','Moisture','watermark','rain',
                   'windSpeed','WindDirection']

df=dataframe.dropna(axis=1, thresh=539702).iloc[:,8:]
del dataframe
gc.collect()
df = df.dropna()
df = df.reset_index(drop=True)

# Random select a chunk of data
from random import randint
start=randint(0, len(df)-6000)
print(start)
dataset = df.iloc[start:start+6000,]


# # normalize and add 5% noise

# normalize data
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(dataset.iloc[:3600,].values)
test_scaled = scaler.transform(dataset.iloc[3600:,].values)

# add 5% noises as anomalies into train and test data in order to evaluate the method
import math
    
qty=math.floor(len(train_scaled)*0.05)
train_anomalies=np.random.choice(train_scaled.shape[0],size = qty,replace=False)
print(train_anomalies)
temp_data=train_scaled[train_anomalies,:]+np.random.normal(0,1,size=train_scaled.shape[1])
i=0
for row in train_anomalies:
    train_scaled[row,:]=temp_data[i,:]
    i+=1

qty=math.floor(len(test_scaled)*0.05)
test_anomalies=np.random.choice(test_scaled.shape[0],size = qty,replace=False)
print(test_anomalies)
temp_data=test_scaled[test_anomalies,:]+np.random.normal(0,1,size=test_scaled.shape[1])
i=0
for row in test_anomalies:
    test_scaled[row,:]=temp_data[i,:]
    i+=1

# multivariate output data prep

# split a multivariate parallel sequence into samples
def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

# fit Vanilla model on dataset
def fit_model(trainX, trainy,n_features):
    n_steps=trainX.shape[1]
    # define model
    model = Sequential()
    model.add(LSTM(100, activation='relu', input_shape=(n_steps, n_features)))
    model.add(Dense(n_features))
    model.compile(optimizer='adam', loss='mse',metrics=['mae'])
    # fit model
    model.fit(trainX, trainy, epochs=500, verbose=0)
    return model

# make an ensemble prediction 
def ensemble_predictions(members, weights, testX):
    # make predictions
    temp=[model.predict(testX[i]) for model,i in zip(members,range(n_members))]
    for i in range(n_members-1):
        temp[i]=np.delete(temp[i], slice(0, n_members-i-1), axis=0)
    yhats = array(temp)
    result = tensordot(yhats, weights, axes=((0),(0)))
    return result

# # evaluate a specific number of members in an ensemble
def evaluate_ensemble(members, weights, testX, testy):
    # make prediction
    yhat = ensemble_predictions(members, weights, testX)
    # calculate MAE
    return mean_absolute_error(testy, yhat), yhat

n_members = 5
X_train=[]
y_train=[]
# convert into input/output, splitting data points into five multivariate parallel sequences with sequence length from 1 to 5
for i in range(1,n_members+1):
    X_, y_ = split_sequences(train_scaled, i)
    X_train.append(X_)
    y_train.append(y_)

X_test=[]
y_test=[]
# convert into input/output, splitting data points into five multivariate parallel sequences with sequence length from 1 to 5
for i in range(1,n_members+1):
    X_, y_ = split_sequences(test_scaled, i)
    X_test.append(X_)
    y_test.append(y_)

n_features = X_train[0].shape[2]
members = [fit_model(X_train[i], y_train[i],n_features) for i in range(n_members)]
# evaluate each single model on the test set
for i in range(n_members):
    _, test_acc = members[i].evaluate(X_test[i], y_test[i], verbose=0)
    print('Model %d: %.3f' % (i+1, test_acc))

# normalize a vector to have unit norm
def normalize(weights):
    # calculate l1 vector norm
    result = norm(weights, 1)
    # check for a vector of all zeros
    if result == 0.0:
        return weights
    # return normalized vector (unit norm)
    return weights / result

# loss function for optimization process, designed to be minimized
def loss_function(weights, members, testX, testy):
    # normalize weights
    normalized = normalize(weights)
    # calculate error rate
    return evaluate_ensemble(members, normalized, testX, testy)[0]

# define bounds on each weight
bound_w = [(0.0, 1.0)  for _ in range(n_members)]
# arguments to the loss function
search_arg = (members, X_test, y_test[n_members-1])
# global optimization of ensemble weights
result = differential_evolution(loss_function, bound_w, search_arg, maxiter=1000, tol=1e-7)
# get the chosen weights
weights = normalize(result['x'])
print('Optimized Weights: %s' % weights)
# evaluate chosen weights
score, prediction = evaluate_ensemble(members, weights, X_test, y_test[n_members-1])
print('Optimized Weights Score: %.3f' % score)


# # Calculate Euclidean distance & Detect anomalies

temp = [] #temporary list
y_test_final=y_test[n_members-1]
for j in range(len(y_test_final)):
    dis = sum([pow(y_test_final[j][i] - prediction[j][i], 2) for i in range(n_features)])
    temp.append(round(pow(dis, 0.5),4))

array_dis=np.array(temp)
thold= np.percentile(array_dis,95)
outcome=[]
for a in array_dis:
    if a <= thold:
        outcome.append(1.0) #1 is normal
    else:
        outcome.append(0.0) #0 is abnormals

# Create a array to indicate the position of actual anomalies
b = np.ones((len(test_scaled),1))
rows=test_anomalies
b[rows] = 0
test_scaled_anomolies=np.hstack((test_scaled,b))

# Output detection results to excel
scaled_test_df = pd.DataFrame({'act_AmbientTemp': y_test_final[:, 0], 'act_SurfaceTemp': y_test_final[:, 1],'act_RH': y_test_final[:, 2], 
                               'act_rain': y_test_final[:, 3],'act_WindDirection': y_test_final[:, 4], 'prd_AmbientTemp': prediction[:, 0], 
                               'prd_SurfaceTemp': prediction[:, 1],'prd_RH': prediction[:, 2], 'prd_rain': prediction[:, 3],
                               'prd_WindDirection': prediction[:, 4], 'Euclidean distance': array_dis, 
                               'act_class':test_scaled_anomolies[5:,-1], 'prd_class':outcome})
scaled_test_df.to_excel("scaled_test_df_ensemble5(v)-(r5).xlsx")

# ROC curve
fpr, tpr, thresholds = roc_curve(scaled_test_df['act_class'], scaled_test_df['Euclidean distance'],pos_label=0)
roc_auc = auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve_ensembleLSTM5 (area = %0.3f)' % roc_auc)
#plt.plot(fpr1, tpr1, color='green',lw=lw, label='ROC curve_LSTM7 (area = %0.3f)' % roc_auc1)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


# # Visualize data over time

aa=[x for x in range(200)]
plt.figure(figsize=(15,4))
plt.plot(aa, y_test[n_members-1][-200:,0], 'b:', label="actual")
plt.plot(aa, prediction[-200:,0], 'g', label="prediction")
outcome200=scaled_test_df['prd_class'][-200:].to_numpy()

indexes = [index for index in range(len(test_scaled_anomolies[-200:,-1])) if test_scaled_anomolies[-200:,-1][index] == 0]
plt.plot(indexes,[prediction[-200:,0][i] for i in indexes], ls="", marker="o", label="actual anomalies",alpha=0.6)
p_indexes = [index for index in range(len(outcome200)) if outcome200[index] == 0]
plt.plot(p_indexes,[prediction[-200:,0][i] for i in p_indexes], ls="", marker="*", label="predicted anomalies")

plt.tight_layout()
sns.despine(top=True)
plt.subplots_adjust(left=0.07)
plt.xlim([0, 210])
plt.ylabel('AmbientTemp', size=15)
plt.xlabel('Time stamp', size=15)
plt.legend(fontsize=12)
plt.show();
