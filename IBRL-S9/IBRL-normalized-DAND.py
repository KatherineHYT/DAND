
# coding: utf-8

# In[1]:


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


# In[2]:


IBRL_df = pd.read_csv('data.txt.gz', compression='gzip', sep=' ', header=None, 
                                 names=['date','time','epoch', 'moteid','temp','hum','lum','vol'])
IBRL_S9=IBRL_df.loc[IBRL_df['moteid'] == 9]


# In[3]:


IBRL_S9=IBRL_S9.dropna()


# In[4]:


from random import randint
start=randint(0, len(IBRL_S9)-6000)
print(start)
dataset = IBRL_S9.iloc[start:start+6000,4:8]


# # Normalize data

# In[5]:


# normalize data
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(dataset.iloc[:3600,:].values)
test_scaled = scaler.transform(dataset.iloc[3600:,:].values)


# # Add 5% noise as anomolies

# In[6]:


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


# In[7]:


qty=math.floor(len(test_scaled)*0.05)
test_anomalies=np.random.choice(test_scaled.shape[0],size = qty,replace=False)
print(test_anomalies)
temp_data=test_scaled[test_anomalies,:]+np.random.normal(0,1,size=test_scaled.shape[1])
i=0
for row in test_anomalies:
    test_scaled[row,:]=temp_data[i,:]
    i+=1


# In[8]:


# multivariate output data prep

# split a multivariate sequence into samples
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


# In[9]:


# fit model on dataset
def fit_model(trainX, trainy,n_features):
    n_steps=trainX.shape[1]
    # define model
    model = Sequential()
    model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
    model.add(Dense(n_features))
    model.compile(optimizer='adam', loss='mse',metrics=['mae'])
    # fit model
    model.fit(trainX, trainy, epochs=500, verbose=0)
    return model


# In[10]:


# make an ensemble prediction for multi-class classification
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


# In[11]:


n_members = 5
X_train=[]
y_train=[]
# convert into input/output
for i in range(1,n_members+1):
    X_, y_ = split_sequences(train_scaled, i)
    X_train.append(X_)
    y_train.append(y_)

#print(X_train.shape, y_train.shape)
print(X_train[0].shape)


# In[12]:


X_test=[]
y_test=[]
# convert into input/output
#X_train, y_train = [split_sequences(train_scaled, i) for i in range(n_members)]
#print(X_train.shape, y_train.shape)
#print(y_train[1])
for i in range(1,n_members+1):
    X_, y_ = split_sequences(test_scaled, i)
    X_test.append(X_)
    y_test.append(y_)

#print(X_train.shape, y_train.shape)
print(X_test[0].shape)


# In[13]:


n_features = X_train[0].shape[2]

members = [fit_model(X_train[i], y_train[i],n_features) for i in range(n_members)]
# evaluate each single model on the test set

for i in range(n_members):
    _, test_acc = members[i].evaluate(X_test[i], y_test[i], verbose=0)
    print('Model %d: %.3f' % (i+1, test_acc))
# evaluate averaging ensemble (equal weights)
#weights = [1.0/n_members for _ in range(n_members)]
#score, prediction = evaluate_ensemble(members, weights, X_test, y_test[n_members-1])
#print('Equal Weights Score: %.3f' % score)


# In[14]:


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


# In[15]:


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
score_e, prediction_e = evaluate_ensemble(members, weights, X_test, y_test[n_members-1])
print('Optimized Weights Score: %.3f' % score_e)


# In[16]:


import seaborn as sns
aa=[x for x in range(200)]
plt.figure(figsize=(8,4))
plt.plot(aa, y_test[n_members-1][-200:,0], marker='.', label="actual")
plt.plot(aa, prediction_e[-200:,0], 'r', label="prediction")
# plt.tick_params(left=False, labelleft=True) #remove ticks
plt.tight_layout()
sns.despine(top=True)
plt.subplots_adjust(left=0.07)
plt.ylabel('Temp', size=15)
plt.xlabel('Time step', size=15)
plt.legend(fontsize=15)
plt.show();


# # Calculate Euclidean distance & Detect anomalies

# In[17]:


temp = [] #temporary list
y_test_final=y_test[n_members-1]
for j in range(len(y_test_final)):
    dis = sum([pow(y_test_final[j][i] - prediction_e[j][i], 2) for i in range(n_features)])
    temp.append(round(pow(dis, 0.5),4))
print(len(temp))


# In[18]:


array_dis=np.array(temp)
thold= np.percentile(array_dis,95)
outcome=[]
for a in array_dis:
    if a <= thold:
        outcome.append(1.0) #1 is normal
    else:
        outcome.append(0.0) #0 is abnormals


# In[19]:


b = np.ones((len(test_scaled),1))
rows=test_anomalies
b[rows] = 0
#print(b)


# In[20]:


test_scaled_anomolies=np.hstack((test_scaled,b))


# In[21]:


scaled_test_df = pd.DataFrame({'act_temp': y_test_final[:, 0], 'act_hum': y_test_final[:, 1],'act_lum': y_test_final[:, 2], 
                               'act_vol': y_test_final[:, 3], 'prd_temp': prediction_e[:, 0], 'prd_hum': prediction_e[:, 1],
                               'prd_lum': prediction_e[:, 2], 'prd_vol': prediction_e[:, 3],
                               'Euclidean distance': array_dis, 'act_class':test_scaled_anomolies[5:,-1],'prd_class':outcome})
#scaled_test_df
scaled_test_df.to_excel("scaled_test_df_ensemble5(v)-addnoise(r7).xlsx")


# # ROC cureve

# In[22]:


from sklearn.metrics import roc_curve,auc
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

# In[25]:


import seaborn as sns
aa=[x for x in range(200)]
plt.figure(figsize=(15,4))
plt.plot(aa, y_test[n_members-1][-200:,0], marker='.', label="actual")
plt.plot(aa, prediction_e[-200:,0], 'r', label="prediction")

indexes = [index for index in range(len(test_scaled_anomolies[-200:,-1])) if test_scaled_anomolies[-200:,-1][index] == 0]
plt.plot(indexes,[y_test[n_members-1][-200:,0][i] for i in indexes], ls="", marker="o", label="actual anomalies")
p_indexes = [index for index in range(len(outcome[-200:])) if outcome[-200:][index] == 0]
plt.plot(p_indexes,[prediction_e[-200:,0][i] for i in p_indexes], ls="", marker="*", label="predict anomalies")

plt.tight_layout()
sns.despine(top=True)
plt.subplots_adjust(left=0.07)
plt.ylabel('Temp', size=15)
plt.xlabel('Time step', size=15)
plt.legend(fontsize=15)
plt.show();

