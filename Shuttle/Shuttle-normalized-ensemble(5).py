
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


# #Importing Files from Google Drive

# In[2]:


from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials


# In[3]:


auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)


# In[7]:


#https://drive.google.com/open?id=1M_VyfJ3We265BENCpfr2_S9zsi93deaH
file_id='1M_VyfJ3We265BENCpfr2_S9zsi93deaH'
downloaded = drive.CreateFile({'id':file_id})   # replace the id with id of file you want to access
downloaded.GetContentFile('shuttle.trn')        # replace the file name with your file

#https://drive.google.com/file/d/1QCdpAeMQpx4qgp3NUNjNISPyu1Qp_J2N
file_id='1QCdpAeMQpx4qgp3NUNjNISPyu1Qp_J2N'
downloaded = drive.CreateFile({'id':file_id})   # replace the id with id of file you want to access
downloaded.GetContentFile('shuttle.tst')        # replace the file name with your file


# In[8]:


train_df = pd.read_csv('shuttle.trn',sep=" ", header=None, names=['time','m1','m2', 'm3','m4','m5','m6','m7','m8','status'])
train_df = train_df[train_df.status!= 4]
test_df= pd.read_csv('shuttle.tst',sep=" ", header=None, names=['time','m1','m2', 'm3','m4','m5','m6','m7','m8','status'])
test_df = test_df[test_df.status!= 4]


# In[9]:


all_df=pd.concat([train_df, test_df], ignore_index=True)
all_df.head()


# In[10]:


from random import randint
start=randint(0, len(all_df)-6000)
print(start)
dataset = all_df.iloc[start:start+6000,]


# # Normalize data

# In[11]:


# normalize data
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(dataset.iloc[:3600,1:9].values)
test_scaled = scaler.transform(dataset.iloc[3600:,1:9].values)


# In[12]:


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


# In[13]:


# fit Vanilla model on dataset
def fit_model(trainX, trainy,n_features):
    # define model
    model = Sequential()
    model.add(LSTM(100, activation='relu', input_shape=(n_steps, n_features)))
    model.add(Dense(n_features))
    model.compile(optimizer='adam', loss='mse',metrics=['mae'])
    # fit model
    model.fit(trainX, trainy, epochs=500, verbose=0)
    return model


# In[14]:


# make an ensemble prediction for multi-class classification
def ensemble_predictions(members, weights, testX, testy):
    # make predictions
    yhats = [model.predict(testX) for model in members]
    yhats = array(yhats)
    a_testy=array(testy)
    # weighted sum across ensemble members
    #summed = tensordot(abs(yhats-a_testy), weights, axes=((0),(0)))
    # argmax across classes
    #result = argmax(summed, axis=1)
    result = tensordot(yhats, weights, axes=((0),(0)))
    return result

# # evaluate a specific number of members in an ensemble
def evaluate_ensemble(members, weights, testX, testy):
    # make prediction
    yhat = ensemble_predictions(members, weights, testX, testy)
    # calculate MAE
    return mean_absolute_error(testy, yhat), yhat


# In[15]:


n_steps = 5
# convert into input/output
X_train, y_train = split_sequences(train_scaled, n_steps)
print(X_train.shape, y_train.shape)
print(y_train[1])


# In[16]:


n_steps = 5
# convert into input/output
X_test, y_test = split_sequences(test_scaled, n_steps)
print(X_test.shape, y_test.shape)
print(y_test[1])


# In[17]:


n_features = X_train.shape[2]
n_members = 5
members = [fit_model(X_train, y_train,n_features) for _ in range(n_members)]
# evaluate each single model on the test set

for i in range(n_members):
    _, test_acc = members[i].evaluate(X_test, y_test, verbose=0)
    print('Model %d: %.3f' % (i+1, test_acc))
# evaluate averaging ensemble (equal weights)
weights = [1.0/n_members for _ in range(n_members)]
score, prediction = evaluate_ensemble(members, weights, X_test, y_test)
print('Equal Weights Score: %.3f' % score)


# In[18]:


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


# In[19]:


# define bounds on each weight
bound_w = [(0.0, 1.0)  for _ in range(n_members)]
# arguments to the loss function
search_arg = (members, X_test, y_test)
# global optimization of ensemble weights
result = differential_evolution(loss_function, bound_w, search_arg, maxiter=1000, tol=1e-7)
# get the chosen weights
weights = normalize(result['x'])
print('Optimized Weights: %s' % weights)
# evaluate chosen weights
score, prediction = evaluate_ensemble(members, weights, X_test, y_test)
print('Optimized Weights Score: %.3f' % score)


# # Visualize the prediction of one varialble with actual class

# In[20]:


aa=[x for x in range(50)]

fig = plt.figure(figsize=(12,4))
ax = fig.add_subplot(111)
ax2 = ax.twinx()

#plt.figure(figsize=(12,4))
ax.plot(aa, y_test[-50:,7], '-g', label="actual_scaled")
ax.plot(aa, prediction[-50:,7], '-r', label="prediction")

ax2.plot(aa, train_df.iloc[4950:5000,-1].values, marker='.')
ax.plot(np.nan, marker='.', label="actual")
# plt.tick_params(left=False, labelleft=True) #remove ticks
#plt.tight_layout()
#sns.despine(top=True)
#plt.subplots_adjust(left=0.07)
#plt.ylabel('Hum', size=15)
#plt.xlabel('Time step', size=15)
ax.legend(fontsize=15)

ax.set_xlabel("Time step", size=12)
ax.set_ylabel(r"m8")
ax2.set_ylabel(r"class")
ax2.set_ylim(0, 8)
#ax.set_ylim(-20,100)
plt.show();


# # Calculate Euclidean distance & Detect anomalies

# In[21]:


# The percentage of normaly in testing 
# Class#1 is normal instances
my_list=dataset.iloc[3605:6000,-1].values
one=sum(1 for item in my_list if item==(1))
print('Number of class1 in testing set:',one)
print('Anomaly rate in testing set:', 1-(one/len(my_list)))
print(round((one/len(my_list)*100)))


# In[22]:


temp = [] #temporary list
for j in range(len(y_test)):
    dis = sum([pow(y_test[j][i] - prediction[j][i], 2) for i in range(n_features)])
    temp.append(round(pow(dis, 0.5),4))
print(len(temp))


# In[23]:


array_dis=np.array(temp)
thold= np.percentile(array_dis,round((one/len(my_list)*100)))#align with the Anomaly rate in testing set
outcome=[]
for a in array_dis:
    if a <= thold:
        outcome.append(1.0)# normality
    else:
        outcome.append(0.0)# anomaly
        
sum(a == b for a,b in zip(outcome, train_df.iloc[3006:5000,-1].values))/len(outcome)


# In[30]:


scaled_test_df = pd.DataFrame({'act_m1': y_test[:, 0], 'act_m2': y_test[:, 1],'act_m3': y_test[:, 2], 'act_m4': y_test[:, 3],
                        'act_m5': y_test[:, 4], 'act_m6': y_test[:, 5],'act_m7': y_test[:, 6], 'act_m8': y_test[:, 7],
                        'prd_m1': prediction[:, 0], 'prd_m2': prediction[:, 1],'prd_m3': prediction[:, 2], 'prd_m4': prediction[:, 3],
                               'prd_m5': prediction[:, 4], 'prd_m6': prediction[:, 5],'prd_m7': prediction[:, 6], 'prd_m8': prediction[:, 7],
                              'act_class':dataset.iloc[3605:6000,-1].values, 'Euclidean distance': array_dis, 'prd_class':outcome})
scaled_test_df['act_class'].loc[scaled_test_df['act_class'] != 1] = 0 


# In[31]:


from google.colab import files
scaled_test_df.to_excel("Shuttle-scaled_test_df_ensemble(5)-(r5).xlsx")
files.download('Shuttle-scaled_test_df_ensemble(5)-(r5).xlsx')


# In[32]:


from sklearn.metrics import roc_curve,auc
fpr, tpr, thresholds = roc_curve(scaled_test_df['act_class'], scaled_test_df['Euclidean distance'],pos_label=0)
roc_auc = auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve_ensembleLSTM5 (area = %0.3f)' % roc_auc)
#plt.plot(fpr1, tpr1, color='green',lw=lw, label='ROC curve_ensemble(5) (area = %0.3f)' % roc_auc1)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


# # Visualize data over time

# In[34]:


aa=[x for x in range(200)]
plt.figure(figsize=(15,4))
plt.plot(aa, y_test[-200:,0], 'b:', label="actual")
plt.plot(aa, prediction[-200:,0], 'g', label="prediction")
outcome200=scaled_test_df['prd_class'][-200:].to_numpy()
actual200=scaled_test_df['act_class'][-200:].to_numpy()

indexes = [index for index in range(len(actual200)) if actual200[index] == 0]
plt.plot(indexes,[prediction[-200:,0][i] for i in indexes], ls="", marker="o", label="actual anomalies",alpha=0.6)
p_indexes = [index for index in range(len(outcome200)) if outcome200[index] == 0]
plt.plot(p_indexes,[prediction[-200:,0][i] for i in p_indexes], ls="", marker="*", label="predicted anomalies")

plt.tight_layout()
sns.despine(top=True)
plt.subplots_adjust(left=0.07)
plt.xlim([0, 210])
plt.ylabel('m1', size=15)
plt.xlabel('Time stamp', size=15)
plt.legend(fontsize=12)
plt.show();

