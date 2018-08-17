import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Dense


import time


"""
print("Reading IoT dataset")
X1 = pd.read_csv("bengin_traffic.csv",header=0).as_matrix()
X1_length = X1.shape[0]



m1 = pd.read_csv("syn.csv",header=0).as_matrix() 
m1_length = m1.shape[0]
Xm1 = np.concatenate((X1, m1), axis=0)
X_train, X_test = Xm1[:40000, ...], Xm1[40000:, ...]

"""

df1 = pd.read_csv("bengin_traffic.csv")

df2 = pd.read_csv("syn.csv")

frames = [df1, df2]

df = pd.concat(frames)
#Normalize
normalized_df=(df-df.min())/(df.max()-df.min()) #But it is not an online way 
#Also max min shouldnt change after training phase.Change above code and see result.



#
X_train, X_test = train_test_split(normalized_df,test_size=0.7 ,shuffle = False, stratify = None)


#X_train, X_test = train_test_split(X1, test_size=0.4, random_state=0)




# In[17]:

"""
input_layer = Input(shape=(input_dim, ))

encoder = Dense(encoding_dim, activation="tanh", )(input_layer)
encoder = Dense(int(encoding_dim / 2), activation="sigmoid")(encoder)

decoder = Dense(int(encoding_dim / 2), activation='tanh')(encoder)
decoder = Dense(input_dim, activation='sigmoid')(decoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)
"""

input_layer = Input(shape=(115,))
encoded = Dense(86, activation='sigmoid')(input_layer)
encoded = Dense(57, activation='sigmoid')(encoded)
encoded = Dense(37, activation='sigmoid')(encoded)
encoded = Dense(28, activation='sigmoid')(encoded)

decoded = Dense(37, activation='sigmoid')(encoded)
decoded = Dense(57, activation='sigmoid')(decoded)
decoded = Dense(86, activation='sigmoid')(decoded)
decoded = Dense(115, activation='sigmoid')(decoded)

autoencoder = Model(input_layer, decoded)
#the problem with keras is it want whole dataset as an input parameter
start = time.time()

autoencoder.compile(optimizer='adam', 
                    loss='mean_squared_logarithmic_error')	#change loss to without log


autoencoder.fit(X_train, X_train,
                epochs=2,
                batch_size=1,
                shuffle=True)
#https://stackoverflow.com/questions/46308374/what-is-validation-data-used-for-in-a-keras-sequential-model
predictions = autoencoder.predict(X_test)


stop = time.time()
print("Total time taken: "+ str(stop - start))

rmse = np.mean(np.power(X_test - predictions, 2), axis=1).as_matrix(columns=None)

###################
#Calculating threshold

threshold1 = 0
index =0
for i in range(0,10000):
    if(rmse[i]> threshold1):
        threshold1 = rmse[i]
        index = i

    
#creating timestamps
observations = df.shape[0]
templist = []
for i in range(0,observations):
    temp = i/25000
    templist.append(temp)

temparr = np.array(templist)


print("Plotting results")

from matplotlib import cm
plt.figure(figsize=(10,5))
fig = plt.scatter(temparr[36274:],rmse[:],s=0.01,c=None,cmap=None)
#plt.yscale("log")
plt.title("Anomaly Scores for Execution Phase")
plt.ylabel("RMSE ")
plt.xlabel("Time elapsed [min]")
plt.show()

