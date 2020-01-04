import pandas
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.preprocessing import text
import numpy as np


####################################################################################################

g = pandas.read_csv("autos.csv",encoding = "ISO-8859-1")

g["price"]  = g["price"]/100
g    = g[g["price"]>0.10]
g    = g[g["price"]<350]


g["kilometer"]          = g["kilometer"]/1000
g["yearOfRegistration"] = g["yearOfRegistration"]/1000
g["powerPS"]            = g["powerPS"]/100


brands       = pandas.get_dummies(g["brand"]).values
fuel         = pandas.get_dummies(g["fuelType"]).values
vehicleType  = pandas.get_dummies(g["vehicleType"]).values
gearbox      = pandas.get_dummies(g["gearbox"]).values
brand        = pandas.get_dummies(g["brand"]).values
model_car    = pandas.get_dummies(g["model"]).values
repaired     = pandas.get_dummies(g["notRepairedDamage"]).values


X = g[["powerPS","kilometer","yearOfRegistration"]].values
Y = g["price"].values


####################################################################################################
 
model = Sequential()
model.add(Dense(44, input_dim=312, init='normal', activation='relu'))
model.add(Dense(8, init='normal', activation='relu'))
model.add(Dense(4, init='normal', activation='relu'))
model.add(Dense(1, init='normal'))

sgd = keras.optimizers.Adam(lr=0.07, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(loss='mean_absolute_error',
              optimizer=sgd,  
              metrics=["mean_absolute_error"])


######################################################


blocksize = np.int64(X.shape[0] / 20)
for w in range(1,50):
    current   = 0
    ds = 0
    for q in range(1,21):
        nextt = blocksize*q
        if (q==20):
            nextt = X.shape[0]
        
        Q           = X[current:nextt,:]
        vehicleTypeq = vehicleType[current:nextt,:]
        gearboxq     = gearbox[current:nextt,:]
        fuelq        = fuel[current:nextt,:]
        brandq       = brand[current:nextt,:]
        model_carq   = model_car[current:nextt,:]
        repairedq    = repaired[current:nextt,:]

        Q = np.concatenate((Q,vehicleTypeq),axis=1)
        Q = np.concatenate((Q,gearboxq),axis=1)
        Q = np.concatenate((Q,fuelq),axis=1)
        Q = np.concatenate((Q,brandq),axis=1)
        Q = np.concatenate((Q,model_carq),axis=1)
        Q = np.concatenate((Q,repairedq),axis=1)
        
        
        Y = g["price"].values[current:nextt]
        ds = model.train_on_batch(Q,Y)
        current = nextt
    print(ds[0])

#########################################################################################################   

gB = 0
current   = 0
for q in range(1,21):        
    nextt = blocksize*q
    if (q==20):
            nextt = X.shape[0]
    Q           = X[current:nextt,:]
    print(q)
    vehicleTypeq = vehicleType[current:nextt,:]
    gearboxq     = gearbox[current:nextt,:]
    fuelq        = fuel[current:nextt,:]
    brandq       = brand[current:nextt,:]
    model_carq   = model_car[current:nextt,:]
    repairedq    = repaired[current:nextt,:]
    Q = np.concatenate((Q,vehicleTypeq),axis=1)
    Q = np.concatenate((Q,gearboxq),axis=1)
    Q = np.concatenate((Q,fuelq),axis=1)
    Q = np.concatenate((Q,brandq),axis=1)
    Q = np.concatenate((Q,model_carq),axis=1)
    Q = np.concatenate((Q,repairedq),axis=1)
    print("#")
    p = model.predict(Q)
    
    current = nextt
    if (q!=1):
        gB = np.concatenate((gB,p),axis=0)
    else:
        gB = p
        
print("finish")            
g["predicted"] = gB
g["mae"] = abs(g["price"]-g["predicted"])
print(g["mae"].mean())
print(g[["predicted","price"]])

