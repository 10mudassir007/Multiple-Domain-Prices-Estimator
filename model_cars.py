import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score,KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error

df = pd.read_csv('https://raw.githubusercontent.com/10mudassir007/Multiple-Domain-Prices-Estimator/main/cars/carscleaned.csv')


df['horsepower'] /= 100
df['peakrpm'] /= 1000
df[['citympg','highwaympg']] /= 10

X = np.array(df[['doornumber', 'horsepower', 'peakrpm',
       'citympg', 'highwaympg','fueltype_diesel', 'fueltype_gas',
       'carbody_convertible', 'carbody_hardtop', 'carbody_hatchback',
       'carbody_sedan', 'carbody_wagon', 'drivewheel_4wd', 'drivewheel_fwd',
       'drivewheel_rwd']])
y = np.array(df['price'])

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


model = AdaBoostRegressor(estimator=DecisionTreeRegressor(random_state=42,criterion='squared_error', splitter='best', max_depth=6, min_samples_split=2, min_samples_leaf=1, max_features=0.70),n_estimators=430,learning_rate=0.1,loss='exponential',random_state=42)
model.fit(x_train,y_train)
predictions = model.predict(x_test)


if __name__ == "__main__":
       kf = KFold(n_splits=5,shuffle=True,random_state=42)
       scores = cross_val_score(model,X,y,cv=kf,scoring='neg_mean_squared_error')
       mse_scores = -scores
       mean_mse = mse_scores.mean()
       print('Mean squared error',mean_squared_error(y_test,predictions))
       print('Mean absolute error',mean_absolute_error(y_test,predictions))
       print('Model score train set',model.score(x_train,y_train))
       print('Model score test set',model.score(x_test,y_test))
       print('Cross val score',scores)
       print('Cross val mse',mse_scores)
       print('Cross val mse mean',mean_mse)

def predict_cars(door=0,horsepower=0,peakrpm=0,citympg=0,highwaympg=0,fueltype=0,carbody=0,drivewheel=0):
       fueltypes = ['diesel','gas']
       carbodys = ['convertible', 'hardtop', 'hatchback','sedan', 'wagon']
       drivewheels = ['4wd','fwd','rwd']

       for j in range(len(fueltypes)):
              if fueltype == fueltypes[j]:
                     fueltypes[j] = 1
              else:
                     fueltypes[j] = 0
       
       for j in range(len(carbodys)):
              if carbody == carbodys[j]:
                     carbodys[j] = 1
              else:
                     carbodys[j] = 0
       
       for j in range(len(drivewheels)):
              if drivewheel == drivewheels[j]:
                     drivewheels[j] = 1
              else:
                     drivewheels[j] = 0
       features = [door,(horsepower/100),(peakrpm/1000),citympg,highwaympg]
       features.extend(fueltypes)
       features.extend(carbodys)
       features.extend(drivewheels)
       features = np.array([features])
       return f'The predicted price of car is {int(((model.predict(features)) //10) * 10)} dollars'
        
        
    