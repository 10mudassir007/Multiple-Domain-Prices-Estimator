#IMPORTING NECESSARY LIBRARIES
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score,KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error


#LOADING DATA , THE CSV FILE SHOULD BE IN THE SAME LOCATION AS THE TERMINAL LOCATION OR CHANGE THE LOCATION OF THE TERMINAL TO WHERE THE FILE IS SAVED USING cd commands
df = pd.read_csv('Multiple-Domain-Prices-Estimator/cleanedlaptops.csv')
df.fillna(0, inplace=True)
df = df[['RAM','Storage','Screen','Price','Brand_Acer','Brand_Alurin','Brand_Apple','Brand_Asus','Brand_Deep Gaming','Brand_Denver','Brand_Dynabook Toshiba','Brand_Gigabyte','Brand_HP','Brand_Innjoo','Brand_LG', 'Brand_Lenovo',
       'Brand_PcCom', 'Brand_Primux', 'Brand_Prixton', 'Brand_Razer',
       'Brand_Realme', 'Brand_Samsung', 'Brand_Thomson', 'Brand_Toshiba',
       'Brand_Vant', 'Cores', 'External VRAM']].head(900)

df[['RAM','Screen']] /= 10

df['Storage'] /= 1000

X = np.array(df[['RAM','Storage','Screen','Brand_Acer','Brand_Alurin','Brand_Apple','Brand_Asus','Brand_Deep Gaming','Brand_Denver','Brand_Dynabook Toshiba','Brand_Gigabyte','Brand_HP','Brand_Innjoo','Brand_LG', 'Brand_Lenovo',
       'Brand_PcCom', 'Brand_Primux', 'Brand_Prixton', 'Brand_Razer',
       'Brand_Realme', 'Brand_Samsung', 'Brand_Thomson', 'Brand_Toshiba',
       'Brand_Vant', 'Cores']])

y = np.array(df['Price'])



x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=123)
#Creating the model and training it 
model = GradientBoostingRegressor(n_estimators=430)
model.fit(x_train,y_train)
predictions = model.predict(x_test)


#Evaluation metrics on test data to see how well the model is predicting the data



kf = KFold(n_splits=5,shuffle=True,random_state=42)
scores = cross_val_score(model,X,y,cv=kf,scoring='neg_mean_squared_error')
mse_scores = -scores
mean_mse = mse_scores.mean()
if __name__ == "__main__":
       print('Mean squared error',mean_squared_error(y_test,predictions))
       print('Mean absolute error',mean_absolute_error(y_test,predictions))
       print('Model score train set',model.score(x_train,y_train))
       print('Model score test set',model.score(x_test,y_test))
       print('Cross val score',scores)
       print('Cross val mse',mse_scores)
       print('Cross val mse mean',mean_mse)




def predict_laptops(brandname=0,storage=0,ram=0,screen=0,no_of_cores=0):
       brandnames_ = ['Brand_Acer','Brand_Alurin','Brand_Apple','Brand_Asus','Brand_Deep Gaming','Brand_Denver','Brand_Dynabook Toshiba','Brand_Gigabyte','Brand_HP','Brand_Innjoo','Brand_LG', 'Brand_Lenovo',
       'Brand_PcCom', 'Brand_Primux', 'Brand_Prixton', 'Brand_Razer','Brand_Realme', 'Brand_Samsung', 'Brand_Thomson', 'Brand_Toshiba','Brand_Vant']

       brandnames = []

       for i in brandnames_:
              x = brandnames_.index(i)
              i = i.split('_')
              brandnames.append(i[1])


    
       for j in range(len(brandnames)):
              if brandname == brandnames[j]:
                     brandnames[j] = 1
              else:
                     brandnames[j] = 0 

       features = [(ram/10),(storage/1000),(screen/10)]
       features.extend(brandnames)
       features.append(no_of_cores)
       features = np.array([features])
       return f'The predicted price of laptop is {int(((model.predict(features)) //10) * 10)} dollars'





        



