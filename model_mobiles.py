#IMPORTING NECESSARY LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,cross_val_score,KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score


#LOADING DATA , THE CSV FILE SHOULD BE IN THE SAME LOCATION AS THE TERMINAL LOCATION OR CHANGE THE LOCATION OF THE TERMINAL TO WHERE THE FILE IS SAVED USING cd commands
df = pd.read_csv('Multiple-Domain-Prices-Estimator/cleanedmobiles.csv').head(1339)
df = df[['Brand_10.or',
       'Brand_Alcatel', 'Brand_Apple',
       'Brand_Asus',  'Brand_BlackBerry', 'Brand_Comio',
       'Brand_Coolpad', 'Brand_Gionee', 'Brand_Google', 'Brand_HP',
       'Brand_HTC', 'Brand_Homtom', 'Brand_Honor', 'Brand_Huawei',
       'Brand_InFocus', 'Brand_Infinix', 'Brand_Intex', 'Brand_Itel',
       'Brand_Jio', 'Brand_Jivi', 'Brand_Karbonn', 'Brand_Kult', 'Brand_LG',
       'Brand_Lava', 'Brand_LeEco', 'Brand_Lenovo', 'Brand_Lephone',
       'Brand_Lyf', 'Brand_M-tech', 'Brand_Meizu', 'Brand_Micromax',
       'Brand_Microsoft', 'Brand_Mobiistar', 'Brand_Motorola', 'Brand_Nokia',
       'Brand_Nubia', 'Brand_Nuu Mobile', 'Brand_OnePlus', 'Brand_Onida',
       'Brand_Oppo', 'Brand_Panasonic', 'Brand_Phicomm', 'Brand_Philips',
       'Brand_Poco', 'Brand_Razer', 'Brand_Reach', 'Brand_Realme',
       'Brand_Samsung', 'Brand_Sansui', 'Brand_Smartron', 'Brand_Sony',
       'Brand_Spice', 'Brand_Swipe', 'Brand_TCL', 'Brand_Tambo', 'Brand_Tecno',
       'Brand_Videocon', 'Brand_Vivo', 'Brand_Xiaomi', 'Brand_Xolo',
       'Brand_ZTE', 'Brand_Zen', 'Brand_Ziox', 'Brand_Zopo',
 'Brand_iBall', 'Brand_mPhone','Screen size (inches)', 'Resolution x','Price', 'Resolution y', 'Processor',
       'RAM (GB)', 'Internal storage (GB)', 'Rear camera', 'Front camera','Battery capacity (mAh)']]
df[['Battery capacity (mAh)','Resolution x' , 'Resolution y']] = df[['Battery capacity (mAh)','Resolution x' , 'Resolution y']] / 1000
df[['RAM (GB)']] = df[['RAM (GB)']] /100
df[['Internal storage (GB)']] /= 1000
df[['Rear camera','Front camera']] /= 10
X = np.array(df[['Brand_Apple',
       'Brand_Asus',  'Brand_BlackBerry', 
       'Brand_Coolpad', 'Brand_Gionee', 'Brand_Google', 'Brand_Honor', 'Brand_Huawei',
       'Brand_InFocus', 'Brand_Infinix', 'Brand_Intex',
       'Brand_Lenovo',  'Brand_Motorola', 'Brand_Nokia',
       'Brand_Nubia', 'Brand_OnePlus','Brand_Oppo', 'Brand_Realme',
       'Brand_Samsung', 'Brand_Sony','Brand_Swipe', 'Brand_Tecno',
       'Brand_Videocon', 'Brand_Vivo', 'Brand_Xiaomi', 'Brand_Xolo',
       'Brand_ZTE', 'Brand_Zen', 'Brand_Zopo','Brand_iBall','Battery capacity (mAh)','Screen size (inches)', 'Resolution x', 'Resolution y', 'Processor',
       'RAM (GB)', 'Internal storage (GB)', 'Rear camera', 'Front camera']])

y = np.array(df['Price'])


x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


#Creating the model and training it 


model = DecisionTreeRegressor(random_state=42,criterion='squared_error', max_depth=6, max_features=0.7)
model.fit(x_train,y_train)
predictions = model.predict(x_test)


if __name__ == "__main__":
       kf = KFold(n_splits=5,shuffle=True,random_state=42)
       scores = cross_val_score(model,X,y,cv=kf,scoring='neg_mean_squared_error')
       mse_scores = -scores
       mean_mse = mse_scores.mean()

#Evaluation metrics on test data to see how well the model is predicting the data
       print('Mean squared error',mean_squared_error(y_test,predictions))
       print('Mean absolute error',mean_absolute_error(y_test,predictions))
       print('Model score train set',model.score(x_train,y_train))
       print('Model score test set',model.score(x_test,y_test))
       print('Cross val score',scores)
       print('Cross val mse',mse_scores)
       print('Cross val mse mean',mean_mse)


def adjusted_r2_score(y_true, y_pred, n, k):
    r2 = r2_score(y_true, y_pred)
    adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - k - 1))
    return adjusted_r2

#print('Adjusted R2 score',adjusted_r2_score(y_test,predictions,len(y_test),(len(df.columns)-1)))

def predict_mobiles(brandname,battery,screensize,resolutionx,resolutiony,processorcores,ram,storage,rearcamera,frontcamera):
       brandnames_ = ['Brand_Apple',
       'Brand_Asus',  'Brand_BlackBerry', 
       'Brand_Coolpad', 'Brand_Gionee', 'Brand_Google', 'Brand_Honor', 'Brand_Huawei',
       'Brand_InFocus', 'Brand_Infinix', 'Brand_Intex',
       'Brand_Lenovo',  'Brand_Motorola', 'Brand_Nokia',
       'Brand_Nubia', 'Brand_OnePlus','Brand_Oppo', 'Brand_Realme',
       'Brand_Samsung', 'Brand_Sony','Brand_Swipe', 'Brand_Tecno',
       'Brand_Videocon', 'Brand_Vivo', 'Brand_Xiaomi', 'Brand_Xolo',
       'Brand_ZTE', 'Brand_Zen', 'Brand_Zopo','Brand_iBall']
    
       brandname = f'Brand_{brandname}'

       for i in range(len(brandnames_)):
              if brandname == brandnames_[i]:
                     brandnames_[i] = 1
              else:
                     brandnames_[i] = 0
       features = brandnames_
       features_ = [battery/1000,screensize,resolutionx/1000,resolutiony/1000,processorcores,ram/100,storage/100,rearcamera/10,frontcamera/10]
       features += features_
       features = np.array([features])

       return f'The predicted price of mobile is {(int((model.predict(features)*3.37)//10)*10)} rupees '




