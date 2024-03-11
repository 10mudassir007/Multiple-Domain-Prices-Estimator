import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split,cross_val_score,KFold
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.linear_model import SGDRegressor

df = pd.read_csv('F:\Files\Portfolio\Multiple-Domain-Prices-Estimator\Housing1.csv')
#df['Area'] = df['Area'] /1000


#plt.show()

#imputer = SimpleImputer()
#imputer.fit_transform(df)

#plt.show()

encoder = OneHotEncoder()
encoded = encoder.fit_transform(df[['mainroad','guestroom','basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']])
df_encoded = pd.DataFrame(encoded.toarray(),columns=encoder.get_feature_names_out(['mainroad','guestroom','basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']))
df = df.drop(['mainroad','guestroom','basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus'],axis=1)

df = pd.concat([df,df_encoded],axis=1)

df['area'] /= 1000
X = np.array(df.drop('price',axis=1))
y = np.array(df['price'])

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)






model = SGDRegressor(loss='squared_error',penalty='l1',alpha=0.1,epsilon=0.1,eta0=0.01,max_iter=400,learning_rate='adaptive',random_state=42)
model.fit(x_train,y_train)

preds = model.predict(x_test)


if __name__ == "__main__":
     print('Mean squared error',mean_squared_error(y_test,preds))
     print('Mean absolute error',mean_absolute_error(y_test,preds))
     print('Model score train set',model.score(x_train,y_train))

     print('R2 score',r2_score(y_test,preds))



def adjusted_r2_score(y_true, y_pred, n, k):
    r2 = r2_score(y_true, y_pred)
    adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - k - 1))
    return adjusted_r2



def predict_houses(sqft=0,bedrooms=0,bathrooms=0,stories=0,parking=0,mainroad=0,guestroom=0,basement=0,hotwaterheating=0,airconditioning=0,prefarea=0,furnishingstatus=0):
        features = [sqft/1000,bedrooms,bathrooms,stories,parking]
        
        yes_no = ['No','Yes']
        list_to_add = []
        furnishingstatus_ = ['Furnished','Semi-furnished','Unfurnished']

        for i in range(len(yes_no)):
            if mainroad == yes_no[i]:
                 list_to_add.append(1)
            else:
                 list_to_add.append(0)
            
        for i in range(len(yes_no)):
            if guestroom== yes_no[i]:
                 list_to_add.append(1)
            else:
                 list_to_add.append(0)
            
        for i in range(len(yes_no)):
            if basement == yes_no[i]:
                 list_to_add.append(1)
            else:
                 list_to_add.append(0)

        for i in range(len(yes_no)):
            if hotwaterheating == yes_no[i]:
                 list_to_add.append(1)
            else:
                 list_to_add.append(0)
            
        for i in range(len(yes_no)):
            if airconditioning == yes_no[i]:
                 list_to_add.append(1)
            else:
                 list_to_add.append(0)
            
        for i in range(len(yes_no)):
            if prefarea == yes_no[i]:
                 list_to_add.append(1)
            else:
                 list_to_add.append(0)
        for i in range(len(furnishingstatus_)):
            if furnishingstatus == furnishingstatus_[i]:
                 list_to_add.append(1)
            else:
                 list_to_add.append(0)
        features.extend(list_to_add)


        features = np.array([features])
        prediction = model.predict(features)
        return f'The predicted price of house is {int((prediction//1000)*1000)}$ dollars'
        



