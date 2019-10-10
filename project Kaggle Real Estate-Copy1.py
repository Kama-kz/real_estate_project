#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import csv
import sklearn as sk


# In[3]:


kaggle=pd.read_csv(r"C:\Users\kamal\Desktop\104\project\kaggle 1\hp/train.csv")


# In[3]:


kaggle.head()


# In[4]:


na=kaggle.isna().sum()


# In[5]:


len(kaggle)


# In[6]:


kaggle.describe()


# In[7]:


kaggle.info()


# In[8]:


na=pd.DataFrame(na)


# In[9]:


#looking for NA in data and calculating the %% of empty cells in a column. Above 80% empty cells = remove the column
empty=[]

for i in range(len(na)):
    if na.iloc[i,0]>0:
        f=na.iloc[i,0]
        c="{:10.2f}".format(f/len(kaggle)*100)
        print(na.index[i], "___",f,c,"%")
        if float(c)>80:
            empty.append(na.index[i])


# In[10]:


#more than 80% of the cells are empty cells
empty


# In[11]:


kaggle=kaggle.drop(empty,axis=1)


# In[12]:


kaggle.head(9)


# In[13]:


#find average proportion of LotFrontage to LotArea so we can use the ratio to fill in empty cells in LotFrontage
aver=[]
for i in range(len(kaggle)): 
    if kaggle.iloc[i]["LotFrontage"] == "nan":
        continue
    a=kaggle.iloc[i]["LotFrontage"]/kaggle.iloc[i]["LotArea"]
    aver.append(a)


# In[14]:


aver=pd.DataFrame(aver).dropna()


# In[15]:


s=aver.sum()
s


# In[16]:


aver=s/(1460-259)
aver


# In[17]:


kaggle["LotFrontage"].fillna(0,inplace=True)


# In[18]:


kaggle1=kaggle.copy()


# In[19]:


#insert in zeroes new calculated values
for i in range(len(kaggle1)):
    if kaggle1.iloc[i]["LotFrontage"]==0:
        kaggle1.at[i,"LotFrontage"]=aver*kaggle1.iloc[i]["LotArea"]


# In[20]:


kaggle1["LotFrontage"].head(13)


# In[21]:


kaggle1["MasVnrType"]=kaggle1["MasVnrType"].fillna("None")


# In[22]:


kaggle1["MasVnrType"].isna().sum()


# In[23]:


kaggle1.iloc[529]["MasVnrType"]


# In[24]:


kaggle1["MasVnrArea"]=kaggle1["MasVnrArea"].fillna(0)


# In[25]:


kaggle1["MasVnrArea"].isnull().sum()


# In[26]:


kaggle1["MasVnrArea"].head()


# In[27]:


kaggle1.iloc[234]["MasVnrArea"]


# In[28]:


#replace NA in the basement&garage categories with None category as NA is not missing data but just no basement, no garage
kaggle1["BsmtQual"]=kaggle1["BsmtQual"].fillna("None")


# In[29]:


kaggle1["BsmtCond"]=kaggle1["BsmtCond"].fillna("None") 


# In[30]:


kaggle1["BsmtExposure"]=kaggle1["BsmtExposure"].fillna("None") 


# In[31]:


kaggle1["BsmtFinType1"]=kaggle1["BsmtFinType1"].fillna("None") 


# In[32]:


kaggle1["BsmtFinType2"]=kaggle1["BsmtFinType2"].fillna("None") 


# In[33]:


kaggle1["Electrical"]=kaggle1["Electrical"].fillna("Mix") 


# In[34]:


kaggle1["GarageType"]=kaggle1["GarageType"].fillna("None") 


# In[35]:


kaggle1["FireplaceQu"]=kaggle1["FireplaceQu"].fillna("None") 


# In[36]:


min1=kaggle1.min()["GarageYrBlt"]
min1


# In[37]:


kaggle1["GarageYrBlt"]=kaggle1["GarageYrBlt"].fillna(min1)


# In[38]:


kaggle1["GarageFinish"]=kaggle1["GarageFinish"].fillna("None") 


# In[39]:


kaggle1["GarageQual"]=kaggle1["GarageQual"].fillna("None") 


# In[40]:


kaggle1["GarageCond"]=kaggle1["GarageCond"].fillna("None") 


# In[41]:


na2=kaggle1.isna().sum()


# In[42]:


#checking that there is no more NAs in the data
for i in range(len(na2)):
    if na2.iloc[i]>0:
        print(na2.index[i])


# In[43]:


na2.head()


# In[44]:


kaggle_ob=kaggle1.select_dtypes(include=['object'])


# In[45]:


kaggle_ob.head()


# In[46]:


import seaborn as sns
from scipy import stats
from scipy.stats import iqr


# In[47]:


kaggle_num=kaggle1.select_dtypes(exclude=['object'])
kaggle_num.head()


# In[48]:


col=kaggle_num.columns.values.tolist()


# In[49]:


out1=[2,3,9,12,13]
out2=[]
for i in out1:
    out2.append(col[i])
out2


# In[50]:


kaggle1[out2].head(2)


# In[51]:


for i in range(len(out2)):
    z1 = np.abs(stats.zscore(kaggle1[out2[i]].to_numpy()))
    z2=np.where(z1>3)
    for n in z2:
        kaggle1.loc[n,out2[i]]=2*kaggle1[out2[i]].mean()


# In[52]:


sns.boxplot(x=kaggle_num[col[3]])


# In[53]:


sns.boxplot(x=kaggle1[col[3]])


# In[54]:


#kaggle1 is clean of NAs and outliers but before coding categorical values
kaggle2=kaggle1.copy()


# In[55]:


kaggle_ob.columns.values


# In[56]:


for i in kaggle_ob.columns.values:
    kaggle2=pd.get_dummies(kaggle2, columns=[i])


# In[57]:


kaggle2.head()


# In[58]:


kaggle2.select_dtypes(include=['object']).head(2)


# In[59]:


kaggle2.info()


# In[60]:


#looking for outliers
z = np.abs(stats.zscore(kaggle2,axis=1))


# In[61]:


dd=np.where(z > 3)
np.shape(dd)


# In[62]:


kaggle3=kaggle2.copy()


# In[63]:


y=kaggle3["SalePrice"]


# In[64]:


#y.column=["SalePrice"]


# In[65]:


#y=pd.DataFrame(y)


# In[66]:


x=kaggle3.drop("SalePrice",axis=1)
x.info()


# In[67]:


np.where(np.isnan(x))


# In[68]:


x.columns[8]


# In[69]:


x.head(2)


# In[70]:


from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
x1=scaler.fit_transform(x)


# In[71]:


x1=pd.DataFrame(x1)
x1.columns=x.columns


# In[72]:


x1.head(2)


# In[73]:


x1.shape


# In[74]:


y.shape


# In[75]:


y.head(2)


# In[76]:


from sklearn.model_selection import train_test_split

x_train1, x_test, y_train1, y_test = train_test_split(x1, y, test_size=0.25)


# In[77]:


x_train1.shape


# In[78]:


x_test.shape


# In[79]:


y_train1.head(2)


# In[80]:


x_train1.head(2)


# In[81]:


x_test.head(2)


# In[82]:


x_train, x_val, y_train, y_val = train_test_split(x_train1, y_train1, test_size=0.2)


# # Start applying different classifiers

# # 1.LINEAR REGRESSION 

# In[83]:


from sklearn.linear_model import LinearRegression


# In[84]:


np.where(np.isnan(x_train))


# In[85]:


lr = LinearRegression()


# In[86]:


from sklearn.feature_selection import RFECV, RFE


# In[87]:


rfecv = RFECV(lr, step=1, cv=5, scoring="neg_mean_squared_error")


# In[88]:


res=rfecv.fit(x_train, y_train)


# In[89]:


rfecv.n_features_


# In[90]:


res.score(x_train,y_train)


# In[91]:


res.score(x_test,y_test)


# In[92]:


pred1 = res.predict(x_test)
pred11 = res.predict(x_val)


# In[93]:


from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_squared_error


# In[94]:


(pred1 <= 0).all() 


# In[95]:


np.sqrt(mean_squared_log_error(y_test, pred1))


# In[96]:


score={}
for i in range(9,30):
    rfe = RFE(lr, i, step=1)
    res1=rfe.fit(x_train, y_train)
    pred1 = res1.predict(x_test)
    a=np.sqrt(mean_squared_error(y_test, pred1))
    score.update({i:a})
print("Number of features:", min(score, key=score.get),"//", "RMSE:", min(score.values()))   


# In[97]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[98]:


pos = np.arange(len(list(score.keys())))
plt.bar(pos, score.values(), align='center', alpha=0.9)
plt.xticks(pos, list(score.keys()))
plt.ylabel('root_mean_squared_error')
plt.title('# of features')
plt.show()


# In[99]:


rfe1 = RFE(lr, 28, step=1)
res2=rfe1.fit(x_train, y_train)
pred2 = res2.predict(x_test)
np.sqrt(mean_squared_log_error(y_test, pred2))


# In[100]:


res2.score(x_train,y_train)


# In[101]:


res2.score(x_test,y_test)


# In[102]:


# there are more than 4300 participants in this kaggle competition. Our result is bad and gives us the bottom 10% rank...


# # 2. RIDGE LINEAR REGRESSION

# In[103]:


from sklearn.linear_model import RidgeClassifierCV


# In[104]:


ridge = RidgeClassifierCV(alphas=[1e-3, 1e-2, 1e-1, 1,2,5,10,15,20,25,30,35,40,45,50,75,100,200],scoring="neg_mean_squared_error").fit(x_train, y_train)


# In[105]:


ridge.score(x_train,y_train)


# In[106]:


ridge.score(x_test,y_test)


# In[107]:


pred3 = ridge.predict(x_test)
pred33 = ridge.predict(x_val)
np.sqrt(mean_squared_log_error(y_test, pred3))


# In[108]:


#the result is still not good, but a bit better (going down). To be in top 1% in the competition we need to have at least 0.106.


# # 3. Lasso Regression

# In[109]:


from sklearn.linear_model import LassoCV


# In[110]:


lasso = LassoCV(cv=5).fit(x_train, y_train)


# In[111]:


lasso.score(x_train,y_train)


# In[112]:


lasso.score(x_test,y_test)


# In[113]:


pred4 = lasso.predict(x_test)
pred44 = lasso.predict(x_val)
np.sqrt(mean_squared_log_error(y_test, pred4))


# In[114]:


#the result is much better. We have got now apprx. top 37% in the kaggle competition.


# # 4. Elastic Net Regression

# In[115]:


from sklearn.linear_model import ElasticNetCV


# In[116]:


elastic = ElasticNetCV(cv=5).fit(x_train, y_train)


# In[117]:


elastic.score(x_train,y_train)


# In[118]:


elastic.score(x_test,y_test)


# In[119]:


pred5 = elastic.predict(x_test)
pred55 = elastic.predict(x_val)
np.sqrt(mean_squared_log_error(y_test, pred5))


# In[120]:


#not the best result. Not the best algorithm for this data set. Elastic regression generally works well when we have a big dataset. Gives us the bottop 5% rank.


# # now let_s try one advanced technique

# # 5. Gradient Boosting Algorithm

# In[121]:


from sklearn.ensemble import GradientBoostingRegressor


# In[122]:


gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, loss='ls').fit(x_train, y_train) 


# In[123]:


gb.score(x_train,y_train)


# In[124]:


gb.score(x_test,y_test)


# In[125]:


pred6 = gb.predict(x_test)
pred66 = gb.predict(x_val)
np.sqrt(mean_squared_log_error(y_test, pred6))


# In[126]:


#not bad result. We have got now apprx. top 33% in the kaggle competition. But there is more room to grow...


# # 6. RandomForest

# In[127]:


from sklearn.ensemble import RandomForestRegressor


# In[128]:


rf=RandomForestRegressor(n_estimators=100).fit(x_train,y_train)  


# In[129]:


rf.score(x_train,y_train)


# In[130]:


rf.score(x_test,y_test)


# In[131]:


pred7 = rf.predict(x_test)
pred77 = rf.predict(x_val)
np.sqrt(mean_squared_log_error(y_test, pred7))


# In[132]:


#result is close to the best previously achieved rersult. But could not beat it.


# In[133]:


#pip install xgboost


# # 7. XGBoost

# In[134]:


from xgboost import XGBRegressor


# In[135]:


xg=XGBRegressor().fit(x_train,y_train) 


# In[136]:


xg.score(x_train,y_train)


# In[137]:


xg.score(x_test,y_test)


# In[138]:


pred8 = xg.predict(x_test)
pred88 = xg.predict(x_val)
np.sqrt(mean_squared_log_error(y_test, pred8))


# In[139]:


# so far the best result on level 1.


# # now let_s try to do level 2 - "STACKING" to impove our result

# In[140]:


prediction=np.stack((pred1, pred3,pred4,pred5,pred6,pred7,pred8))


# In[141]:


prediction1=pd.DataFrame(prediction).T


# In[142]:


prediction1.head()


# In[143]:


prediction_train=np.stack((pred11, pred33,pred44,pred55,pred66,pred77,pred88))


# In[144]:


prediction_train1=pd.DataFrame(prediction_train).T


# In[145]:


prediction_train1.head(2)


# # will use again Gradient Boosting as meta-clasifier

# In[146]:


gb1 = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, loss='ls').fit(prediction_train1, y_val)


# In[147]:


gb1.score(prediction_train1,y_val)


# In[148]:


gb1.score(prediction1,y_test)


# In[149]:


pred8 = gb1.predict(prediction1)
np.sqrt(mean_squared_log_error(y_test, pred8))


# In[150]:


#unfortunately it did not improve the best result...


# # let_s try linear regression as meta-clasifier

# In[151]:


lr1 = lr.fit(prediction_train1, y_val)


# In[152]:


lr1.score(prediction_train1,y_val)


# In[153]:


lr1.score(prediction1,y_test)


# In[154]:


pred9 = lr1.predict(prediction1)
np.sqrt(mean_squared_log_error(y_test, pred9))


# In[155]:


#it helped. Our result is now better. Apprx. top 6.5% rank. // unstable //


# # let_s try lasso regression as meta-clasifier

# In[156]:


lasso1 = LassoCV(cv=5).fit(prediction_train1, y_val)


# In[157]:


lasso1.score(prediction_train1,y_val)


# In[158]:


lasso1.score(prediction1,y_test)


# In[159]:


pred10 = lasso1.predict(prediction1)
np.sqrt(mean_squared_log_error(y_test, pred10))


# In[160]:


#again better. Apprx. top 5.6% rank.


# # try RFECV with linear regression

# In[161]:


res1=rfecv.fit(prediction_train1, y_val)


# In[162]:


res1.n_features_


# In[163]:


res1.score(prediction_train1,y_val)


# In[164]:


res1.score(prediction1,y_test)


# In[165]:


pred11 = res1.predict(prediction1)
np.sqrt(mean_squared_log_error(y_test, pred11))


# In[166]:


#good result. 


# # RF as meta-clasifier

# In[167]:


rf1=RandomForestRegressor(n_estimators=100).fit(prediction_train1, y_val)  


# In[168]:


rf1.score(prediction_train1,y_val)


# In[169]:


rf1.score(prediction1,y_test)


# In[170]:


pred12 = rf1.predict(prediction1)
np.sqrt(mean_squared_log_error(y_test, pred12))


# In[171]:


#could not improve the result.


# # 7. XGBoost as meta-clasifier

# In[172]:


xg1=XGBRegressor().fit(prediction_train1, y_val) 


# In[173]:


xg1.score(prediction_train1,y_val)


# In[174]:


xg1.score(prediction1,y_test)


# In[175]:


pred13 = xg1.predict(prediction1)
np.sqrt(mean_squared_log_error(y_test, pred13))


# # simple averaging predictions

# In[176]:


ad=[0,1,3]
prediction=prediction1.drop(columns=ad)


# In[177]:


prediction.head(2)


# In[178]:


prediction[7]=prediction[6]
prediction[8]=prediction[4]


# In[179]:


prediction.head(2)


# In[180]:


pred_mean=prediction.mean(axis=1)


# In[181]:


pred_mean.head()


# In[182]:


np.sqrt(mean_squared_log_error(y_test, pred_mean))


# In[183]:


pred_mean1=prediction1.mean(axis=1)


# In[184]:


pred_mean1.head()


# In[185]:


np.sqrt(mean_squared_log_error(y_test, pred_mean1))


# In[ ]:




