#!/usr/bin/env python
# coding: utf-8

# Internship_Id: H2HBABBA1022
# 
# House        :Monica
# 
# Name         :Sasi Kumar Reddy Thota

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")


# In[2]:


df1=pd.read_csv('H2HBABBA1022.csv') #data reading


# In[3]:


df1.shape


# In[4]:


df1.head()


# ### Slicing records with Clearing date Null into a separate DataFrame.(Train+Test)

# In[5]:


df1.isnull().sum()


# In[6]:


df1.isnull().mean()


# In[7]:


null_data = df1[df1['clear_date'].isnull()]
null_data.shape


# In[8]:


Test=pd.DataFrame(null_data)
Test.head()


# In[9]:


notnull_data=df1[df1['clear_date'].notnull()]


# In[10]:


notnull_data.shape


# In[11]:


Train=pd.DataFrame(notnull_data)
Train.shape


# ### Preprocessing

# In[12]:


Train_t=Train.T #finding Transpose matrix for interchanging rows and columns
Train_t.duplicated().sum()


# In[13]:


const_col=[x for x in Train.columns if Train[x].nunique()==1] #finding constant columns
print(const_col)


# In[14]:


Train.shape #before droping constant columns


# In[15]:


Train.drop(const_col,axis=1,inplace=True) #droping constant columns because no pattern will generated


# In[16]:


Train.shape #after droping constant columns


# #### Null Imputation

# In[17]:


Train.isnull().sum()


# In[18]:


Train.drop('area_business',axis=1,inplace=True) #above area_business is Null in all observations so we have to remove


# In[19]:


Train.shape #after removing area_business column 


# In[20]:


Train.isnull().sum()


# In[21]:


(Train['invoice_id'].isin(Train['doc_id'])).sum() #here we can see the out of 40000,39996 are same so we can conclude that both are same


# In[22]:


Train.invoice_id.fillna(Train['doc_id'],inplace=True) #filling invoice_id null values with doc_id because both are same


# In[23]:


Train.drop(['doc_id'],axis=1,inplace=True)#droping doc_id column because it is duplicated column


# In[24]:


Train.info()


# In[25]:


Train.info()


# In[26]:


Train.head()


# #### converting date columns to datetime format

# In[27]:


Train['clear_date']=pd.to_datetime(Train['clear_date'])
Train['posting_date']=pd.to_datetime(Train['posting_date'])
Train['due_in_date']=pd.to_datetime(Train['due_in_date'], format='%Y%m%d')
Train['document_create_date']=pd.to_datetime(Train['document_create_date.1'], format='%Y%m%d')
Train['document_create_date.1']=pd.to_datetime(Train['document_create_date.1'], format='%Y%m%d')
Train['baseline_create_date']=pd.to_datetime(Train['baseline_create_date'], format='%Y%m%d')
Train['buisness_year']=Train['buisness_year'].astype('int')


# In[28]:


Train.info()


# In[29]:


Train_t=Train.T#checking duplicate columns exists or not after converting datetime format
Train_t.duplicated().sum()


# In[30]:


Train_t.duplicated()#finding duplicate column


# In[31]:


Train


# In[32]:


Train.drop('document_create_date.1',axis=1,inplace=True)#deleting duplicate column


# In[33]:


print(Train.columns)#cross verifying whether duplicated variable deleted or not


# #### Addding Target column

# In[34]:


Train['delay']=Train['clear_date']-Train['due_in_date']
Train['delay']=Train['delay'].dt.days.astype('int')


# In[35]:


Train.drop('clear_date',axis=1,inplace=True) #we have to remove this clear_date column because we have to predict this in test


# #### Sorting the data in ascending order based on the Posting Date Column.(training on past data and testing on future data)

# In[36]:


Train.sort_values(by=['posting_date'],inplace=True)


# #### Splitting Main Train into train,validation and local test

# In[37]:


from sklearn.model_selection import train_test_split
x=Train.drop(['delay'],axis=1)
y=Train['delay']
x.shape,y.shape


# In[38]:


x_train,x_midtest,y_train,y_midtest=train_test_split(x,y,test_size=0.3,random_state=0,shuffle=False)  


# In[39]:


x_train.shape,x_midtest.shape,y_train.shape,y_midtest.shape


# In[40]:


x_val,x_test,y_val,y_test=train_test_split(x_midtest,y_midtest,test_size=0.5,random_state=0,shuffle=False)


# In[41]:


x_val.shape,x_test.shape,y_val.shape,y_test.shape


# ### Exploratory data analysis (EDA)

# In[42]:


sns.distplot(y_train)


# In[43]:


sns.boxplot(x=y_train)


# distribution of target column is approximately normldistribution only few outlayers are present but not a problem

# In[44]:


Train.info()


# In[45]:


x_train.rename({'document type': 'document_type'}, axis=1, inplace=True)#changing document type to document_type
x_val.rename({'document type': 'document_type'}, axis=1, inplace=True)
x_test.rename({'document type': 'document_type'}, axis=1, inplace=True)


# ### Feature Engineering

# In[46]:


x_train.head()


# In[47]:


x_train.info()


# In[48]:


from sklearn import preprocessing
encoder=preprocessing.LabelEncoder()


# In[49]:


def addCatagories(encoder,type_set,column):#function for adding new catogories of val,test sets
    diff=set(type_set[column])-set(encoder.classes_)
    for i in diff:
        encoder.classes_=np.append(encoder.classes_,i)
    return encoder.classes_


# In[50]:


def train_val_test_labelencoding(encoder,column,train=x_train,val=x_val,test=x_test):
    train[column]=encoder.fit_transform(train[column])
    encoder.classes_=addCatagories(encoder,val,column)#function for label encoding for val,train,test sets
    val[column]=encoder.fit_transform(val[column])
    encoder.classes_=addCatagories(encoder,test,column)
    test[column]=encoder.fit_transform(test[column])
    return train,val,test   


# In[51]:


x_train,x_val,x_test=train_val_test_labelencoding(encoder,'business_code')#function calls for label encoding
x_train,x_val,x_test=train_val_test_labelencoding(encoder,'cust_number')
x_train,x_val,x_test=train_val_test_labelencoding(encoder,'name_customer')
x_train,x_val,x_test=train_val_test_labelencoding(encoder,'invoice_currency')
x_train,x_val,x_test=train_val_test_labelencoding(encoder,'cust_payment_terms')
x_train,x_val,x_test=train_val_test_labelencoding(encoder,'document_type')


# In[52]:


print(x_train['business_code'].unique(),x_val['business_code'].unique(),x_test['business_code'].unique())


# In[53]:


print(x_train.info(),x_val.info(),x_test.info()) #checking whether objects are converted into int or float


# #### Extracting date,month and year from datetime columns

# In[54]:


def extract_date_month_year(column,train=x_train,val=x_val,test=x_test):#function for extracting date,month and year for every date time 
    train['day_of_'+column] = train[column].dt.day                      #columns in train,val,test data sets
    train['month_of_'+column] =train[column].dt.month
    train['year_of_'+column] =train[column].dt.year
    
    val['day_of_'+column] =val[column].dt.day
    val['month_of_'+column] =val[column].dt.month
    val['year_of_'+column] =val[column].dt.year

    test['day_of_'+column] =test[column].dt.day
    test['month_of_'+column] =test[column].dt.month
    test['year_of_'+column] =test[column].dt.year    


# In[55]:


extract_date_month_year('posting_date') #fuction calls for extracting date,month,year for every date time in train,val and test
extract_date_month_year('document_create_date')
extract_date_month_year('due_in_date')
extract_date_month_year('baseline_create_date')


# #### Droping datetime columns in train,val and test

# In[56]:


x_train.drop(x_train.loc[:,['posting_date','document_create_date','due_in_date','baseline_create_date']],axis=1,inplace=True)
x_val.drop(x_val.loc[:,['posting_date','document_create_date','due_in_date','baseline_create_date']],axis=1,inplace=True)
x_test.drop(x_test.loc[:,['posting_date','document_create_date','due_in_date','baseline_create_date']],axis=1,inplace=True)


# In[57]:


print(x_train.info(),x_val.info(),x_test.info()) #checking whether these are updated or not 


# In[58]:


colormap = plt.cm.RdBu
plt.figure(figsize=(30,15))
plt.title('Pearson Correlation of Features', y=1.05, size=30)
sns.heatmap(x_train.merge(y_train , on = x_train.index ).corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)


# In[59]:


x_train.drop(x_train.loc[:,['buisness_year']],axis=1,inplace=True)#business year is constant column hence have to drop
x_val.drop(x_val.loc[:,['buisness_year']],axis=1,inplace=True)
x_test.drop(x_test.loc[:,['buisness_year']],axis=1,inplace=True)


# In[60]:


x_train.shape #checking the whether the column is deleted


# ### Building Machine Learning Models

# #### LinearRegression

# In[61]:


from sklearn.linear_model import LinearRegression
linear_model = LinearRegression()
linear_model.fit(x_train,y_train)


# In[62]:


#Prediction on validation set
y_predict = linear_model.predict(x_val)


# In[63]:


#caluculating mean_squared_error between prediction of linear regression
from sklearn.metrics import mean_squared_error
mean_squared_error(y_val, y_predict, squared=False)


# In[64]:


print(y_val.head(),y_predict)


# #### Decision Tree Regressor

# In[65]:


from sklearn.tree import DecisionTreeRegressor
reg_model = DecisionTreeRegressor(random_state=0 , max_depth=3.2)


# In[66]:


reg_model.fit(x_train, y_train)


# In[67]:


y_predict_reg = reg_model.predict(x_val)


# In[68]:


mean_squared_error(y_val, y_predict_reg, squared=False)


# ### Prediction on x_test

# In[69]:


y_test_prediction = reg_model.predict(x_test)
mean_squared_error(y_test, y_test_prediction, squared=False)


# # Working on data set where clear_data is NULL(i.e Test)

# In[70]:


#Creating another Test as X_Test for preprocessing and all..
X_Test=Test.drop('clear_date',axis=1)
Test.drop('clear_date',axis=1,inplace=True)


# In[71]:


X_Test.info()


# In[72]:


#changing feature name document to document_type for easy operations
X_Test.rename({'document type': 'document_type'}, axis=1, inplace=True)


# In[73]:


#Finding number of null values for each feature
X_Test.isna().sum()


# In[74]:


#Before droping area_business
X_Test.shape


# In[75]:


#Droping area_business because all the observations are null hence there is no use
X_Test.drop(X_Test.loc[:,['area_business']],axis=1,inplace=True)


# In[76]:


#After droping area_business
X_Test.shape


# In[77]:


#These were the constant columns present in Train data set
const_columns=['posting_id', 'isOpen']
X_Test.drop(const_col,axis=1,inplace=True)


# #### converting date columns to datetime format

# In[78]:



X_Test['posting_date']=pd.to_datetime(X_Test['posting_date'])
X_Test['document_create_date']=pd.to_datetime(X_Test['document_create_date'], format='%Y%m%d')
X_Test['due_in_date']=pd.to_datetime(X_Test['due_in_date'], format='%Y%m%d')
X_Test['document_create_date.1']=pd.to_datetime(X_Test['document_create_date.1'], format='%Y%m%d')
X_Test['baseline_create_date']=pd.to_datetime(X_Test['baseline_create_date'], format='%Y%m%d')
X_Test['buisness_year']=X_Test['buisness_year'].astype('int')


# In[79]:


X_Test.info()


# In[80]:


#sorting by posting_date
X_Test.sort_values(by=['posting_date'],inplace=True)


# In[81]:


X_Test_t=X_Test.T#checking duplicate columns exists or not after converting datetime format
X_Test_t.duplicated().sum()


# In[ ]:





# In[82]:


X_Test_t.duplicated()#finding duplicate column


# In[83]:


X_Test.drop(['document_create_date.1','doc_id'],axis=1,inplace=True)#deleting duplicate column


# In[84]:


print(X_Test.shape)#cross verifying whether duplicated variable deleted or not


# ### Feature Engineering

# In[85]:


from sklearn import preprocessing
encoder=preprocessing.LabelEncoder()


# In[86]:


X_Test['business_code']=encoder.fit_transform(X_Test['business_code'])
X_Test['cust_number']=encoder.fit_transform(X_Test['cust_number'])
X_Test['name_customer']=encoder.fit_transform(X_Test['name_customer'])
X_Test['invoice_currency']=encoder.fit_transform(X_Test['invoice_currency'])
X_Test['cust_payment_terms']=encoder.fit_transform(X_Test['cust_payment_terms'])
X_Test['document_type']=encoder.fit_transform(X_Test['document_type'])


# In[87]:


X_Test.info()


# In[88]:


#function for extracting date month and year
def Extract_date_month_year(column):
    X_Test['day_of_'+column] = X_Test[column].dt.day
    X_Test['month_of_'+column] =X_Test[column].dt.month
    X_Test['year_of_'+column] =X_Test[column].dt.year


# In[89]:


Extract_date_month_year('posting_date')
Extract_date_month_year('document_create_date')
Extract_date_month_year('due_in_date')
Extract_date_month_year('baseline_create_date')


# In[90]:


X_Test.drop(X_Test.loc[:,['posting_date','document_create_date','due_in_date','baseline_create_date']],axis=1,inplace=True)


# In[91]:


X_Test.shape


# In[92]:


X_Test.info()


# In[93]:


#droping Business year from X_Test
X_Test.drop('buisness_year',axis=1,inplace=True)


# In[94]:


#number of features in train and test are same
print(x_train.shape,X_Test.shape)


# ## Prediction on Final Test

# In[95]:


End_result = reg_model.predict(X_Test)


# In[96]:


End_result = pd.Series(End_result,name='Delay')#adding Delay to Test


# In[97]:


Test.reset_index(drop=True,inplace=True)


# In[98]:


Final = Test.merge(End_result , on = X_Test.index )
Final


# In[99]:


#Ading PDP to Final Test Set 
Final['predicted_payment_date(PDP)'] = pd.to_datetime(Test['due_in_date'], format='%Y%m%d') + pd.to_timedelta(pd.np.ceil(Final.Delay), unit="D")


# In[100]:


#Adding Aging bucket to the Final Test Set
bins = [-1,0, 15,30,45,60]
labels = ['0-15', '16-30', '31-45', '46-60','>60']
Final['Aging_bucket'] = pd.cut(Final.Delay, bins, labels = labels,include_lowest = True)


# In[101]:


Final


# In[ ]:




