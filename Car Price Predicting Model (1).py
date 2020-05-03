#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
sns.set()
import pandas as pd
import statsmodels.api as sm 
from sklearn.linear_model import LinearRegression


# ## Importing Data 

# In[2]:


cardata = pd.read_csv('1.04. Real-life example.csv')


# In[3]:


cardata 


# In[4]:


cardata.describe(include='all')


# **A Few Observations**
# 
# **1) PRESENCE OF OUTLIERS IN SOME FEATURES:** Outliers are present in the distributions of a few features (Price, Mileage, EngineV, Year). The most significant indicators being
# a) The distance of the maximum. 
# b) Minimum samples from the mean. 
# ***Solution: The outliers will be removed.***
# 
# **2) LITTLE OR NO VARIATION IN 'REGISTRATION' FIELD:** Registration seems to be yes for almost all datasets.
# 
# **3) UNIQUE NUMBER OF SAMPLES IN THE 'MODEL' FIELD:** The number of unique Cars models(Too High Variance) is too many for a regression model.
# ***Solution: For performance sakes, we will consider all other feature except Car models.***

# In[5]:


cardata =cardata.drop(['Model'], axis=1)


# In[6]:


cardata.hist()


# The diagram above shows the distribution of the numerical features. its shows that this features are affected by outliers 

# ## Data Preprocessing

# **Removing Outliers**

# In[7]:


data_no_outliers = cardata[cardata['Price']< cardata['Price'].quantile(0.95)]
data_no_outliers = data_no_outliers[data_no_outliers['EngineV']< 6.6]
data_no_outliers = data_no_outliers[data_no_outliers['Mileage']< data_no_outliers['Mileage'].quantile(0.99)]
data_no_outliers = data_no_outliers[data_no_outliers['Year']>data_no_outliers['Year'].quantile(0.03)]
data_no_outliers.describe(include='all')


# In[8]:


data_no_outliers.hist()# AfterOutliers have been removed in Price 


# **Dealing With Missing Values**

# In[9]:


data_no_outliers.isnull().sum()


# We can see we dont have any missing values 

# **Checking OLS**

# In[10]:


f, (plt1, plt2, plt3) = plt.subplots(1,3, sharey=True, figsize = [20, 3])
plt1.scatter(data_no_outliers['EngineV'],data_no_outliers['Price'] )
plt1.set_title('Price Vs EngineV')
plt2.scatter(data_no_outliers['Mileage'],data_no_outliers['Price'] )
plt2.set_title('Price Vs Mileage')
plt3.scatter(data_no_outliers['Year'],data_no_outliers['Price'] )
plt3.set_title('Year Vs Mileage')


# The High level view shows thats the relationship of features are not linear to Price. To use a Linear regression in this type of case we have to convert the features with log 

# In[11]:


Price = np.log(data_no_outliers['Price'])


data_no_outliers['Pricelog'] =Price


# In[12]:


data_no_outliers


# In[13]:


f, (plt1, plt2, plt3) = plt.subplots(1,3, sharey=True, figsize = [20, 3])
plt1.scatter(data_no_outliers['EngineV'],data_no_outliers['Pricelog'] )
plt1.set_title('Price Vs EngineV')
plt2.scatter(data_no_outliers['Mileage'],data_no_outliers['Pricelog'] )
plt2.set_title('Price Vs Mileage')
plt3.scatter(data_no_outliers['Year'],data_no_outliers['Pricelog'] )
plt3.set_title('Price Vs Year')


# **Checking for MultiCollinearity**

# In[14]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
features = data_no_outliers[['Year', 'Mileage', 'EngineV']]
vif = pd.DataFrame()
vif['VLF'] = [variance_inflation_factor(features.values, i) for i in range(features.shape[1])]
vif['features'] = features.columns


# In[15]:


vif


# We will drop the Year column due to a high VIF of above 10. This shows it is correlated with the other variables 

# In[16]:


data_no_outliers_collinearity = data_no_outliers.drop(['Year'], axis=1)


# **Dealing with the Categorical Features** 

# In[17]:



dummydata_no_outliers=pd.get_dummies(data_no_outliers_collinearity, drop_first= True)


# In[18]:


dummydata_no_outliers


# In[19]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
features = dummydata_no_outliers[['Mileage','Brand_BMW','Brand_Mercedes-Benz','Brand_Mitsubishi','Brand_Renault','Brand_Toyota','Brand_Volkswagen','Body_hatch','Body_other','Body_sedan','Body_vagon','Body_van','Engine Type_Gas','Engine Type_Other','Engine Type_Petrol','Registration_yes']]
vif = pd.DataFrame()
vif['VLF'] = [variance_inflation_factor(features.values, i) for i in range(features.shape[1])]
vif['features'] = features.columns


# In[20]:


vif


# # Linear Regression Model

# In[21]:


Finaldata=dummydata_no_outliers[['Mileage','Brand_BMW','Brand_Mercedes-Benz','Brand_Mitsubishi','Brand_Renault','Brand_Toyota','Brand_Volkswagen','Body_hatch','Body_other','Body_sedan','Body_vagon','Body_van','Engine Type_Gas','Engine Type_Other','Engine Type_Petrol','Registration_yes','EngineV', 'Pricelog']]



# In[22]:


Finaldata


# In[23]:


y = Finaldata['Pricelog']
x = Finaldata.drop(['Pricelog'], axis = 1)


# **Scale the data**

# In[24]:


from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler()
scaler.fit(x)
x_scaled=scaler.transform(x)


# In[25]:


from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test=train_test_split(x_scaled,y, test_size=0.2, random_state= 42)


# In[26]:


reg = LinearRegression()
reg.fit(x_train, y_train)


# In[27]:


y_hat =reg.predict(x_train)


# In[28]:


y_hat


# In[29]:


plt.scatter(y_train, y_hat, alpha=0.2)
plt.xlabel('y_train', fontweight='bold')
plt.ylabel('y_hat', fontweight= 'bold')
plt.xlim(6,11.5)
plt.ylim(6,11.5)
plt.title("Test Plot", fontweight='bold')


# **Residual plot**

# In[30]:


sns.distplot(y_hat- y_train)
plt.title("Residual plot", fontweight ="bold")


# The prediciton is generally fine, However it more likely to overestimate the Price than underestimate it, due to the distribution 

# **Weights and bias**

# In[31]:


reg.score(x_train, y_train)


# In[32]:


reg.intercept_


# In[33]:


reg.coef_


# In[34]:


reg_summary = pd.DataFrame(x.columns.values, columns=['Features'])
reg_summary['Weights'] = reg.coef_
reg_summary


# ## Testing

# In[35]:


y_hat_test = reg.predict(x_test)


# In[36]:


plt.scatter(y_test, y_hat_test, alpha=0.2)
plt.xlabel('Actual', fontweight='bold', fontsize=18) 
plt.ylabel('Predicted', fontweight='bold', fontsize=18)
plt.title('Actual Vs Predicted')
plt.show()


# In[37]:


sns.distplot(y_hat_test-y_test)


# We have to convert the Pricelog back to normal
# 

# In[38]:


FirstModelPrice =pd.DataFrame(np.exp(y_hat_test), columns = ['Prediction'])
FirstModelPrice.head()
FirstModelPrice['Actual'] = np.exp(y_test.reset_index(drop=True))
FirstModelPrice['Residual'] = FirstModelPrice['Actual'] -FirstModelPrice['Prediction']
FirstModelPrice['%'] = FirstModelPrice['Residual']*100 / FirstModelPrice['Actual']


# In[39]:


FirstModelPrice.round(2)


# In[40]:


FirstModelPrice.describe()


# ## Improving Our Model
# After seeing the stats of the first model. It is seen that the model is good, however it can still be improved. We will thereby bring back the feature 'Year' that was previously dropped.

# In[41]:


FeatureScaling=pd.get_dummies(data_no_outliers, drop_first= True)
plt.figure(figsize=(17,10))
cor = FeatureScaling.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()


# From the Heatmap, we can see that some of the fields are not necessarily correlated with the Price and as such we might as well drop them.

# In[42]:


cor_target =abs(cor['Pricelog'])
relevant_features = cor_target[cor_target>0.5]


# In[43]:


relevant_features


# We can notice here than the highest features that the bigest corelations exists in Year, Then registration_yes, then Mileage 

# In[44]:


print(FeatureScaling[["Mileage","Year"]].corr())
print(FeatureScaling[["Mileage","Registration_yes"]].corr())
print(FeatureScaling[["Year","Registration_yes"]].corr())


# We already know that Year correlates with one of the other variables, so we try to correlate the other features to know which feature corrleates with which. We discover that the Year correlates with Mileage, makes sense. We will in this case keep 'Year' Since its more correlated than Mileage to Price.
# 
# 
# The other fields are dummy data will be added. As the R-square was a lot better after the where added. Now the other steps will be carried to get a better model 
# 
# 

# In[45]:


x_afterscaling=FeatureScaling[['Year','Brand_BMW','Brand_Mercedes-Benz','Brand_Mitsubishi','Brand_Renault','Brand_Toyota','Brand_Volkswagen','Body_hatch','Body_other','Body_sedan','Body_vagon','Body_van','Engine Type_Gas','Engine Type_Other','Engine Type_Petrol','Registration_yes']]
y_aftersaling=FeatureScaling[['Pricelog']]


# In[46]:


from sklearn.model_selection import train_test_split 
x_second_train, x_second_test, y_second_train, y_second_test=train_test_split(x_afterscaling,y_aftersaling, test_size=0.2, random_state= 42)
reg = LinearRegression()
reg.fit(x_second_train, y_second_train)


# In[47]:


y_second_hat =reg.predict(x_second_train)


# In[48]:


reg.score(x_second_train, y_second_train)


# The R square value of the new model is better (0.83(New) vs 0.76(Old))

# ## Comparing the two models 

# In[49]:


f, (plt1, plt2) = plt.subplots(1,2, sharey=True, figsize = [18, 9])
plt1.scatter(y_second_hat,y_second_train)
plt1.title.set_text('Second Model Test Plot')
plt2.scatter(y_train, y_hat)
plt2.title.set_text('First Model Test Plot')


# In[50]:


f, axs = plt.subplots(ncols=2, figsize =[20,5])


sns.distplot(y_second_hat- y_second_train, ax=axs[0])
axs[0].title.set_text('Second Model')
sns.distplot(y_hat- y_train, ax=axs[1])
axs[1].title.set_text('First Model')


# The residual distribution of the second model is much more central around 0.00. Therefore its should be more accurate 

# In[51]:


y_second_predict_hat=reg.predict(x_second_test)


# In[52]:


f, (plt1, plt2) = plt.subplots(1,2, sharey=True, figsize = [18, 9])
plt1.scatter(y_second_test,y_second_predict_hat, alpha=0.2)
plt1.title.set_text('Second Model Testing Plot')
plt.xlabel('Actual', fontweight='bold', fontsize=18) 
plt.ylabel('Predicted', fontweight='bold', fontsize=18)


plt2.scatter(y_test, y_hat_test, alpha=0.2)

plt2.title.set_text('First Model Test Plot')

plt.show()


# In[60]:


FirstModelPrice['Second_Prediction'] =np.exp(y_second_predict_hat)
FirstModelPrice['Second_Actual'] = np.exp(y_second_test.reset_index(drop=True))
FirstModelPrice['Second_Residual'] = FirstModelPrice['Second_Actual'] -FirstModelPrice['Second_Prediction']
FirstModelPrice['2nd%'] = FirstModelPrice['Second_Residual']*100 / FirstModelPrice['Second_Actual']


# In[62]:


FirstModelPrice.describe()


# The second model has been placed side to side with the first model in terms of stats. It is seen that the variance from the target is generally lower and as such we can conclude that it is a much more accurate model. 

# In[ ]:




