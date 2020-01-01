# Predicting Critic Scores using Video Game Statistics

## Team Members: Ram S IMT2017521, Rathin Bhargava IMT201722

## Reading Data and Importing Libraries

import random
from math import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import pandas as pd
import seaborn as sns
# %matplotlib inline

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("Video_Games_Sales_as_at_22_Dec_2016.csv")
data.shape

## Data Visualization and Preprocessing

##print(len(data.columns))
data.head(2)

#### Since we are predicting MetaCritic score for games, we will drop all the data points which do not have a MetaCritic Score.

data = data.dropna(subset=['Critic_Score'])
data.shape

data.describe()

#### Number of Null values in each column

for i in list(data.columns):
        print (i,":",data[i].isnull().sum())

#### None of these columns seems to have more than 20% of their values missing. Therefore, we don't think we should be removing any more columns from the data set

#### Let us see the data type of each column

for i in list(data.columns):
    #print(i,":",data[i].dtypes)

#### As we can see, every column has the required data type except User_Score which is an object instead of float64

### Sampler Function

#### When it comes to replacing null values, we can replace them with the mean value of that column, or the median.
#### We can also create a probability distribution based on the given data in that column and can perform Markov Chain Monte Carlo (MCMC) sampling on that distribution to fill the null values.
#### However, the sampler itself is good for generating data but might not perform well for replacing null values

#### Code for MCMC sampling

def sampler(s,data):
    d = dict(data[s].value_counts())
    sum1 = sum(d.values())
    m = d.keys()
    prob = [d[x]/float(sum1) for x in d.keys()]
#     #print(prob)
    return [d.keys(),prob]

def sample(composite):
    key = list(composite[0])
    prob = composite[1]
    a = random.random()
    sofar = 0
    for i in range(len(key)):
        sofar += prob[i]
        if(a <= sofar):
            return key[i]
    return key[len(key)-1]

def replace_null(s,data):
    composite = sampler(s,data)
    l = list(data[s])
    for i in range(len(l)):
        if(pd.isna(l[i])):
            l[i] = sample(composite)
    data[s]=l

### Columnwise Analysis

### Year of Release

#print(len(data['Year_of_Release'].unique()))
#print(sorted(list(data['Year_of_Release'].unique())))

#### The year of release might have some importance when it comes to Critic_Score. However, there are 26 values and the order of importance of this feature might not correlate chronologically. 

#### We will now populate the nan values in the data set.

# replace_null("Year_of_Release",data)
# data['Year_of_Release'] = data['Year_of_Release'].replace(0.0,2002.0)
data['Year_of_Release'] = data['Year_of_Release'].replace(0.0,data['Year_of_Release'].mean())

data['Year_of_Release'].plot(kind='hist')

#### Outlier detection

sns.boxplot(x=data['Year_of_Release'])

### Feature Engineering - Year_of_Release

#### We will be performing some feature engineering. We will create a new feature which reflect games released in that generation. This is because the exact year seems to be not very related to the score and we do not wish to suppose a 

#1985 - 1995
data['Year_198595'] = data[data['Year_of_Release'] <= 1995]['Year_of_Release']
data['Year_198595'] = data['Year_198595'].replace(np.nan, 0)
#print(sorted(data['Year_198595'].unique()))
for i in sorted(data['Year_198595'].unique()):
    if(i!=0.0):
        data['Year_198595'] = data['Year_198595'].replace(i, 1)
#print(sorted(data['Year_198595'].unique()))

#1996 - 2005
data['Year_199505'] = data[data['Year_of_Release'] <= 2005]['Year_of_Release']
data['Year_199505'] = data[data['Year_199505'] > 1995]['Year_199505']
data['Year_199505'] = data['Year_199505'].replace(np.nan, 0)
#print(sorted(data['Year_199505'].unique()))
for i in sorted(data['Year_199505'].unique()):
    if(i!=0.0):
        data['Year_199505'] = data['Year_199505'].replace(i, 1)
# #print(sorted(data['Year_199505'].unique()))

#2006 - 2010
data['Year_200510'] = data[data['Year_of_Release'] <= 2010]['Year_of_Release']
data['Year_200510'] = data[data['Year_200510'] > 2005]['Year_200510']
data['Year_200510'] = data['Year_200510'].replace(np.nan, 0)
#print(sorted(data['Year_200510'].unique()))
for i in sorted(data['Year_200510'].unique()):
    if(i!=0.0):
        data['Year_200510'] = data['Year_200510'].replace(i, 1)
# #print(sorted(data['Year_200510'].unique()))

#2010 - beyond
data['Year_201016'] = data[data['Year_of_Release'] > 2010]['Year_of_Release']
data['Year_201016'] = data['Year_201016'].replace(np.nan, 0)
#print(sorted(data['Year_201016'].unique()))
for i in sorted(data['Year_201016'].unique()):
    if(i!=0.0):
        data['Year_201016'] = data['Year_201016'].replace(i, 1)
# #print(sorted(data['Year_201016'].unique()))


data.shape

# platform_data.mean()

group_by_year = data.groupby(by=['Year_of_Release'])
year_data_avg = group_by_year.mean()
year_data_count = group_by_year.count()


data_count_series = year_data_count.iloc[:,0]

features_of_interest = pd.DataFrame({'Critic_Score':year_data_avg['Critic_Score'],'No. Games':data_count_series,})

features_of_interest

#### We still believe that we cannot drop the Year_of_Release feature because it will help with the training. However, the partitions we have made were big turning points in gaming technology as well as general trends in gamer expectations.

data.head(2)

### Platform

platform_data = data.groupby(by=['Platform'])
platform_data.count()


group_by_platform = data.groupby(by=['Platform'])
platform_data_avg = group_by_platform.mean()
platform_data_count = group_by_platform.count()


data_count_series = platform_data_count.iloc[:,0]

features_of_interest = pd.DataFrame({'Critic_Score':platform_data_avg['Critic_Score'],'No. Games':data_count_series,})

features_of_interest

#### It might be relevant if we could classify the consoles into four families - Sony, Microsoft, Nintendo, Others. Also we could engineer 3 new features called 'Type' which will specify if the device is a handheld/console/pc. 

#### Device Type

#Handheld
data['Handheld'] = data[data['Platform'].isin(['3DS','DS','GBA','GC','PSP','PSV'])]['Platform']
data['Handheld'] = data['Handheld'].replace(np.nan, 0)
#print((data['Handheld'].unique()))
for i in (data['Handheld'].unique()):
    if(i!=0.0):
        data['Handheld'] = data['Handheld'].replace(i, 1)
# #print((data['Handheld'].unique()))

#Console
data['Console'] = data[data['Platform'].isin(['DC','PS','PS2','PS3','PS4','Wii','WiiU','X360','XB','XOne'])]['Platform']
data['Console'] = data['Console'].replace(np.nan, 0)
#print((data['Console'].unique()))
for i in (data['Console'].unique()):
    if(i!=0.0):
        data['Console'] = data['Console'].replace(i, 1)
# #print((data['Console'].unique()))

#PC
data['PC'] = data[data['Platform'].isin(['PC'])]['Platform']
data['PC'] = data['PC'].replace(np.nan, 0)
#print((data['PC'].unique()))
for i in (data['PC'].unique()):
    if(i!=0.0):
        data['PC'] = data['PC'].replace(i, 1)
# #print((data['PC'].unique()))

#### Company

#Microsoft
data['Microsoft'] = data[data['Platform'].isin(['XB','XOne','X360'])]['Platform']
data['Microsoft'] = data['Microsoft'].replace(np.nan, 0)
#print((data['Microsoft'].unique()))
for i in (data['Microsoft'].unique()):
    if(i!=0.0):
        data['Microsoft'] = data['Microsoft'].replace(i, 1)
# #print((data['Microsoft'].unique()))

#Sony
data['Sony'] = data[data['Platform'].isin(['PS','PS2','PS3','PS4','PSP','PSV'])]['Platform']
data['Sony'] = data['Sony'].replace(np.nan, 0)
#print((data['Sony'].unique()))
for i in (data['Sony'].unique()):
    if(i!=0.0):
        data['Sony'] = data['Sony'].replace(i, 1)
# #print((data['Sony'].unique()))

#Nintendo
data['Nintendo'] = data[data['Platform'].isin(['3DS','DS','GBA','GC','Wii','WiiU'])]['Platform']
data['Nintendo'] = data['Nintendo'].replace(np.nan, 0)
#print((data['Nintendo'].unique()))
for i in (data['Nintendo'].unique()):
    if(i!=0.0):
        data['Nintendo'] = data['Nintendo'].replace(i, 1)
# #print((data['Nintendo'].unique()))

#PCCom
data['PCCom'] = data[data['Platform'].isin(['PC'])]['Platform']
data['PCCom'] = data['PCCom'].replace(np.nan, 0)
#print((data['PCCom'].unique()))
for i in (data['PCCom'].unique()):
    if(i!=0.0):
        data['PCCom'] = data['PCCom'].replace(i, 1)
# #print((data['PCCom'].unique()))

num=[]
for i in data['Platform'].unique():
    if( isinstance(i, int) == 1 or isinstance(i, float) ==1):
        #print(i)
        num.append(i)

#### This shows that there is a 0 in the dataset where there should be a string. We shall replace it with the string 'zero'
for i in num:
    data['Platform'] = data['Platform'].replace(i,"zero")

lc = LabelEncoder()

data['Platform'] = lc.fit_transform(data['Platform'])

data.head(2)

### Genre

group_by_genre = data.groupby(by=['Genre'])
genre_data_avg = group_by_genre.mean()
genre_data_count = group_by_genre.count()
genre_data_count

data_count_series = genre_data_count.iloc[:,0]

features_of_interest = pd.DataFrame({'Critic_Score':genre_data_avg['Critic_Score'],'No. Games':data_count_series,})

features_of_interest

#### While One Hot Encoding for this feature is very amenable, 12 new features will be added to the data set. It is necessary to check if we can handle such a data set or not

encoded_columns = pd.get_dummies(data['Genre'])

for i in encoded_columns:
    data[i+"Genre"] = encoded_columns[i]

data = data.drop(columns=['Genre'],axis=1)
data.shape

data.head(2)

### Publisher

#print("Size of Domain: ",len(data['Publisher'].unique()))
#print("Type of Domain: ",data['Publisher'].dtypes)
# #print(data['Publisher'].unique())
#print("Number of null values: ",data['Publisher'].isnull().sum())

#### Since we have too many publishers, we shall Label Encode this feature. 

num=[]
for i in data['Publisher'].unique():
    if( isinstance(i, int) == 1 or isinstance(i, float) ==1):
        #print(i)
        num.append(i)

#### This shows that there is a 0 in the dataset where there should be a string. We shall replace it with the string 'zero'

for i in num:
    data['Publisher'] = data['Publisher'].replace(i,"zero")

lc = LabelEncoder()

data['Publisher'] = lc.fit_transform(data['Publisher'])

data['Publisher'].plot(kind='hist')

### Developer

#print("Size of Domain: ",len(data['Developer'].unique()))
#print("Type of Domain:",data['Developer'].dtypes)
# #print(data['Publisher'].unique())
#print("Number of null values: ",data['Developer'].isnull().sum())

#### We see that we need to do something very similar to what we did for Publisher, which is Label Encoding

num=[]
for i in data['Developer'].unique():
    if( isinstance(i, int) == 1 or isinstance(i, float) ==1):
        #print(i)
        num.append(i)

for i in num:
    data['Developer'] = data['Developer'].replace(i,"zero")

lc = LabelEncoder()

data['Developer'] = lc.fit_transform(data['Developer'])

data['Developer'].plot(kind='hist')

### Rating

#print(data['Rating'].unique())
data['Rating'].dtypes

group_by_rating = data.groupby(by=['Rating'])
rating_data_avg = group_by_rating.mean()
rating_data_count = group_by_rating.count()
rating_data_count

data_count_series = rating_data_count.iloc[:,0]

features_of_interest = pd.DataFrame({'Critic_Score':rating_data_avg['Critic_Score'],'No. Games':data_count_series})

features_of_interest

#### Encoding: Since the size of the domain is less, we can safely One Hot Encode this data.

num=[]
for i in data['Rating'].unique():
    if( isinstance(i, int) == 1 or isinstance(i, float) ==1):
        #print(i)
        num.append(i)

for i in num:
    data['Rating'] = data['Rating'].replace(i,"zero")

encoded_columns = pd.get_dummies(data['Rating'])

for i in encoded_columns:
    data[i] = encoded_columns[i]
data = data.drop(columns=['Rating'],axis=1)
data.shape

### User Score

data['User_Score'].unique()

#### We can see that except for nan, 'tbd' the rest of the values are all numbers.

#print(data['User_Score'].isnull().sum())

#### We can convert User_Score column into a numerical column. We replace the illegal values with mean as there are very few values.

data['User_Score'] = pd.to_numeric(data['User_Score'], errors='coerce')

data['User_Score'] = data['User_Score'].replace(np.nan, 0 , regex=True)  
data['User_Score'] = data['User_Score'].astype(float)

sns.boxplot(x=data['User_Score'])

data.head(2)

### Other Numerical features - Global_Sales, EU_Sales ...

data['Platform'].unique()

#### We can also use our sampler/mean to fill the missing values in all the numerical columns

# Using Sampler/mean for each column 

for i in list(data.columns):
    if(data[i].dtypes != 'object'):
        data[i] = data[i].fillna(data[i].mean())
#         replace_null(i,data)

sns.boxplot(x=data['Critic_Score'])

sns.boxplot(x=data['User_Score'])

#### It can be seen that generally people are a little more generous with their scores compared to critics.

sns.boxplot(x=data['Critic_Count'])

sns.boxplot(x=data['User_Count'])

### Illegal value analysis for features

sns.boxplot(x=data['Global_Sales'])

data[data['Global_Sales'] > 60]

#### This game is a special case as it was given for free to all customers who bought the Wii console (which is one of the most popular consoles of all time). It is an outlier.

indexNames = data[ data['Global_Sales'] > 60 ].index
 
# Delete these row indexes from dataFrame
data.drop(indexNames , inplace=True)

plotting = pd.DataFrame(data['Global_Sales'])
plotting['NA_Sales'] = data['NA_Sales']
plotting['EU_Sales'] = data['EU_Sales']
plotting['JP_Sales'] = data['JP_Sales']
plotting.head(10).plot(kind='bar')

### Correlation matrix and heat maps

corr = data.corr()
corr.style.background_gradient(cmap='coolwarm')

sns.heatmap(data.corr(), annot=True)

### Train Test Split for Critic Score

train, test = train_test_split(data, test_size=0.33,random_state=6)

x_train = train.drop(['Name','User_Score','Global_Sales','NA_Sales','EU_Sales','JP_Sales','Other_Sales'],axis=1)
y_train1 = train['User_Score']
y_train2 = train['Global_Sales']
y_train3 = train['NA_Sales']
y_train4 = train['EU_Sales']
y_train5 = train['JP_Sales']

x_test = test.drop(['Name','User_Score','Global_Sales','NA_Sales','EU_Sales','JP_Sales','Other_Sales'],axis=1)
y_test1 = test['User_Score']
y_test2 = test['Global_Sales']
y_test3 = test['NA_Sales']
y_test4 = test['EU_Sales']
y_test5 = test['JP_Sales']

### Scaling the Data set

#### If we perform only Linear Regression, it is not necessary to scale the feature set, however, it is very useful for interpretability fo the data

scaler = StandardScaler() 

scaler.fit(x_train)

x_train = scaler.transform(x_train)

x_test = scaler.transform(x_test)

## Model Training and Error Evaluation

#Creating model
modellir = Ridge()

# Fitting the model with prepared data
modellir.fit(x_train,y_train1)
y_predlir = modellir.predict(x_test)


print("Ridge Regression RMSE (User Score):", sqrt(mean_squared_error(y_test1,y_predlir)))
print("Ridge Regression R2 Score (User Score):", r2_score(y_test1,y_predlir))

modellir.fit(x_train,y_train2)
y_predlir = modellir.predict(x_test)

print("Ridge Regression RMSE (Global Sales):", sqrt(mean_squared_error(y_test2,y_predlir)))
print("Ridge Regression R2 Score (Global Sales):", r2_score(y_test2,y_predlir))

modellir.fit(x_train,y_train3)
y_predlir = modellir.predict(x_test)

print("Ridge Regression RMSE (NA Sales):", sqrt(mean_squared_error(y_test3,y_predlir)))
print("Ridge Regression R2 Score (NA Sales):", r2_score(y_test3,y_predlir))

modellir.fit(x_train,y_train4)
y_predlir = modellir.predict(x_test)

print("Ridge Regression RMSE (EU Sales):", sqrt(mean_squared_error(y_test4,y_predlir)))
print("Ridge Regression R2 Score (EU Sales):", r2_score(y_test4,y_predlir))

modellir.fit(x_train,y_train5)
y_predlir = modellir.predict(x_test)

print("Ridge Regression RMSE (JP Sales):", sqrt(mean_squared_error(y_test5,y_predlir)))
print("Ridge Regression R2 Score (JP Sales):", r2_score(y_test5,y_predlir))

### Residual Plot

plt.scatter(range(0,len(y_test1-y_predlir)),(y_test1-y_predlir)) 

plt.xlabel('indices') 
plt.ylabel('difference') 
   
plt.title('User Scores') 
  
# plt.show() 

plt.scatter(range(0,len(y_test2-y_predlir)),(y_test2-y_predlir)) 

plt.xlabel('indices') 
plt.ylabel('difference') 
   
plt.title('Global Sales') 
  
# plt.show() 

plt.scatter(range(0,len(y_test3-y_predlir)),(y_test3-y_predlir)) 

plt.xlabel('indices') 
plt.ylabel('difference') 
   
plt.title('NA Sales') 
  
# plt.show() 

plt.scatter(range(0,len(y_test4-y_predlir)),(y_test4-y_predlir)) 

plt.xlabel('indices') 
plt.ylabel('difference') 
   
plt.title('EU Sales') 
  
# plt.show() 

plt.scatter(range(0,len(y_test5-y_predlir)),(y_test5-y_predlir)) 

plt.xlabel('indices') 
plt.ylabel('difference') 
   
plt.title('JP Sales') 
  
# plt.show() 

#### This plot gives us a much better idea as to how large a difference the errors are.

## The END

### Creating Pickle file

import pickle

file_Name = "model"

fileObject = open(file_Name, 'wb')

pickle.dump(modellir, fileObject)

fileObject.close()

## EXTRA:

### Error Evaluation without Feature Engineering and Preprocessing

# data = pd.read_csv("Video_Games_Sales_as_at_22_Dec_2016.csv")

# data = data.dropna(subset=['Critic_Score'])

# for i in list(data.columns):
#     if(data[i].dtypes != 'object'):
#         data[i].fillna(data[i].mean())

# data['User_Score'] = pd.to_numeric(data['User_Score'], errors='coerce')

# data = data.replace(np.nan, 0 , regex=True)  
# data['User_Score'] = data['User_Score'].astype(float)

# data['User_Score'].dtypes

# for s in list(data.columns):
#     if(data[s].dtypes == 'object'):
#         num=[]
#         for i in data[s].unique():
#             if( isinstance(i, int) == 1 or isinstance(i, float) ==1):
# #                 #print(i)
#                 num.append(i)

# #### This shows that there is a 0 in the dataset where there should be a string. We shall replace it with the string 'zero'

#         for i in num:
#             data[s] = data[s].replace(i,"zero")




# lc = LabelEncoder()
# for s in list(data.columns):
#     if(data[s].dtypes == 'object'):
# #         #print(data[s])
#         data[s] = lc.fit_transform(data[s])

# train, test = train_test_split(data, test_size=0.33,random_state=6)

# x_train = train.drop(['Name','User_Score','Global_Sales','NA_Sales','EU_Sales','JP_Sales','Other_Sales'],axis=1)
# y_train1 = train['User_Score']
# y_train2 = train['Global_Sales']
# y_train3 = train['NA_Sales']
# y_train4 = train['EU_Sales']
# y_train5 = train['JP_Sales']

# x_test = test.drop(['Name','User_Score','Global_Sales','NA_Sales','EU_Sales','JP_Sales','Other_Sales'],axis=1)
# y_test1 = test['User_Score']
# y_test2 = test['Global_Sales']
# y_test3 = test['NA_Sales']
# y_test4 = test['EU_Sales']
# y_test5 = test['JP_Sales']
# scaler = StandardScaler() 

# scaler.fit(x_train)

# x_train = scaler.transform(x_train)

# x_test = scaler.transform(x_test)

# ## Model Training and Error Evaluation

# #Creating model
# modellir = Lasso()

# # Fitting the model with prepared data
# modellir.fit(x_train,y_train1)
# y_predlir = modellir.predict(x_test)


# #print("Linear Regression RMSE (User Score):", sqrt(mean_squared_error(y_test1,y_predlir)))
# #print("Linear Regression R2 Score (User Score):", r2_score(y_test1,y_predlir))

# modellir.fit(x_train,y_train2)
# y_predlir = modellir.predict(x_test)

# #print("Linear Regression RMSE (Global Sales):", sqrt(mean_squared_error(y_test2,y_predlir)))
# #print("Linear Regression R2 Score (Global Sales):", r2_score(y_test2,y_predlir))

# modellir.fit(x_train,y_train3)
# y_predlir = modellir.predict(x_test)

# #print("Linear Regression RMSE (NA Sales):", sqrt(mean_squared_error(y_test3,y_predlir)))
# #print("Linear Regression R2 Score (NA Sales):", r2_score(y_test3,y_predlir))

# modellir.fit(x_train,y_train4)
# y_predlir = modellir.predict(x_test)

# #print("Linear Regression RMSE (EU Sales):", sqrt(mean_squared_error(y_test4,y_predlir)))
# #print("Linear Regression R2 Score (EU Sales):", r2_score(y_test4,y_predlir))

# modellir.fit(x_train,y_train5)
# y_predlir = modellir.predict(x_test)

# #print("Linear Regression RMSE (JP Sales):", sqrt(mean_squared_error(y_test5,y_predlir)))
# #print("Linear Regression R2 Score (JP Sales):", r2_score(y_test5,y_predlir))

