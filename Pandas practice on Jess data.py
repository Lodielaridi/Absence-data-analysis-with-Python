#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
Excel_file = (r"C:\Users\lodiel-aridi\Desktop\Python_test.xlsx")
df = pd.read_excel(Excel_file, sheet_name=1, index_col=0)
df.head(30)


# In[2]:


df.describe()


# In[3]:


df.dtypes


# In[4]:


df.describe(include="all")


# In[5]:


df.info()


# In[6]:


#Nothing I did showed the "N/a" values which I know should be NaN - so I just ran head(30) to see all the data
#I think it's because they occur in the salary column of type object
#This isn't ideal, I should look for a solution that identifies N/A or missing values for type object columns
#but for this instance, I'll just replace "N/a" with np.nan and then count how many missing values I have
import numpy as np
df.replace("N/a", np.nan,inplace=True)
df.head(30)


# In[7]:


#then I'll repeat the missing data steps to count
missing_data =df.isnull()
missing_data.head()


# In[8]:


for column in missing_data.columns.values.tolist():
    print(column)
    print(missing_data[column].value_counts())
    print("")


# In[9]:


#I want to calculate the mean  salary by department and replace missing values with those, but first I need to change the data type to float
#The below is just a trial run to see what I get for avg salary
#Running it made me realise that it is now identified as an Integer because the N/a values were changed to missing
avg_salary=df["Please mention your annual salary base"].astype("float").mean(axis=0)
print(avg_salary)


# In[10]:


#This is where I reached the Module 1 lab section on missing data replace with mean, and realised they dont 
#provide a solution for when you need to replace with mean of group
#TBC another time
df['Please mention your annual salary base'] = df.groupby(['In which department do you work?', 'In which age group do you fall?'])['Please mention your annual salary base']    .transform(lambda x: x.fillna(x.mean()))


# In[11]:


missing_data=df.isnull()
missing_data.head(30)


# In[12]:


for column in missing_data.columns.values.tolist():
    print(column)
    print(missing_data[column].value_counts())
    print("")


# In[13]:


#yay - done the replace missing data bit - now I need to make sure the fields are all in the correct format


# In[14]:


df.dtypes


# In[15]:


avg_salary=df["Please mention your annual salary base"].astype("float").mean(axis=0)
print(avg_salary)


# In[16]:


df[['Absences']]=df[['Absences']].astype("float")


# In[17]:


df.dtypes


# In[18]:


#now I need to normalize the data


# In[19]:


#I will transform all the float data to be from 0 to 1, by dividing the values by their maximum 


# In[20]:


df['Individual Average']=df['Individual Average']/df['Individual Average'].max()


# In[21]:


df['Individual Average'].head()


# In[21]:


df['Please mention your annual salary base']=df['Please mention your annual salary base']/df['Please mention your annual salary base'].max()
df['Absences']=df['Absences']/df['Absences'].max()


# In[22]:


df[['Absences','Please mention your annual salary base']].head()


# In[23]:


#Binning
#First thing is to look at the distributions or historgrams of the different absences,and bin them into high,medium, low- this is just for practice, I won't be using the binned values in the analysis 
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as plt
from matplotlib import pyplot
plt.pyplot.hist(df["Absences"])

#this sets x/y labels and plot title
plt.pyplot.xlabel("Absences")
plt.pyplot.ylabel("Count")
plt.pyplot.title("Absence Bins")


# In[24]:


bins=np.linspace(min(df["Absences"]),max(df["Absences"]),4)
bins


# In[25]:


Absence_bins = ['Low','Medium', 'High']


# In[26]:


df['Absence-buckets'] =pd.cut(df['Absences'],bins,labels= Absence_bins, include_lowest=True)
df[['Absences','Absence-buckets']].head(30)


# In[27]:


df[['In which department do you work?','Absence-buckets']].head(30)


# In[28]:


df['Absence-buckets'].value_counts()


# In[29]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as plt
from matplotlib import pyplot
pyplot.bar(Absence_bins,df["Absence-buckets"].value_counts())

#set x/y labels and plot title
plt.pyplot.xlabel("Absences")
plt.pyplot.ylabel("count")
plt.pyplot.title("Absence buckets")


# In[30]:


#below I wil plot a histogram to show not just count of absences in the high bucket etc, but to show the boundaries of the buckets (in terms of absence dayes)


# In[31]:


#%matplotlib inline 
#import matplotlib as plt
#from matplotlib import pyplot

a=(0,1,2)

#draw a histogram of attribute "absences" with bins = 3
plt.pyplot.hist(df["Absences"],bins=3)
plt.pyplot.xlabel("Absence days")
plt.pyplot.ylabel("count")
plt.pyplot.title("Absence bins")


# In[32]:


#done with bins now dummy variables (will use it on gender, age group and department.
#We use dummy variables so we can do regression, as regression analysis doesn't understand words.


# In[33]:


Age_dummy = pd.get_dummies(df["In which age group do you fall?"])
Age_dummy.head()


# In[34]:


Gender_dummy = pd.get_dummies(df["Please specify your gender"])
Gender_dummy


# In[35]:


Department_dummy=pd.get_dummies(df['In which department do you work?'])
Department_dummy


# In[36]:


#I need to add the dummy variables to the main data frame, but will not include the gender one as it doesnt make sense to use it for analysis


# In[37]:


#merge data frame df and dummy variables
df=pd.concat([df,Age_dummy,Department_dummy],axis=1)


# In[38]:


df.head()


# In[39]:


df.to_csv("Pandas Jess Data normalized.csv")


# In[40]:


import numpy as np


# In[41]:


path="Pandas Jess Data normalized.csv"
df = pd.read_csv(path)
df.head()


# In[ ]:


#start of lab 3 for exploratory analysis - we will look at correlations and descriptive statistics and ANOVA
%%capture
get_ipython().system(' pip install seaborn')


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df.corr()
#the below already shows us some interesting coorelations 
#(eg. satisfaction score is lowest in the youngest group, which also happens to be the age group correlated to transport department)
#Satisfaction is not highly correlated with absence, but there is a negative correlation of -0.2, and no correlation with salary, but a slightly higher correlation in sales and -ve correlation in transport
#Absences were negatively moderately correlated to salary, and negatively correlated with growth department.and highly correlated to transport department
#salary was highly correlated with mid age group and negatively correlated with younger group (also transport), and postively correlated with tech, and lowest paid group is transport followed by operations


# In[ ]:


#Linear relationships:
#In order to start understanding the (linear) relationship between
#an individual variable and the price. We can do this by using "regplot",
#which plots the scatterplot plus the fitted regression line for the data.
#Let's see several examples of different linear relationships:

sns.regplot(x="Absences", y="Individual Average", data=df)
plt.ylim(0,)
df[["Absences","Individual Average"]].corr()


# In[ ]:


sns.regplot(x="Absences", y="Please mention your annual salary base", data=df)
plt.ylim(0,)
df[["Absences","Please mention your annual salary base"]].corr()


# In[ ]:


#Categorical variables -The categorical variables can have the type "object" or "int64"
#A good way to visualize categorical variables is by using boxplots.

df.dtypes


# In[ ]:


sns.boxplot(x="In which age group do you fall?", y="Absences", data=df)


# In[ ]:


sns.boxplot(x="In which age group do you fall?", y="Individual Average", data=df)


# In[ ]:


sns.boxplot(x="Absence-buckets", y="Individual Average", data=df)


# In[ ]:


sns.boxplot(x="In which department do you work?", y="Individual Average", data=df)


# In[ ]:


sns.boxplot(x="In which department do you work?", y="Absences", data=df)


# In[ ]:


df.describe(include=["object"])


# In[ ]:


df.describe()


# In[ ]:


#Value-counts is a good way of understanding how many units of each characteristic/variable
#we have. We can apply the "value_counts" method on the column 'drive-wheels'.
#Don’t forget the method "value_counts" only works on Pandas series, not Pandas Dataframes. 
#As a result, we only include one bracket "df['drive-wheels']" not two brackets "df[['drive-wheels']]".


# In[ ]:


#reached value counts - need to try those on a few categorical variables like age group and department to figure how many of each we have

