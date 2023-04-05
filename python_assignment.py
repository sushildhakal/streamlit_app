#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os


# In[3]:


df = pd.read_csv('nyc-rolling-sales.csv')


# In[5]:


df.dtypes


# In[7]:


df.columns = df.columns.str.strip()


# In[8]:


df = df.dropna(subset=['SALE PRICE'])


# In[9]:


df['SALE PRICE'] = pd.to_numeric(df['SALE PRICE'].replace('[\$,]', '', regex=True), errors='coerce')


# In[14]:


df = df.dropna(subset=['SALE PRICE'])


# In[15]:


df


# In[16]:


df['SALE PRICE'] = df['SALE PRICE'].astype(int)


# In[18]:


df.dtypes


# In[19]:


df['GROSS SQUARE FEET'] = pd.to_numeric(df['GROSS SQUARE FEET'].str.replace(',', '').str.replace('$', '').replace('-', np.nan), errors='coerce')


# In[21]:


df.dtypes


# In[22]:


df['LAND SQUARE FEET'] = pd.to_numeric(df['LAND SQUARE FEET'].str.replace(',', '').str.replace('$', '').replace('-', np.nan), errors='coerce')


# In[23]:


df.dtypes


# In[24]:


borough_neighborhood_df = df.groupby(['BOROUGH', 'NEIGHBORHOOD'])['SALE PRICE'].mean().reset_index()


# In[25]:


df['PRICE PER SQFT'] = df['SALE PRICE'] / df['GROSS SQUARE FEET']


# In[27]:


df['AGE OF BUILDING'] = 2023 - df['YEAR BUILT']


# In[28]:


building_class_df = df.groupby('BUILDING CLASS CATEGORY')['SALE PRICE'].mean().reset_index()
tax_class_df = df.groupby('TAX CLASS AT TIME OF SALE')['SALE PRICE'].mean().reset_index()


# In[29]:


borough_groups = df.groupby('BOROUGH').agg({'SALE PRICE': 'mean', 'PRICE PER SQFT': 'mean', 'NEIGHBORHOOD': 'nunique', 'BUILDING CLASS CATEGORY': 'nunique'})
neighborhood_groups = df.groupby('NEIGHBORHOOD').agg({'SALE PRICE': 'mean', 'PRICE PER SQFT': 'mean', 'BUILDING CLASS CATEGORY': 'nunique', 'GROSS SQUARE FEET': 'sum'})


# In[30]:


borough_counts = df['BOROUGH'].value_counts()


# In[31]:


borough_counts.plot.bar()


# In[32]:


plt.title('Number of Sales by Borough')
plt.xlabel('Borough')
plt.ylabel('Number of Sales')
borough_counts.plot.bar()


# In[33]:


# group data by borough and calculate mean sale price
borough_prices = df.groupby('BOROUGH')['SALE PRICE'].mean()


# In[34]:


# create bar chart of average sale price by borough
fig, ax = plt.subplots()
borough_prices.plot(kind='bar', ax=ax)
ax.set_title('Average Sale Price by Borough')
ax.set_xlabel('Borough')
ax.set_ylabel('Average Sale Price ($)')
plt.show()


# In[36]:


sns.scatterplot(data=df, x='GROSS SQUARE FEET', y='SALE PRICE', alpha=0.50, hue='NEIGHBORHOOD')
plt.title('Sale Price vs Gross Square Feet')
plt.xlabel('Gross Square Feet')
plt.ylabel('Sale Price')
plt.show()


# In[39]:


import matplotlib.pyplot as plt

# Plot scatter plot with hue
plt.scatter(df['GROSS SQUARE FEET'], df['SALE PRICE'], c=df['BOROUGH'], alpha=0.5)
plt.xlabel('Gross Square Feet')
plt.ylabel('Sale Price')
plt.title('Sale Price vs Gross Square Feet')
plt.show()


# In[40]:


plt.boxplot([df[df['TAX CLASS AT TIME OF SALE'] == 1]['SALE PRICE'], 
             df[df['TAX CLASS AT TIME OF SALE'] == 2]['SALE PRICE'],
             df[df['TAX CLASS AT TIME OF SALE'] == 4]['SALE PRICE']])
plt.xticks([1, 2, 3], ['Class 1', 'Class 2', 'Class 4'])
plt.xlabel('Tax Class')
plt.ylabel('Sale Price')
plt.title('Sale Price by Tax Class')
plt.show()


# In[41]:


plt.hist(df['SALE PRICE'], bins=50)
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.title('Price Distribution')
plt.show()


# In[42]:


sns.barplot(x='BOROUGH', y='SALE PRICE', data=df)
plt.xlabel('Borough')
plt.ylabel('Average Sale Price')
plt.title('Average Sale Price by Borough')
plt.show()


# In[43]:


sns.barplot(x='NEIGHBORHOOD', y='SALE PRICE', data=df)
plt.xticks(rotation=90)
plt.xlabel('Neighborhood')
plt.ylabel('Average Sale Price')
plt.title('Average Sale Price by Neighborhood')
plt.show()


# In[44]:


sns.barplot(x='BUILDING CLASS CATEGORY', y='SALE PRICE', data=df)
plt.xticks(rotation=90)
plt.xlabel('Building Class Category')
plt.ylabel('Average Sale Price')
plt.title('Average Sale Price by Building Class Category')
plt.show()


# In[45]:


plt.hist(2023 - df['YEAR BUILT'], bins=100)
plt.xlabel('Age of Building')
plt.ylabel('Frequency')
plt.title('Age Distribution')
plt.show()


# In[46]:


sns.barplot(x='TAX CLASS AT TIME OF SALE', y='SALE PRICE', data=df)
plt.xlabel('Tax Class')
plt.ylabel('Average Sale Price')
plt.title('Average Sale Price by Tax Class')
plt.show()


# In[ ]:




