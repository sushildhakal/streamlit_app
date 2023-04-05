from turtle import width

import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')
import os

st.set_page_config(page_title="Data visualisation of the NEWYORK HOUSING MARKET", page_icon=":rocket:", layout="wide")
st.write("# New York City Housing Prices Visualisation")

st.markdown("Identifying How do apartment prices in New York City vary by **location, size, and amenities**, and what are the most important factors that influence apartment prices in different neighborhoods? ")


@st.cache_data
def load_data():

    df = pd.read_csv("nyc-rolling-sales.csv")

    return df


df = load_data()


df.columns = df.columns.str.strip()

df = df.dropna(subset=['SALE PRICE'])

df['SALE PRICE'] = pd.to_numeric(df['SALE PRICE'].replace('[\$,]', '', regex=True), errors='coerce')
df['SALE PRICE'] = df['SALE PRICE'].astype(float)
df['GROSS SQUARE FEET'] = pd.to_numeric(df['GROSS SQUARE FEET'].str.replace(',', '').str.replace('$', '').replace('-', np.nan), errors='coerce')
df['LAND SQUARE FEET'] = pd.to_numeric(df['LAND SQUARE FEET'].str.replace(',', '').str.replace('$', '').replace('-', np.nan), errors='coerce')
#borough_neighborhood_df = df.groupby(['BOROUGH', 'NEIGHBORHOOD'])['SALE PRICE'].mean().reset_index()
df['PRICE PER SQFT'] = df['SALE PRICE'] / df['GROSS SQUARE FEET']
df['AGE OF BUILDING'] = 2023 - df['YEAR BUILT']
building_class_df = df.groupby('BUILDING CLASS CATEGORY')['SALE PRICE'].mean().reset_index()
tax_class_df = df.groupby('TAX CLASS AT TIME OF SALE')['SALE PRICE'].mean().reset_index()
borough_groups = df.groupby('BOROUGH').agg({'SALE PRICE': 'mean', 'PRICE PER SQFT': 'mean', 'NEIGHBORHOOD': 'nunique', 'BUILDING CLASS CATEGORY': 'nunique'})
neighborhood_groups = df.groupby('NEIGHBORHOOD').agg({'SALE PRICE': 'mean', 'PRICE PER SQFT': 'mean', 'BUILDING CLASS CATEGORY': 'nunique', 'GROSS SQUARE FEET': 'sum'})



# Chart 1: Number of Sales by Borough


fig1, ax1 = plt.subplots()
borough_counts = df['BOROUGH'].value_counts()
ax1.bar(borough_counts.index, borough_counts.values)
ax1.set_title('Number of Sales by Borough')
ax1.set_xlabel('Borough')
ax1.set_ylabel('Number of Sales')




# Chart 2: Average Sale Price by Borough
fig2, ax2 = plt.subplots()
# group data by borough and calculate mean sale price
borough_prices = df.groupby('BOROUGH')['SALE PRICE'].mean()
ax2.bar(borough_prices.index, borough_prices.values)
ax2.set_title('Average Sale Price by Borough')
ax2.set_xlabel('Borough')
ax2.set_ylabel('Average Sale Price ($)')

# create bar scatter chart of sale price vs gross square feet
# Chart 3: Sale Price vs Gross Square Feet
fig3, ax3 = plt.subplots()
sns.scatterplot(data=df, x='SALE PRICE', y='GROSS SQUARE FEET', ax=ax3, s=10, alpha=0.5,hue=df['BOROUGH'])


ax3.set_xlim([0, 100000000]) # set x axis limit from 0 to 10,000,0000
ax3.set_ylim([0, 1000000]) # set y axis limit from 0 to 100,000


#ax3.scatter(df['GROSS SQUARE FEET'], df['SALE PRICE'], c=df['BOROUGH'], alpha=0.5, hue= "NEIGHBORHOOD")
ax3.set_xlabel('Gross Square Feet')
ax3.set_ylabel('Sale Price')
ax3.set_title('Sale Price vs Gross Square Feet')

# Create a figure and axis object
 #Chart 4: Sale Price by Tax Class
# Create a figure object
#fig4, ax4 = plt.subplots()

# Create the boxplot
#fig4 = plt.figure()
#plt.boxplot([df[df['TAX CLASS AT TIME OF SALE'] == 1]['SALE PRICE'], 
#             df[df['TAX CLASS AT TIME OF SALE'] == 2]['SALE PRICE'],
#             df[df['TAX CLASS AT TIME OF SALE'] == 4]['SALE PRICE']])
#plt.xticks([1, 2, 3], ['Class 1', 'Class 2', 'Class 4'])
#plt.xlabel('Tax Class')
#plt.ylabel('Sale Price')
#plt.title('Sale Price by Tax Class')

#boxplot_data = [df[df['TAX CLASS AT TIME OF SALE'] == 1]['SALE PRICE'], 
#                df[df['TAX CLASS AT TIME OF SALE'] == 2]['SALE PRICE'],
#                df[df['TAX CLASS AT TIME OF SALE'] == 4]['SALE PRICE']]
#ax4.boxplot(boxplot_data)

## Set the axis labels and title
#ax4.set_xticks([1, 2, 3])
#ax4.set_xticklabels(['Class 1', 'Class 2', 'Class 4'])
#ax4.set_xlabel('Tax Class')
#ax4.set_ylabel('Sale Price')
#ax4.set_title('Sale Price by Tax Class')


#fig4, ax4 = plt.subplots()
#boxplot_data = [df[df['TAX CLASS AT TIME OF SALE'] == 1]['SALE PRICE'], 
#                df[df['TAX CLASS AT TIME OF SALE'] == 2]['SALE PRICE'],
#                df[df['TAX CLASS AT TIME OF SALE'] == 4]['SALE PRICE']]
#ax4.boxplot(boxplot_data)
#ax4.set_xticklabels(['Class 1', 'Class 2', 'Class 4'])
#ax4.set_xlabel('Tax Class')
#ax4.set_ylabel('Sale Price')
#ax4.set_title('Sale Price by Tax Class')

# Chart 5: Scatter plot of Sale Price vs Year Built
fig5, ax5 = plt.subplots()
ax5.scatter(df['YEAR BUILT'], df['SALE PRICE'], c=df['BOROUGH'], alpha=0.5)
ax5.set_xlim([1750, 2023]) # set x axis limit from 1750 to 2023
ax5.set_ylim([0, 100000000]) # set y axis limit from 0 to 100,000,000
ax5.set_xlabel('Year Built')
ax5.set_ylabel('Sale Price')
ax5.set_title('Sale Price vs Year Built')

# Chart 6: Sale Price distribution
fig6, ax6 = plt.subplots()
ax6.hist(df['SALE PRICE'], bins=50)
ax6.set_xlabel('Sale Price')
ax6.set_ylabel('Count')
ax6.set_title('Sale Price Distribution')


# Chart 7: Sale Price distribution

fig7 = plt.figure(figsize=(15, 10))
sns.barplot(x='BUILDING CLASS CATEGORY', y='SALE PRICE', data=df)
plt.xticks(rotation=90)
plt.xlabel('Building Class Category')
plt.ylabel('Average Sale Price')
plt.title('Average Sale Price by Building Class Category')


# Chart 8: average sales by borough in colour and barplot
def create_borough_price_plot():
    fig, ax = plt.subplots()
    sns.barplot(x='BOROUGH', y='SALE PRICE', data=df, ax=ax)
    ax.set_xlabel('Borough')
    ax.set_ylabel('Average Sale Price')
    ax.set_title('Average Sale Price by Borough')
    return fig

# Call the function to create the plot
fig8 = create_borough_price_plot()


fig9, ax = plt.subplots()

sns.barplot(x='NEIGHBORHOOD', y='SALE PRICE', data=df, ax=ax)

ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_xlabel('Neighborhood')
ax.set_ylabel('Average Sale Price')
ax.set_title('Average Sale Price by Neighborhood')




fig10, ax = plt.subplots()
sns.barplot(x='TAX CLASS AT TIME OF SALE', y='SALE PRICE', data=df, ax=ax)
ax.set_xlabel('Tax Class')
ax.set_ylabel('Average Sale Price')
ax.set_title('Average Sale Price by Tax Class')




# create figure
fig11 = plt.figure()

# plot histogram
plt.hist(2023 - df['YEAR BUILT'], bins=100)

# set axis labels and title
plt.xlabel('Age of Building')
plt.ylabel('Frequency')
plt.title('Age Distribution')


# Group the charts in 2 columns
#with st.container():
col1, col2 = st.columns(2)


#Display the charts in the left column
with col1:
    st.write('## Data')
    st.pyplot(fig1)
    st.pyplot(fig2)
    st.pyplot(fig3)
   
    st.pyplot(fig9)
    st.pyplot(fig7)

# Display the charts in the right column
with col2:
    st.write('## Visualization')
    #st.pyplot(fig4)
    st.pyplot(fig5)
    st.pyplot(fig6)
    st.pyplot(fig8)
    st.pyplot(fig10)
    st.pyplot(fig11)