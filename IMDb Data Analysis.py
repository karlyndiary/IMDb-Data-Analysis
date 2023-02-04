#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

#Data viz packages
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import plotly.express as px
import plotly.offline as pyo


# In[2]:


df = pd.read_csv(r'C:\Users\KAREN J FERNANDES\anaconda3\Files\IMDb Analysis\imdb_processed.csv')


# In[3]:


df.head()


# ## Data Pre-processing

# In[4]:


#view the list of columns
df.columns


# In[5]:


#dropping unnamed column
df = df.drop("Unnamed: 0",axis=1)


# In[6]:


df.shape


# In[7]:


#check for no of duplicates present
len(df)-len(df.drop_duplicates())


# In[8]:


#Dropping duplicates based on title column
df = df.drop_duplicates('title').sort_index()


# In[9]:


df.shape


# In[10]:


#check for any nulls
df.isnull().any()


# In[11]:


#replacing Nan with Missing in string columns
df[['genre', 'country', 'language', 'cast', 'director', 'composer', 'writer']] = df[['genre', 'country', 'language', 'cast', 'director', 'composer', 'writer']].fillna('Missing')


# In[12]:


#replacing Nan with 0 in numeric columns
df[['rating', 'vote', 'runtime']] = df[['rating', 'vote', 'runtime']].fillna('0')


# In[13]:


#check for any nulls
df.isnull().any()


# In[14]:


#check datatypes
df.dtypes


# In[15]:


#converting 'year' from float to int
df['year'] = df['year'].astype(int)
df['rating'] = df['rating'].astype(float)


# In[16]:


#check datatypes
df.dtypes


# In[17]:


#remove brackets from genre, country, language, director, cast and writer columns
df['genre'] = df['genre'].str.strip('[]')
df['country'] = df['country'].str.strip('[]')
df['language'] = df['director'].str.strip('[]')
df['director'] = df['director'].str.strip('[]')
df['cast'] = df['cast'].str.strip('[]')
df['writer'] = df['writer'].str.strip('[]')


# In[18]:


#remove quotes from genre, country, language, director columns
df['genre'] = df['genre'].str.replace(r"\'","", regex=True)
df['country'] = df['country'].str.replace(r"\'","", regex=True)
df['language'] = df['language'].str.replace(r"\'","", regex=True)
df['director'] = df['director'].str.replace(r"\'","", regex=True)


# In[19]:


#splitting the genre column into sub-categories. Only the first genre category will be considered for this analysis.
N = 8

df[[f'genre {x+1}' for x in range(N)]] = (
    df['genre'].str.split(',', n=N+1, expand=True).iloc[:, :N])


# In[20]:


df.head()


# In[21]:


#rename columns
df.rename(columns = {'genre 1':'genre_1'}, inplace = True)
df.rename(columns = {'kind':'category'}, inplace = True)


# In[22]:


df.columns


# In[23]:


df["category"].value_counts()


# In[24]:


#replacing 'tv mini series' to 'tv series' and 'tv movies', 'video movie' to 'movie'
df.replace(regex=r'tv mini series', value='tv series', inplace=True)
df.replace(regex=r'tv movie', value='movie', inplace=True)
df.replace(regex=r'video movie', value='movie', inplace=True)


# In[25]:


df["category"].value_counts()


# In[26]:


df.columns


# In[27]:


df.to_csv('cleaned_imdb_data.csv')


# # Exploratory Data Analysis (EDA)

# ## Top 10 Best Perfoming Movies

# In[28]:


#setting the theme for all the plots
sns.set_theme()


# In[29]:


fig = plt.figure(figsize = (10, 5))
ax = sns.barplot(x = 'title', y = 'rating', data = df.sort_values('rating', ascending=False)[0:10])
plot = ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)
plt.xlabel('Movie Title')
plt.ylabel('Rating')
plt.title('Top 10 Best Performing Movies')
plt.show()


# ## Distribution of Category

# In[30]:


plt.figure(figsize = (10, 5))
sns.countplot(data = df, x = "category")
plt.title("Movies and TV Shows Release Categories", fontsize = 15)
plt.xlabel('Category')
plt.ylabel('Count')


# ## Genre with the most releases

# In[31]:


pyo.init_notebook_mode()

fig_tree = px.treemap(df, path=[px.Constant("Distribution of Geners"),'genre_1'])
fig_tree.update_layout(title='Highest release in Geners',
                  margin=dict(t=50, b=0, l=70, r=40),
                  plot_bgcolor='#333', paper_bgcolor='#333',
                  title_font=dict(size=25, color='#fff', family="Lato, sans-serif"),
                  font=dict(color='#8a8d93'),
                  hoverlabel=dict(bgcolor="#444", font_size=13, font_family="Lato, sans-serif"))


# ## Total number of releases each year

# In[32]:


sns.displot(df, x="year", hue="category", kind="kde", fill=True)
plt.title("Number of movie releases each year")
plt.ylabel('Number of releases')
plt.xlabel('Year')


# ## Top 10 countries with most releases

# In[33]:


df["countries"] = df["country"].apply(lambda x : True if x.find(',') != -1 else False)
df["countries"].value_counts()


# In[34]:


country_df = df.where(df['countries'] == False)['country'].value_counts().to_frame()
country_df.reset_index(inplace = True)
country_df.rename(columns = {"index" : "country", "country" : "count"}, inplace = True)
country_df.head(11)


# In[35]:


plt.figure(figsize = (16, 9))
plt.title("Top 10 Countries with most releases", fontsize = 25)
plt.ylabel('Countries', fontsize = 15)
plt.xlabel('Count', fontsize = 15)
data = country_df[-(country_df.country == 'Missing')]
g = data.groupby('country', as_index=False)['count'].sum().sort_values(by='count', ascending=False).head(10)
ax = sns.barplot(data=g, y = 'country', x='count')


# ## Top 10 Directors

# In[36]:


plt.figure(figsize = (16, 9))
ax = sns.countplot(data = df[-(df.director == 'Missing')], y = 'director', order=df['director'].value_counts().index[1:11])
plt.title("Top 10 Directors", fontsize = 15)
plt.xlabel('Count')
plt.ylabel('Directors')

