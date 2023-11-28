# IMDb Data Analysis

Data analysis for the case study follows the following steps:

# <span> Table of Contents </span>
* [1. Prepare](#1-prepare)
* [2. Process](#2-process)
* [3. Analyze and Share](#3-analyze-and-share)
* [4. Tableau Dashboard](#4-tableau-dashboard)

## 1. Prepare
The data used is stored in Kaggle under [Netflix Prize Shows Information (9000 Shows)](https://www.kaggle.com/datasets/akashguna/netflix-prize-shows-information). The dataset contains information like the movie's duration, cast, director, genre, and languages present.

## 2. Process

### 2.1 Loading Libraries

```bash
import pandas as pd
import numpy as np

#Data viz packages
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import plotly.express as px
import plotly.offline as pyo

#NLP
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from wordcloud import WordCloud
```

### 2.2 Loading Dataset
```
df = pd.read_csv(r'/kaggle/input/netflix-prize-shows-information/imdb_processed.csv')
```

### 2.3 Exploring Dataset
```
df.head()
```

### 2.4 Data Cleaning
```
#view the list of columns
df.columns
```
```
#dropping unnamed column
df = df.drop("Unnamed: 0",axis=1)
```
```
#check for any nulls
df.isnull().any()
```
```
#replacing Nan with Missing in string columns
df[['genre', 'country', 'language', 'cast', 'director', 'composer', 'writer']] = df[['genre', 'country', 'language', 'cast', 'director', 'composer', 'writer']].fillna('Missing')
```
```
#replacing Nan with 0 in numeric columns
df[['rating', 'vote', 'runtime']] = df[['rating', 'vote', 'runtime']].fillna('0')
```
```
#check datatypes
df.dtypes
```
```
#converting 'year' from float to int and 'rating' from object to float
df['year'] = df['year'].astype(int)
df['rating'] = df['rating'].astype(float)
```
```
#creating a id, using title and release year
#add the 'Composite_Key' column
df['id'] = df['title'] + '_' + df['year'].astype(str)

#move 'id' column to the beginning
df = df[['id'] + [col for col in df.columns if col != 'id']]

#view the list of columns
df.columns
```
```
#check for no of duplicates present
len(df)-len(df.drop_duplicates())
```
```
#dropping duplicates based on id column
df = df.drop_duplicates('id').sort_index()
```
```
df.shape
```
```
#remove brackets from genre, country, language, director, cast and writer columns
df['genre'] = df['genre'].str.strip('[]')
df['country'] = df['country'].str.strip('[]')
df['language'] = df['director'].str.strip('[]')
df['director'] = df['director'].str.strip('[]')
df['cast'] = df['cast'].str.strip('[]')
df['writer'] = df['writer'].str.strip('[]')
```
```
#remove quotes from genre, country, language, director columns
df['genre'] = df['genre'].str.replace(r"\'","", regex=True)
df['country'] = df['country'].str.replace(r"\'","", regex=True)
df['language'] = df['language'].str.replace(r"\'","", regex=True)
df['director'] = df['director'].str.replace(r"\'","", regex=True)
df['cast'] = df['cast'].str.replace(r"\'","", regex=True)
df['writer'] = df['writer'].str.replace(r"\'","", regex=True)
```
```
#splitting the genre column into sub-categories. Only the first genre category will be considered for this analysis.
N = 8

df[[f'genre {x+1}' for x in range(N)]] = (
    df['genre'].str.split(',', n=N+1, expand=True).iloc[:, :N])
```
```
#rename columns
df.rename(columns = {'genre 1':'genre_1'}, inplace = True)
df.rename(columns = {'kind':'category'}, inplace = True)
```
```
#check for the total no of values in each category
df["category"].value_counts()
```
```
#replacing 'tv mini series' to 'tv series' and 'tv movies', 'video movie' to 'movie'
df.replace(regex=r'tv mini series', value='tv series', inplace=True)
df.replace(regex=r'tv movie', value='movie', inplace=True)
df.replace(regex=r'video movie', value='movie', inplace=True)
```
```
#save to csv
df.to_csv('cleaned_imdb_dataset.csv')
```

 ## 3. Analyze and Share
### 3.0 Setting the theme
```
sns.set_theme()
```
### 3.1 Top 10 Best Performing Movies and TV Shows
```
fig = plt.figure(figsize = (15, 8))
ax = sns.barplot(x = 'title', y = 'rating', data = df.sort_values('rating', ascending=False)[0:10])

# Plot the ratings on top of the bars
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.1f'), 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha = 'center', va = 'center', 
                xytext = (0, 10), 
                textcoords = 'offset points')
    
plot = ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)
plt.xlabel('Movie Title and TV Shows')
plt.ylabel('Rating')
plt.title('Top 10 Best Performing Movies and TV Shows', fontsize = 15)

# Set y-axis ticks
plt.yticks([0, 2, 4, 6, 8, 10])

plt.show()
```
![download](https://github.com/karlyndiary/IMDb-Data-Analysis/assets/116041695/8357967d-85a1-4188-9b6f-9b7f24363247)

### 3.2 Distribution of Category
```
plt.figure(figsize = (10, 5))
sns.countplot(data = df, x = "category")
plt.title("Movies and TV Shows Release Categories", fontsize = 15)
plt.xlabel('Category')
plt.ylabel('Count')
```
![download (1)](https://github.com/karlyndiary/IMDb-Data-Analysis/assets/116041695/20753991-fb8b-48ae-b049-469c02061f34)

### 3.3 Genre with the most releases

```
pyo.init_notebook_mode()

fig_tree = px.treemap(df, path=[px.Constant("Distribution of Geners"),'genre_1'])
fig_tree.update_layout(title='Highest release in Geners',
                  margin=dict(t=50, b=0, l=70, r=40),
                  plot_bgcolor='#333', paper_bgcolor='#333',
                  title_font=dict(size=25, color='#fff', family="Lato, sans-serif"),
                  font=dict(color='#8a8d93'),
                  hoverlabel=dict(bgcolor="#444", font_size=13, font_family="Lato, sans-serif"))
```
![newplot (2)](https://user-images.githubusercontent.com/116041695/216749501-c894c66a-d9b6-4295-a11f-139728c348e1.png)

### 3.4 Total number of releases each year
```
sns.displot(df, x="year", hue="category", kind="kde", fill=True)
plt.title("Number of movie releases each year", fontsize = 15)
plt.ylabel('Number of releases', fontsize = 12)
plt.xlabel('Year', fontsize = 12)
```
![download (2)](https://github.com/karlyndiary/IMDb-Data-Analysis/assets/116041695/53ec23cf-250d-4d48-820a-d8757831e435)

### 3.5 Top 10 countries with most releases

```
df["countries"] = df["country"].apply(lambda x : True if x.find(',') != -1 else False)
df["countries"].value_counts()
```
```
country_df = df.where(df['countries'] == False)['country'].value_counts().to_frame()
country_df.reset_index(inplace = True)
country_df.rename(columns = {"index" : "country", "country" : "count"}, inplace = True)
country_df.head(11)
```
```
plt.figure(figsize = (16, 9))
plt.title("Top 10 Countries with most releases", fontsize = 25)
data = country_df[-(country_df.country == 'Missing')]
g = data.groupby('country', as_index=False)['count'].sum().sort_values(by='count', ascending=False).head(10)
ax = sns.barplot(data=g, y = 'country', x='count')
plt.ylabel('Countries', fontsize = 15)
plt.xlabel('Count', fontsize = 15)
```
![download (3)](https://github.com/karlyndiary/IMDb-Data-Analysis/assets/116041695/2886ce4a-3085-41d5-80ec-76dc2af94c8a)

### 3.6 Top 10 Directors

```
plt.figure(figsize = (16, 9))
ax = sns.countplot(data = df[-(df.director == 'Missing')], y = 'director', order=df['director'].value_counts().index[1:11])
plt.title("Top 10 Directors", fontsize = 25)
plt.xlabel('Count', fontsize = 15)
plt.ylabel('Directors', fontsize = 15)
```
![download (4)](https://github.com/karlyndiary/IMDb-Data-Analysis/assets/116041695/6c363f4e-7d20-416f-a2e0-1d8abd919ac1)

### 3.7 Wordcloud for Cast

```
stop_words = set(stopwords.words('english'))
df['cast_no_stopwords'] = df['cast'].apply(lambda x: [item for item in str(x).split() if item not in stop_words])

all_words = list([a for b in df['cast_no_stopwords'].tolist() for a in b])
all_words_str = ' '.join(all_words) 

def plot_cloud(wordcloud):
    plt.figure(figsize=(30, 20))
    plt.imshow(wordcloud) 
    plt.axis("off");

wordcloud = WordCloud(width = 2000, height = 1000, random_state=1, background_color='white', 
                      colormap='viridis', collocations=False).generate(all_words_str)
plot_cloud(wordcloud)
```
![download (5)](https://github.com/karlyndiary/IMDb-Data-Analysis/assets/116041695/0f64888b-2a9f-4329-bc30-7ddd175e39db)

### 3.8 Time series of Ratings

```
# Create a line chart using Seaborn
plt.figure(figsize=(15, 8))
ax = sns.lineplot(x='year', y='rating', data=df, marker='o')

# Set x-axis ticks for every 5 years
plt.xticks(range(df['year'].min(), df['year'].max() + 1, 5))

plt.xlabel('Years')
plt.ylabel('Rating')
plt.title('Ratings Over the Years', fontsize=15)
plt.grid(True)
plt.tight_layout()  # Adjust spacing between labels
plt.show()
```
![download (6)](https://github.com/karlyndiary/IMDb-Data-Analysis/assets/116041695/5c6a9351-7ffd-4580-91e6-5668928a41f5)

### 3.9 Relationship between ratings and votes

```
plt.figure(figsize=(10, 6))
plt.scatter(x = 'rating', y = 'vote', data = df)
plt.xlabel('Vote')
plt.ylabel('Rating')
plt.title('Relationship between Rating and Vote')
plt.grid(True)
plt.show()
```
![download (7)](https://github.com/karlyndiary/IMDb-Data-Analysis/assets/116041695/4d20d767-9770-40b1-980f-ba461da43943)

## 4. Tableau Dashboard
![Dashboard](https://github.com/karlyndiary/IMDb-Data-Analysis/assets/116041695/fe31ac64-5937-4fe7-9821-8640c2aca4d4)
