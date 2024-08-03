import numpy as np
import pandas as pd 
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
import string
import nltk
from nltk.corpus import stopwords
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
import streamlit as st

# Suppressing warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# Load data
data = pd.read_csv('Vadodara_data.csv')
zomato=data.drop(['Sr No.','URL','Phone'],axis=1)
zomato.drop_duplicates(inplace=True)
zomato.dropna(how='any',inplace=True)
zomato = zomato.rename(columns={'Cost (for two)':'Cost'})

# Cleaning data
zomato['Cost'] = zomato['Cost'].astype(str).apply(lambda x: x.replace(',','.')).astype(float)
zomato = zomato.loc[zomato.Ratings !='NEW']
zomato = zomato.loc[zomato.Ratings !='-'].reset_index(drop=True)
remove_slash = lambda x: x.replace('/5', '') if type(x) == str else x
zomato.Ratings = zomato.Ratings.apply(remove_slash).str.strip().astype('float')

# Computing Mean Rating
restaurants = list(zomato['Name'].unique())
zomato['Mean Rating'] = 0

for i in range(len(restaurants)):
    zomato['Mean Rating'][zomato['Name'] == restaurants[i]] = zomato['Ratings'][zomato['Name'] == restaurants[i]].mean()

# Scaling ratings
scaler = MinMaxScaler(feature_range = (1,5))
zomato[['Mean Rating']] = scaler.fit_transform(zomato[['Mean Rating']]).round(2)

# Cleaning reviews
zomato["Reviews"] = zomato["Reviews"].str.lower()

# Removal of Punctuation
PUNCT_TO_REMOVE = string.punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

zomato["Reviews"] = zomato["Reviews"].apply(lambda text: remove_punctuation(text))

# Function to ensure NLTK stopwords are downloaded
def download_nltk_stopwords():
    try:
        nltk.data.find('corpora/stopwords.zip')
    except LookupError:
        nltk.download('stopwords')

# Ensure NLTK stopwords are downloaded
download_nltk_stopwords()

# Removal of Stopwords
STOPWORDS = set(stopwords.words('english'))
def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

zomato["Reviews"] = zomato["Reviews"].apply(lambda text: remove_stopwords(text))

# Set index
zomato = zomato.drop(['ReviewsCount','City', 'Online_order'],axis=1)
zomato.set_index('Name', inplace=True)

# Indices
indices = pd.Series(zomato.index)

# TF-IDF vectorization for cuisine
tfidf_cuisine = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=1, stop_words='english')
tfidf_matrix_cuisine = tfidf_cuisine.fit_transform(zomato['Cuisine'])
cosine_similarities_cuisine = linear_kernel(tfidf_matrix_cuisine, tfidf_matrix_cuisine)

# TF-IDF vectorization for reviews
tfidf_reviews = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=1, stop_words='english')
tfidf_matrix_reviews = tfidf_reviews.fit_transform(zomato['Reviews'])
cosine_similarities_reviews = linear_kernel(tfidf_matrix_reviews, tfidf_matrix_reviews)

# Function to recommend by cuisine
def recommend_by_cuisine(Name, cosine_similarities=cosine_similarities_cuisine):
    recommend_restaurant = []
    idx = indices[indices == Name].index[0]
    score_series = pd.Series(cosine_similarities[idx]).sort_values(ascending=False)
    top10_indexes = list(score_series.iloc[1:11].index)
    for each in top10_indexes:
        recommend_restaurant.append(list(zomato.index)[each])
    df_new = []
    for each in recommend_restaurant:
        df_new.append(zomato.loc[each, ['Cuisine', 'Mean Rating', 'Cost', 'Reviews']])
    df_new = pd.DataFrame(df_new)
    df_new = df_new.drop_duplicates(subset=['Cuisine', 'Mean Rating', 'Cost', 'Reviews'], keep=False)
    df_new = df_new.sort_values(by='Mean Rating', ascending=False).head(3)
    
    st.write('TOP %s RESTAURANTS WITH SIMILAR CUISINES TO %s: ' % (str(len(df_new)), Name))
    st.write(df_new)

# Function to recommend by reviews
def recommend_by_reviews(Name, cosine_similarities=cosine_similarities_reviews):
    recommend_restaurant = []
    idx = indices[indices == Name].index[0]
    if not idx:
        st.write("No restaurants found with name '%s'." % Name)
        return None
    score_series = pd.Series(cosine_similarities[idx]).sort_values(ascending=False)
    top10_indexes = list(score_series.iloc[1:11].index)
    for each in top10_indexes:
        recommend_restaurant.append(list(zomato.index)[each])
    df_new = []
    for each in recommend_restaurant:
        df_new.append(zomato.loc[each, ['Cuisine', 'Location','Mean Rating', 'Cost', 'Reviews']])
    df_new = pd.DataFrame(df_new)
    df_new = df_new.drop_duplicates(subset=['Cuisine','Location', 'Mean Rating', 'Cost', 'Reviews'], keep=False)
    df_new = df_new.sort_values(by='Mean Rating', ascending=False).head(3)
    
    st.write('TOP %s RESTAURANTS WITH SIMILAR REVIEWS TO %s: ' % (str(len(df_new)), Name))
    st.write(df_new)

df_tfidf_cuisine = pd.DataFrame(tfidf_matrix_cuisine.toarray(), columns=tfidf_cuisine.get_feature_names_out())
df_tfidf_reviews = pd.DataFrame(tfidf_matrix_reviews.toarray(), columns=tfidf_reviews.get_feature_names_out())

# Save DataFrames to Excel
with pd.ExcelWriter('tfidf_matrices.xlsx') as writer:
    df_tfidf_cuisine.to_excel(writer, sheet_name='TF-IDF Cuisine', index=False)
    df_tfidf_reviews.to_excel(writer, sheet_name='TF-IDF Reviews', index=False)

# Streamlit UI
st.title("Restaurant Recommendation System")

option = st.sidebar.selectbox(
    'Select Recommendation Type',
    ('By Cuisine', 'By Reviews'))

if option == 'By Cuisine':
    name = st.text_input('Enter a Restaurant Name for Cuisine Recommendation')
    if st.button('Recommend'):
        recommendations = recommend_by_cuisine(name)
elif option == 'By Reviews':
    name = st.text_input('Enter a Restaurant Name for Reviews Recommendation')
    if st.button('Recommend'):
        recommendations = recommend_by_reviews(name)

# Checkbox to show dataframe
if st.sidebar.checkbox('Show DataFrame'):
    st.write(zomato)
