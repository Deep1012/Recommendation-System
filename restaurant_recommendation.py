import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import linear_kernel
import streamlit as st

# Set page config with restaurant icon
st.set_page_config(page_title="Restaurant Recommendation System", page_icon="🍽️")

# Load data
data = pd.read_csv('Vadodara_data.csv')
zomato = data.drop(['Sr No.', 'URL', 'Phone'], axis=1)
zomato.drop_duplicates(inplace=True)
zomato.dropna(how='any', inplace=True)
zomato = zomato.rename(columns={'Cost (for two)': 'Cost'})

# Cleaning data
zomato['Cost'] = zomato['Cost'].astype(str).apply(lambda x: x.replace(',', '.')).astype(float)
zomato = zomato.loc[zomato.Ratings != 'NEW']
zomato = zomato.loc[zomato.Ratings != '-'].reset_index(drop=True)
remove_slash = lambda x: x.replace('/5', '') if type(x) == str else x
zomato.Ratings = zomato.Ratings.apply(remove_slash).str.strip().astype('float')

# Computing Mean Rating
restaurants = list(zomato['Name'].unique())
zomato['Mean Rating'] = 0
for i in range(len(restaurants)):
    zomato.loc[zomato['Name'] == restaurants[i], 'Mean Rating'] = zomato.loc[zomato['Name'] == restaurants[i], 'Ratings'].mean()

# Scaling ratings
scaler = MinMaxScaler(feature_range=(1, 5))
zomato[['Mean Rating']] = scaler.fit_transform(zomato[['Mean Rating']]).round(2)

# Cleaning reviews
PUNCT_TO_REMOVE = string.punctuation
zomato["Reviews"] = zomato["Reviews"].str.lower().apply(lambda text: text.translate(str.maketrans('', '', PUNCT_TO_REMOVE)))

# Set index
zomato.set_index('Name', inplace=True)
indices = pd.Series(zomato.index)

# TF-IDF vectorization for cuisine and reviews
tfidf_cuisine = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), stop_words='english')
tfidf_matrix_cuisine = tfidf_cuisine.fit_transform(zomato['Cuisine'])
cosine_similarities_cuisine = linear_kernel(tfidf_matrix_cuisine, tfidf_matrix_cuisine)

tfidf_reviews = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), stop_words='english')
tfidf_matrix_reviews = tfidf_reviews.fit_transform(zomato['Reviews'])
cosine_similarities_reviews = linear_kernel(tfidf_matrix_reviews, tfidf_matrix_reviews)

# Function to recommend by cuisine
def recommend_by_cuisine(Name, cosine_similarities=cosine_similarities_cuisine):
    if Name not in indices.values:
        st.write(f"Restaurant '{Name}' not found.")
        return
    idx = indices[indices == Name].index[0]
    score_series = pd.Series(cosine_similarities[idx]).sort_values(ascending=False)
    top10_indexes = list(score_series.iloc[1:11].index)
    recommend_restaurant = [zomato.index[each] for each in top10_indexes]
    df_new = zomato.loc[recommend_restaurant, ['Cuisine', 'Mean Rating', 'Cost', 'Reviews']].drop_duplicates()
    df_new = df_new.sort_values(by='Mean Rating', ascending=False).head(3)
    st.write(f"Top {len(df_new)} restaurants with similar cuisines to {Name}:")
    st.write(df_new)

# Function to recommend by reviews
def recommend_by_reviews(Name, cosine_similarities=cosine_similarities_reviews):
    if Name not in indices.values:
        st.write(f"Restaurant '{Name}' not found.")
        return
    idx = indices[indices == Name].index[0]
    score_series = pd.Series(cosine_similarities[idx]).sort_values(ascending=False)
    top10_indexes = list(score_series.iloc[1:11].index)
    recommend_restaurant = [zomato.index[each] for each in top10_indexes]
    df_new = zomato.loc[recommend_restaurant, ['Cuisine', 'Mean Rating', 'Cost', 'Reviews']].drop_duplicates()
    df_new = df_new.sort_values(by='Mean Rating', ascending=False).head(3)
    st.write(f"Top {len(df_new)} restaurants with similar reviews to {Name}:")
    st.write(df_new)

# Streamlit UI
st.title("Restaurant Recommendation System")

option = st.sidebar.selectbox('Select Recommendation Type', ('By Cuisine', 'By Reviews'))

name = st.text_input('Enter a Restaurant Name')

if st.button('Recommend'):
    if option == 'By Cuisine':
        recommend_by_cuisine(name)
    elif option == 'By Reviews':
        recommend_by_reviews(name)

# Checkbox to show dataframe
if st.sidebar.checkbox('Show DataFrame'):
    st.write(zomato)