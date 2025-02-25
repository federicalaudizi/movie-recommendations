import streamlit as st
import pandas as pd
import numpy as np
from movie_recommender import MovieVAE, load_and_preprocess_data

def load_movie_data(movies_path='ml-100k/ml-100k/u.item'):
    """Load movie titles and genres."""
    column_names = [
        'movie_id', 
        'title', 
        'release_date',
        'video_release_date',
        'imdb_url'
    ] + [
        'unknown', 'action', 'adventure', 'animation',
        'children', 'comedy', 'crime', 'documentary', 'drama', 'fantasy',
        'film_noir', 'horror', 'musical', 'mystery', 'romance', 'sci_fi',
        'thriller', 'war', 'western'
    ]
    
    movies = pd.read_csv(
        movies_path,
        sep='|',
        encoding='latin-1',
        header=None,
        names=column_names
    )
    
    return movies

def get_user_ratings(movies_df, n_movies_to_rate=10):
    """Get user ratings for a sample of movies."""
    st.subheader("Rate some movies")
    
    # Initialize session state for ratings if it doesn't exist
    if 'user_ratings' not in st.session_state:
        st.session_state.user_ratings = {}
    
    # Sample random movies - but only once!
    if 'sample_movies' not in st.session_state:
        st.session_state.sample_movies = movies_df.sample(n=n_movies_to_rate)
    
    # Create sliders and store ratings in session state
    for _, movie in st.session_state.sample_movies.iterrows():
        movie_id = movie['movie_id']
        # Get existing rating from session state or default to 0
        current_rating = st.session_state.user_ratings.get(movie_id, 0.0)
        
        rating = st.slider(
            f"{movie['title']}",
            min_value=0.0,
            max_value=5.0,
            step=0.5,
            value=current_rating  # Use the stored rating as the current value
        )
        st.session_state.user_ratings[movie_id] = rating
    
    return st.session_state.user_ratings

def create_user_vector(user_ratings, n_movies):
    """Convert user ratings to vector format."""
    user_vector = np.zeros(n_movies)
    if user_ratings:
        for movie_id, rating in user_ratings.items():
            movie_idx = int(movie_id) - 1
            user_vector[movie_idx] = rating / 5.0  # Normalize to [0,1]
    return user_vector

def get_recommendations(model, user_vector, movies_df, n_recommendations=10):
    """Get movie recommendations for user."""
    # Reshape user vector for prediction
    user_vector_reshaped = user_vector.reshape(1, -1)
    
    # Get predictions
    predictions = model.predict(user_vector_reshaped)[0]

    # Get indices of top recommended movies
    # Exclude movies the user has already rated
    rated_movies = np.where(user_vector > 0)[0]
    predictions[rated_movies] = -1
    
    top_movie_indices = np.argsort(predictions)[::-1][:n_recommendations]
    
    # Get movie details
    recommendations = []
    for idx in top_movie_indices:
        movie = movies_df.iloc[idx]
        recommendations.append({
            'title': movie['title'],
            'predicted_rating': predictions[idx] * 5  # Denormalize
        })
    
    return recommendations

def main():
    st.title("Movie Recommender System")
    
    # Load data
    try:
        user_movie_matrix = load_and_preprocess_data()
        movies_df = load_movie_data()
        
        # Initialize and load model
        n_users, n_movies = user_movie_matrix.shape
        model = MovieVAE(n_users, 1682)
        model.build_model()
        
        # Initialize session state for showing recommendations
        if 'show_recommendations' not in st.session_state:
            st.session_state.show_recommendations = False
        
        # Get user ratings
        user_ratings = get_user_ratings(movies_df)
        
        # Save ratings button
        if st.button("Save Ratings"):
            st.session_state.user_ratings = user_ratings
            st.session_state.show_recommendations = True
            st.success("Ratings saved! Showing recommendations below.")
        
        # Show recommendations section if ratings are saved
        if st.session_state.show_recommendations and 'user_ratings' in st.session_state:
            st.markdown("---")
            st.subheader("Your Recommended Movies")
            
            user_vector = create_user_vector(
                st.session_state.user_ratings,
                n_movies
            )
            
            recommendations = get_recommendations(
                model,
                user_vector,
                movies_df
            )
            
            for rec in recommendations:
                st.write(
                    f"{rec['title']} "
                    f"(Predicted Rating: {rec['predicted_rating']:.1f})"
                )
    
    except Exception as e:
        import traceback
        st.error(f"Error loading data or model: {str(e)}")
        st.error(traceback.format_exc())

if __name__ == "__main__":
    main()