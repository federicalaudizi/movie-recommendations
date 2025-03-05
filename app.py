import streamlit as st
import pandas as pd
import numpy as np
from movie_recommender import MovieVAE, load_and_preprocess_data

# Dummy user credentials 
USER_CREDENTIALS = {
    "user1": "password123",
    "user2": "movierec2024"
}

def init_session_state():
    """Initialize session state variables"""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "page" not in st.session_state:
        st.session_state.page = "login"
    if "username" not in st.session_state:
        st.session_state.username = None   
    if "guest" not in st.session_state:
        st.session_state.guest = False
    if "selected_genres" not in st.session_state:
        st.session_state.selected_genres = []
    if "user_ratings" not in st.session_state:
        st.session_state.user_ratings = {}
    if "sample_movies" not in st.session_state:
        st.session_state.sample_movies = None
    if "show_recommendations" not in st.session_state:
        st.session_state.show_recommendations = False

def navigate_to(page):
    """Helper function to navigate between pages"""
    st.session_state.page = page

def auth_page():
    """Authentication page with user and guest login options"""
    st.title("Movie Recommender Login")
    
    login_option = st.radio("Login as:", ["User", "Guest"])
    
    if login_option == "User":
        # User login form
        st.subheader("User Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
                st.session_state.authenticated = True
                st.session_state.username = username
                st.session_state.guest = False
                navigate_to("rating")
                st.success(f"Welcome, {username}!")
                st.rerun()
            else:
                st.error("Invalid username or password. Try again.")
    else:
        # Guest login with genre preferences
        st.subheader("Guest Login")
        st.write("Select your preferred genres to receive recommendations.")

        genres = ["Action", "Comedy", "Drama", "Sci-Fi", "Romance", "Thriller"]
        selected_genres = st.multiselect("Choose genres", genres)

        if st.button("Continue as Guest"):
            st.session_state.authenticated = True
            st.session_state.guest = True
            st.session_state.selected_genres = selected_genres
            navigate_to("rating")
            st.success("You are logged in as a guest!")
            st.rerun()

def load_movie_data(movies_path='ml-100k/ml-100k/u.item'):
    """Load movie titles and genres."""
    column_names = [
        'movie_id', 
        'title', 
        'release_date',
        'video_release_date',
        'imdb_url'
    ] + [
        'Unknown', 'Action', 'Adventure', 'Animation',
        'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
        'Film_noir', 'horror', 'musical', 'mystery', 'Romance', 'Sci-Fi',
        'Thriller', 'war', 'western'
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
    
    # Sample random movies if not already sampled
    if st.session_state.sample_movies is None:
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

def rating_page():
    """Movie rating page"""
    # Show logout button
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("Movie Recommender System")
        if st.session_state.guest:
            st.write(f"Logged in as: Guest")
        else:
            st.write(f"Logged in as: {st.session_state.username}")
    
    with col2:
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.page = "login"
            st.session_state.username = None
            st.session_state.guest = False
            st.session_state.user_ratings = {}
            st.session_state.sample_movies = None
            st.session_state.show_recommendations = False
            st.rerun()

    # Load data
    user_movie_matrix = load_and_preprocess_data()
    movies_df = load_movie_data()

    # Initialize model
    n_users, n_movies = user_movie_matrix.shape
    model = MovieVAE(n_users, 1682)
    model.build_model()

    # Show content based on user type
    if st.session_state.guest:
        st.subheader("Recommendations Based on Your Selected Genres")
        st.write("You chose:", ", ".join(st.session_state.selected_genres))
        # Here you would implement genre-based recommendations
        movies_df = load_movie_data()
        
        # Show genre-based recommendations
        if st.session_state.selected_genres:
            genre_movies = get_genre_based_movies(movies_df, st.session_state.selected_genres)
            st.write(genre_movies)
        else:
            # Show popular movies as a fallback
            st.write("Top Rated Movies:")
            popular_movies = get_popular_movies(movies_df)
            st.write(popular_movies)
        
    else:
        # Get user ratings
        user_ratings = get_user_ratings(movies_df)
        
        if st.button("Get Recommendations"):
            st.session_state.user_ratings = user_ratings
            st.session_state.show_recommendations = True
            navigate_to("recommendations")
            st.rerun()

def get_popular_movies(movies_df, n=10):
    """Get top-rated movies for guests."""
    ratings = pd.read_csv('ml-100k/ml-100k/u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])
    top_movies = ratings.groupby('movie_id').mean()['rating'].sort_values(ascending=False).head(n)
    return movies_df[movies_df['movie_id'].isin(top_movies.index)][['title']]

def get_genre_based_movies(movies_df, selected_genres, n=10):
    """Recommend movies based on selected genres."""
    filtered_movies = movies_df[movies_df[selected_genres].sum(axis=1) > 0]
    return filtered_movies.sample(n=min(n, len(filtered_movies)))[['title']]

def recommendations_page():
    """Recommendations page"""
    st.title("Your Movie Recommendations")
    
    # Show navigation options
    cols = st.columns([1, 1, 2])
    with cols[0]:
        if st.button("Back to Ratings"):
            navigate_to("rating")
            st.rerun()
    with cols[1]:
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.page = "login"
            st.session_state.username = None
            st.session_state.guest = False
            st.session_state.user_ratings = {}
            st.session_state.sample_movies = None
            st.session_state.show_recommendations = False
            st.rerun()
    
    # Load data and model
    user_movie_matrix = load_and_preprocess_data()
    movies_df = load_movie_data()
    
    n_users, n_movies = user_movie_matrix.shape
    model = MovieVAE(n_users, 1682)
    model.build_model()
    
    # Display recommendations
    user_vector = create_user_vector(st.session_state.user_ratings, n_movies)
    recommendations = get_recommendations(model, user_vector, movies_df)
    
    st.subheader("Movies You Might Like")
    for i, rec in enumerate(recommendations, 1):
        st.write(f"{i}. {rec['title']} (Predicted Rating: {rec['predicted_rating']:.1f})")
    
    # Show what the user rated
    st.subheader("Movies You Rated")
    rated_movies = []
    for movie_id, rating in st.session_state.user_ratings.items():
        if rating > 0:
            movie = movies_df[movies_df['movie_id'] == movie_id]
            if not movie.empty:
                rated_movies.append({
                    'title': movie['title'].values[0],
                    'rating': rating
                })
    
    for rated in sorted(rated_movies, key=lambda x: x['rating'], reverse=True):
        st.write(f"{rated['title']} - Your Rating: {rated['rating']}")

def main():
    # Initialize session state
    init_session_state()
    
    # Check authentication
    if not st.session_state.authenticated:
        auth_page()
    else:
        if st.session_state.page == "login":
            auth_page()
        elif st.session_state.page == "rating":
            rating_page()
        elif st.session_state.page == "recommendations":
            recommendations_page()

if __name__ == "__main__":
    main()