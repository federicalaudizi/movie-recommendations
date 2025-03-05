import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from movie_recommender import MovieVAE
# Reuse the data loading function from the original code
def load_and_preprocess_data(ratings_path='ml-100k/ml-100k/u.data', 
                             item_path='ml-100k/ml-100k/u.item'):
    """Load and preprocess MovieLens data."""
    # Load ratings
    columns = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings = pd.read_csv(ratings_path, sep='\t', names=columns)
    
    # Create user-movie matrix
    user_movie_matrix = ratings.pivot(
        index='user_id',
        columns='movie_id',
        values='rating'
    ).fillna(0)
    
    # Load movie metadata for knowledge-based filtering
    movie_columns = ['movie_id', 'title', 'release_date', 'video_release_date', 
                     'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation',
                     'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                     'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                     'Thriller', 'War', 'Western']
    
    # Try to load movie metadata if available
    try:
        movies = pd.read_csv(item_path, sep='|', names=movie_columns, encoding='latin-1')
        # Extract genre columns (binary indicators)
        genre_columns = movie_columns[5:]
        # Create a movie feature matrix
        movie_features = movies[genre_columns].values
    except:
        # If file not available, we'll create dummy features
        movie_features = np.random.randint(0, 2, size=(user_movie_matrix.shape[1], 19))
    
    return user_movie_matrix, ratings, movie_features


class CollaborativeFiltering:
    """User-based collaborative filtering"""
    
    def __init__(self, user_movie_matrix, k=10):
        """
        Initialize the collaborative filtering model.
        
        Parameters:
        user_movie_matrix: DataFrame with users as rows and movies as columns
        k: Number of nearest neighbors to consider
        """
        self.user_movie_matrix = user_movie_matrix
        self.k = k
        self.similarity_matrix = None
        
    def fit(self):
        """Compute user similarity matrix based on cosine similarity"""
        # Fill NaN values with 0
        matrix = self.user_movie_matrix.fillna(0).values
        
        # Compute cosine similarity between users
        self.similarity_matrix = cosine_similarity(matrix)
        
        # Set self-similarity to 0 to avoid recommending already watched movies
        np.fill_diagonal(self.similarity_matrix, 0)
        
    def predict(self, user_idx):
        """
        Predict ratings for all movies for a given user.
        
        Parameters:
        user_idx: Index of the user in the user_movie_matrix
        
        Returns:
        Array of predicted ratings for all movies
        """
        if self.similarity_matrix is None:
            self.fit()
            
        # Get user's ratings
        user_ratings = self.user_movie_matrix.iloc[user_idx].values
        
        # Get similarity scores for this user with all other users
        similarities = self.similarity_matrix[user_idx]
        
        # Find top-k similar users
        most_similar_users = np.argsort(similarities)[::-1][:self.k]
        
        # Get the ratings from these similar users
        similar_user_ratings = self.user_movie_matrix.iloc[most_similar_users].values
        
        # Weight the ratings by similarity
        weighted_ratings = similar_user_ratings * similarities[most_similar_users].reshape(-1, 1)
        
        # Sum of weights
        sum_of_weights = np.sum(np.abs(similarities[most_similar_users]))
        
        if sum_of_weights > 0:
            # Weighted average of ratings
            weighted_average = np.sum(weighted_ratings, axis=0) / sum_of_weights
        else:
            # If sum of weights is 0, use average rating
            weighted_average = np.mean(similar_user_ratings, axis=0)
        
        # Don't recommend already rated movies
        weighted_average[user_ratings > 0] = user_ratings[user_ratings > 0]
        
        return weighted_average


class KnowledgeBasedRecommender:
    """Knowledge-based recommender using movie features"""
    
    def __init__(self, movie_features, user_movie_matrix):
        """
        Initialize the knowledge-based recommender.
        
        Parameters:
        movie_features: Matrix where each row represents a movie and columns are features
        user_movie_matrix: DataFrame with users as rows and movies as columns
        """
        self.movie_features = movie_features
        self.user_movie_matrix = user_movie_matrix
        self.user_profiles = None
        
    def fit(self):
        """Build user profiles based on their rated movies"""
        # Create user profiles based on weighted average of movie features
        n_users = self.user_movie_matrix.shape[0]
        n_features = self.movie_features.shape[1]
        
        self.user_profiles = np.zeros((n_users, n_features))
        
        for u in range(n_users):
            # Get user's ratings
            user_ratings = self.user_movie_matrix.iloc[u].values
            
            # Skip if user has no ratings
            if np.sum(user_ratings > 0) == 0:
                continue
                
            # Calculate weighted average of movie features
            for m in range(len(user_ratings)):
                if user_ratings[m] > 0:
                    self.user_profiles[u] += self.movie_features[m] * user_ratings[m]
                    
            # Normalize profile
            rating_sum = np.sum(user_ratings[user_ratings > 0])
            if rating_sum > 0:
                self.user_profiles[u] /= rating_sum
    
    def predict(self, user_idx):
        """
        Predict ratings for a user based on similarity between user profile and movie features.
        
        Parameters:
        user_idx: Index of the user in the user_movie_matrix
        
        Returns:
        Array of predicted ratings for all movies
        """
        if self.user_profiles is None:
            self.fit()
            
        # Get user profile
        user_profile = self.user_profiles[user_idx]
        
        # Calculate similarity between user profile and each movie
        similarities = np.zeros(self.movie_features.shape[0])
        
        for m in range(self.movie_features.shape[0]):
            # Cosine similarity
            dot_product = np.dot(user_profile, self.movie_features[m])
            user_norm = np.linalg.norm(user_profile)
            movie_norm = np.linalg.norm(self.movie_features[m])
            
            if user_norm > 0 and movie_norm > 0:
                similarities[m] = dot_product / (user_norm * movie_norm)
        
        # Scale similarities to the rating range [0, 5]
        predictions = similarities * 5
        
        # Keep original ratings for already rated movies
        user_ratings = self.user_movie_matrix.iloc[user_idx].values
        predictions[user_ratings > 0] = user_ratings[user_ratings > 0]
        
        return predictions


class HybridRecommender:
    """Hybrid recommender combining collaborative and knowledge-based approaches"""
    
    def __init__(self, cf_model, kb_model, alpha=0.6):
        """
        Initialize the hybrid recommender.
        
        Parameters:
        cf_model: Collaborative filtering model
        kb_model: Knowledge-based model
        alpha: Weight for the collaborative filtering predictions (0-1)
        """
        self.cf_model = cf_model
        self.kb_model = kb_model
        self.alpha = alpha
        
    def fit(self):
        """Fit both underlying models"""
        self.cf_model.fit()
        self.kb_model.fit()
        
    def predict(self, user_idx):
        """
        Generate hybrid predictions.
        
        Parameters:
        user_idx: Index of the user in the user_movie_matrix
        
        Returns:
        Array of predicted ratings for all movies
        """
        # Get predictions from both models
        cf_predictions = self.cf_model.predict(user_idx)
        kb_predictions = self.kb_model.predict(user_idx)
        
        # Combine predictions
        hybrid_predictions = self.alpha * cf_predictions + (1 - self.alpha) * kb_predictions
        
        return hybrid_predictions


# Evaluation functions (reused from the original code)
def compute_precision_recall_f1(test_data, predictions, k):
    """Compute precision, recall, and F1 score."""
    n_users = test_data.shape[0]
    
    precisions = []
    recalls = []
    f1s = []
    
    for i in range(n_users):
        top_k_preds = np.argsort(-predictions[i])[:k]
        true_positives = np.intersect1d(np.where(test_data[i] > 0), top_k_preds)
        
        precision = len(true_positives) / k
        recall = len(true_positives) / len(np.where(test_data[i] > 0)[0])
        
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    
    return np.mean(precisions), np.mean(recalls), np.mean(f1s)

def evaluate_model(test_data, predictions, model_name):
    """Calculate evaluation metrics."""
    mse = np.mean(np.square(test_data - predictions))
    mae = np.mean(np.abs(test_data - predictions))
    rmse = np.sqrt(mse)
    
    # Calculate precision, recall, and F1 for top-10 recommendations
    precision, recall, f1 = compute_precision_recall_f1(test_data, predictions, 10)

    return {
        'model': model_name,
        'rmse': rmse,
        'mae': mae,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def compare_models():
    """Compare different recommendation approaches"""
    # Load and preprocess data
    user_movie_matrix, ratings, movie_features = load_and_preprocess_data()
    
    # Split data
    train_matrix, test_matrix = train_test_split(
        user_movie_matrix,
        test_size=0.2,
        random_state=42
    )

    #Initialize VAE model
    n_users, n_movies = user_movie_matrix.shape
    model = MovieVAE(n_users, n_movies)
    model.build_model()
    model.train(train_matrix.values, test_matrix.values)
    model_predictions = model.predict(test_matrix.values)
    vae_results = evaluate_model(test_matrix.values, model_predictions, "VAE")
    
    # Initialize collaborative filtering model
    cf_model = CollaborativeFiltering(train_matrix, k=20)
    cf_model.fit()
    
    # Initialize knowledge-based model
    kb_model = KnowledgeBasedRecommender(movie_features, train_matrix)
    kb_model.fit()
    
    # Initialize hybrid model
    hybrid_model = HybridRecommender(cf_model, kb_model, alpha=0.7)
    hybrid_model.fit()
    
    # Make predictions and evaluate
    results = []
    results.append(vae_results)

    # Evaluate collaborative filtering
    cf_predictions = np.array([cf_model.predict(i) for i in range(test_matrix.shape[0])])
    cf_results = evaluate_model(test_matrix.values, cf_predictions, "Collaborative Filtering")
    results.append(cf_results)
    
    # Evaluate knowledge-based
    kb_predictions = np.array([kb_model.predict(i) for i in range(test_matrix.shape[0])])
    kb_results = evaluate_model(test_matrix.values, kb_predictions, "Knowledge-Based")
    results.append(kb_results)
    
    # Evaluate hybrid
    hybrid_predictions = np.array([hybrid_model.predict(i) for i in range(test_matrix.shape[0])])
    hybrid_results = evaluate_model(test_matrix.values, hybrid_predictions, "Hybrid")
    results.append(hybrid_results)
    
    # Return results as DataFrame
    results_df = pd.DataFrame(results)
    return results_df

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from movie_recommender import MovieVAE

# Reuse all the existing code from the original file
# [... All the existing code remains the same ...]

# Add visualization functions
def plot_model_comparison(results_df):
    """Create bar charts comparing model performance metrics."""
    metrics = ['rmse', 'mae', 'precision', 'recall', 'f1']
    _, axes = plt.subplots(len(metrics), 1, figsize=(10, 15))

    for i, metric in enumerate(metrics):
        sns.barplot(x='model', y=metric, data=results_df, ax=axes[i])
        axes[i].set_title(f'Model Comparison - {metric.upper()}')
        axes[i].set_ylabel(metric.upper())
        axes[i].set_xlabel('')
        
        # Add value labels on top of each bar
        for p in axes[i].patches:
            axes[i].annotate(f'{p.get_height():.4f}', 
                             (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.close()

def plot_similarity_matrix(cf_model, n_users=20):
    """Plot the user similarity matrix from collaborative filtering."""
    if cf_model.similarity_matrix is None:
        cf_model.fit()
    
    # Get a subset of the similarity matrix for better visualization
    similarity_subset = cf_model.similarity_matrix[:n_users, :n_users]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_subset, cmap="coolwarm", annot=False, linewidths=.5)
    plt.title(f'User Similarity Matrix (Sample of {n_users} users)')
    plt.xlabel('User ID')
    plt.ylabel('User ID')
    plt.savefig('similarity_matrix.png')
    plt.close()

def plot_alpha_sensitivity(user_movie_matrix, movie_features, test_matrix):
    """Plot the sensitivity of hybrid model to alpha parameter."""
    # Initialize models
    cf_model = CollaborativeFiltering(user_movie_matrix, k=20)
    cf_model.fit()
    
    kb_model = KnowledgeBasedRecommender(movie_features, user_movie_matrix)
    kb_model.fit()
    
    # Test various alpha values
    alphas = np.arange(0, 1.1, 0.1)
    rmse_values = []
    precision_values = []
    recall_values = []
    f1_values = []
    
    for alpha in alphas:
        hybrid_model = HybridRecommender(cf_model, kb_model, alpha=alpha)
        hybrid_model.fit()
        
        hybrid_predictions = np.array([hybrid_model.predict(i) for i in range(test_matrix.shape[0])])
        
        # Calculate metrics
        mse = np.mean(np.square(test_matrix.values - hybrid_predictions))
        rmse = np.sqrt(mse)
        precision, recall, f1 = compute_precision_recall_f1(test_matrix.values, hybrid_predictions, 10)
        
        rmse_values.append(rmse)
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
    
    # Plot results
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(alphas, rmse_values, 'o-', linewidth=2)
    plt.title('RMSE vs Alpha')
    plt.xlabel('Alpha')
    plt.ylabel('RMSE')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(alphas, precision_values, 'o-', linewidth=2)
    plt.title('Precision@10 vs Alpha')
    plt.xlabel('Alpha')
    plt.ylabel('Precision')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(alphas, recall_values, 'o-', linewidth=2)
    plt.title('Recall@10 vs Alpha')
    plt.xlabel('Alpha')
    plt.ylabel('Recall')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(alphas, f1_values, 'o-', linewidth=2)
    plt.title('F1@10 vs Alpha')
    plt.xlabel('Alpha')
    plt.ylabel('F1 Score')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('alpha_sensitivity.png')
    plt.close()

def generate_all_visualizations():
    """Generate all visualization plots."""
    # Load and preprocess data
    user_movie_matrix, ratings, movie_features = load_and_preprocess_data()
    
    # Split data
    train_matrix, test_matrix = train_test_split(
        user_movie_matrix, 
        test_size=0.2, 
        random_state=42
    )
    
    
    # Initialize and train models
    # Initialize VAE model
    n_users, n_movies = user_movie_matrix.shape
    vae_model = MovieVAE(n_users, n_movies)
    vae_model.build_model()
    vae_model.train(train_matrix.values, test_matrix.values)
    vae_predictions = vae_model.predict(test_matrix.values)
    vae_results = evaluate_model(test_matrix.values, vae_predictions, "VAE")
    
    # Initialize collaborative filtering model
    cf_model = CollaborativeFiltering(train_matrix, k=20)
    cf_model.fit()
    
    # Initialize knowledge-based model
    kb_model = KnowledgeBasedRecommender(movie_features, train_matrix)
    kb_model.fit()
    
    # Initialize hybrid model
    hybrid_model = HybridRecommender(cf_model, kb_model, alpha=0.7)
    hybrid_model.fit()
    
    # Plot similarity matrix
    plot_similarity_matrix(cf_model)
    
    # Make predictions and evaluate
    results = []
    results.append(vae_results)
    
    # Evaluate collaborative filtering
    cf_predictions = np.array([cf_model.predict(i) for i in range(test_matrix.shape[0])])
    cf_results = evaluate_model(test_matrix.values, cf_predictions, "Collaborative Filtering")
    results.append(cf_results)
    
    # Evaluate knowledge-based
    kb_predictions = np.array([kb_model.predict(i) for i in range(test_matrix.shape[0])])
    kb_results = evaluate_model(test_matrix.values, kb_predictions, "Knowledge-Based")
    results.append(kb_results)
    
    # Evaluate hybrid
    hybrid_predictions = np.array([hybrid_model.predict(i) for i in range(test_matrix.shape[0])])
    hybrid_results = evaluate_model(test_matrix.values, hybrid_predictions, "Hybrid")
    results.append(hybrid_results)
    
    # Create DataFrame of results
    results_df = pd.DataFrame(results)
    
    # Generate visualizations
    plot_model_comparison(results_df)
    plot_alpha_sensitivity(train_matrix, movie_features, test_matrix)
    
    return results_df

if __name__ == "__main__":
    results = compare_models()
    print("\nResults Summary:")
    print(results)

    # Generate visualizations
    print("\nGenerating visualizations...")
    generate_all_visualizations()
    print("Visualizations saved to disk.")