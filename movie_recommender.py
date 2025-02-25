import numpy as np
import pandas as pd
from tf_keras.layers import Input, Dense, Lambda
from tf_keras.models import Model
from tf_keras.losses import binary_crossentropy
from tf_keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import tf_keras.backend as K
from tf_keras.losses import binary_crossentropy 


class MovieVAE:
    def __init__(self, n_users, n_movies, latent_dim=50):
        self.n_users = n_users
        self.n_movies = n_movies
        self.latent_dim = latent_dim
        self.encoder = None
        self.decoder = None
        self.vae = None
    
    # def sampling(args):
    #     """Reparameterization trick."""
    #     z_mean, z_log_var = args
    #     batch = tf.shape(z_mean)[0]
    #     dim = tf.shape(z_mean)[1]
    #     epsilon = tf.random.normal(shape=(batch, dim))
    #     return z_mean + tf.exp(0.5 * z_log_var) * epsilon
        
    def build_model(self):
        # Encoder
        input_layer = Input(shape=(self.n_movies,))

        x = Dense(1024, activation='relu')(input_layer)
        x = Dense(512, activation='relu')(x)
        x = Dense(256, activation='relu')(x)

        z_mean = Dense(self.latent_dim)(x)
        z_log_var = Dense(self.latent_dim)(x)
        
        #define the sampling layer

        z = Lambda(lambda args: args[0] + K.exp(0.5 * args[1]) * K.random_normal(shape=K.shape(args[0])))([z_mean, z_log_var])

        self.encoder = Model(input_layer, [z_mean, z_log_var, z])

        # Decoder
        latent_input = Input(shape=(self.latent_dim,))
        x = Dense(256, activation='relu')(latent_input)
        x = Dense(512, activation='relu')(x)
        x = Dense(1024, activation='relu')(x)
        output_layer = Dense(self.n_movies, activation='sigmoid')(x)
        
        self.decoder = Model(latent_input, output_layer)
        
        # VAE
        outputs = self.decoder(self.encoder(input_layer)[2])
        self.vae = Model(input_layer, outputs)
        
        # Loss function
        reconstruction_loss = self.n_movies * K.mean(K.sum(binary_crossentropy(input_layer, outputs), axis=-1))

        kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        kl_loss = K.mean(kl_loss)
     
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        
        self.vae.add_loss(vae_loss)
        self.vae.compile(optimizer='adam')
        
    def train(self, train_data, test_data, epochs=10, batch_size=64):
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=8,
            restore_best_weights=True
        )
        
        return self.vae.fit(
            train_data,
            train_data,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(test_data, test_data),
            callbacks=[early_stopping]
        )
        
    def predict(self, user_data):
        return self.vae.predict(user_data)

def load_and_preprocess_data(ratings_path='ml-100k/ml-100k/u.data'):
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
    
    # Normalize ratings to [0,1]
    user_movie_matrix = user_movie_matrix / 5.0
    
    return user_movie_matrix

def evaluate_model(test_data, predictions):
    """Calculate evaluation metrics."""
    mse = np.mean(np.square(test_data - predictions))
    mae = np.mean(np.abs(test_data - predictions))
    rmse = np.sqrt(mse)
    
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    
    return rmse, mae

def main():
    # Load and preprocess data
    user_movie_matrix = load_and_preprocess_data()
    
    # Split data
    train_data, test_data = train_test_split(
        user_movie_matrix.values,
        test_size=0.2,
        random_state=42
    )
    
    # Initialize and build model
    n_users, n_movies = user_movie_matrix.shape
    print(n_users, n_movies)
    model = MovieVAE(n_users, n_movies)
    print("Building model...")
    model.build_model()
    print("Model built.")
    
    # Train model
    history = model.train(train_data, test_data)
    
    # Make predictions and evaluate
    predictions = model.predict(test_data)
    rmse, mae = evaluate_model(test_data, predictions)
    
    return model, history, (rmse, mae)

if __name__ == "__main__":
    model, history, metrics = main()