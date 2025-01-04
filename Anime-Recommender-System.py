# %% [markdown]
# # 1. Mengimpor Library
# 
# ---
# 
# Pada bab ini, kita akan mengimpor berbagai library yang diperlukan untuk analisis data dan pembuatan model. Library yang digunakan antara lain `pandas` untuk manipulasi data, `numpy` untuk operasi numerik, `seaborn` dan `matplotlib` untuk visualisasi data, serta `tensorflow` dan `keras` untuk pembuatan model machine learning. Selain itu, kita juga akan menggunakan `sklearn` untuk ekstraksi fitur dan pengukuran kesamaan.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import seaborn as sns
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# %% [markdown]
# # 2. Data Loading
# 
# ---
# 
# Data yang digunakan dalam proyek ini diambil dari [Anime Recommendations Database](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database/data) yang tersedia di Kaggle. Dataset ini terdiri dari dua file utama: `anime.csv` yang berisi informasi tentang berbagai anime, dan `rating.csv` yang berisi rating yang diberikan oleh pengguna untuk setiap anime.

# %%
df_anime = pd.read_csv("data/anime.csv")
df_anime.head()

# %%
df_rating = pd.read_csv("data/rating.csv")
df_rating.head()

# %% [markdown]
# # 3. Data Understanding
# 
# ---
# 
# Dataset ini berisi informasi tentang preferensi pengguna dari 73.516 pengguna pada 12.294 anime. Setiap pengguna dapat menambahkan anime ke daftar yang telah mereka selesaikan dan memberikan rating, dan dataset ini merupakan kompilasi dari rating tersebut. Sumber dataset ini dapat ditemukan di [Anime Recommendations Database](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database/data).

# %% [markdown]
# ## 3.1. Melihat ukuran data

# %%
anime_rows, anime_cols = df_anime.shape
rating_rows, rating_cols = df_rating.shape

print(f"Jumlah baris data anime: {anime_rows}")
print(f"Jumlah kolom data anime: {anime_cols}")
print()
print(f"Jumlah baris data rating: {rating_rows}")
print(f"Jumlah kolom data rating: {rating_cols}")

# %% [markdown]
# Dataset `anime.csv` memiliki jumlah baris sebanyak **12.294** dan jumlah kolom sebanyak **7**. Ini menunjukkan bahwa terdapat 12.294 entri anime yang berbeda dengan 7 atribut yang mendeskripsikan setiap anime tersebut. Sementara itu, dataset `rating.csv` memiliki jumlah baris sebanyak **7.813.737** dan jumlah kolom sebanyak **3**. Berikut adalah penjelasan dari masing-masing kolom pada kedua dataset tersebut:
# 
# **anime.csv**
# 
# - anime_id : id unik myanimelist.net yang mengidentifikasi sebuah anime.
# - name : nama lengkap anime.
# - genre : daftar genre yang dipisahkan dengan koma untuk anime ini.
# - type : film, TV, OVA, dll.
# - episodes : berapa banyak episode dalam anime ini. (1 jika film).
# - rating : rata-rata rating dari 10 untuk anime ini.
# - members : jumlah anggota komunitas yang ada dalam "grup" anime ini.
# 
# **rating.csv**
# 
# - user_id : id unik myanimelist.net yang mengidentifikasi seorang pengguna.
# - anime_id : anime yang telah dinilai oleh pengguna ini.
# - rating : rating dari 10 yang diberikan pengguna ini (-1 jika pengguna menontonnya tetapi tidak memberikan rating).
# 

# %% [markdown]
# ## 3.2. Jumlah Data Unik

# %%
unique_anime_ids = df_anime['anime_id'].nunique()
unique_user_ids = df_rating['user_id'].nunique()

print(f"Jumlah data anime ID: {unique_anime_ids}")
print(f"Jumlah data user ID: {unique_user_ids}")

# %% [markdown]
# Berdasarkan hasil perhitungan, terdapat sebanyak **12.294** data anime unik yang tersedia dalam dataset `anime.csv`. Selain itu, terdapat sebanyak **73.515** pengguna unik yang memberikan rating pada berbagai anime dalam dataset `rating.csv`.

# %% [markdown]
# ## 3.3. Informasi Data

# %%
df_anime.info()

# %% [markdown]
# Pada dataset `anime.csv`, terdapat beberapa kolom yang memiliki tipe data yang berbeda. Kolom `anime_id` dan `members` memiliki tipe data integer, kolom `name`, `genre`,  `type`, dan `episodes` memiliki tipe data string, sedangkan kolom `rating` memiliki tipe data float.

# %%
df_rating.info()

# %% [markdown]
# Sementara itu, pada dataset `rating.csv`, semua kolom memiliki tipe data integer.

# %% [markdown]
# ## 3.4. Statistik Deskriptif

# %%
df_anime.describe()

# %% [markdown]
# Deskripsi statistik dari dataset `anime.csv` menunjukkan bahwa rata-rata rating anime adalah sebesar 6.47, dengan rating terendah sebesar 1.67 dan rating tertinggi sebesar 10. Sementara itu, anime dengan jumlah anggota terendah memiliki 5 anggota, sedangkan anime dengan jumlah anggota terbanyak memiliki 1.013.917 anggota.

# %%
df_rating.describe()

# %% [markdown]
# Deskripsi statistik dari dataset `rating.csv` menunjukkan bahwa rata-rata rating yang diberikan oleh pengguna adalah sebesar 6.14, dengan rating terendah sebesar -1 dan rating tertinggi sebesar 10.

# %% [markdown]
# ## 3.5. Exploratory Data Analysis (EDA)
# 
# Exploratory Data Analysis (EDA) adalah proses untuk memahami data dengan cara menganalisis karakteristik utama dataset. Pada bagian ini, kita akan memvisualisasikan data untuk memahami distribusi dan pola dalam dataset.

# %% [markdown]
# ### 3.5.1. Dataset `anime.csv`

# %% [markdown]
# #### 3.5.1.1. Top Anime Berdasarkan Jumlah Anggota

# %%
# Select top 10 anime based on the number of members
top_anime = df_anime.nlargest(10, 'members')

# Create a vertical bar plot
plt.figure(figsize=(10, 5))
sns.barplot(x='name', y='members', data=top_anime, hue='name', palette='viridis', legend=False)
plt.title('Top 10 Anime Berdasarkan Jumlah Anggota')
plt.xlabel('Nama Anime')
plt.ylabel('Jumlah Anggota')
plt.xticks(rotation=25, ha='right')
plt.show()

# %% [markdown]
# Dari visualisasi di atas, kita dapat melihat bahwa anime dengan jumlah anggota terbanyak adalah anime dengan judul "`Death Note`" yang memiliki lebih dari 1 juta anggota, diikuti oleh anime "`Shingeki no Kyojin`" dan "`Sword Art Online`".

# %% [markdown]
# #### 3.5.1.2. Distribusi Tipe Anime

# %%
# Menghitung distribusi tipe anime
type_counts = df_anime['type'].value_counts()

# Membuat pie chart
plt.figure(figsize=(10, 8))
plt.pie(type_counts, labels=type_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('viridis', len(type_counts)))
plt.title('Distribusi Tipe Anime')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.legend(type_counts.index, title="Tipe Anime", bbox_to_anchor=(1.05, 1), loc='best')
plt.show()

# %% [markdown]
# Dari visualisasi di atas, kita dapat melihat bahwa anime dengan tipe `TV` adalah yang paling banyak, diikuti oleh anime dengan tipe `OVA`, `Movie`, `Special`, `ONA`, dan `Music` sebagai tipe anime yang paling sedikit.

# %% [markdown]
# #### 3.5.1.3. Distribusi Genre Anime

# %%
# Menghitung distribusi genre anime
genre_counts = df_anime['genre'].str.split(',').explode().value_counts()

# Convert the genre_counts Series to a DataFrame
df_genre_counts = genre_counts.reset_index()
df_genre_counts.columns = ['Genre', 'Count']
df_genre_counts

# %% [markdown]
# Dari tabel di atas, kita dapat melihat bahwa genre `Comey` adalah genre yang paling banyak, diikuti oleh genre `Action`, `Sci-Fi`, `Fantasy`, dan `Shounen`.

# %% [markdown]
# #### 3.5.1.4. Distribusi Rating Anime

# %%
# Create a histogram for the distribution of average anime ratings
plt.figure(figsize=(10, 5))
sns.histplot(df_anime['rating'].dropna(), bins=30, kde=True, color='green')
plt.title('Distribusi Rata-rata Rating Anime')
plt.xlabel('Rating')
plt.ylabel('Frekuensi')
plt.show()

# %% [markdown]
# Distribusi rating anime sebagian besar tersebar antara `5` hingga `8.0`. Hal ini menunjukkan bahwa mayoritas anime memiliki rating yang cukup baik, dengan sedikit anime yang memiliki rating sangat rendah atau sangat tinggi. Distribusi ini cenderung **left-skewed**, yang berarti lebih banyak anime yang memiliki rating di atas rata-rata dibandingkan dengan yang memiliki rating di bawah rata-rata. Skewness ini menunjukkan bahwa penonton cenderung memberikan rating yang lebih tinggi untuk anime yang mereka tonton.

# %% [markdown]
# ### 3.5.2. Dataset `rating.csv`

# %% [markdown]
# #### 3.5.2.1. Pengguna dengan Rating Terbanyak

# %%
# Group by user_id and count the number of ratings for each user
user_rating_counts = df_rating.groupby('user_id').size().reset_index(name='rating_count')

# Sort the result by rating_count in descending order
user_rating_counts = user_rating_counts.sort_values(by='rating_count', ascending=False)

# Reset the index of the result DataFrame
user_rating_counts.reset_index(drop=True, inplace=True)

# Return the result as a DataFrame
user_rating_counts

# %% [markdown]
# Dari tabel di atas, kita dapat melihat bahwa pengguna dengan id `48766` memberikan rating terbanyak, yaitu sebanyak `10.227` anime.

# %% [markdown]
# #### 3.5.2.2. Distribusi Rating Pengguna

# %%
# Create a histogram for the distribution of user ratings
plt.figure(figsize=(10, 5))
sns.histplot(df_rating['rating'], bins=30, kde=True, color='green')
plt.title('Distribusi Rating Pengguna')
plt.xlabel('Rating')
plt.ylabel('Frekuensi')
plt.show()

# %% [markdown]
# Dari visualisasi di atas, kita dapat melihat bahwa sebagian besar pengguna memberikan rating antara `6` hingga `10`. Hal ini menunjukkan bahwa mayoritas pengguna memberikan rating yang cukup baik untuk anime yang mereka tonton. Selain itu, terdapat sejumlah pengguna yang memiliki rating `-1`, hal menunjukkan bahwa mereka menonton anime tersebut tetapi tidak memberikan rating. Data rating `-1` ini kemungkinan besar akan dihapus karena tidak memberikan informasi yang berguna dalam pembuatan model rekomendasi.

# %% [markdown]
# # 4. Data Preparation
# 
# ---
# 
# Pada tahap ini, kita akan melakukan beberapa langkah data pre-preparation untuk mempersiapkan data sebelum digunakan dalam pembuatan model rekomendasi. Langkah-langkah pre-preparation yang akan dilakukan antara lain adalah filtering data, handling inrelevant data, dan handling duplicate data.

# %% [markdown]
# ## 4.1. Filter Pengguna dengan Rating Sebanyak 500 atau Lebih
# 
# Filtering data rating sangat penting dilakukan karena dataset rating memiliki jumlah data yang sangat besar, mencapai jutaan entri. Dengan jumlah data yang sangat besar ini, penggunaan resource komputasi menjadi sangat tinggi dan dapat menyebabkan proses analisis data menjadi tidak efisien. Oleh karena itu, dilakukan filtering data untuk mengurangi jumlah data yang akan diproses. Salah satu cara yang digunakan adalah dengan menyaring pengguna yang memberikan rating sebanyak 500 atau lebih. Dengan melakukan filtering ini, kita dapat mengurangi jumlah data yang harus diproses tanpa mengorbankan kualitas analisis, sehingga penggunaan resource menjadi lebih efisien dan proses analisis data dapat berjalan lebih cepat.

# %%
rows, cols = df_rating.shape
print(f"Jumlah baris data rating sebelum filtering: {rows}")

# %%
# Group by user_id and count the number of rated anime for each user
user_rating_counts = df_rating.groupby('user_id').size().reset_index(name='anime_count')

# Sort the result by anime_count in descending order
user_rating_counts = user_rating_counts.sort_values(by='anime_count', ascending=False)

# Display the result
user_rating_counts

# %%
# Menghitung jumlah pengguna dengan anime_count >= 500
users_with_500_or_more = user_rating_counts[user_rating_counts['anime_count'] >= 500].shape[0]

# Menghitung jumlah pengguna dengan anime_count < 500
users_with_less_than_500 = user_rating_counts[user_rating_counts['anime_count'] < 500].shape[0]

print(f"Jumlah pengguna dengan anime_count >= 500: {users_with_500_or_more}")
print(f"Jumlah pengguna dengan anime_count < 500: {users_with_less_than_500}")

# %%
filtered_users = user_rating_counts[user_rating_counts['anime_count'] >= 500]

# Filter df_rating to include only users with anime_count >= 500
df_rating = df_rating[df_rating['user_id'].isin(filtered_users['user_id'])]

# Display the shape of the filtered dataframe
rows, cols = df_rating.shape
print(f"Jumlah baris data rating setelah filtering: {rows}")

# %% [markdown]
# Dari hasil filtering data, kita dapat melihat bahwa terdapat sebanyak **1,853** pengguna yang memberikan rating sebanyak 500 atau lebih. Dengan melakukan filtering ini, kita dapat mengurangi jumlah data yang harus diproses dari **7.813.737** menjadi **1.384.631**.

# %% [markdown]
# ## 4.2. Menghapus Data yang Tidak Relevan
# 
# Pada dataset `rating.csv`, terdapat beberapa anime yang memiliki rating `-1`. Data ini akan dihapus karena tidak memberikan informasi yang berguna dalam pembuatan model rekomendasi.

# %%
# Melihat value count dari rating user
df_rating['rating'].value_counts()

# %% [markdown]
# Terdapat sebanyak `289.627` data rating yang memiliki rating `-1`. Data ini akan dihapus dari dataframe `df_rating`.

# %%
df_rating = df_rating[df_rating['rating'] != -1]

# Melihat value count dari rating user
df_rating['rating'].value_counts()

# %% [markdown]
# Terlihat bahwa rating `-1` telah dihapus dari dataframe `df_rating`.

# %% [markdown]
# ## 4.3. Menangani Data Duplikat

# %%
# Check for duplicate rows in the dataframe
duplicate_rows = df_anime[df_anime.duplicated()]

# Print the number of duplicate rows
print(f"Jumlah baris duplikat dalam dataframe anime: {duplicate_rows.shape[0]}")

# %%
# Check for duplicate rows in the dataframe
duplicate_rows = df_rating[df_rating.duplicated()]

# Print the number of duplicate rows
print(f"Jumlah baris duplikat dalam dataframe rating: {duplicate_rows.shape[0]}")

# %% [markdown]
# Pada penanganan data duplikat, terlihat bahwa tidak ada data duplikat yang ditemukan dalam dataset `anime.csv` maupun `rating.csv`.

# %% [markdown]
# # 5. Data Modeling
# 
# ---
# 
# Pada tahap ini, kita akan membuat model rekomendasi menggunakan metode **Content-Based Filtering** dan **Collaborative Filtering**. Kita akan menggunakan algoritma machine learning `K-Nearest Neighbors (KNN)` untuk membuat model rekomendasi **Content-Based Filtering**, dan algoritma deep learning `RecommenderNet` untuk membuat model rekomendasi **Collaborative Filtering**.

# %% [markdown]
# ## 5.1. Content-based Filtering (K-Nearest Neighbors)
# 
# Content-based Filtering adalah metode rekomendasi yang berfokus pada konten dari item yang direkomendasikan. Pada metode ini, rekomendasi diberikan berdasarkan kesamaan antara item yang direkomendasikan dengan item yang telah disukai oleh pengguna. Salah satu algoritma yang dapat digunakan dalam Content-based Filtering adalah algoritma `K-Nearest Neighbors (KNN)`.

# %%
# Hapus baris dengan nilai kosong pada kolom 'genre'
anime = df_anime.dropna(subset=['genre'])
anime.head(1)

# %% [markdown]
# Kita akan menggunakan TF-IDF (Term Frequency-Inverse Document Frequency) untuk mengekstraksi fitur dari data anime. TF-IDF adalah metode yang digunakan untuk mengekstraksi fitur dari teks dengan memberikan bobot pada kata-kata yang muncul dalam teks. Dalam kasus ini, kita akan menggunakan kolom `genre` dari dataset `anime.csv` sebagai teks yang akan diekstraksi fiturnya.

# %%
# Membuat TF-IDF Vectorizer untuk mentransformasi kolom 'genre'
tfidf = TfidfVectorizer()
# Fit dan transform kolom 'genre'
tfidf_matrix = tfidf.fit_transform(anime['genre'])

# %%
# Mengubah matriks TF-IDF menjadi array
tfidf_matrix.toarray()

# %% [markdown]
# Setelah melakukan ekstraksi fitur, kita akan menggunakan algoritma `K-Nearest Neighbors (KNN)` untuk membuat model rekomendasi. Pada parameter `metric`, kita akan menggunakan `cosine` untuk mengukur kesamaan antara item, dan pada parameter `algorithm`, kita akan menggunakan `brute` untuk menghitung jarak antara item.

# %%
# Inisialisasi NearestNeighbors model
knn_model = NearestNeighbors(metric='cosine', algorithm='brute')

# %%
# Latih model NearestNeighbors
knn_model.fit(tfidf_matrix)

# %%
# Fungsi untuk mendapatkan rekomendasi berdasarkan model NearestNeighbors
def get_knn_recommendations(title, knn_model, anime, n=10):
    # Mendapatkan indeks anime yang sesuai dengan judul
    idx = anime[anime['name'] == title].index[0]

    # Mendapatkan vektor TF-IDF untuk anime yang diberikan
    tfidf_vector = tfidf_matrix[idx]

    # Menemukan n tetangga terdekat
    distances, indices = knn_model.kneighbors(tfidf_vector, n_neighbors=n+1)

    # Mendapatkan indeks dari n anime yang paling mirip
    anime_indices = indices.flatten()[1:]

    # Mengembalikan n anime yang paling mirip
    return anime.iloc[anime_indices][['anime_id', 'name', 'genre', 'rating', 'members']]

# %%
# Mendapatkan judul anime random dari dataset
judul_random = random.choice(anime['name'].tolist())
# Mencetak judul anime random dan genrenya
genre_random = anime[anime['name'] == judul_random]['genre'].values[0]

print(f"Judul anime random: {judul_random}")
print(f"Genre dari anime random: {genre_random}")

# Mendapatkan rekomendasi untuk judul anime random menggunakan KNN
rekomendasi_knn = get_knn_recommendations(judul_random, knn_model, anime, n=5)
rekomendasi_knn

# %% [markdown]
# Dari hasil rekomendasi, kita dapat melihat bahwa anime yang direkomendasikan memiliki genre yang mirip dengan anime random yang terpilih. Hal ini menunjukkan bahwa model rekomendasi Content-based Filtering menggunakan algoritma K-Nearest Neighbors (KNN) telah berhasil memberikan rekomendasi yang sesuai dengan preferensi pengguna.

# %% [markdown]
# ## 5.2. Collaborative Filtering (RecommenderNet)
# 
# Collaborative Filtering adalah metode rekomendasi yang berfokus pada hubungan antara pengguna dan item. Pada metode ini, rekomendasi diberikan berdasarkan kesamaan antara pengguna dan item yang telah disukai oleh pengguna.
# 
# Di sini, kita akan membuat class RecommenderNet menggunakan class Model dari Keras. Kode class RecommenderNet ini diadaptasi dari tutorial di situs Keras dengan beberapa penyesuaian untuk kasus yang sedang kita selesaikan.

# %%
class RecommenderNet(Model):
    def __init__(self, num_users, num_anime, embedding_size, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.user_embedding = Embedding(num_users, embedding_size, embeddings_initializer='he_normal', embeddings_regularizer='l2')
        self.anime_embedding = Embedding(num_anime, embedding_size, embeddings_initializer='he_normal', embeddings_regularizer='l2')
        self.user_bias = Embedding(num_users, 1)
        self.anime_bias = Embedding(num_anime, 1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        anime_vector = self.anime_embedding(inputs[:, 1])
        user_bias = self.user_bias(inputs[:, 0])
        anime_bias = self.anime_bias(inputs[:, 1])
        dot_user_anime = tf.tensordot(user_vector, anime_vector, 2)
        x = dot_user_anime + user_bias + anime_bias
        return tf.nn.sigmoid(x)

# %% [markdown]
# Selanjutnya kita akan menyandikan (encode) data user_id dan anime_id ke dalam indeks integer. Selain itu, kita juga akan mencari data rating terendah dan tertinggi yang akan digunakan dalam proses splitting data nanti.

# %%
rating = df_rating

# Create encoded mapping for user_id and anime_id
user_id_mapping = {id:i for i, id in enumerate(rating['user_id'].unique())}
print(f'user_id_mapping: {user_id_mapping}')
anime_id_mapping = {id:i for i, id in enumerate(rating['anime_id'].unique())}
print(f'anime_id_mapping: {anime_id_mapping}')

# Create reverse mapping for user_id and anime_id
inverse_user_id_mapping = {i:id for id, i in user_id_mapping.items()}
print(f'inverse_user_id_mapping: {inverse_user_id_mapping}')
inverse_anime_id_mapping = {i:id for id, i in anime_id_mapping.items()}
print(f'inverse_anime_id_mapping: {inverse_anime_id_mapping}')

# Map user_id and anime_id to their respective indices
rating.loc[:, 'user_id'] = rating['user_id'].map(user_id_mapping)
rating.loc[:, 'anime_id'] = rating['anime_id'].map(anime_id_mapping)

# Number of unique users and anime in the dataset
num_users = rating['user_id'].nunique()
num_anime = rating['anime_id'].nunique()

# Minimum and maximum ratings in the dataset
min_rating = min(rating['rating'])
max_rating = max(rating['rating'])

# Create a DataFrame to display the number of users, number of anime, minimum rating, and maximum rating
df_counts = pd.DataFrame({
    'num_users': [num_users],
    'num_anime': [num_anime],
    'min_rating': [min_rating],
    'max_rating': [max_rating]
})

df_counts

# %% [markdown]
# Lalu, kita akan membagi data menjadi data training dan data testing dengan rasio 70:30. Data training akan digunakan untuk melatih model, sedangkan data testing akan digunakan untuk mengevaluasi model.

# %%
# Prepare the training data
X_train = rating[['user_id', 'anime_id']].values
y_train = rating['rating'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

# Print the shape of the training and testing sets
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# %% [markdown]
# Setelah itu, kita akan membuat model Collaborative Filtering menggunakan class RecommenderNet yang telah kita buat sebelumnya. Model ini akan dilatih menggunakan data training yang telah kita bagi sebelumnya.
# 
# Dalam proses training, kita akan menggunakan `Adam` sebagai optimizer, `mean_squared_error` sebagai loss function, dan `mean_absolute_error` sebagai metrics. Selain itu, kita juga akan menggunakan `EarlyStopping` untuk menghentikan proses training jika tidak terjadi peningkatan performa model.

# %%
%%time

# Define the model
model = RecommenderNet(num_users, num_anime, embedding_size=50)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mean_absolute_error'])

# Define Early Stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model with Early Stopping
history = model.fit(x=X_train, y=y_train, batch_size=512, epochs=20, verbose=1, validation_data=(X_test, y_test), callbacks=[early_stopping])

# %% [markdown]
# Dari output di atas, pelatihan model berhenri setelah `15` epoch dengan nilai mean squared error terendah sebesar `0.0165` dan nilai mean absolute error sebesar `0.0969`

# %%
# Save the model
model.save('model/recommender_net_model.keras')

# %%
# Plot the training history
plt.figure(figsize=(12, 6))

# Plot the training and validation loss (MSE)
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss (MSE)')
plt.plot(history.history['val_loss'], label='Validation Loss (MSE)')
plt.title('Model Loss (MSE)')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()

# Plot the training and validation metric (MAE)
plt.subplot(1, 2, 2)
plt.plot(history.history['mean_absolute_error'], label='Training MAE')
plt.plot(history.history['val_mean_absolute_error'], label='Validation MAE')
plt.title('Model Metric (MAE)')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error (MAE)')
plt.legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# Dari hasil evaluasi model, kita dapat melihat bahwa model Collaborative Filtering yang telah kita buat memiliki nilai mean squared error sebesar `0.0165` dan nilai mean absolute error sebesar `0.0969`. Nilai ini menunjukkan bahwa model Collaborative Filtering yang telah kita buat memiliki performa yang baik dalam memprediksi rating yang diberikan oleh pengguna.

# %%
def get_model_recommendations(user_id, model, anime, anime_id_mapping, inverse_anime_id_mapping, rating, n=10):
    # Map the user_id to the internal user_id used in the model
    mapped_user_id = user_id_mapping.get(user_id, None)
    if mapped_user_id is None:
        raise ValueError(f"User ID {user_id} tidak ditemukan dalam user_id_mapping.")

    # Get the list of all anime IDs
    all_anime_ids = np.array(list(anime_id_mapping.values()))

    # Create an array with the user_id repeated for each anime_id
    user_anime_array = np.array([[mapped_user_id, anime_id] for anime_id in all_anime_ids])

    # Predict the ratings for all anime for the given user
    predicted_ratings = model.predict(user_anime_array).flatten()

    # Get the indices of the top n anime with the highest predicted ratings
    top_n_indices = predicted_ratings.argsort()[-n:][::-1]

    # Get the anime IDs for the top n recommendations
    top_n_anime_ids = [inverse_anime_id_mapping[idx] for idx in top_n_indices]

    # Get the top-rated anime for the user
    user_top_rated = rating[rating['user_id'] == mapped_user_id].sort_values(by='rating', ascending=False).head(n)
    user_top_rated_anime_ids = user_top_rated['anime_id'].map(inverse_anime_id_mapping).tolist()

    # Return the top-rated anime and the top n recommended anime
    top_rated_anime = anime[anime['anime_id'].isin(user_top_rated_anime_ids)][['anime_id', 'name', 'genre', 'rating', 'members']]
    recommended_anime = anime[anime['anime_id'].isin(top_n_anime_ids)][['anime_id', 'name', 'genre', 'rating', 'members']]
    
    return top_rated_anime, recommended_anime

# %%
# Mendapatkan ID pengguna random dari dataset
user_id = random.choice(list(user_id_mapping.keys()))
print(f"ID pengguna random: {user_id}")

# Mendapatkan anime dengan rating tertinggi dan rekomendasi untuk ID pengguna random
top_rated, recommendations = get_model_recommendations(user_id, model, anime, anime_id_mapping, inverse_anime_id_mapping, rating, n=5)

# %%
print(f"Anime dengan rating tertinggi dari User {user_id}:")
top_rated

# %%
print(f"Rekomendasi anime untuk User {user_id}:")
recommendations

# %% [markdown]
# Dari hasil rekomendasi Collaborative Filtering, kita dapat melihat bahwa anime yang direkomendasikan memiliki rating yang tinggi dan sesuai dengan preferensi pengguna. Selain itu, genre dari anime yang direkomendasikan juga memiliki kemiripan dengan genre anime yang telah ditonton oleh pengguna. Hal ini menunjukkan bahwa model Collaborative Filtering yang telah kita buat telah berhasil memberikan rekomendasi yang sesuai dengan preferensi pengguna.

# %% [markdown]
# # 6. Kesimpulan
# 
# ---
# 
# Pada proyek ini, kita telah melakukan analisis data dan pembuatan model rekomendasi anime menggunakan dua pendekatan utama: **Content-Based Filtering** dan **Collaborative Filtering**. 
# 
# 1. **Content-Based Filtering**:
#     - Menggunakan algoritma **K-Nearest Neighbors (KNN)** untuk memberikan rekomendasi berdasarkan kesamaan genre anime.
#     - Model ini berhasil memberikan rekomendasi anime yang memiliki genre mirip dengan anime yang telah ditonton oleh pengguna.
# 
# 2. **Collaborative Filtering**:
#     - Menggunakan model **RecommenderNet** yang dibangun dengan **TensorFlow** dan **Keras** untuk memberikan rekomendasi berdasarkan kesamaan preferensi pengguna.
#     - Model ini dilatih menggunakan data rating pengguna dan berhasil memberikan rekomendasi anime yang sesuai dengan preferensi pengguna.
# 
# Hasil dari kedua pendekatan ini menunjukkan bahwa model rekomendasi yang telah dibuat memiliki performa yang baik dalam memberikan rekomendasi anime yang relevan dan sesuai dengan preferensi pengguna. Dengan demikian, proyek ini berhasil mencapai tujuan utamanya yaitu membuat sistem rekomendasi anime yang efektif.

# %%



