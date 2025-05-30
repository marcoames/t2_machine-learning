from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# Inicializa o app Flask e habilita CORS
app = Flask(__name__)
CORS(app)

# Carrega os datasets principais e auxiliares

# músicas
df_main = pd.read_csv('data/data.csv')
# gêneros por artista
df_genres = pd.read_csv('data/data_w_genres.csv')
# médias por artista
df_artist = pd.read_csv('data/data_by_artist.csv')
# médias por ano
df_year = pd.read_csv('data/data_by_year.csv')
# médias por gênero
df_genre_avg = pd.read_csv('data/data_by_genres.csv')

# Corrige o nome da coluna de gêneros para remover espaços extras
colunas_corrigidas = [col.strip() for col in df_genre_avg.columns]
df_genre_avg.columns = colunas_corrigidas

# Normaliza 'artists' para facilitar merges
df_main['artists'] = df_main['artists'].apply(lambda x: x.strip("[]").replace("'", "").replace('"', '').strip())
df_genres['artists'] = df_genres['artists'].apply(lambda x: x.strip("[]").replace('"', '').strip())
df_artist['artists'] = df_artist['artists'].apply(lambda x: x.strip("[]").replace('"', '').strip())

# Merge com gêneros (adiciona coluna 'genres' ao dataset principal)
df_enriched = df_main.merge(df_genres[['artists', 'genres']], on='artists', how='left')

# Merge com médias por artista (adiciona colunas *_artist_avg)
df_enriched = df_enriched.merge(df_artist, on='artists', suffixes=('', '_artist_avg'), how='left')

# Merge com médias por ano (adiciona colunas *_year_avg)
if 'year' in df_enriched.columns:
    df_enriched = df_enriched.merge(df_year, on='year', suffixes=('', '_year_avg'), how='left')

# Merge com médias por gênero (adiciona colunas *_genre_avg usando o primeiro gênero)
def get_first_genre(genres):
    if isinstance(genres, str) and genres.strip():
        return genres.split(',')[0].replace('[','').replace(']','').replace("'",'').replace('"','').strip()
    return ''
df_enriched['main_genre'] = df_enriched['genres'].apply(get_first_genre)
df_genre_avg['genres'] = df_genre_avg['genres'].str.strip()
df_enriched = df_enriched.merge(df_genre_avg, left_on='main_genre', right_on='genres', suffixes=('', '_genre_avg'), how='left')


# Seleção e normalização das features enriquecidas para recomendação
feature_cols = [
    'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
    'popularity',
    # médias por artista
    'danceability_artist_avg', 'energy_artist_avg', 'loudness_artist_avg', 'speechiness_artist_avg',
    'acousticness_artist_avg', 'instrumentalness_artist_avg', 'liveness_artist_avg', 'valence_artist_avg', 'tempo_artist_avg',
    # médias por ano
    'acousticness_year_avg', 'danceability_year_avg', 'energy_year_avg', 'instrumentalness_year_avg',
    'liveness_year_avg', 'loudness_year_avg', 'speechiness_year_avg', 'tempo_year_avg', 'valence_year_avg',
    # médias por gênero
    'acousticness_genre_avg', 'danceability_genre_avg', 'energy_genre_avg', 'instrumentalness_genre_avg',
    'liveness_genre_avg', 'loudness_genre_avg', 'speechiness_genre_avg', 'tempo_genre_avg', 'valence_genre_avg'
]

feature_cols = [col for col in feature_cols if col in df_enriched.columns]
df_enriched_clean = df_enriched.dropna(subset=feature_cols)  # remove linhas com valores ausentes

# Normaliza as features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(df_enriched_clean[feature_cols])

# Agrupamento KMeans para segmentação de músicas
kmeans = KMeans(n_clusters=20, random_state=42, n_init=10)
clusters = kmeans.fit_predict(features_scaled)
df_enriched_clean['cluster'] = clusters

# Rota para fazer a recomendação de músicas
@app.route('/recommend', methods=['POST'])
def recommend():
    body = request.json
    song_name = body.get('song')
    artist_name = body.get('artist')

    # Busca pela combinação de nome e artista
    row = df_enriched_clean[(df_enriched_clean['name'].str.lower() == song_name.lower()) & (df_enriched_clean['artists'].str.lower() == artist_name.lower())]
    if row.empty:
        return jsonify({'error': 'Song+Artist not found'}), 404

    idx = row.index[0]
    cluster_id = df_enriched_clean.loc[idx, 'cluster']

    # Seleciona músicas do mesmo cluster, exceto a própria
    cluster_data = df_enriched_clean[(df_enriched_clean['cluster'] == cluster_id) & (df_enriched_clean.index != idx)]

    song_vector = scaler.transform(df_enriched_clean.loc[[idx], feature_cols])
    cluster_vectors = scaler.transform(cluster_data[feature_cols])

    similarities = cosine_similarity(song_vector, cluster_vectors)[0]
    top_indices = similarities.argsort()[::-1]

    # Remove qualquer recomendação da própria música 
    recommended_indices = []
    for i in top_indices:
        if not cluster_data.iloc[i]['name'] == song_name:
            recommended_indices.append(i)
        if len(recommended_indices) == 5:
            break

    # Remove recomendações duplicadas (nome+artista)
    seen = set()
    unique_recommended = []
    for i in top_indices:
        rec_name = cluster_data.iloc[i]['name']
        rec_artist = cluster_data.iloc[i]['artists']
        if (rec_name.lower(), rec_artist.lower()) == (song_name.lower(), artist_name.lower()):
            continue  # nunca recomenda a própria música
        if (rec_name.lower(), rec_artist.lower()) not in seen:
            seen.add((rec_name.lower(), rec_artist.lower()))
            unique_recommended.append(i)
        if len(unique_recommended) == 5:
            break

    # Monta a resposta com as recomendações
    recommended = cluster_data.iloc[unique_recommended][[
        'name', 'artists', 'genres', 'year', 'popularity', 'danceability', 'energy', 'valence', 'tempo',
        'loudness', 'acousticness', 'instrumentalness', 'speechiness', 'liveness'
    ]].to_dict(orient='records')

    selected_song_info = df_enriched_clean.loc[idx, [
        'name', 'artists', 'genres', 'year', 'popularity', 'danceability', 'energy', 'valence', 'tempo',
        'loudness', 'acousticness', 'instrumentalness', 'speechiness', 'liveness'
    ]].to_dict()

    return jsonify({
        'selected_song': selected_song_info,
        'recommendations': recommended
    })


if __name__ == '__main__':
    app.run(debug=True)
