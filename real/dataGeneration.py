from scipy.spatial.distance import cdist
import pandas as pd

arquivo = 'LatLongEscolasCageData.csv'

df = pd.read_csv(arquivo)
df = df.dropna()

# coordenadas = df[['Latitude', 'Longitude']]

# # Calcular a distância euclidiana entre os pontos
# distancias = cdist(coordenadas, coordenadas, metric='euclidean')

# distancias

arquivo = 'LatLongEscolasCageData.csv'

df = pd.read_csv(arquivo)
df = df.dropna(subset=['Latitude', 'Longitude'])

# Converta as colunas Latitude e Longitude para tipos numéricos
df[['Latitude', 'Longitude']] = df[['Latitude', 'Longitude']].apply(pd.to_numeric, errors='coerce')

# Use cdist diretamente no DataFrame
distancias = cdist(df[['Latitude', 'Longitude']], df[['Latitude', 'Longitude']], metric='euclidean')

distancias
