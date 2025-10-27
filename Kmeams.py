import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# 1. Generar datos de ejemplo
# Creamos un conjunto de datos ficticio con 4 clústeres distintos.
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# 2. Aplicar el algoritmo K-means
# Creamos una instancia de KMeans con 4 clústeres.
# n_init='auto' es una buena práctica para elegir los centroides iniciales.
kmeans = KMeans(n_clusters=4, random_state=0, n_init='auto')
kmeans.fit(X)

# Obtenemos las etiquetas de los clústeres para cada punto de datos
etiquetas_cluster = kmeans.labels_

# Obtenemos las coordenadas de los centroides
centroides = kmeans.cluster_centers_

# 3. Visualizar los resultados
plt.figure(figsize=(10, 8))

# Graficamos los puntos de datos, coloreados por sus clústeres
plt.scatter(X[:, 0], X[:, 1], c=etiquetas_cluster, s=50, cmap='viridis', alpha=0.8, label='Puntos de datos')

# Graficamos los centroides de los clústeres
plt.scatter(centroides[:, 0], centroides[:, 1], c='red', s=200, marker='X', label='Centroides', edgecolors='black')

plt.title('Visualización del Algoritmo K-means')
plt.xlabel('Característica 1')
plt.ylabel('Característica 2')
plt.legend()
plt.grid(True)
plt.show()

# Opcional: Imprimir los centroides calculados
print("Centroides de los clústeres:")
print(centroides)