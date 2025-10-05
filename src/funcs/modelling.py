from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np


class Modelling:
  
  @staticmethod
  def clustering(model, data: pd.DataFrame, n_clusters: int = None, use_pca: bool = False, n_components: int = 2):
    
        
        model = KMeans(n_clusters=n_clusters, random_state=42)
        
        labels  = model.fit_predict(data)
          
        reduced_data = None
        if use_pca:
            pca = PCA(n_components=n_components)
            reduced_data = pca.fit_transform(data)

        return labels, model, reduced_data
      
  
      
      
