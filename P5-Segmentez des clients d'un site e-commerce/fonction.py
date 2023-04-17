# Fonction pour la vectorisation des variables categorielle

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import warnings
import scipy.stats
import statistics
import scipy.stats as st
import scipy.stats as stats
import operator

# Visualisation
import seaborn as sns
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Circle
from matplotlib.pyplot import cm
from matplotlib.path import Path
from matplotlib.axis import Axis

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.cluster.hierarchy import linkage, dendrogram, cut_tree
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import silhouette_score, silhouette_samples

from scipy.stats import norm
from scipy.stats import chi2_contingency, stats, chisquare

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import cdist
from scipy.spatial.distance import pdist, squareform


from builtins import input

def taux_remplissage(df, min_threshold,label_min_threshold,max_threshold,label_max_threshold):
    """
    Cette fonction permet d'afficher le taux de remplissage du dataset avec un graphique en barreshorizontale
    avec pour abscisses la liste des variables
    et pour ordonnées le taux de remplissage en pourcentage
    
    Arguments : dataframe, le seuil minimum , le seuil maximum, 
    Résultat : Diagramme en bar horizontale avec le % du taux de remplissage avec affichage du seuil renseignée
    """
# Calcul du taux de remplissage des colonnes
    rate = pd.DataFrame(
        df.count()/df.shape[0]*100,
        columns=["Taux de remplissage"]).sort_values("Taux de remplissage", 
                                                 ascending=False).reset_index()

    # Plot
    fig, axs = plt.subplots(figsize=(10,35))
    sns.barplot(y=rate["index"], x="Taux de remplissage", 
                data=rate, palette="rainbow_r")# Permuter x et y change le sens du graph
    plt.title("Taux de remplissage des variables", fontsize=14)
    plt.xlabel("Taux de remplissage (%)")
    plt.ylabel("")
    plt.axvline(x=min_threshold, color='g')
    plt.text(min_threshold, -1, label_min_threshold, color='g', fontsize=14)
    plt.axvline(x=max_threshold, color='r')
    plt.text(max_threshold, -1, label_max_threshold, color='r', fontsize=14)
    return plt.show()

# Function to calculate missing values by column# Funct 
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Le Dataframe à " + str(df.shape[1]) + " colonness.\n"      
            "Il y a " + str(mis_val_table_ren_columns.shape[0]) +
              " colonnes qui ont des valeurs manquantes.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns

def resume_col_data(df):
    '''
    * Nom : resume_col_data.
    * Paramètres : df
    * Utilisation : resume_col_data(df)
    * Warnings :
    * Résumé : Fonction permettant d'analyser les valeurs manquante d'un dataset
  
               Paramètre à donner en entrée de la fonction :
                - df: Dataframe d'entrée.
               
               Paramètre de sortie:
               Retourne un dataFrame
               
    * Packages nécessaires : 
                - Aucun
    '''
    # Valeurs calculées pour chaque colonne
    val_unique = df.nunique(dropna=False) # Nb Valeurs uniques, y compris les valeurs manquantes
    val_non_nulle = df.count(axis=0) # Nb de valeurs Non-nulles
    val_manq = df.isnull().sum() # Nb de valeurs manquantes
    val_manq_pourcent = 100 * df.isnull().sum() / len(df) # % de valeurs manquantes
    val_zero = (df == 0.00).astype(int).sum(axis=0) # Nb de valeurs à zéro
    val_zero_pourcent = 100 *((df == 0.00).astype(int).sum(axis=0)) / len(df)
    # Création d'une table de donnée ayant pour colonne les valeurs calculées
    ts_table = pd.concat([val_unique, val_non_nulle, val_manq, val_manq_pourcent, val_zero, \
                              val_zero_pourcent], axis=1)
    # Libellés explicites pour chaque colonne
    ts_table = ts_table.rename(
    columns = {0 : "Valeurs uniques", 1 : "Valeurs non-nulles", 2 : "Valeurs manquantes", \
                   3 : "% Valeurs manquantes", 4 : "Valeurs à zéro", 5 : "% Zéro"})
    # Ajout de la colonne de type de données à la table de données
    ts_table['Type Données'] = df.dtypes
    # Création d'un filtre sur les valeurs manquantes
    filtre_vm = ts_table[ts_table["Valeurs manquantes"] != 0]
    # Tri décroissant sur la colonne "Valeurs manquantes"
    ts_table = ts_table[
            ts_table.iloc[:,0] != 0].sort_values("% Valeurs manquantes", ascending=False).round(1)
    # Afficher le nombre de colonnes et de lignes au total et le nombre de colonnes avec vlr manquantes
    print("Le DataFrame a",'\033[1m' + str(df.shape[1]),"colonnes" + '\033[0m', "et",'\033[1m' +str(df.shape[0]),"lignes"+ '\033[0m',"(y compris les en-têtes).\n"    
            "Il y a " + '\033[1m' + str(filtre_vm.shape[0]) + " colonnes" + '\033[0m', "ayant des valeurs manquantes.")
    print('\033[1m' + "Les types de données :\n" + '\033[0m',df.dtypes.value_counts())

    return ts_table

from sklearn.neighbors import LocalOutlierFactor

def delete_univariate_outliers(dataframe):
    '''Suppression des valeurs extrêmes du dataset - on exclut le centile le plus extreme
    Entree: objet dataframe
    Traitement : Supression Nan univariés
    Sortie : objet dataframe
    '''
    #valeurs extremes
    index_nan = []
    index_nan_flat = []
    for column in dataframe.select_dtypes(include = ['int32','float64']).columns.tolist() :

        
        index_nan.append(dataframe.loc[dataframe[column] > dataframe[
            column].quantile(0.99)].index.tolist())
        index_nan.append(dataframe.loc[dataframe[column] < dataframe[
            column].quantile(0.01)].index.tolist())

    for sublist in index_nan:
        for item in sublist:
            index_nan_flat.append(item)
                
    #suppression des doublons
    index_nan_flat = list(dict.fromkeys(index_nan_flat))
    dataframe[column].loc[index_nan_flat] = np.nan

    return dataframe.dropna(axis=0)

def delete_multivariate_outliers(dataframe):
    '''Suppression des outliers multivariés 
    (1% le plus éloigné par le calcul de la distance aux 5 plus proches voisins)
    Entree : objet dataframe
    Sortie : objet dataframe
    '''
    
    lof = LocalOutlierFactor(n_neighbors = 5, n_jobs=-1)
    lof.fit_predict(dataframe.select_dtypes(['float64','int32']).dropna())
    indices = dataframe.select_dtypes(['float64','int32']).dropna().index
    df_lof = pd.DataFrame(index = indices,
                           data = lof.negative_outlier_factor_, columns=['lof'])
    index_to_drop = df_lof[df_lof['lof']< np.quantile(
        lof.negative_outlier_factor_, 0.01)].index
    return dataframe.drop(index_to_drop, axis=0)

def clean_outliers(dataframe):
    dataframe = delete_univariate_outliers(dataframe)

    dataframe = delete_multivariate_outliers(dataframe)
    return dataframe


def pearson_corr_heatmap(df, mask_upper=True,
                         figsize=(10,10), font=13,
                         cmap='coolwarm'):
    
    """Plot Pearson correlation with or without upper triangle
    Set triange value to True to mask the upper triangle,
    to False to keep the upper triangle"""
    
    # Compute the correlation matrix
    corr_matrix = df.corr()
    
    # Generate a mask for the upper triangle
    # for lower triangle, transpose the mask with .T
    mask = np.zeros_like(corr_matrix)
    mask[np.triu_indices_from(mask)] = True
    
    # Delete empty cell of top and bottom
    mask = mask[1:, :-1]
    corr = corr_matrix.iloc[1:, :-1]
    
    plt.figure(figsize=figsize)
    
    if mask_upper:
        with sns.axes_style('white'):
            sns.heatmap(corr, mask=mask,
                        annot=True, annot_kws={'size': font},
                        fmt='.2f', cmap=cmap)
    else:
        sns.heatmap(corr_matrix, annot=True, fmt='.2f',
                    cmap=cmap)
    
    plt.tick_params(labelsize=font)
    plt.title('Pearson correlation Matrix',
              fontsize=15, fontweight='bold',
              x=0.5, y=1.05)
    plt.show()

def plot_histograms(df, cols, bins=30, figsize=(9, 4), color = 'lightgrey',
                    skip_outliers=False, z_thresh=3, layout=(3,3)):
    
    """Plot histogram subplots with multi-parameters:
    ------------------------------------------------------------
    ARGUMENTS :
    df = Dataframe
    cols = only numeric columns as list
    color = lightgrey, lightblue, etc.
    skip_outliers = True or False (plot with/without outliers)
    # outliers are computed with scipy.stats z-score
    z_thresh = z-score threshold, default=3
    layout : nb_rows, nb_columns
    ------------------------------------------------------------
    """

    fig = plt.figure(figsize=figsize)

    for i, c in enumerate(cols,1):
        ax = fig.add_subplot(*layout,i)
        features = df[c][np.abs(st.zscore(df[c]))< z_thresh]
        plt.suptitle('Distribution',
                     fontsize=18, fontweight='bold',
                     y=1.05) 
    
    plt.tight_layout()
    plt.show()

def graph_distrib(df, features, title):
    """
    Cette fonction permet de visualiser la distribution de chaque variable à l'aide d'un graphique (histogramme) qui comprend la visualisation de 
    la moyenne, la mediane et de la courbes de densité

    Arguments : dataframe, colonnes , titre du graphique   
    Résultat : un graphique de visualisation de la distribution pour chaque variable 
    """
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(30,15))

    sub = 0
    for i in range(len(features)):
        fig.add_subplot(2,5,i+1)
    
        left, width = 0, 1
        bottom, height = 0, 1
        right = left + width
        top = bottom + height
    
        colonne = numerical_features[i]
        kstest = stats.kstest(df[colonne].notnull(),'norm')
        ax = sns.distplot(df[colonne], fit=stats.norm, kde=False)
        ax.set_title("Distribution vs loi normale : {}".format(colonne))
        ax.text(right, top, 'Test Kolmogorov-Smirnov \n Pvalue: {:.2} \n Stat: {:.2}'.format(kstest.pvalue, kstest.statistic),
            horizontalalignment='right',
            verticalalignment='top',
            style='italic', transform=ax.transAxes, fontsize = 12,
            bbox={'facecolor':'#00afe6', 'alpha':0.5, 'pad':0})
        ax.axvline(df[colonne].median(), color="r", ls="-", label="med")
        #ax.text((gauche+.15), (bas_page+.6), "med", color="r", transform=axs.transAxes)
        ax.axvline(df[colonne].mean(), color="g", ls="--", label="moy")
        #ax.text((gauche+.15), (bas_page+.4), "moy", color="g", transform=axs.transAxes)
        plt.xticks(fontsize=14)
        ax.legend(loc ="center right") 
    plt.suptitle(title, fontsize=20)
    plt.show()

def merge_fct(data_1, data_2 , row_1, row_2):
    data_final = pd.merge(left = data_1,
             right =  data_2,
             left_on = row_1,
             right_on = row_2,
             how = "left")
             
    return data_final

def plot_histograms(df, cols, bins=30, figsize=(9, 4), color = 'lightgrey',
                    skip_outliers=False, z_thresh=3, layout=(3,3)):
    
    """Plot histogram subplots with multi-parameters:
    ------------------------------------------------------------
    ARGUMENTS :
    df = Dataframe
    cols = only numeric columns as list
    color = lightgrey, lightblue, etc.
    skip_outliers = True or False (plot with/without outliers)
    # outliers are computed with scipy.stats z-score
    z_thresh = z-score threshold, default=3
    layout : nb_rows, nb_columns
    ------------------------------------------------------------
    """

    fig = plt.figure(figsize=figsize)

    for i, c in enumerate(cols,1):
        ax = fig.add_subplot(*layout,i)

        # Choice to skip outliers or not
        if skip_outliers:
            features = df[c][np.abs(st.zscore(df[c]))< z_thresh]
        else:
            features = df[c]
        ax.hist(features,  bins=bins, color=color)
        ax.set_title(c)
        ax.vlines(df[c].mean(), *ax.get_ylim(),
                  color='red', ls='-', lw=1.5)
        ax.vlines(df[c].median(), *ax.get_ylim(),
                  color='green', ls='-.', lw=1.5)
        ax.vlines(df[c].mode()[0], *ax.get_ylim(),
                  color='goldenrod', ls='--', lw=1.5)
        ax.legend(['mean', 'median', 'mode'])
        ax.title.set_fontweight('bold')
        # xmin, xmax = ax.get_xlim()
        # ax.set(xlim=(0, xmax/5))

    # Title linked to skip_outliers
    if skip_outliers:
        plt.suptitle('Distribution hors valeurs aberrantes',
                     fontsize=18, fontweight='bold',
                     y=1.05)
    else:
        plt.suptitle('Distribution incluant les valeurs aberrantes',
                     fontsize=18, fontweight='bold',
                     y=1.05) 
    
    plt.tight_layout()
    plt.show()


def preprocess(df_init, var_quali=None, stdScale=False):
  '''
    * Nom : preprocess.
    * Paramètres : df_init, var_quali=None
    * Utilisation : vectorisation(df_init, var_quali)
    * Warnings :
    * Résumé : Fonction permettant de faire le preprocessing en transformant les variables catégorielle en variable numérique (O et 1)
  
               Paramètre à donner en entrée de la fonction :
                - df_init: Dataframe d'entrée.
                - var_quali: Dataframe (si les données sont déjà encodées) 
               
               Paramètre de sortie:
               Retourne un dataFrame de variable de type numérique
               
    * Packages nécessaires : 
                - warnings (import warnings)
    
    '''
    
  
  if var_quali is not None:
    
    #Selection des variables categorielle
    categorical = df_init.select_dtypes(['category','object']).columns
    idx_categorical = df_init[categorical].copy()
    idx_categorical = pd.concat([idx_categorical, var_quali], axis=1)
    
    #Selection des variables numérique
    numerical = df_init.select_dtypes(['int32','float64','uint8','int64']).columns
    idx_numerical = df_init[numerical].copy()
    idx_numerical = idx_numerical.drop(var_quali, axis=1)
    
  else:
    #Selection des variables categorielle
    categorical = df_init.select_dtypes(['category','object']).columns
    idx_categorical = df_init[categorical]
    
    #Selection des variables numérique
    numerical = df_init.select_dtypes(['int32','float64','uint8','int64']).columns
    idx_numerical = df_init[numerical]
 
  if idx_categorical.shape[1] > 0 and idx_numerical.shape[1] > 0:     
    df_cat = pd.get_dummies(idx_categorical)
    X = idx_numerical.values
    names = idx_numerical.index
    if stdScale:
        std_scale = preprocessing.StandardScaler().fit(idx_numerical)
        X_scaled = std_scale.transform(idx_numerical)
        X_scaled_df = pd.DataFrame(X_scaled, index=names, columns=numerical)
        df = pd.concat([df_cat, X_scaled_df], axis=1)
    else:
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(idx_numerical)
        X_scaled_df = pd.DataFrame(X_scaled, index=names, columns=numerical)
        df = pd.concat([df_cat, X_scaled_df], axis=1)
    
  elif idx_categorical.shape[1] != 0:
    df = pd.get_dummies(idx_categorical)
    
  elif idx_numerical.shape[1] > 0:    
    X = idx_numerical.values
    names = idx_numerical.index
    if stdScale:
        std_scale = preprocessing.StandardScaler().fit(idx_numerical)
        X_scaled = std_scale.transform(idx_numerical)
        df = pd.DataFrame(X_scaled, index=names, columns=numerical)
    else:
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(idx_numerical)
        df = pd.DataFrame(X_scaled, index=names, columns=numerical)
 
  else:
    warnings.warn('votre dataframe est vide !')
  
  return df

def segmentation(data_scale=None,data_init=None,cah=False, method= 'ward', metric= 'euclidean', criterion= 'distance', Kmeans=False, init='k-means++',n_init=50, mixte_kmeans_cah=False, mixte_cah_kmeans=False, random_state=False, silhouette=False):
  '''  
       * Nom : segmentation.  
       * Paramètres : data_scale=None,data_init=None,cah=False, method= 'ward', metric= 'euclidean', criterion= 'distance', Kmeans=False, init='k-means++', mixte_kmeans_cah=False, mixte_cah_kmeans=False.  
       * Utilisation : segmentation(data_scale=None,data_init=None,cah=False, method= 'ward', metric= 'euclidean', criterion= 'distance', Kmeans=False, init='k-means++', mixte_kmeans_cah=False, mixte_cah_kmeans=False                  
                         
       * Warnings :  
       * Résumé : Fonction permettant de réaliser une segmentation en utilisant :
                  - CAH (Paramètre par default : method= 'ward', metric= 'euclidean', criterion= 'distance')
                  - KMeans (Paramètre par default : init='k-means++', mixte_kmeans_cah=False, mixte_cah_kmeans=False)
                  - CAH + KMeans
                  - KMeans + CAH 
                  * ATTENTION: Pour utiliser une segmentation, il faut changer le paramètre en entrée de la fonction sur True 
                  lors de l'appel de la fonction dans votre script (Exemple: segmentation(data_scale=data_scale,cah=True))
                  
                  Avec un dataframe en entrée aux choix :
                  - un dataframe de donnée standardisée (mettre en paramètre le dataframe dans data_scale)
                  - un dataframe de donnée non transformé (mettre en paramètre le dataframe dans data_init)
                  
                  Retourne une liste de clusters .             
                              
      * Packages nécessaires :                             
        - six (from six.moves import input)          
        - warnings (import warnings)    
      '''
      
  # définir des variables global
  global Z_cah
  global clusters_cah
  global clusters_kmeans_cah
  global clusters_cah_kmeans
  
  if data_init is None:
    if cah:
      Z = linkage(data_scale, method=method, metric=metric)
      Z_cah = Z.copy()
      fig = plt.figure(figsize=(25, 10))
      dn = dendrogram(Z)
      plt.show()
      plt.close()
      # Coupage du dendrogramme en n clusters
      nb_clusters = int(input('Veuillez définir la hauteur de t :'))
      clusters_cah = fcluster(Z, t=nb_clusters, criterion=criterion)
      #Ajout de la colonne cluster au dataset
      data = data_scale.copy()
      data['cluster_cah'] = clusters_cah
      segmentation.groupe_cah = clusters_cah
    
    if Kmeans:
      inertia=[]
      valeur = round(int(input("Veuillez définir la plage de valeur de K (2,...) pour l'affichage de l'EB:")))
      for k in range(2,valeur):
        kmeans = KMeans(n_clusters=k,init='k-means++', n_init=50, random_state=random_state).fit(data_scale)
        labels = kmeans.predict(data_scale)
        if silhouette:
            print(f'Silhouette Score(n={k}): {silhouette_score(data_scale, labels)}')
        inertia.append(kmeans.inertia_)
      fig = plt.figure(figsize=(20, 10))
      title = fig.suptitle("La méthode du coude utilisant l'inertie", fontsize=18)
      sns.pointplot(x=list(range(2,valeur,1)), y=inertia)
      plt.tight_layout()
      plt.show()
      plt.close()
      '''
      plt.scatter(range(2,valeur,1),inertia)
      plt.title('Elbow-method')
      plt.xlabel("Nombre de clusters")
      plt.ylabel("Cout du modele (Inertia)")
      plt.tight_layout()
      plt.show()
      plt.close()
      '''
      nb_clusters = int(input('Veuillez définir le nombre de clusters :'))
      model=KMeans(n_clusters=nb_clusters,init=init).fit(data_scale)
      clusters_kmeans = model.labels_
      data = data_scale.copy()
      data['cluster_kmeans'] = clusters_kmeans
      segmentation.groupe_kmeans = clusters_kmeans
      
    if mixte_kmeans_cah:
      nb_clusters = int(input('Veuillez définir le nombre de clusters :'))
      model=KMeans(n_clusters=nb_clusters,init=init).fit(data_scale)
      labels = model.labels_
      centroids = model.cluster_centers_
      kmeans_subset = data_scale.iloc[model.labels_]
      
      Z_mixte = linkage(kmeans_subset, method=method, metric=metric)
      
      fig = plt.figure(figsize=(25, 10))
      dn = dendrogram(Z_mixte)
      plt.show()
      plt.close()
      
      nb_clusters = int(input('Veuillez définir la hauteur de t :'))
      clusters_kmeans_cah = fcluster(Z_mixte, t=nb_clusters, criterion=criterion)
      data = data_scale.copy()
      data['cluster_cah_mixte'] = clusters_kmeans_cah
      
      segmentation.clusters_kmeans_cah = clusters_kmeans_cah
      
    if mixte_cah_kmeans:
      Z = linkage(data_scale, method=method, metric=metric)
      
      fig = plt.figure(figsize=(25, 10))
      dn = dendrogram(Z)
      plt.show()
      plt.close()
      
      nb_clusters = int(input('Veuillez définir la hauteur de t :'))
      clusters_cah = fcluster(Z, t=nb_clusters, criterion=criterion)
      cah_subset = data_scale.iloc[clusters_cah]
      
      inertia=[]
      valeur = round(int(input("Veuillez définir la plage de valeur de K (2,...) pour l'affichage de l'EB:")))
      for k in range(2,valeur,1):
        kmeans = KMeans(n_clusters=k,init='k-means++').fit(data_scale)
        inertia.append(kmeans.inertia_)
      plt.scatter(list(range(2,valeur,1)),inertia)
      plt.title('Elbow-method')
      plt.xlabel("Nombre de clusters")
      plt.ylabel("Cout du modele (Inertia)")
      plt.tight_layout()
      plt.show()
      plt.close()
      
      nb_clusters = int(input('Veuillez définir le nombre de clusters :'))
      model=KMeans(n_clusters=nb_clusters,init=init).fit(cah_subset)
      clusters_cah_kmeans = model.labels_
      data = data_scale.copy()
      data['cluster_cah_kmeans'] = clusters_cah_kmeans
      segmentation.clusters_cah_kmeans = clusters_cah_kmeans
      
  elif data_scale is None :
    mess = int(input('ATTENTION: Etes_vous sur de vouloir utiliser un dataframe avec des données sans preprocessing pour réaliser la segmentation ? (O=NON, 1=OUI)'))
    if mess == 1:
      if cah:
        Z = linkage(data_init, method=method, metric=metric)
        fig = plt.figure(figsize=(25, 10))
        dn = dendrogram(Z)
        plt.show()
        plt.close()
        # Coupage du dendrogramme en n clusters
        nb_clusters = int(input('Veuillez définir la hauteur de t :'))
        clusters_cah = fcluster(Z, t=nb_clusters, criterion=criterion)
        cah_subset = data_init.iloc[clusters_cah]
        
        #Ajout de la colonne cluster au dataset
        data = data_init.copy()
        data['cluster_cah'] = clusters_cah
        segmentation.groupe_cah = clusters_cah
      
      if Kmeans:
        inertia=[]
        valeur = round(int(input("Veuillez définir la plage de valeur de K (2,...) pour l'affichage de l'EB:")))
        for k in range(2,valeur):
          kmeans = KMeans(n_clusters=k,init='k-means++').fit(data_init)
          inertia.append(kmeans.inertia_)
        plt.scatter(range(2,valeur),inertia)
        plt.title('Elbow-method')
        plt.xlabel("Nombre de clusters")
        plt.ylabel("Cout du modele (Inertia)")
        plt.tight_layout()
        plt.show()
        plt.close()
        
        nb_clusters = int(input('Veuillez définir le nombre de clusters :'))
        model=KMeans(n_clusters=nb_clusters,init=init).fit(data_init)
        clusters_kmeans = model.labels_
        data = data_init.copy()
        data['cluster_kmeans'] = clusters_kmeans.copy()
        segmentation.groupe_kmeans = clusters_kmeans
        
      if mixte_kmeans_cah:
        nb_clusters = int(input('Veuillez définir le nombre de clusters :'))
        model=KMeans(n_clusters=nb_clusters,init=init).fit(data_init)
        labels = model.labels_
        centroids = model.cluster_centers_
        kmeans_subset = data_init.iloc[model.labels_]
        
        Z_mixte = linkage(kmeans_subset, method=method, metric=metric)
        
        fig = plt.figure(figsize=(25, 10))
        dn = dendrogram(Z_mixte)
        plt.show()
        plt.close()
        
        nb_clusters = int(input('Veuillez définir la hauteur de t :'))
        clusters_kmeans_cah = fcluster(Z_mixte, t=nb_clusters, criterion=criterion)
        data = data_init.copy()
        data['cluster_cah_mixte'] = clusters_kmeans_cah
        segmentation.groupe_cah_mixte = clusters_kmeans_cah
        
      if mixte_cah_kmeans:
        Z = linkage(data_init, method=method, metric=metric)
        
        fig = plt.figure(figsize=(25, 10))
        dn = dendrogram(Z)
        plt.show()
        plt.close()
        
        nb_clusters = int(input('Veuillez définir la hauteur de t :'))
        clusters_cah = fcluster(Z, t=nb_clusters, criterion=criterion)
        cah_subset = data_init.iloc[clusters_cah]
        
        inertia=[]
        valeur = round(int(input("Veuillez définir la plage de valeur de K (2,...) pour l'affichage de l'EB:")))
        for k in range(2,valeur):
          kmeans = KMeans(n_clusters=k,init='k-means++').fit(cah_subset)
          inertia.append(kmeans.inertia_)
        plt.scatter(range(2,valeur),inertia)
        plt.title('Elbow-method')
        plt.xlabel("Nombre de clusters")
        plt.ylabel("Cout du modele (Inertia)")
        plt.tight_layout()
        plt.show()
        plt.close()
        
        nb_clusters = int(input('Veuillez définir le nombre de clusters :'))
        model=KMeans(n_clusters=nb_clusters,init=init).fit(cah_subset)
        clusters_cah_kmeans = model.labels_
        data = data_init.copy()
        data['cluster_cah_kmeans'] = clusters_cah_kmeans
        segmentation.groupe_cah_kmeans = clusters_cah_kmeans
    else:
      data =  'Veuillez modifier le dataframe en entrée en utilisant le paramètre data_scale='
      
  else:
    warnings.warn("ATTENTION, Vous n'avez pas défini de dataframe dans les parametres d'entrée ou le dataframe utilisée est erroné")
  
  return data


def cluster_info(data_init, clusters, seuil_importance_variable = None, seuil_cramer = 30, correlations=False, topVar=False, chi2=False, cramer=False):
  '''  
       * Nom : cluster_info.  
       * Paramètres : df_init, clusters, seuil_importance_variable = None, seuil_cramer = 30.  
       * Utilisation : cluster_info(data, clusters, seuil_importance_variable = valeur(entier), seuil_cramer = valeur(entier))                  
                       cluster_info.nom_de_l'attribut (afficher le dataframe voir définition des attributs ci-dessous)  
       * Warnings :  
       * Résumé : Fonction permettant d'afficher plusieurs dataframe pour l'interpretabilité des résultats d'une segmentation(clustering)             
                  Ces 4 paramètres sont à donner en entrée de la fonction :              
                  - df_init: Dataframe d'entrée,               
                  - clusters : Nom de la variable qui identifie les groupes obtenues et leurs labels ,              
                  - seuil_importance_variable : seuil du R2 des variables les plus significative pour le clustering.
                  - seuil_cramer = 30 : seuil du V_cramer,
                  
                  Retourne un dataFrame global avec la moyenne des variables par cluster, la moyenne générale et le R2 de chaque variable.  
                  
                  la ligne suivante: 'cluster_info.nom_de_l'attribut' permet d'afficher les dataframes suivant :    
                  
                  - cluster_info.mean: Affiche un dataframe de la moyenne des variables les plus significatives par cluster (seuil superieur à 50),               
                  - cluster_info.topVar: Affiche un dataframe des variables les plus significative pour la création de la segmentation ,               
                  - cluster_info.distance: Affiche un dataframe des distances entres les groupes  
                  - cluster_info.std :  Affiche un dataframe avec les Ecart type 
                  - cluster_info.chi2 :  Affiche un dataframe des corrélations entre les variables catégorielle avec la p-value du chi2 et le V_cramer
                  - cluster_info.correlation : Affiche un dataframe des corrélations entre les variables numérique
             
      * Packages nécessaires :                             
        - scipy (from scipy.spatial.distance import pdist, squareform).              
        - scipy (from scipy.stats import chi2_contingency, stats, chisquare)
        - scipy (import scipy.stats)
        - statistics (import statistics)
        - warnings (import warnings)
  '''
  #Selection des variable numerique
  numerical = data_init.select_dtypes(['int32','float64','uint8','int64']).columns
  idx_numerical = data_init[numerical].copy()
  
  #Selection des variables categorielle
  categorical = data_init.select_dtypes(['category','object']).columns
  idx_categorical = data_init[categorical].copy()
  
  # Vectorisation des variables categorielle
  if idx_categorical.shape[1] != 0:
    df = pd.get_dummies(data_init.copy())
  else:
    df = data_init.copy()
  #Ajout de la colonne cluster au dataset
  df['cluster'] = clusters
  #moyenne globale pour chaque variable
  m = df.mean().round(2)
  #TSS (somme totale des carrés)
  TSS = df.shape[0]*df.var(ddof=0)
  ##dataframe conditionnellement aux groupes
  gb = df.groupby('cluster')
  #taille des groupes conditionnels
  nk = gb.size()
  #moyenne conditionnel
  mk = gb.mean()
  #(différence entre les moyennes cond. et la moyenne globale) ²
  EMk = (mk-m)**2
  #ponderation par la taille des groupes
  EM = EMk.multiply(nk,axis=0)
  #somme => BSS (entre la somme des carrés)
  BSS = np.sum(EM,axis=0)
  #carré du rapport de corrélation
  #variance expliquée par l'appartenance aux groupes pour chaque variable
  R2 = BSS/TSS*100
  
  #Ajout moyenne generale au dataset
  gen = m.to_frame(name='Gen').T
  #gen = round(gen*100,2)
  df_gen = mk.append(gen, ignore_index=False)
  
  #Ajout R2 au dataset
  r2 = R2.to_frame(name='%R2').T
  #r2 = round(r2,2)
  df_r2 = df_gen.append(r2, ignore_index=False)
  df_r2 = df_r2.drop(['cluster'], axis=1)
  #df_r2 = round(df_r2.columns*100,2)
  
  #Nombre de clients en %
  clients = nk.to_frame(name='clients')
  df_r2['clients'] = clients.copy()
  df_r2['clients (%)'] = round(clients/clients.sum()*100,2)
  
  #Moyenne des variables les plus significatives par cluster
  liste_mean = []
  lst_mean = []
  for index, row in mk.iterrows():
    for cols in mk.columns:
      if mk.loc[index,cols] > 0.50:
        liste_mean.append("{}".format(index, cols))
        lst_mean.append(["{}".format(index),"{}".format(cols), mk.loc[index, cols]])
  df_mean = pd.DataFrame(lst_mean, columns=["G","Var","mean"])
  
  #Ecart type
  list_VarQuanti_gp = {}
  list_VarQuali_gp = {}
  for i in np.unique(df['cluster']):
    if idx_numerical.shape[1] > 0:
      df_numerical = idx_numerical
      df_numerical['cluster'] = clusters
      sdgen = np.std(df_numerical)
      df_i = df_numerical.loc[df_numerical['cluster'] == i].copy()
      sd_grp_i = np.std(df_i)
      moy = mk.loc[mk.index.isin([i])].copy().T
      moyGen = m
      #print("GRP :", i)
      df_std = pd.concat([moy, moyGen,sd_grp_i,sdgen], join="inner",axis=1)
      df_std.columns = ['MOY','MOY_GEN','ECART-TYPE','ECART-TYPE_GEN']
      list_VarQuanti_gp["gp{}".format(i)] = df_std
    
    if idx_categorical.shape[1] != 0:
      data_categorical = idx_categorical
      data_categorical['cluster'] = clusters
      data_categ_dum = pd.get_dummies(data_categorical)
      df_i = pd.get_dummies(data_categorical[data_categorical['cluster'] == i])
      moy = mk[mk.index.isin([i])].T
      moyGen = m
      somme_group = np.sum(df_i, axis=0)
      somme = np.sum(data_categ_dum, axis=0)
      sum_grp_idx = somme_group.index.tolist()
      sum_idx = somme.index.tolist()
      if len(sum_idx) != len(sum_grp_idx) or sum_idx != sum_grp_idx:
        somme_news = somme.filter(items=sum_grp_idx, axis=0)
        somme_grp_news = somme_group.filter(items=sum_idx, axis=0)
      else:
        somme_news = somme
        somme_grp_news = somme_group
        
        
      c_mode = somme_grp_news/somme_news.apply(lambda x: float(x))
      cla_mode = c_mode.T
      df_desc_quali = pd.concat([cla_mode, moy, moyGen], join="inner",axis=1)
      #print("GRP :", i)
      df_desc_quali.columns = ['CLA_MODE','MOY','MOY_GEN']
      list_VarQuali_gp["gp{}".format(i)] = df_desc_quali
      
  #Calcul de la distance
  distances = pdist(df_r2, metric='euclidean')
  dist_matrix = squareform(distances)
  df_dist= pd.DataFrame(dist_matrix)
  
  # Distance entre les groupes
  liste_dist = []
  liste = []
  for index, row in df_dist.iterrows():
    for cols in df_dist.columns:
      if index != cols and "G {} G {}".format(cols, index) not in liste_dist:
        liste_dist.append("G {} G {}".format(index, cols))
        liste.append(["G {}".format(index),"G {}".format(cols), df_dist.loc[index, cols].copy()])
  df_distance = pd.DataFrame(liste, columns=["Num_gp1","Num_gp2","dist_euclidean"])
  df_distance = df_distance.dropna(axis=0)
  
  # Définition des attributs
  cluster_info.r2 = df_r2.copy()
  cluster_info.mean = df_mean.copy()
  cluster_info.std = list_VarQuanti_gp
  cluster_info.distance = df_distance.copy()
  cluster_info.desc_quali = list_VarQuali_gp
  
  
  # Corrélation entre les variables et variables les plus significatives
  
  if idx_numerical.shape[1] > 0:
    df_numerical = idx_numerical.copy()
      
    # Variables quantitative les plus significatives
    top_var = R2.loc[R2>seuil_importance_variable].copy()
    df_top_Var = top_var.to_frame(name='%R2_cramer')
    
    if correlations:
      # Corrélation entre les variables quantitatives
      liste_corr = []
      lst = []
      for cols1 in df_numerical.columns:
        for cols2 in df_numerical.columns:
          if cols1 != cols2 and "var {} var {}".format(cols2, cols1) not in liste_corr:
            liste_corr.append("var {} var {}".format(cols1, cols2))
            corr, p_values = stats.spearmanr(df_numerical[cols1], df_numerical[cols2])
            if p_values <= 0.05:
              lst.append(["var {}".format(cols1),"var {}".format(cols2), p_values])
      df_correlation = pd.DataFrame(lst, columns=["Num_var1","Num_var2","p_values"])
      # Attribut correlation
      cluster_info.correlation = df_correlation.copy()
      
  if idx_categorical.shape[1] != 0:
    data_categorical = idx_categorical.copy()
    data_categorical['cluster'] = clusters
    
    # Variables quantitative les plus significatives
    top_var = R2.loc[R2>seuil_importance_variable].copy()
    df_top_Var = top_var.to_frame(name='%R2_cramer')
    
    # Variables qualitative les plus significatives
    if topVar:
      lst_r2_cramer = []
      for cols in data_categorical.columns:
        contingence = pd.crosstab(data_categorical['cluster'] , data_categorical[cols])
        chi2 = chi2_contingency(contingence)[0]
        n = sum(contingence.sum())
        if n*(min(contingence.shape)-1) > 0.0:
          cramer = np.sqrt(chi2 / (n*(min(contingence.shape)-1)))*100
        else :
          cramer = 0
        if cramer >= seuil_cramer:
          lst_r2_cramer.append(["{}".format(cols), cramer])
      df_R2_cramer = pd.DataFrame(lst_r2_cramer, columns=["{}","%R2_cramer"])
      df_R2_cramer.set_index('{}',inplace = True)
      df_R2_cramer = df_R2_cramer.drop(['cluster'], axis=0)
      df_topVar = df_R2_cramer.append(df_top_Var, ignore_index=False)
      
      # Attribut Top variable
      cluster_info.topVar = df_topVar.copy()
    
    # Corrélation entre les variables qualitative
    if chi2:
      liste_chi2 = []
      lst_chi2 = []
      for cols1 in data_categorical.columns:
        for cols2 in data_categorical.columns:
          if cols1 != cols2 and "var {} var {}".format(cols2, cols1) not in liste_chi2:
            liste_chi2.append("var {} var {}".format(cols1, cols2))
            contingence = pd.crosstab(data_categorical[cols1], data_categorical[cols2])
            chi2, p_values, dof, ex = chi2_contingency(contingence)
            chi2 = chi2_contingency(contingence)[0]
            n = sum(contingence.sum())
            if n*(min(contingence.shape)-1) > 0.0:
              cramer = np.sqrt(chi2 / (n*(min(contingence.shape)-1)))*100
            else :
              cramer = 0
            if p_values <= 0.05:
              lst_chi2.append(["var {}".format(cols1),"var {}".format(cols2), p_values, cramer])
      df_chi2 = pd.DataFrame(lst_chi2, columns=["Num_var1","Num_var2","pvalues_chi2","V_cramer"])
      #Attribut chi2
      cluster_info.chi2 = df_chi2.copy()
      
    if cramer:
      liste_cramer = []
      lst_cramer = []
      for cols1 in data_categorical.columns:
        for cols2 in data_categorical.columns:
          if cols1 != cols2 and "var {} var {}".format(cols2, cols1) not in liste_cramer:
            liste_cramer.append("var {} var {}".format(cols1, cols2))
            contingence = pd.crosstab(data_categorical[cols1], data_categorical[cols2])
            chi2 = chi2_contingency(contingence)[0]
            n = sum(contingence.sum())
            if n*(min(contingence.shape)-1) > 0.0:
              cramer = np.sqrt(chi2 / (n*(min(contingence.shape)-1)))*100
            else :
              cramer = 0
            if cramer >= seuil_cramer:
              lst_cramer.append(["var {}".format(cols1),"var {}".format(cols2), cramer])
      df_cramer = pd.DataFrame(lst_cramer, columns=["Num_var1","Num_var2","V_cramer"])
      #Attribut cramer 
      cluster_info.cramer = df_cramer.copy()
  
  return df_r2

def compare_cluster(arbre_cah, data_init, k_max=10, seuil_cramer=30):
  
  lst_grp = []
  lst_mean = []
  
  for i in range(2,k_max+1):
    clusters_cah = fcluster(arbre_cah, i, criterion='maxclust')
    lst_grp.append([clusters_cah])
    clusters_grp = cluster_info(data_init, clusters_cah, seuil_importance_variable = 50, seuil_cramer = seuil_cramer, correlations=False, topVar=True, chi2=False, cramer=False).copy()
    mean = cluster_info.r2.copy()
    df_mean = mean.copy()
    df_mean = df_mean.drop(['clients', 'clients (%)'], axis=1)
    df_mean = df_mean.drop(['Gen', '%R2'], axis=0)
    lst_mean.append(df_mean)
  grp = pd.DataFrame(lst_grp, index=['partition_gp_{}'.format(j) for j in range(2,k_max+1)])
  df_grp = grp.T
  
  info_grp2 = cluster_info(data_init, clusters_cah, seuil_importance_variable = 50, seuil_cramer = seuil_cramer, correlations=False, topVar=True, chi2=False, cramer=False).copy()
  topVar = cluster_info.topVar.copy()
  df_topVar = topVar.index.tolist()
  labels_annotate = [' \n '.join(df_topVar)]
  lst_desc_quali = [cluster_info.desc_quali.copy()]
  lst_desc_quanti = [cluster_info.std.copy()]
  
  
  for i in range(3, k_max+1):
    gp_divis_n = []
    gp_presence_n_1 = []
    
    for j in range(1,(i+1)):
      clust_i = True
      info_cluster_i = lst_mean[i-2].loc[j].tolist()
      
      for k in range(1,i):
        info_cluster_i_1 = lst_mean[i-3].loc[k].tolist()
        
        if info_cluster_i == info_cluster_i_1:
          clust_i = False
          gp_presence_n_1.append(k)
    
      if clust_i:
        gp_divis_n.append(j)
    
    gp_divis_n_1 = [x for x in range(1,i+1) if x not in gp_presence_n_1][0]
    data_init_b = data_init.copy()
    data_init_b['partition_gp_{}'.format(i)] = df_grp['partition_gp_{}'.format(i)][0]
    data_init_b["cond_pres"] = [True if a in gp_divis_n else False for a in data_init_b['partition_gp_{}'.format(i)]].copy()
    X = data_init_b[data_init_b["cond_pres"]].iloc[:, 0:-2]
    y = data_init_b[data_init_b["cond_pres"]]['partition_gp_{}'.format(i)].copy()
    info_part_clus_div = cluster_info(X, y, seuil_importance_variable = 50, seuil_cramer = seuil_cramer, correlations=False, topVar=True, chi2=False, cramer=False).copy()
    lst_desc_quali.append(cluster_info.desc_quali.copy())
    lst_desc_quanti.append(cluster_info.std.copy())
    
    topVar = cluster_info.topVar.copy()
    df_topVar = topVar.index.tolist()
    labels_annotate.append(' \n '.join(df_topVar))
  ddata = dendrogram(Z_cah, truncate_mode='lastp', p=k_max+1, leaf_font_size=10)
  k = 0
  ddata['d_coord_trie'] = sorted(ddata['dcoord'], key=operator.itemgetter(1))
  
  while k<k_max-1:
    #leaves = ddata["ivl"]
    d =  ddata['d_coord_trie'][len(ddata['icoord'])-k-1]
    d_1 =  ddata['d_coord_trie'][len(ddata['icoord'])-k-2]
    indice_y = ddata['dcoord'].index(d)
    i = ddata['icoord'][indice_y]
    d_1 =  ddata['d_coord_trie'][len(ddata['icoord'])-k-2]
    c =  ddata['color_list'][indice_y]
    x = sum(i[1:3])/2
    y = d[1]
    plt.plot(x, y, 'o', c=c, markeredgewidth=0)
    plt.annotate(labels_annotate[k], (x, y), xytext=(0,-10),textcoords='offset points',va='top', ha='center',fontweight='bold',color=c, fontsize=7)
    k+=1
    ax = plt.gca()
    bounds = ax.get_xbound()
    ax.plot(bounds, [y - ((y-d_1[1])/2), y - ((y-d_1[1])/2)] , '--', c=c, alpha=0.3)
    ax.text(bounds[1],y - ((y-d_1[1])/2),'{} clusters'.format(k+1) , va='center', fontdict={'size': 10}, fontweight='bold',linespacing=240)
    plt.title('Dendrogram comparaison clusters')
  plt.xlabel('Effectifs')
  plt.ylabel('Distance')
  #ax.set_xlim(k))
  #ax.set_xticklabels(leaves, fontsize=10, rotation=45)
  ax.set_ylim(y-k_max,ax.get_ylim()[1])
  #if k_max <= 8:
  #  ax.set_ylim(y+2-k_max,ax.get_ylim()[1]-2)
  #else:
  #  ax.set_ylim(0, ax.get_ylim()[1]-2)
  plt.tight_layout()
  plt.show()
  plt.close()
  
  k_choix_num = 5
  if k_choix_num in range(1, k_max):
    print('Variable quali',lst_desc_quali[k_choix_num-1])
    print('Variable quanti',lst_desc_quanti[k_choix_num-1])
  
  # boucle a terminé
  k_choix_num = int(input("Quelle partition souhaitez vous voir en détail ? (pour arrêter saisissez le numéro 0)"))
  while k_choix_num != 0:
    for k_choix_num in range(1, k_max):
      print(lst_desc_quali[k_choix_num])
      print(lst_desc_quanti[k_choix_num])
      cluster_info.desc_quali['gp5']


  nb_clusters = int(input('Veuillez définir le nombre de clusters :'))
  clusters_cah = fcluster(arbre_cah, nb_clusters, criterion='maxclust')
  df_result = cluster_info(data_init, clusters_cah, seuil_importance_variable = 50, seuil_cramer = seuil_cramer, correlations=False, topVar=True, chi2=False, cramer=False).copy()
  return df_result

def RScore(x,p,d):
    if x <= d[p][0.25]:
        return 1
    elif x <= d[p][0.50]:
        return 2
    elif x <= d[p][0.75]: 
        return 3
    else:
        return 4
    
def FMScore(x,p,d):
    if x <= d[p][0.25]:
        return 4
    elif x <= d[p][0.50]:
        return 3
    elif x <= d[p][0.75]: 
        return 2
    else:
        return 1

def drawHistAndBoxPlot(data, columns, dims_fig):
    nbr_rows = int(len(columns))
    index = 1
    plt.figure(figsize=dims_fig)
    for column in columns:

        #log = False if column == "recence" else True
        plt.subplot(nbr_rows, 2, index)
        left, width = 0, 1
        bottom, height = 0, 1
        right = left + width
        top = bottom + height
    
        kstest = stats.kstest(data[column].notnull(),'norm')
        ax = sns.distplot(data[column], fit=stats.norm, kde=False)
        ax.set_title("Distribution vs loi normale : {}".format(column))
        ax.text(right, top, 'Test Kolmogorov-Smirnov \n Pvalue: {:.2} \n Stat: {:.2}'.format(kstest.pvalue, kstest.statistic),
                horizontalalignment='right',
                verticalalignment='top',
                style='italic', transform=ax.transAxes, fontsize = 12,
                bbox={'facecolor':'#00afe6', 'alpha':0.5, 'pad':0})
        ax.axvline(data[column].median(), color="r", ls="-", label="med")
        #ax.text((gauche+.15), (bas_page+.6), "med", color="r", transform=axs.transAxes)
        ax.axvline(data[column].mean(), color="g", ls="--", label="moy")
        #ax.text((gauche+.15), (bas_page+.4), "moy", color="g", transform=axs.transAxes)
        plt.xticks(fontsize=14)
        ax.legend(loc ="center right") 
        index += 1
        '''
        plt.subplot(nbr_rows, 2, index)
        plt.hist(data[column], log=log, bins=200)
        plt.xlabel(f"{column}")
        plt.ylabel("Count")
        plt.title(f"Histogramme - {column}")
        index += 1
        '''

        plt.subplot(nbr_rows, 2, index)
        sns.boxplot(x=data[column])
        plt.xlabel(column)
        plt.title(f"Boite à moustaches pour {column}")
        index += 1
    plt.show()


def snake_plot(normalised_df_rfm, clusters, df_rfm_original):
    '''
    normalised_df_rfm = pd.DataFrame(normalised_df_rfm, 
                                       index=df_rfm_original.index, 
                                       columns= df_rfm_original.columns)
    normalised_df_rfm['Cluster'] = df_rfm_kmeans[col_cluster]
    '''
    normalised_df_rfm['cluster'] = clusters
    # Melt data into long format
    df_melt = pd.melt(normalised_df_rfm.reset_index(), 
                        id_vars=['customer_unique_id', 'cluster'],
                        value_vars=['recence', 'frequence', 'montant_cumulé'], 
                        var_name='Metric', 
                        value_name='Value')

    plt.xlabel('Metric')
    plt.ylabel('Value')
    sns.pointplot(data=df_melt, x='Metric', y='Value', hue='cluster')
    
    return

def kmeans_TSNE_analysis(data_init, cluster_labels, k,var_montant, heatmap=False):
    
    #Selection des variable numerique
    numerical = data_init.select_dtypes(['int32','float64','uint8','int64']).columns
    idx_numerical = data_init[numerical].copy()
    
    #Selection des variables categorielle
    categorical = data_init.select_dtypes(['category','object']).columns
    idx_categorical = data_init[categorical].copy()
   
    # Vectorisation des variables categorielle
    if idx_categorical.shape[1] != 0:
        df = pd.get_dummies(data_init.copy())
    else:
        df = data_init.copy()
    
    #kmeans = KMeans(n_clusters = clusters_number, random_state = 1)
    #kmeans.fit(normalised_df_rfm)

    # Extract cluster labels
    #cluster_labels = kmeans.labels_
    
        
    # Create a cluster label column in original dataset
    df_new = df.assign(cluster = cluster_labels)
    
    # Initialise TSNE
    model = TSNE(random_state=1)
    transformed = model.fit_transform(df_new)
    
    # Plot t-SNE
    plt.title('TSNE pour {} clusters'.format(k))
    sns.scatterplot(x=transformed[:,0], y=transformed[:,1], hue=cluster_labels, style=cluster_labels, palette="Set1")
    #plt.legend(bbox_to_anchor =(0.75, 1.15), ncol = 2)


    # Calculer la moyenne pour chaque caractéristique par cluster
    kmeans_averages = df_new.groupby(['cluster']).mean().round(2)
    
    # Calculer la moyenne  de chaque caractéristique
    kmeans_averages.loc['mean'] = df_new.mean().round(2)

    
    # Display kmeans table
    display(kmeans_averages)
    print('\n')

       
    # Calculer l'importance relative de chaque caractéristique par cluster
    temp = kmeans_averages[kmeans_averages.index.isin(range(0,k))]
    relative_imp = (temp / cluster_labels.mean() - 1)
    relative_imp.round(2)
    

    # Prepare figures configuration
    if heatmap:
        fig, ax1 = plt.subplots(1, 1, figsize=(10,10))
        # Plot features importance for each segment  
        sns.heatmap(data=relative_imp, annot=True, fmt='.2f',
                    cmap='RdYlBu', ax=ax1)
        ax1.set_title(f'Importance relative des caractéristiques pour {k} clusters',
                          fontweight='bold', pad=30)
        ax1.set_ylabel(ylabel='Segment')
    
        print('\n')
        
    fig, (ax2, ax3) = plt.subplots(1, 2, figsize=(10,10))
    # Compute clusters proportion
    pop_perc = (
        df_new['cluster'].value_counts() / len(df_new) * 100)
    pop_perc.sort_index(inplace=True)

    # Plot clusters proportion
    _,_, autotexts = ax2.pie(pop_perc, autopct='%1.0f%%',
                             textprops=dict(color="w"))
    plt.setp(autotexts, size=12, weight='bold')
    c_circle=plt.Circle((0,0), 0.40, color='white')
    ax2.add_patch(c_circle)
    ax2.set_title('% de clients par cluster',
                  fontweight='bold', pad=30)

    
    
    # Compute revenue for each segment
    cluster_value = df_new.groupby('cluster')[var_montant].sum()

    # Plot contribution to revenue
    _,_, autotexts = ax3.pie(cluster_value, autopct='%1.0f%%',
                             textprops=dict(color="w"))
    plt.setp(autotexts, size=12, weight='bold')
    c_circle=plt.Circle((0,0), 0.40, color='white')
    ax3.add_patch(c_circle)
    ax3.legend(labels=cluster_value.index, loc='center left',
               bbox_to_anchor=(-0.4, 0.5))
    ax3.set_title("Contribution au chiffre d'affaire par cluster",
                  fontweight='bold', pad=30)
    
    plt.show()
    
    return df_new



def pca_correlation_circle(X, dim_a=0, dim_b=1):
    """Plot the PCA correlation circle related to provided dimension,
    a=0, b=1 for 1st two dimensions"""
    
    # Instantiate and compute PCA
    pca = PCA(random_state=42)
    pca.fit(X)

    # Get explained variance ratio (EVR) for each component
    evr_pct = pca.explained_variance_ratio_ * 100 # percentages
    cum_evr = np.cumsum(evr_pct) # cumulative percentages

    
    # Get original features names
    features_names = X.columns
    
    # Format figsize, xy labels, title
    plt.figure(figsize=(15, 15))
    plt.xlabel('Dim{} ({:.2f}%],'.format(dim_a+1, evr_pct[dim_a]))
    plt.ylabel('Dim{} ({:.2f}%],'.format(dim_b+1, evr_pct[dim_b]))
    plt.title('Correlation Circle - PC%d & ' % (dim_a+1) + 'PC%d' % (dim_b+1))
    
    # Plot the circle
    ax = plt.gca()
    ax.add_patch(Circle([0,0], radius=1,
                        color='k', linestyle='-',
                        fill=False, clip_on=False))
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    
    # Plot radius line for each axis
    plt.plot([-1,1], [0,0], color='grey',
             linestyle='dotted', alpha=0.5)
    plt.plot([0,0], [-1,1], color='grey',
             linestyle='dotted', alpha=0.5)
    
    # Plot explained variance in feature space
    x_pca = pca.components_[dim_a]
    y_pca = pca.components_[dim_b]

    sns.scatterplot(x=x_pca, y=y_pca,
                    color='blue', alpha=0.1)

    for x, y, col in zip(x_pca, y_pca, features_names):
        plt.annotate(col, (x,y),
                     textcoords='offset points',
                     xytext=(0, 5+np.random.randint(-20,20)),
                     ha='center')
        ax.arrow(0, 0, x, y,
                 head_width=0.05, head_length=0.02,
                 fc='grey', ec='blue', alpha=0.5)
    plt.grid(False)
    plt.show()

def pca_correlation_matrix(X, n_comp=None, random_state=42,
                           figsize=(15,9)):
    
    if n_comp:
        pca = PCA(n_components=n_comp,
                  random_state=random_state)
    else:
        pca = PCA(random_state=random_state)
    
    # Compute PCA
    pca.fit(X)

    # Get explained variance ratio (EVR) pourcentage
    evr_pct = pca.explained_variance_ratio_ * 100 # percentages
    
    # Plot correlation matrix (heatmap)
    # display all components of selected dimensions
    all_components = [np.abs(pca.components_)[i] for i in range(len(evr_pct))]
    
    # Get original features names
    features_names = X.columns
    
    # Set dimension names
    dimension_names = {'DIM{}'.format(i+1): all_components[i] for i in range(len(evr_pct))}
    
    # Create the correlation matrix dataframe
    correlation_matrix = pd.DataFrame(all_components,
                                      columns=features_names,
                                      index=dimension_names)

    # Plot correlation matrix heatmap
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(correlation_matrix.T, annot=True, fmt='.2f', cmap='YlGnBu')
    ax.xaxis.set_ticks_position('top')
    plt.show()

def display_customers(data, labels, col):
    """Display clusters by % of customer and by % of revenue
    Arguments:
        data {DataFrame} -- Data Frame
        labels {List} -- Cluster list
        col {string} -- Columns for sum
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,20))
    

    monetary_sum = data[col].sum()
    clusters = data.groupby('cluster').agg({
        col: 'sum'
    })    
    col_percent = clusters.groupby(level=0).apply(lambda x: 100 * x / monetary_sum)
    
    ax1.set_title("Pourcentage du chiffre d'affaire au sein des groupes")
    ax1.pie(col_percent,
        autopct='%1.1f%%',
        shadow=True,
        startangle=180,
        labels=labels
    )

    clusters = data.groupby('cluster').mean()        
    ax2.set_title("Pourcentage des clients au sein des groupes")
    ax2.pie(clusters['Note_Moyenne_Commentaire'],
        autopct='%1.1f%%',
        shadow=True,
        startangle=180,
        labels=labels
    )
    plt.show()

def R_Class(r_value, col, quantiles):
        """Recency. set Recency classification by quantille.
        from old (1) to recent (4)
        Arguments:
            r_value {float} -- Recency value
            col {string} -- column name for Recency
            quantiles {dict} -- Quantiles dict
        Returns:
            int -- Recency classification
        """
        if r_value <= quantiles[col][0.25]:
            return 4
        elif r_value <= quantiles[col][0.50]:
            return 3
        elif r_value <= quantiles[col][0.75]:
            return 2
        else:
            return 1

def FM_Class(fm_value, col, quantiles):
    """Frequency or Monetary. set Frequency or Monetary classification by quantille.
    from high (1) to low (4)
    Arguments:
        fm_value {float} -- Frequency or Monetary value
        col {string} -- Column name
        quantiles {dict} -- Quantiles dict
    Returns:
        int -- Recency classification
    """
    if fm_value <= quantiles[col][0.25]:
        return 1
    elif fm_value <= quantiles[col][0.50]:
        return 2
    elif fm_value <= quantiles[col][0.75]:
        return 3
    else:
        return 4

def RFM_level(df):
    """
    Classement des clients en fonction de leur score de
    Recency, Frequency et Monetary
    """
    if df['RFM_Score'] >= 11:
        return 'Meilleur client' 
    if (df['RFM_Score'] >= 9) & (df['RFM_Score'] <= 10):
        return 'Client à forte valeur ajoutée'
    if (df['RFM_Score'] >= 7) & (df['RFM_Score'] <= 8):
        return 'Client de valeur moyenne'
    if (df['RFM_Score'] >= 5) & (df['RFM_Score'] <= 6):
        return 'Client de faible valeur'
    if df['RFM_Score'] <= 4:
        return 'Client perdu' 

    return 'ERROR'

def RFM_map(df):
    """
    Classement des clients en fonction de leur score de
    Recency, Frequency et Monetary
    """
    if df['cluster'] == 'Meilleur client' :
        return 4 
    if df['cluster'] == 'Client à forte valeur ajoutée' :
        return 3
    if df['cluster'] == 'Client perdu' :
        return 2
    if df['cluster'] == 'Client de valeur moyenne' :
        return 1
    if df['cluster'] == 'Client de faible valeur' :
        return 0 
    return 'ERROR'

def silhouette_pca_umap_tsne(X, X_proj, T, original_df_rfm, k_range=[4,5,6]):
    
    """Plot silhouette per k cluster and visualize with pca and tsne"""
    
    for k in k_range:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        fig.set_size_inches(20, 5)
        
        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(X)
        
        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)
        
        y_lower = 10
        for i in range(k):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

            cluster_silhouette_values.sort()

            # Compute the (new) y_upper
            size_cluster = cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster

            # Assign a color per cluster i in range k
            color = cm.nipy_spectral(float(i) / k)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, cluster_silhouette_values,
                              facecolor=color, edgecolor=color,
                              alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples
    
        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        ax1.set_title('Avg silhouette: {:.3f}'.format(silhouette_avg),
                      fontsize=12, fontweight='bold')
        ax1.set_xlabel('Silhouette coefficient values')
        ax1.set_ylabel('Cluster label')

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1
        # For this one, we go from -0.2 to 0.6
        ax1.set_xlim([-0.2, 0.6])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (k + 1) * 10])

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color='red', linestyle="--")
        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.2, 0, 0.2, 0.4, 0.6])

        for data, axs, visu in zip([X_proj, T],
                                   [ax2, ax3],
                                   ['PCA', 'T-SNE']):
            # Visualisation
            sns.scatterplot(x=data[:,0],
                            y=data[:,1],
                            hue=cluster_labels,
                            legend='full',
                            palette=sns.color_palette('hls', k),
                            alpha=0.8,
                            ax=axs)
            # Set layout
            axs.set_title('Projection: '+visu,
                          fontsize=12, fontweight='bold')
            axs.set_xlabel('DIM 1')
            axs.set_ylabel('DIM 2')

        # Labeling the clusters for PCA projection
        centers = kmeans.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')
        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')
    
        # Add suptitle for each cluster plot
        plt.suptitle('Silhouette analysis for KMeans with {} clusters'.format(k),
                     fontsize=16, fontweight='bold')
        print('\n')
        # Create a cluster label column in original dataset
        df_new = original_df_rfm.assign(Cluster = cluster_labels)
    
        # Initialise TSNE
        model = TSNE(random_state=1)
        transformed = model.fit_transform(df_new)

        # Plot t-SNE
        plt.title('TSNE pour {} Clusters'.format(k))
        sns.scatterplot(x=transformed[:,0], y=transformed[:,1], hue=cluster_labels, style=cluster_labels, palette="Set1")


        # Calculer la moyenne pour chaque caractéristique par cluster
        kmeans_averages = df_new.groupby(['Cluster']).mean().round(2)

        # Calculer la moyenne  de chaque caractéristique
        kmeans_averages.loc['mean'] = original_df_rfm.mean().round(2)


        # Display kmeans table
        display(kmeans_averages)
        print('\n')

            
        # Calculer l'importance relative de chaque caractéristique par cluster
        temp = kmeans_averages[kmeans_averages.index.isin(range(0,k))]
        relative_imp = (temp / original_df_rfm.mean() - 1)
        relative_imp.round(2)

        # Prepare figures configuration
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18,4))

        # Plot features importance for each segment  
        sns.heatmap(data=relative_imp, annot=True, fmt='.2f',
                    cmap='RdYlBu', ax=ax1)
        ax1.set_title(f'Importance relative des caractéristiques pour {k} clusters',
                        fontweight='bold', pad=30)
        ax1.set_ylabel(ylabel='Segment')

        # Compute clusters proportion
        pop_perc = (
            df_new['Cluster'].value_counts() / len(df_new) * 100)
        pop_perc.sort_index(inplace=True)

        # Plot clusters proportion
        _,_, autotexts = ax2.pie(pop_perc, autopct='%1.0f%%',
                                    textprops=dict(color="w"))
        plt.setp(autotexts, size=12, weight='bold')
        c_circle=plt.Circle((0,0), 0.40, color='white')
        ax2.add_patch(c_circle)
        ax2.set_title('% de clients par cluster',
                        fontweight='bold', pad=30)


        # Compute revenue for each segment
        cluster_value = df_new.groupby('Cluster')['montant_cumulé'].sum()

        # Plot contribution to revenue
        _,_, autotexts = ax3.pie(cluster_value, autopct='%1.0f%%',
                                    textprops=dict(color="w"))
        plt.setp(autotexts, size=12, weight='bold')
        c_circle=plt.Circle((0,0), 0.40, color='white')
        ax3.add_patch(c_circle)
        ax3.legend(labels=cluster_value.index, loc='center left',
                    bbox_to_anchor=(-0.4, 0.5))
        ax3.set_title("Contribution au chiffre d'affaire par cluster",
                        fontweight='bold', pad=30)

        plt.show()

    
def display_customers(data, labels, col):
    """Display clusters by % of customer and by % of revenue
    Arguments:
        data {DataFrame} -- Data Frame
        labels {List} -- Cluster list
        col {string} -- Columns for sum
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30,20))
    

    monetary_sum = data[col].sum()
    clusters = data.groupby('cluster').agg({
        col: 'sum'
    })    
    pop_perc = (data['cluster'].value_counts() / len(data) * 100)
    pop_perc.sort_index(inplace=True)
    ax1.set_title("Pourcentage du chiffre d'affaire au sein des groupes")
    ax1.pie(pop_perc,
        autopct='%1.1f%%',
        shadow=True,
        startangle=180,
        labels=labels
    )

    cluster_value = data.groupby('cluster')[col].sum()       
    ax2.set_title("Pourcentage des clients au sein des groupes")
    ax2.pie(cluster_value,
        autopct='%1.1f%%',
        shadow=True,
        startangle=180,
        labels=labels
    )
    plt.show()
    
def groupby_unique_customer(df):
    '''Iterate groupby per customer_unique_id'''
    
    # Round all floats to 2 decimal places
    pd.options.display.float_format = '{:.2f}'.format
    
    # Initialize customers grouping
    gp_df = df.groupby('customer_unique_id')
    
    # Create the final df
    main_df = gp_df.date_achat.max().rename('max_date_achat')
    
    # Create specific features to be aggregated to final dataframe
    specific_features = []
    specific_features.append(gp_df.cout.mean().rename('cout'))
    specific_features.append(gp_df.insatisfaction.mean().rename('insatisfaction'))
    specific_features.append(gp_df.frequence.mean().rename('frequence'))
    specific_features.append(gp_df.credit.mean().rename('credit'))
    specific_features.append(gp_df.delai_livraison.mean().rename('delai_livraison'))
    specific_features.append(gp_df.frais_livraison.mean().rename('frais_livraison'))
    specific_features.append(gp_df.categorie_maison.mean().rename('categorie_maison'))
    specific_features.append(gp_df.categorie_electronique_beaute.mean().rename('categorie_electronique_beaute'))
    
    
    # Merge
    for feature_series in specific_features:
        main_df = pd.merge(main_df, feature_series,
                           on='customer_unique_id')
        
    # df max date against customers' max date of purchase
    main_df['recence'] = ((
        (main_df['max_date_achat'].max() -
         main_df['max_date_achat']).dt.days)/30).apply(np.floor)
    
    
    return main_df


def silhouette_pca_tsne(X, X_proj, T,  k_range=[4,5,6]):
    
    for k in k_range:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        fig.set_size_inches(20, 5)
        
        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(X)
        
        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)
        
        y_lower = 10
        for i in range(k):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

            cluster_silhouette_values.sort()

            # Compute the (new) y_upper
            size_cluster = cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster

            # Assign a color per cluster i in range k
            color = cm.nipy_spectral(float(i) / k)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, cluster_silhouette_values,
                              facecolor=color, edgecolor=color,
                              alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples
    
        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        ax1.set_title('Avg silhouette: {:.3f}'.format(silhouette_avg),
                      fontsize=12, fontweight='bold')
        ax1.set_xlabel('Silhouette coefficient values')
        ax1.set_ylabel('Cluster label')

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1
        # For this one, we go from -0.2 to 0.6
        ax1.set_xlim([-0.2, 0.6])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (k + 1) * 10])

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color='red', linestyle="--")
        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.2, 0, 0.2, 0.4, 0.6])

        for data, axs, visu in zip([X_proj,T],
                                   [ax2, ax3],
                                   ['PCA', 'T-SNE']):
            # Visualisation
            sns.scatterplot(x=data[:,0],
                            y=data[:,1],
                            hue=cluster_labels,
                            legend='full',
                            palette=sns.color_palette('hls', k),
                            alpha=0.8,
                            ax=axs)
            # Set layout
            axs.set_title('Projection: '+visu,
                          fontsize=12, fontweight='bold')
            axs.set_xlabel('DIM 1')
            axs.set_ylabel('DIM 2')

        # Labeling the clusters for PCA projection
        centers = kmeans.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')
        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')
    
        # Add suptitle for each cluster plot
        plt.suptitle('Analyse de silhouette pour le regroupement de KMeans sur des échantillons de données avec {} clusters'.format(k),
                     fontsize=16, fontweight='bold')

    plt.show()
    
def plot_radars(data, stdScale=False):
    '''
    scaler = MinMaxScaler()
    data = pd.DataFrame(scaler.fit_transform(data), 
                        index=data.index,
                        columns=data.columns).reset_index()
    '''
    data = preprocess(data, var_quali=None, stdScale=False)
    
    fig = go.Figure()

    for k in data.index:
        fig.add_trace(go.Scatterpolar(
            r=data[data.index==k].iloc[:,0:].values.reshape(-1),
            theta=data.columns[0:],
            fill='toself',
            name='Cluster '+str(k)
        ))

    fig.update_layout(
        polar=dict(
        radialaxis=dict(
          visible=True,
          range=[0, 1]
        )),
        showlegend=True,
        title={
            'text': "Comparaison des moyennes par variable des clusters",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        title_font_color="blue",
        title_font_size=18)

    fig.show()

    
def plot_chart_clusters(kmeans_data, kmeans_df, label,
                               k_range=range(0,6), rows=2, cols=3):
    
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.io as pio
    import plotly.offline as pyo
    pyo.init_notebook_mode()
    #pio.renderers.default='notebook'
    
    # Prepare the name and the weight of each cluster
    var = (kmeans_data[label].value_counts(normalize=True)*100).round(0)
    cls_weight = var.sort_index().values
    cls_name = ['Cluster{}: {}%'.format(i,cls_weight[i]) for i in k_range]
    
    # Set scatterpolar variables
    cls = {}
    cls_rlist = {}
    feat_to_plot = kmeans_df.columns
    df = kmeans_df.reset_index()

    # Plot
    fig = plt.figure(figsize=(20, 40))
    fig = make_subplots(rows=rows, cols=cols,
                        specs=[[{'type': 'polar'}]*cols]*rows,
                        subplot_titles=cls_name)

    for i in k_range:
        # Create scatterpolar data
        cls[i] = df.loc[df[label] == i]
        cls_rlist[i]  = cls[i][feat_to_plot].to_numpy()
        # Add trace for each cluster
        if i < cols:
            row = 1
            col = i+1
        else:
            row = 2
            col = i-2
        fig.add_trace(go.Scatterpolar(name=cls_name[i],
                                      r=cls_rlist[i].tolist()[0],
                                      theta=feat_to_plot,
                                      fill='toself',
                                      hoverinfo='name+r'),
                      row=row, col=col)
        # Note : fixed polar size with radialaxis 
        # and min / max values of data
        # run fig.layout to check the values
        fig.update_layout(title={'text':'Comparaison Cluster',
                                 'y':0.95, 'x':0.5,
                                 'xanchor':'center',
                                 'yanchor':'top'},
                          height=500, width=1300,
                          polar=dict(radialaxis=dict(range=[0,1],
                                                     showticklabels=False,
                                                     ticks='')),
                          polar2=dict(radialaxis=dict(range=[0,1],
                                                      showticklabels=False,
                                                      ticks='')),
                          polar3=dict(radialaxis=dict(range=[0,1],
                                                      showticklabels=False,
                                                      ticks='')),
                          polar4=dict(radialaxis=dict(range=[0,1],
                                                      showticklabels=False,
                                                      ticks='')),
                          polar5=dict(radialaxis=dict(range=[0,1],
                                                      showticklabels=False,
                                                      ticks='')),
                          polar6=dict(radialaxis=dict(range=[0,1],
                                                      showticklabels=False,
                                                      ticks='')),
                          showlegend=False)
        
        if i < cols:
            fig.layout.annotations[i].update(y=1.03)
        else:
            fig.layout.annotations[i].update(y=0.4)
        
    fig.show()