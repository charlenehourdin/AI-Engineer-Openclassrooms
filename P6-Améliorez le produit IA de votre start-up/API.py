# Libraries and Configurations
import pandas as pd
import requests
import json
import os
from urllib.parse import urlparse
from dotenv import load_dotenv

pd.set_option('display.max_colwidth', 200)

load_dotenv()
# Récupérer les valeurs des variables d'environnement pour l'authentification
api_key = "Bearer " + os.getenv("YELP_API_KEY")

endpoint = "https://api.yelp.com/v3/businesses"
path_search = "/search"
path_reviews = "/reviews"

# Créer une liste pour stocker le résultat
cities = ['NYC']
businesses_id = []
reviews = []
def businesses_request():
    constructed_url = endpoint + path_search
    headers = {'Authorization': api_key}
    # Définir un dictionnaire de paramètres
    #endpoint_1 = constructed_url + '/search'
    for city in cities:
        #query = {'categories': 'restaurants', 'location': city, 'offset':0, 'limit': '50'}
        print(f"Requete des entreprises pour la ville {city}")
        params = {'categories':'restaurants',
                'location':city,
                'offset':0,
                'limit':50}
        
        # Définir des variables
        current_offset=0
        current_limit=50 
        total_response = 200
    
        while current_offset <= total_response:
         # Fournir la preuve que l'API fonctionne
            params['offset'] = current_offset
            response = requests.get(url=constructed_url,
                                    headers=headers,
                                    params=params)
            if response.status_code == 200:
                data = response.json()

                for business in data['businesses']:
                    if business['id'] not in businesses_id:
                        businesses_id.append(business['id'])
        # S'assurez que nous avons un identifiant d'entreprise unique
        # .json décoder le format en objet Python (dictionnaire)
            current_offset += current_limit

#Créer le dataframe qui recevra les données

#Récupérer les avis associés
def reviews_request():
    headers = {'Authorization': api_key}
    i=0
    for id in businesses_id:
        constructed_url = endpoint + f"/{id}" + path_reviews
        #endpoint_2 = constructed_url + '/' + str(id) + "/reviews"
        response = requests.get(url=constructed_url,
                                headers=headers)

        if response.status_code == 200:
                data = response.json()
                for review in data['reviews']:
                # On ne garde que les mauvais commentaires
                    #if review['rating'] <= 2:
                    reviews.append([id, 'Restaurants', review['rating'], review['text'].replace('\n', ' ')])
            
        else:
            print(f"Une erreur s'est produite : {response.content}")
        i+=1
    '''
    for review in response.json()['reviews']:
        df.loc[i]=[review['id'],
                     id,
                     review['categories'],
                     review['stars'],
                     review['text']]
    '''

def main():
    businesses_request()

    if len(businesses_id) >= 200:
        reviews_request()
    else:
        return print(f"Une erreur s'est produite lors de la saisie de la liste des Businesses id list, length : {len(businesses_id)}")

    if len(reviews) > 0:
        df = pd.DataFrame(data=reviews, columns=['business_id', 'categories', 'stars', 'text'])
        df = df.sample(n=200)
        df.to_csv('/Users/charlenehourdin/Documents/Openclassrooms/Projet/P6/data/bad_reviews_from_api.csv', index=False)
        print(f"{df.shape[0]} bad reviews enregistrées dans le fichier bad_reviews_from_api.csv")
    else:
        return print(f"Une erreur s'est produite lors de la finalisation de la liste des avis, length : {len(reviews)}")

if __name__ == "__main__":
    main()

#df = pd.DataFrame(data=reviews, columns=['business_id', 'categories', 'stars', 'text'])
        #df = df.sample(n=200)
#df.to_csv('/Users/charlenehourdin/Documents/Openclassrooms/Projet/P6/data/bad_reviews_from_api.csv', index=False)
#print(f"{df.shape[0]} bad reviews enregistrées dans le fichier bad_reviews_from_api.csv")