from azure.cognitiveservices.language.luis.runtime import LUISRuntimeClient
from config import DefaultConfig
from msrest.authentication import CognitiveServicesCredentials

CONFIG = DefaultConfig()

# On crée une instance de LUISRuntimeClient en utilisant la clé API et l'endpoint configurés dans DefaultConfig.
runtime_credentials = CognitiveServicesCredentials(CONFIG.LUIS_API_KEY)
client_runtime = LUISRuntimeClient(endpoint=CONFIG.LUIS_API_ENDPOINT, credentials=runtime_credentials)

# Test du intent "GreetingsIntent"
def test_greetings_intent():
    """
    Test de l'intent "GreetingsIntent" avec la phrase "Hello".
    """
    # On envoie une requête avec la phrase "Hello" à LUIS et on stocke la réponse.
    test_request = "Hello"
    test_response = client_runtime.prediction.resolve(CONFIG.LUIS_APP_ID, query=test_request)
    
    # On s'attend à ce que l'intent prédit soit "GreetingsIntent".
    expected_intent = "GreetingsIntent"
    actual_intent = test_response.top_scoring_intent.intent
    assert actual_intent == expected_intent


# Test du intent "NoneIntent"
def test_none_intent():
    """
    Teste si la requête "I need to book a hotel" retourne l'intent "NoneIntent" avec LUIS.
    """
    # On envoie une requête avec la phrase "I need to book a hotel" à LUIS et on stocke la réponse.
    test_request = "I need to book a hotel"
    test_response = client_runtime.prediction.resolve(CONFIG.LUIS_APP_ID, query=test_request)
    
    # On s'attend à ce que l'intent prédit soit "NoneIntent".
    expected_intent = "NoneIntent"
    actual_intent = test_response.top_scoring_intent.intent
    assert actual_intent == expected_intent

# Test du intent "OrderTravelIntent"
def test_intent():
    """
    Teste si la requête "I want to book a trip" retourne l'intent "OrderTravelIntent" avec LUIS.
    """
    # On envoie une requête avec la phrase "I want to book a trip" à LUIS et on stocke la réponse.
    test_request = "I want to book a trip"
    test_response = client_runtime.prediction.resolve(CONFIG.LUIS_APP_ID, query=test_request)
    
    # On s'attend à ce que l'intent prédit soit "OrderTravelIntent".
    expected_intent = "OrderTravelIntent"
    actual_intent = test_response.top_scoring_intent.intent
    assert actual_intent == expected_intent

# Test de l'entity "DepartureCity"
def test_intent_origin_entity():
    """
    Teste si l'entity de type "DepartureCity" renvoie la valeur "tokyo" avec la requête "I have to leave Tokyo" et LUIS.
    """
    # On envoie une requête avec la phrase "I have to leave Tokyo" à LUIS et on stocke la réponse.
    test_request = "I have to leave Tokyo"
    test_response = client_runtime.prediction.resolve(CONFIG.LUIS_APP_ID, query=test_request)
    
    # On s'attend à ce que l'entity de type "DepartureCity" renvoie la valeur "tokyo".
    expected_origin = "tokyo"
    actual_origin = ""
    if test_response.entities[0].type == 'DepartureCity':
        actual_origin = test_response.entities[0].entity

    assert actual_origin == expected_origin

# Test de l'entity "ArrivalCity"
def test_destination_entity():
    """
    Teste si l'entity de type "ArrivalCity" renvoie la valeur "london" avec la requête "I want to travel to London" et LUIS.
    """
    # On envoie une requête avec la phrase "I want to travel to London" à LUIS et on stocke la réponse.
    test_request = "I want to travel to London"
    test_response = client_runtime.prediction.resolve(CONFIG.LUIS_APP_ID, query=test_request)
    
    # On s'attend à ce que l'entity de type "ArrivalCity" renvoie la valeur "london".
    expected_destination = "london"
    actual_destination = ""
    if test_response.entities[0].type == 'ArrivalCity':
        actual_destination = test_response.entities[0].entity
    # On vérifie que la réponse correspond bien à ce qu'on attend.
    assert actual_destination == expected_destination

# Test de l'entity "DepartureDate" et ArrivalDate
def test_dates_entity():
    """
    Test si les entities "DepartureDate" et "ArrivalDate" sont correctement identifiées par LUIS.
    """
    # On envoie une requête avec la phrase "I want to book a trip from 01 July 2023 to 30 July 2023" à LUIS et on stocke la réponse.
    test_request = "I want to book a trip from 01 July 2023 to 30 July 2023"
    test_response = client_runtime.prediction.resolve(CONFIG.LUIS_APP_ID, query=test_request)

    # On s'attend à ce que l'entity de type "DepartureDate" renvoie la valeur "01 july 2023".
    expected_start_travel_date = "01 july 2023"
    actual_start_travel_date = ""
    if test_response.entities[1].type == 'DepartureDate':
        actual_start_travel_date = test_response.entities[1].entity

    # On s'attend à ce que l'entity de type "ArrivalDate" renvoie la valeur "30 july 2023".
    expected_end_travel_date = "30 july 2023"
    actual_end_travel_date = ""
    if test_response.entities[0].type == 'ArrivalDate':
        actual_end_travel_date = test_response.entities[0].entity

    # On vérifie que les réponses correspondent bien à ce qu'on attend.
    assert actual_start_travel_date == expected_start_travel_date
    assert actual_end_travel_date == expected_end_travel_date



