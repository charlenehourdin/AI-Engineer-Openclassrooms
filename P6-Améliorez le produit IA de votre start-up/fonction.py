import numpy as np
import pandas as pd
import emoji

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches
from wordcloud import WordCloud
import itertools

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
from nltk.stem import WordNetLemmatizer

def plot_ratings_dist(data):
    """Plot distribution of ratings.
    
    Parameters
    ----------
    data : pandas DataFrame
        DataFrame where one column is ratings.
    """
    dist = (data.stars.value_counts(normalize=True))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax = dist.plot(kind='bar', color=['g', 'g', 'grey', 'r', 'r'])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_title('Répartition des notes')
    ax.set_xlabel('Sentiments')
    ax.set_ylabel('Pourçentage')
    ax.set_xticklabels(range(1, 6)[::-1], rotation=0)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.yaxis.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    ax.legend(handles=[mpatches.Patch(color='r', label='Negative'), 
                       mpatches.Patch(color='g', label='Positive')], loc=1)
    for i, v in dist.reset_index(drop=True).items():
        ax.text(i, v, s='{:.2f}%'.format(v*100), ha='center', va='bottom')
    plt.show()
    
def word_freq_hist(freq, number):

    freq_list = tuple(zip(*freq.most_common(number)))
    most_freq = pd.Series(freq_list[1], freq_list[0])

    plt.figure(figsize=(16, 10))
    most_freq.plot.bar()
    plt.title(f'Top {number} des mots les plus fréquents dans le corpus')
    plt.xlabel('Mots')
    plt.ylabel('Fréquence')
    plt.show()
    
def wordcloud(data, max_words):
    extract = []
    for row in data:
        extract+= row

    extract = " ".join(extract)

    wordcloud = WordCloud(background_color = 'white', max_words = max_words, stopwords = []).generate(extract)
    plt.figure(figsize=(12,10))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()

def plot_confusion_matrix(cm, classes,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, list(classes.values()), rotation=45)
    plt.yticks(tick_marks, list(classes.values()))
    plt.title('Confusion matrix', fontweight='bold')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def count_categories(cat):
    categories = {}
    values = [x.strip() for x in cat.split(',')]

    for value in values:
        if value in categories.keys():
            categories[value] += 1
        else:
            categories[value] = 1
        
# Vérifie si une catégorie est dans la liste des catégories sélectionnées
def check_cat(cat, selected_categories):
    values = [x.strip() for x in cat.split(',')]

    for value in values:
        if value in selected_categories:
            return True
    
    return False


def preprocess_nlp(text, df, tokenize=False, lemmatize=False):
    '''
    '''
    #global corpus
    #global freq_totale
    tokenizer = nltk.RegexpTokenizer(r'[a-zA-Z]+')
    if tokenize:
        corpus = []
        for row in df.index:
            corpus.append(tokenizer.tokenize(emoji.demojize(text[row], delimiters=("", "")).lower()))

    lemmatizer = WordNetLemmatizer()
    if lemmatize:
        stopwords = nltk.corpus.stopwords.words('english')

        #SWenglish = set()
        #SWenglish.update(tuple(stopwords.words('english')))
        # Création d'un corpus de tokens
        corpus = []
        for row in df.index:
            tokens = tokenizer.tokenize(emoji.demojize(text[row], delimiters=("", "")).lower())
            corpus.append([lemmatizer.lemmatize(w) for w in tokens if w not in stopwords and len(w) > 3])

    return corpus

# Calcul le coherence score pour plusieurs valeurs de l'hyperparamètres num_topic
def compute_coherence_values(dictionnary, corpus, texts):

    coherences_values = []
    model_list = []

    for num_topic in range(2, 21, 1):
        model = models.LdaModel(corpus=corpus, id2word=dictionnary, num_topics=num_topic)
        model_list.append(model)

        coherence_model = models.CoherenceModel(model=model, texts=texts, dictionary=dictionnary, coherence='c_v')
        coherences_values.append(coherence_model.get_coherence())
    
    return model_list, coherences_values

def build_histogram(kmeans, des, image_num):
    res = kmeans.predict(des)
    hist = np.zeros(len(kmeans.cluster_centers_))
    nb_des=len(des)
    if nb_des==0 : print("problème histogramme image  : ", image_num)    
    for i in res:
        hist[i] += 1.0/nb_des
    return hist
        
def prepare_image(img):
    """
    Use filter on image
     - Gray scale
     - resize 224x224
     - Equilize histogramme
     - Contrast limited adaptive histogram equalization
     Then save the image in img_clean_dir
    """
    filename = path_photos +  "photos/" + img + '.jpg' 
    
    # Lire l'image et définir comme échelle de gris
    image = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
    
    # Redimentionner l'image en 224x224 px
    image = cv2.resize(image, (224, 224))
    
    # Équilibrer l'histogramme
    image = cv2.equalizeHist(image)
    
    
    cv2.imwrite(img_clean_dir  + img +  '.jpg', image) 