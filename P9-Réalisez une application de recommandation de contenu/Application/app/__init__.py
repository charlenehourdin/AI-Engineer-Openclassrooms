from flask import Flask
from app import routes
from app.recommendation import recommend_articles_LMF

app = Flask(__name__)
app.config.from_object('config.Config')