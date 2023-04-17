
from os import environ, path
from dotenv import load_dotenv

BASE_DIR = path.abspath(path.dirname(__file__))
load_dotenv(path.join(BASE_DIR, '.env'))


class Config:
    # Global
    FLASK_APP = environ.get('FLASK_APP')
    FLASK_ENV = environ.get('FLASK_ENV')
    SECRET_KEY = environ.get('SECRET_KEY')
    # Assets
    STATIC_FOLDER = 'app/static'
    TEMPLATES_FOLDER = 'app/templates'
    COMPRESSOR_DEBUG = environ.get('COMPRESSOR_DEBUG')
