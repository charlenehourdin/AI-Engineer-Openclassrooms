#from app import app
from app.routes import app
import sys

sys.path.append('/path/to/recommendation')
if __name__ == '__main__':
    app.run()