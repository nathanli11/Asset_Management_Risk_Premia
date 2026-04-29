import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "app"))

from app import app

if __name__ == "__main__":
    app.run(debug=True, port=8050)
