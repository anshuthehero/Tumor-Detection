import os
from waitress import serve
from app import app  # Ensure 'app' is the Flask instance from your app.py

if __name__ == "__main__":
    # Set the port dynamically if deployed on cloud services like Render, Heroku, etc.
    PORT = int(os.environ.get("PORT", 5000))
    # Run the app using Waitress
    serve(app, host='0.0.0.0', port=PORT)
