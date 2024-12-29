# my-render-app/app.py

from flask import Flask

app = Flask(__name__)

@app.route("/")
def index():
    return "Hello from Render + Flask!"

# ローカル実行用
if __name__ == "__main__":
    app.run(port=5000, debug=True)