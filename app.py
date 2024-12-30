import os
import openai
from flask import Flask, request, jsonify

app = Flask(__name__)

# --- Azure OpenAI 用設定 ---
# RenderのEnvironment Variablesに設定した値を読み込む
openai.api_type = "azure"
openai.api_base = os.getenv("AZURE_OPENAI_API_BASE")       # 例: "https://xxxx.openai.azure.com/"
openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION") # 例: "2023-05-15"
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")

# デプロイしたモデルの名前 (例: gpt-35-turbo)
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

@app.route("/")
def index():
    return "Hello from Render + Flask + Azure OpenAI!"

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    user_question = data.get("question", "")  # ユーザが送った質問

    # Azure OpenAI ChatCompletion呼び出し
    try:
        response = openai.ChatCompletion.create(
            engine=DEPLOYMENT_NAME,  # Azureのデプロイ名
            messages=[
                {"role": "system", "content": "あなたは優秀なアシスタントです。"},
                {"role": "user", "content": user_question}
            ],
            temperature=0.7,
        )
        # 回答(assistant role)のテキストを取り出す
        answer = response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        answer = f"Error occurred: {e}"

    return jsonify({"answer": answer})

# ローカル開発用 (Render では gunicorn で起動する)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
