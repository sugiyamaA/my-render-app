import os
import openai
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import re

from difflib import SequenceMatcher
from flask import Flask, request, jsonify

app = Flask(__name__)

# ======================
# 1. Azure OpenAI 用設定 (必要なら使用)
# ======================
openai.api_type = "azure"
openai.api_base = os.getenv("AZURE_OPENAI_API_BASE")       # 例: "https://xxxx.openai.azure.com/"
openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION") # 例: "2023-05-15" or "2023-06-13"
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")# 例: "gpt-35-turbo" or "gpt-4"

# ======================
# 2. CSV読み込み
# ======================
CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "survey_data.csv")

if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH)
    print("=== CSV Loaded Successfully ===")
    print(f"CSV_PATH: {CSV_PATH}")
else:
    print("=== CSV NOT FOUND, using demo data ===")
    # デモ用のダミーデータ(列名はQ1〜Q7)
    df = pd.DataFrame({
        "Q1 (安全装備パッケージ)": ["BASIC", "STANDARD", "ADVANCE", "BASIC", "PREMIUM"],
        "Q2 (荷台形状)": ["バン/ウィング", "平ボディ", "平ボディ", "ダンプ", "温度管理車"],
        "Q3 (稼働日数)": ["5日", "5日", "6日", "7日", "3~4日"],
        "Q4 (稼働時間)": ["4h~8h", "4h~8h", "8h~12h", "8h~12h", "4h~8h"],
        "Q5 (休憩時間)": ["1h未満", "3h~5h", "3h~5h", "1h未満", "5h以上"],
        "Q6 (走行距離)": ["100km~250km", "100km未満", "500km以上", "250km~500km", "100km~250km"],
        "Q7 (燃費(km/L))": [10.5, 12.3, 9.8, 15.0, 7.6],
    })

print("=== Current df.columns ===", df.columns.tolist())


# ======================
# 3. ファジー検索の仕組み (列名との類似度)
# ======================
def normalized(s: str) -> str:
    """ 空白やカッコなどを除去して小文字化して比較用に正規化 """
    s = s.lower()
    s = re.sub(r"[()\s（）]", "", s)  # カッコや空白を全部除去
    return s

def calc_similarity(a: str, b: str) -> float:
    """ 2文字列の類似度(0.0〜1.0) """
    a_norm = normalized(a)
    b_norm = normalized(b)
    return SequenceMatcher(None, a_norm, b_norm).ratio()

def find_best_column(user_question: str):
    """
    ユーザ入力と df.columns の各列名をファジーマッチし、
    もっとも似ている列とそのスコアを返す。
    スコアが一定の閾値未満なら None。
    """
    best_col = None
    best_score = 0.0
    for col in df.columns:
        score = calc_similarity(user_question, col)
        if score > best_score:
            best_score = score
            best_col = col

    # 閾値を設定 (例: 0.4〜0.7 お好みで調整)
    if best_score < 0.4:
        return None
    else:
        return best_col


def get_distribution_text_and_chart(column_name: str):
    """
    指定カラムの回答分布または統計をテキストとグラフ(Base64)で返す。
    """
    if column_name not in df.columns:
        # ここに来るケースはほぼ無いが、一応対処
        return f"列 '{column_name}' はデータに存在しません。", None

    series = df[column_name]
    # 数値カラム vs 文字列カラム
    if series.dtype in [float, int]:
        # 数値データ → describe & ヒストグラム
        desc = series.describe()
        text_answer = f"【{column_name} の統計情報】\n"
        text_answer += f"- 件数: {desc['count']}\n"
        text_answer += f"- 平均: {desc['mean']:.2f}\n"
        text_answer += f"- 最小: {desc['min']:.2f}\n"
        text_answer += f"- 最大: {desc['max']:.2f}\n"
        counts = pd.cut(series, bins=5).value_counts().sort_index()
    else:
        # 文字列データ → value_counts
        counts = series.value_counts()
        text_answer = f"【{column_name} の回答分布】(多い順)\n"
        for idx, val in counts.items():
            text_answer += f"- {idx}: {val} 件\n"

    # グラフ (matplotlib)
    fig, ax = plt.subplots(figsize=(4,3))
    counts.plot(kind='bar', ax=ax)
    ax.set_title(f"{column_name}")
    ax.set_ylabel("件数")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    chart_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)

    return text_answer, chart_base64


# ======================
# 4. Flaskルート
# ======================
app = Flask(__name__)

@app.route("/")
def index():
    return "Hello from Fuzzy Chat! - CSV columns direct match"

@app.route("/ask", methods=["POST"])
def ask():
    """
    ユーザの入力: { "question": "安全装備パッケージ" } など
    1) df.columns とファジーマッチで最適列を探す
    2) 見つかれば集計結果を返す
    3) なければ「どれを見ればよいかわかりません」と返す
    """
    data = request.json
    user_question = data.get("question", "").strip()
    best_col = find_best_column(user_question)

    if best_col:
        text_ans, chart_base64 = get_distribution_text_and_chart(best_col)
        return jsonify({"answer": text_ans, "image": chart_base64})
    else:
        return jsonify({"answer": "すみません、該当する項目が見つかりませんでした。もう少し具体的に教えてください。", "image": None})

@app.route("/chat")
def chat_page():
    """
    フル画面&スマホ対応のチャットUI
    """
    # ログに列名を出して確認
    print("=== DEBUG df.columns ===", df.columns.tolist())
    return """
    <!DOCTYPE html>
    <html lang="ja">
    <head>
      <meta charset="UTF-8" />
      <title>Fuzzy CSV Chat</title>
      <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
      <style>
        body {
          font-family: sans-serif; margin: 20px;
        }
        #chatArea {
          border: 1px solid #ccc;
          width: 80vw;
          height: 65vh;
          max-width: 1000px;
          margin-bottom: 10px;
          overflow-y: auto;
        }
        @media (min-width: 1024px) {
          #chatArea {
            width: 60vw;
            height: 70vh;
          }
        }
        .msg-user {
          text-align: right;
          margin: 5px;
        }
        .msg-ai {
          text-align: left;
          margin: 5px;
        }
        .bubble-user {
          display: inline-block;
          background-color: #DCF8C6;
          padding: 5px 10px;
          border-radius: 8px;
          max-width: 70%;
          word-wrap: break-word;
        }
        .bubble-ai {
          display: inline-block;
          background-color: #E4E6EB;
          padding: 5px 10px;
          border-radius: 8px;
          max-width: 70%;
          word-wrap: break-word;
        }
        img {
          max-width: 100%;
          margin-top: 5px;
        }
      </style>
    </head>
    <body>
      <h1>アンケート分析チャット(Fuzzy検索版)</h1>
      <div id="chatArea"></div>
      <div>
        <input type="text" id="question" placeholder="質問を入力" style="width:70%;" />
        <button onclick="sendMessage()">送信</button>
      </div>

      <script>
        let messages = [];

        function renderMessages() {
          const chatArea = document.getElementById('chatArea');
          chatArea.innerHTML = "";
          messages.forEach(msg => {
            if(msg.role === "user") {
              chatArea.innerHTML += `
                <div class="msg-user">
                  <div class="bubble-user">${msg.content}</div>
                </div>
              `;
            } else {
              let imageTag = "";
              if (msg.image) {
                imageTag = `<img src="data:image/png;base64,${msg.image}" alt="chart"/>`;
              }
              chatArea.innerHTML += `
                <div class="msg-ai">
                  <div class="bubble-ai">
                    ${msg.content.replace(/\\n/g, "<br/>")}
                    ${imageTag}
                  </div>
                </div>
              `;
            }
          });
          chatArea.scrollTop = chatArea.scrollHeight;
        }

        async function sendMessage() {
          const questionValue = document.getElementById('question').value.trim();
          if(!questionValue) return;

          // ユーザメッセージ
          messages.push({ role: "user", content: questionValue });
          renderMessages();
          document.getElementById('question').value = "";

          // サーバに POST
          const resp = await fetch("/ask", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({ question: questionValue })
          });
          const data = await resp.json();

          // AIメッセージ
          messages.push({
            role: "assistant",
            content: data.answer || "(no response)",
            image: data.image
          });
          renderMessages();
        }
      </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
