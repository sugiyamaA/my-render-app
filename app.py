import os
import openai
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import re
import numpy as np

from difflib import SequenceMatcher
from flask import Flask, request, jsonify

app = Flask(__name__)

# ======================
# 1. Azure OpenAI 用設定
# ======================
openai.api_type = "azure"
openai.api_base = os.getenv("AZURE_OPENAI_API_BASE", "")
openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "")
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "")

# ======================
# 2. CSV読み込み
# ======================
CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "survey_data.csv")
if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH)
    print("=== CSV Loaded Successfully ===", CSV_PATH)
else:
    print("=== CSV NOT FOUND, using demo data ===")
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
# 3. 日本語フォント設定(文字化け対策)
# ======================
# Windows環境やLinux環境で使えるフォントが違うので、ここでは Meiryo を優先
# もしサーバーに無い場合は Noto Sans CJK や IPAexGothic など使えるフォントを指定してください
plt.rcParams['font.sans-serif'] = ['Meiryo','IPAexGothic','Noto Sans CJK JP','Hiragino Maru Gothic Pro']
plt.rcParams['font.family'] = 'sans-serif'

# ======================
# 4. ファジーマッチで列を探す
# ======================
def normalized(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[()\s（）]", "", s)
    return s

def calc_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalized(a), normalized(b)).ratio()

def find_best_column(user_input: str):
    """
    df.columns の中で最も類似度が高い列を探し、閾値未満なら None を返す
    """
    best_col = None
    best_score = 0.0
    for col in df.columns:
        score = calc_similarity(user_input, col)
        if score > best_score:
            best_score = score
            best_col = col
    # 閾値(0.4あたり)はお好みで調整
    if best_score < 0.4:
        return None
    return best_col

# ======================
# 5. データ集計 & グラフ生成
# ======================
def get_distribution_text_and_chart(column_name: str):
    """ 指定カラムの回答分布 or 数値統計を返す + 各バー別色 """
    if column_name not in df.columns:
        return f"列 '{column_name}' はデータに存在しません。", None

    series = df[column_name]
    if series.dtype in [float,int]:
        desc = series.describe()
        text_answer = f"【{column_name} の統計情報】\n"
        text_answer += f"- 件数: {desc['count']}\n"
        text_answer += f"- 平均: {desc['mean']:.2f}\n"
        text_answer += f"- 最小: {desc['min']:.2f}\n"
        text_answer += f"- 最大: {desc['max']:.2f}\n"
        counts = pd.cut(series, bins=5).value_counts().sort_index()
        index_labels = [str(interval) for interval in counts.index]
    else:
        counts = series.value_counts()
        text_answer = f"【{column_name} の回答分布】(多い順)\n"
        for idx, val in counts.items():
            text_answer += f"- {idx}: {val} 件\n"
        index_labels = counts.index.astype(str)

    # distinct color for each bar
    # matplotlib colormapsを使ってユニークカラーリストを作る例
    n = len(counts)
    colors = plt.cm.tab10(np.linspace(0, 1, n))  # tab10 は10色パレット, n>10なら再利用

    fig, ax = plt.subplots(figsize=(4,3))
    ax.bar(x=range(n), height=counts.values, color=colors)
    ax.set_title(column_name)
    ax.set_xticks(range(n))
    ax.set_xticklabels(index_labels, rotation=45, ha="right")
    ax.set_ylabel("件数")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    chart_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)

    return text_answer, chart_base64

# ======================
# 6. Flaskアプリ
# ======================
app = Flask(__name__)

@app.route("/")
def index():
    return "Hello from Fuzzy Chat with color bars & fallback message"

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    user_question = data.get("question","").strip()
    best_col = find_best_column(user_question)

    if best_col:
        text_ans, chart_base64 = get_distribution_text_and_chart(best_col)
        return jsonify({"answer": text_ans, "image": chart_base64})
    else:
        # 質問が曖昧な場合
        fallback_msg = """質問が曖昧です。以下のどれかを含む文言でもう一度質問してください。
- Q1(安全装備パッケージ)
- Q2(荷台形状)
- Q3(稼働日数)
- Q4(稼働時間)
- Q5(休憩時間)
- Q6(走行距離)
- Q7(燃費)
"""
        return jsonify({"answer": fallback_msg, "image": None})

@app.route("/chat")
def chat_page():
    print("=== DEBUG df.columns ===", df.columns.tolist())
    return """
    <!DOCTYPE html>
    <html lang="ja">
    <head>
      <meta charset="UTF-8"/>
      <title>Fuzzy CSV Chat (Colorful Bars)</title>
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
          display: block;
          margin-top: 5px;
        }
      </style>
    </head>
    <body>
      <h1>アンケート分析チャット(多色バー + フォント設定)</h1>
      <div id="chatArea"></div>
      <div>
        <input type="text" id="question" placeholder="質問を入力" style="width:70%;" />
        <button onclick="sendMessage()">送信</button>
      </div>

      <script>
        let messages = [];

        function renderMessages(){
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
              if(msg.image) {
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

          messages.push({ role: "user", content: questionValue });
          renderMessages();
          document.getElementById('question').value = "";

          const resp = await fetch("/ask", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question: questionValue })
          });
          const data = await resp.json();

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
