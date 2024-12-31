import os
import re
import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # サーバー上でmatplotlibを使うため
import matplotlib.pyplot as plt

from difflib import SequenceMatcher
from flask import Flask, request, jsonify

# =====================
# 日本語フォント設定（文字化け対策）
# =====================
# 環境に合わせてインストール済みのフォントを優先リストに追加
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "IPAexGothic", "Meiryo", "Hiragino Sans", "sans-serif"]
plt.rcParams["font.family"] = "sans-serif"

# Flaskアプリ
app = Flask(__name__)

# ===== CSV読み込み =====
CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "survey_data.csv")
if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH)
    print("CSV loaded:", CSV_PATH)
else:
    print("CSV not found. Using demo data.")
    df = pd.DataFrame({
        "Q1 (安全装備パッケージ)": ["BASIC", "STANDARD", "ADVANCE", "BASIC", "PREMIUM"],
        "Q2 (荷台形状)": ["バン/ウィング", "平ボディ", "平ボディ", "ダンプ", "温度管理車"],
        "Q3 (稼働日数)": ["5日", "5日", "6日", "7日", "3~4日"],
        "Q4 (稼働時間)": ["4h~8h", "4h~8h", "8h~12h", "8h~12h", "4h~8h"],
        "Q5 (休憩時間)": ["1h未満", "3h~5h", "3h~5h", "1h未満", "5h以上"],
        "Q6 (走行距離)": ["100km~250km", "100km未満", "500km以上", "250km~500km", "100km~250km"],
        "Q7 (燃費(km/L))": [10.5, 12.3, 9.8, 15.0, 7.6],
    })

print("Columns:", df.columns.tolist())

# ===== ファジーマッチ =====
def normalize_str(s: str) -> str:
    """小文字化＋カッコや空白除去"""
    s = s.lower()
    s = re.sub(r"[()\s（）]", "", s)
    return s

def calc_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_str(a), normalize_str(b)).ratio()

def find_best_column(user_text: str, threshold=0.4):
    """df.columns の中で最も類似度が高い列を返す（threshold未満ならNone）"""
    best_score = 0.0
    best_col = None
    for col in df.columns:
        score = calc_similarity(user_text, col)
        if score > best_score:
            best_score = score
            best_col = col
    if best_score < threshold:
        return None
    return best_col

# ===== データ集計＋グラフ生成 =====
def get_distribution_and_chart(column_name: str):
    """指定カラムの分布or統計を返し、バーを同系色(Blues)で表示。グリッドも薄く。"""
    if column_name not in df.columns:
        return f"列 '{column_name}' は存在しません。", None

    series = df[column_name]
    if series.dtype in [float, int]:
        desc = series.describe()
        text_msg = f"【{column_name} の統計】\n"
        text_msg += f"- 件数: {desc['count']}\n"
        text_msg += f"- 平均: {desc['mean']:.2f}\n"
        text_msg += f"- 最小: {desc['min']:.2f}\n"
        text_msg += f"- 最大: {desc['max']:.2f}\n"
        counts = pd.cut(series, bins=5).value_counts().sort_index()
        labels = [str(interval) for interval in counts.index]
    else:
        counts = series.value_counts()
        text_msg = f"【{column_name} の回答分布】\n"
        for idx, val in counts.items():
            text_msg += f"- {idx}: {val} 件\n"
        labels = counts.index.astype(str)

    # グラフ生成
    fig, ax = plt.subplots(figsize=(4,3))
    n = len(counts)
    # Bluesカラーマップを使用
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, n))  # 0.3~0.9の範囲で濃淡
    ax.bar(range(n), counts.values, color=colors, edgecolor="white")
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title(column_name)
    ax.set_ylabel("件数")

    # グリッドを薄く引く
    ax.grid(axis="y", color="gray", linestyle="--", linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    chart_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)

    return text_msg, chart_base64

# Flask ルート
app = Flask(__name__)

@app.route("/")
def index():
    return "Hello from Chat - with colorful bars & grid"

@app.route("/ask", methods=["POST"])
def ask():
    """
    POST { "question": "安全装備" } など
    1) ファジーマッチで列を探す
    2) 見つかれば集計＆グラフ
    3) なければ、曖昧メッセージ
    """
    data = request.json
    user_text = data.get("question", "").strip()

    if not user_text:
        # 単語だけでも拾いたいが、空なら流石に聞き返す
        return jsonify({"answer": "何について知りたいですか？", "image": None})

    best_col = find_best_column(user_text)
    if best_col:
        text_msg, chart_b64 = get_distribution_and_chart(best_col)
        return jsonify({"answer": text_msg, "image": chart_b64})
    else:
        # 曖昧メッセージ
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
def chat():
    """
    HTML: チャットUI (h1フォントを18px, グラフクリックで拡大)
    """
    return """
<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <title>アンケート分析チャット</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <style>
    body {
      font-family: sans-serif; margin: 20px;
    }
    h1 {
      font-size: 18px; /* タイトルっぽく */
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
      cursor: pointer; /* クリック可能にする */
    }
    /* 拡大表示用モーダル */
    #imgModal {
      display: none; /* 初期は非表示 */
      position: fixed; 
      z-index: 9999;
      left: 0; top: 0;
      width: 100%; height: 100%;
      background-color: rgba(0,0,0,0.8);
    }
    #imgModalContent {
      margin: 10% auto;
      display: block;
      max-width: 90%;
    }
  </style>
</head>
<body>
  <h1>アンケート分析チャット</h1>
  <div id="chatArea"></div>
  <div>
    <input type="text" id="question" placeholder="質問を入力" style="width:70%;" />
    <button onclick="sendMessage()">送信</button>
  </div>

  <!-- 画像拡大用モーダル -->
  <div id="imgModal" onclick="closeModal()">
    <img id="imgModalContent">
  </div>

  <script>
    let messages = [];

    function renderMessages() {
      const chatArea = document.getElementById('chatArea');
      chatArea.innerHTML = "";
      messages.forEach(msg => {
        if (msg.role === "user") {
          chatArea.innerHTML += `
            <div class="msg-user">
              <div class="bubble-user">${msg.content}</div>
            </div>
          `;
        } else {
          let imageTag = "";
          if (msg.image) {
            imageTag = `<img src="data:image/png;base64,${msg.image}" alt="chart" onclick="enlargeImage(this)" />`;
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
        headers: {"Content-Type": "application/json"},
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

    // 画像クリックでモーダル拡大
    function enlargeImage(img) {
      const modal = document.getElementById("imgModal");
      const modalContent = document.getElementById("imgModalContent");
      modalContent.src = img.src;
      modal.style.display = "block";
    }
    function closeModal() {
      const modal = document.getElementById("imgModal");
      modal.style.display = "none";
    }
  </script>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
