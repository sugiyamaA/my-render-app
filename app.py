import os
import openai
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Flaskサーバー上でmatplotlibを使うための設定
import matplotlib.pyplot as plt
import io
import base64

from flask import Flask, request, jsonify

app = Flask(__name__)

# ======================
# 1. Azure OpenAI用設定
# ======================
openai.api_type = "azure"
openai.api_base = os.getenv("AZURE_OPENAI_API_BASE")       # "https://xxxx.openai.azure.com/"
openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION") # "2023-05-15" / "2023-06-13" etc.
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")

DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME") # 例: "gpt-35-turbo" or "gpt-4"

# ======================
# 2. CSV読み込み
# ======================
CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "survey_data.csv")

# CSV列名が不明の場合にデバッグしやすいように、実際に読み込めたかチェック
if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH)
    print("=== CSV Loaded Successfully ===")
    print(f"CSV_PATH: {CSV_PATH}")
else:
    print("=== CSV not found, using demo data. ===")
    df = pd.DataFrame({
        "Q1 (安全装備パッケージ)": ["BASIC", "STANDARD", "ADVANCE", "BASIC", "PREMIUM"],
        "Q2 (荷台形状)": ["バン/ウィング", "平ボディ", "平ボディ", "ダンプ", "温度管理車"],
        "Q3 (稼働日数)": ["5日", "5日", "6日", "7日", "3~4日"],
        "Q4 (稼働時間)": ["4h~8h", "4h~8h", "8h~12h", "8h~12h", "4h~8h"],
        "Q5 (休憩時間)": ["1h未満", "3h~5h", "3h~5h", "1h未満", "5h以上"],
        "Q6 (走行距離)": ["100km~250km", "100km未満", "500km以上", "250km~500km", "100km~250km"],
        "Q7 (燃費(km/L))": [10.5, 12.3, 9.8, 15.0, 7.6],
    })

# デバッグ: 読み込んだ列名をログ出力
print("=== Current df.columns ===")
for col in df.columns:
    print(f"col: '{col}'")


# ======================
# 3. Q列とキーワードのマッピング
# ======================
QUESTION_MAP = [
    {
        "column": "Q1 (安全装備パッケージ)",
        "keywords": ["q1", "安全", "安全装備", "パッケージ", "basic", "standard", "advance", "premium"]
    },
    {
        "column": "Q2 (荷台形状)",
        "keywords": ["q2", "荷台", "形状", "架装", "ウィング", "バン", "平ボディ", "ダンプ"]
    },
    {
        "column": "Q3 (稼働日数)",
        "keywords": ["q3", "稼働日", "稼働日数", "日数", "何日"]
    },
    {
        "column": "Q4 (稼働時間)",
        "keywords": ["q4", "稼働時間", "何時間", "時間"]
    },
    {
        "column": "Q5 (休憩時間)",
        "keywords": ["q5", "休憩", "休憩時間"]
    },
    {
        "column": "Q6 (走行距離)",
        "keywords": ["q6", "走行距離", "距離", "km"]
    },
    {
        "column": "Q7 (燃費(km/L))",
        "keywords": ["q7", "燃費", "km/l", "何km", "何キロ"]
    },
]


def find_best_column(user_question: str):
    """
    質問文に含まれるキーワードとQUESTION_MAPを照合し、
    最もマッチしそうなカラム名を返す。
    0件や複数ヒットの場合は None で返す。
    """
    user_question_lower = user_question.lower()
    matched_columns = []

    for item in QUESTION_MAP:
        col = item["column"]
        for kw in item["keywords"]:
            if kw.lower() in user_question_lower:
                matched_columns.append(col)
                break

    matched_columns = list(set(matched_columns))
    if len(matched_columns) == 1:
        return matched_columns[0]
    else:
        return None


def get_distribution_text_and_chart(column_name: str):
    """
    指定カラムの回答分布または統計をテキストとグラフ(Base64)で返す。
    """
    if column_name not in df.columns:
        return (f"'{column_name}' はデータに存在しません。", None)

    series = df[column_name]

    if series.dtype in [float, int]:
        # 数値データ → describe + ヒストグラム
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

    # matplotlib でグラフ生成
    fig, ax = plt.subplots(figsize=(4,3))
    counts.plot(kind='bar', ax=ax)
    ax.set_title(f"{column_name}")
    if series.dtype in [float, int]:
        ax.set_xlabel("値の区間")
    else:
        ax.set_xlabel("回答内容")
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

@app.route("/")
def index():
    return "Hello from Render + Flask + Azure OpenAI (Q1〜Q7対応)"

@app.route("/ask", methods=["POST"])
def ask():
    """
    ユーザの質問を受け取り、最適なQ列を探して集計結果を返す。
    見つからなければ追加で質問を促す。
    """
    data = request.json
    user_question = data.get("question", "").strip()

    best_col = find_best_column(user_question)
    if best_col is not None:
        text_ans, chart_base64 = get_distribution_text_and_chart(best_col)
        return jsonify({"answer": text_ans, "image": chart_base64})
    else:
        clarification_msg = """質問が曖昧です。以下のいずれかについて知りたいですか？
- Q1(安全装備パッケージ)
- Q2(荷台形状)
- Q3(稼働日数)
- Q4(稼働時間)
- Q5(休憩時間)
- Q6(走行距離)
- Q7(燃費)
そのどれかを含む文言でもう一度質問してください。"""
        return jsonify({"answer": clarification_msg, "image": None})


@app.route("/chat")
def chat_page():
    """
    レスポンシブ対応のHTMLチャットUI
    - フル画面に近い大きさ
    - スマホでも見やすい
    """
    # デバッグ: CSVの列名をログに出す（Renderのログで確認）
    print("=== DEBUG: df.columns ===", df.columns.tolist())

    return """
    <!DOCTYPE html>
    <html lang="ja">
    <head>
      <meta charset="UTF-8">
      <title>アンケート分析チャット(Q1〜Q7対応)</title>
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <style>
        body {
          font-family: sans-serif;
          margin: 20px;
        }
        #chatArea {
          border: 1px solid #ccc;
          width: 80vw;    /* レスポンシブ幅 */
          height: 65vh;   /* レスポンシブ高さ */
          max-width: 1000px;
          margin-bottom: 10px;
          overflow-y: auto;
        }
        /* PCがさらに広い場合は余裕があればサイズUP */
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
      <h1>アンケート分析チャット(Q1〜Q7対応)</h1>
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
                  <div class="bubble-ai">${msg.content.replace(/\\n/g, "<br/>")}${imageTag}</div>
                </div>
              `;
            }
          });
          // 下までスクロール
          chatArea.scrollTop = chatArea.scrollHeight;
        }

        async function sendMessage() {
          const questionValue = document.getElementById('question').value;
          if(!questionValue) return;

          // ユーザメッセージ追加
          messages.push({ role: "user", content: questionValue });
          renderMessages();
          document.getElementById('question').value = "";

          // サーバに問い合わせ
          const resp = await fetch("/ask", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question: questionValue })
          });
          const data = await resp.json();

          // AIメッセージ追加
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

# ======================
# ローカル実行 or Render本番
# ======================
if __name__ == "__main__":
    # ローカルでテストする場合は下記コマンド:
    # python app.py
    app.run(host="0.0.0.0", port=5000, debug=True)
