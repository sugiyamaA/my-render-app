import os
import openai
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

from flask import Flask, request, jsonify

app = Flask(__name__)

# ----- Azure OpenAI用の設定２ -----
openai.api_type = "azure"
openai.api_base = os.getenv("AZURE_OPENAI_API_BASE")       # 例: "https://xxxx.openai.azure.com/"
openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION") # 例: "2023-05-15" or "2023-06-13"
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")# 例: "gpt-35-turbo" or "gpt-4"

# ----- CSV読み込み -----
#   "data/survey_data.csv" に、Q1〜Q7の1000件分の回答があると想定
CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "survey_data.csv")
if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH)
else:
    # CSVがない場合のデモ用ダミーデータ
    df = pd.DataFrame({
        "Q1 (安全装備パッケージ)": ["BASIC", "STANDARD", "ADVANCE", "BASIC", "PREMIUM"],
        "Q2 (荷台形状)": ["バン/ウィング", "平ボディ", "平ボディ", "ダンプ", "温度管理車"],
        "Q3 (稼働日数)": ["5日", "5日", "6日", "7日", "3~4日"],
        "Q4 (稼働時間)": ["4h~8h", "4h~8h", "8h~12h", "8h~12h", "4h~8h"],
        "Q5 (休憩時間)": ["1h未満", "3h~5h", "3h~5h", "1h未満", "5h以上"],
        "Q6 (走行距離)": ["100km~250km", "100km未満", "500km以上", "250km~500km", "100km~250km"],
        "Q7 (燃費(km/L))": [10.5, 12.3, 9.8, 15.0, 7.6],
    })

# ----- Q列とキーワードのマッピング -----
#   質問文に含まれるキーワード→どのQ列にマッチさせるかを定義
#   ここを増やせば曖昧対応・部分一致をカバー可能
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
    最もマッチしそうな1つのカラム名を返す。
    - 複数ヒットした場合や全くヒットしない場合は None で返す。
    """
    user_question_lower = user_question.lower()
    matched_columns = []

    for item in QUESTION_MAP:
        col = item["column"]
        for kw in item["keywords"]:
            if kw.lower() in user_question_lower:
                matched_columns.append(col)
                break  # いったん1キーワードでも当たれば追加し、次のitemへ

    # 重複を排除
    matched_columns = list(set(matched_columns))

    if len(matched_columns) == 1:
        # 1つに特定できた
        return matched_columns[0]
    else:
        # 0 or 複数の場合は曖昧
        return None

def get_distribution_text_and_chart(column_name: str):
    """
    指定カラムの回答分布を数値＋グラフで返す。
    - 多い順にソートしてわかりやすい文章にする
    - 棒グラフをbase64で返す
    """
    if column_name not in df.columns:
        return (f"'{column_name}' はデータに存在しません。", None)

    series = df[column_name]

    # もし燃費など数値カラムの場合は describe とかも使える
    if series.dtype in [float, int]:
        # 数値データとみなして describe 
        desc = series.describe()  # count, mean, std, min, 25%, 50%, 75%, max
        text_answer = f"【{column_name} の統計情報】\n"
        text_answer += f"- 件数: {desc['count']}\n"
        text_answer += f"- 平均: {desc['mean']:.2f}\n"
        text_answer += f"- 最小: {desc['min']:.2f}\n"
        text_answer += f"- 最大: {desc['max']:.2f}\n"
        # 棒グラフというよりヒストグラムを作る
        counts = pd.cut(series, bins=5).value_counts().sort_index()
    else:
        # 文字列とみなし、value_countsの多い順で並べる
        counts = series.value_counts()
        text_answer = f"【{column_name} の回答分布】(多い順)\n"
        for idx, val in counts.items():
            text_answer += f"- {idx}: {val} 件\n"

    # グラフを作る (countsを棒グラフ or ヒストグラム表示)
    fig, ax = plt.subplots(figsize=(4,3))
    counts.plot(kind='bar', ax=ax)
    ax.set_title(f"{column_name}")
    ax.set_xlabel("回答カテゴリ" if series.dtype == object else "区間")
    ax.set_ylabel("件数")
    plt.tight_layout()

    # 画像をBase64エンコード
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    chart_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)

    return text_answer, chart_base64


@app.route("/")
def index():
    return "Hello from Render + Flask + Azure OpenAI, Q1~Q7 version"

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    user_question = data.get("question", "").strip()

    # 1. ユーザの質問から最も合いそうなQ列を探す
    best_col = find_best_column(user_question)

    if best_col is not None:
        # 2. 該当カラムが見つかった場合→アンケート集計を返す
        text_ans, chart_base64 = get_distribution_text_and_chart(best_col)
        return jsonify({"answer": text_ans, "image": chart_base64})
    else:
        # 3. 見つからない or 複数の可能性がある -> ユーザに質問
        #   例: LLMを呼び出して「不明」状態で回答してもいいが、ここでは聞き返し
        clarification_msg = """質問が曖昧です。以下のいずれかについて知りたいですか？
- Q1(安全装備)
- Q2(荷台形状)
- Q3(稼働日数)
- Q4(稼働時間)
- Q5(休憩時間)
- Q6(走行距離)
- Q7(燃費)
そのどれかを含む文言でもう一度質問してください。"""
        return jsonify({"answer": clarification_msg, "image": None})


# --- チャットUI (LINE風) ---
@app.route("/chat")
def chat_page():
    return """
    <!DOCTYPE html>
    <html lang="ja">
    <head>
      <meta charset="UTF-8" />
      <title>アンケート分析チャット(Q1〜Q7対応)</title>
      <style>
        body { font-family: sans-serif; margin: 20px; }
        #chatArea {
          border: 1px solid #ccc; padding: 10px; width: 400px; height: 400px;
          overflow-y: auto; margin-bottom: 10px;
        }
        .msg-user { text-align: right; margin: 5px; }
        .msg-ai { text-align: left; margin: 5px; }
        .bubble-user {
          display: inline-block; background-color: #DCF8C6; 
          padding: 5px 10px; border-radius: 8px;
        }
        .bubble-ai {
          display: inline-block; background-color: #E4E6EB;
          padding: 5px 10px; border-radius: 8px;
        }
        img { max-width: 200px; display: block; margin-top: 5px; }
      </style>
    </head>
    <body>
      <h1>アンケート分析チャット(Q1〜Q7対応)</h1>
      <div id="chatArea"></div>
      <div>
        <input type="text" id="question" placeholder="質問を入力" style="width:300px;" />
        <button onclick="sendMessage()">送信</button>
      </div>

      <script>
        let messages = []; // 画面上のメッセージ履歴

        function renderMessages() {
          const chatArea = document.getElementById('chatArea');
          chatArea.innerHTML = "";
          messages.forEach(msg => {
            if (msg.role === "user") {
              chatArea.innerHTML += `
                <div class="msg-user"><div class="bubble-user">${msg.content}</div></div>
              `;
            } else {
              let imageTag = "";
              if (msg.image) {
                imageTag = `<img src="data:image/png;base64,${msg.image}" alt="chart" />`;
              }
              chatArea.innerHTML += `
                <div class="msg-ai">
                  <div class="bubble-ai">${msg.content.replace(/\\n/g, "<br/>")}${imageTag}</div>
                </div>
              `;
            }
          });
          chatArea.scrollTop = chatArea.scrollHeight;
        }

        async function sendMessage() {
          const questionValue = document.getElementById('question').value;
          if(!questionValue) return;
          // 自分のメッセージを追加
          messages.push({ role: "user", content: questionValue });
          renderMessages();
          document.getElementById('question').value = "";

          // サーバにPOST
          const resp = await fetch("/ask", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question: questionValue })
          });
          const data = await resp.json();

          // AIレスポンスをmessagesに追加
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

# Render上での実行時: gunicorn app:app --bind 0.0.0.0:$PORT
# ローカルなら: python app.py
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
