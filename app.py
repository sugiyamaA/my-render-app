import os
import re
import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # サーバー上でmatplotlibを使うため
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from difflib import SequenceMatcher
from flask import Flask, request, jsonify

# =====================================
# 1) 日本語フォントをサーバーサイドで強制指定
# =====================================
# 例: Noto Sans CJK (Ubuntu系でインストール済み想定)
font_path = "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(font_path):
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams["font.family"] = font_prop.get_name()
    print("Using font:", font_prop.get_name())
else:
    plt.rcParams["font.family"] = "sans-serif"
    print("Warning: custom font not found. Using sans-serif.")


app = Flask(__name__)

# ===== CSV読み込み =====
CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "survey_data.csv")
if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH, encoding="utf-8")
    print("CSV loaded:", CSV_PATH)
else:
    # CSVが無い場合デモ用
    df = pd.DataFrame({
        "安全装備パッケージ": ["STANDARD", "PREMIUM", "BASIC", "ADVANCE"],
        "荷台形状": ["ミキサ", "ダンプ", "その他", "バン/ウィング"],
        "稼働日数": ["5日", "2日以下", "7日", "3～4日"],
        # ...以下省略...
    })
    print("CSV not found. Using small demo data.")
print("Columns:", df.columns.tolist())

# =====================================
# 2) 稼働日数などを数値化するヘルパー
# =====================================
def parse_kadou_nissu(s):
    """'7日', '3～4日', '2日以下' などを数値化するサンプル."""
    if not isinstance(s, str):
        return None

    # 2日以下
    m_le = re.match(r"(\d+)日以下", s)
    if m_le:
        # "2日以下" -> 2 という数字を返す(「以下」はどう扱うか要相談)
        return float(m_le.group(1))

    # "3～4日" -> 平均 (3+4)/2
    m_range = re.match(r"(\d+)～(\d+)日", s)
    if m_range:
        start = float(m_range.group(1))
        end = float(m_range.group(2))
        return (start + end) / 2

    # "5日", "7日" など
    m_single = re.match(r"(\d+)日", s)
    if m_single:
        return float(m_single.group(1))

    return None

# 稼働日数を「稼働日数_num」みたいな列に追加する
if "稼働日数" in df.columns:
    df["稼働日数_num"] = df["稼働日数"].apply(parse_kadou_nissu)

# =====================================
# 3) ファジーマッチで列を特定
# =====================================
def normalize_str(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[()\s（）]", "", s)
    return s

def calc_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_str(a), normalize_str(b)).ratio()

def find_best_column(user_text: str, threshold=0.4):
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

# =====================================
# 4) ユーザ入力から複数条件をパース
# =====================================
def parse_conditions(user_text: str):
    """
    例: 「荷台形状のミキサ で 稼働日数が5日以上 の 安全装備パッケージ のグラフを作成」
     -> filter_dict = {
          '荷台形状': ('==', 'ミキサ'),
          '稼働日数': ('>=', '5日')
        }, target_col = "安全装備パッケージ"
    """
    filter_dict = {}
    target_col = None

    pattern_ge = re.compile(r"(\d+)日以上")
    pattern_le = re.compile(r"(\d+)日以下")
    pattern_eq_day = re.compile(r"(\d+)日")

    tokens = user_text.split()
    col_in_focus = None
    for t in tokens:
        # 列名っぽいものを探す
        c = find_best_column(t)
        if c:
            # 例: "荷台形状" にマッチしたらそこに条件をつける想定
            col_in_focus = c
            continue

        # もし列が確定しているなら、その後に来る文字列を条件とみなす
        if col_in_focus:
            m_ge = pattern_ge.search(t)
            m_le = pattern_le.search(t)
            m_eq = pattern_eq_day.search(t)

            if m_ge:
                val = m_ge.group(1) + "日"
                filter_dict[col_in_focus] = (">=", val)
                col_in_focus = None
            elif m_le:
                # 2日以下 などは CSV にそのまま "2日以下" と書いてある場合もある
                val = m_le.group(1) + "日"
                filter_dict[col_in_focus] = ("<=", val)
                col_in_focus = None
            elif m_eq:
                # 5日 など
                filter_dict[col_in_focus] = ("==", m_eq.group(1) + "日")
                col_in_focus = None
            else:
                # 普通の文字列
                # ex: ミキサ, ダンプ, 車載車...
                filter_dict[col_in_focus] = ("==", t)
                col_in_focus = None

    # 文末付近に再登場する列名をターゲット列とみなす (超簡易)
    for t in reversed(tokens):
        c = find_best_column(t)
        if c:
            target_col = c
            break

    # 見つからなければデフォルト
    if not target_col:
        target_col = "安全装備パッケージ"

    return filter_dict, target_col

# =====================================
# 5) フィルタ適用
# =====================================
def apply_filters(df_in: pd.DataFrame, filter_dict: dict):
    filtered = df_in.copy()
    for col, (op, val) in filter_dict.items():
        if col not in filtered.columns:
            continue

        # 稼働日数など数値比較が必要なら、対応する数値列を使う
        # 例: "稼働日数" なら "稼働日数_num" に置き換える
        use_col = col
        if col == "稼働日数" and "稼働日数_num" in filtered.columns:
            use_col = "稼働日数_num"

        # 数値列なら数値比較
        if filtered[use_col].dtype in [int, float]:
            # val から数字を取り出す
            # 例: "5日" -> 5
            m_num = re.search(r"(\d+)", val)
            if m_num:
                vnum = float(m_num.group(1))
                if op == ">=":
                    filtered = filtered[filtered[use_col] >= vnum]
                elif op == "<=":
                    filtered = filtered[use_col] <= vnum
                    filtered = filtered[filtered[use_col] <= vnum]
                elif op == "==":
                    filtered = filtered[filtered[use_col] == vnum]
        else:
            # 文字列比較(部分一致 or 完全一致)
            # ここでは簡易的に「==」を「.str.contains(val)」で実装
            if op == "==":
                filtered = filtered[filtered[col].astype(str).str.contains(val)]
            elif op == ">=":
                # "5日以上" → CSV に実際 "5日" と書いてあれば拾うが
                # "6日" "7日" は別表記なので注意。数値列でやるべき
                # ここでは一応 partial match
                filtered = filtered[filtered[col].astype(str).str.contains(val)]
            elif op == "<=":
                # "2日以下" → CSV に "2日以下" と書いてあれば拾う
                filtered = filtered[filtered[col].astype(str).str.contains(val)]

    return filtered

# =====================================
# 6) グラフ生成
# =====================================
def get_distribution_and_chart(df_in, column_name):
    if column_name not in df_in.columns:
        return f"列 '{column_name}' は存在しません。", None

    series = df_in[column_name]
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

    fig, ax = plt.subplots(figsize=(5,3))
    n = len(counts)
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, n))
    ax.bar(range(n), counts.values, color=colors, edgecolor="white")
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title(column_name)
    ax.set_ylabel("件数")
    ax.grid(axis="y", color="gray", linestyle="--", linewidth=0.5, alpha=0.3)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    chart_b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)

    return text_msg, chart_b64


@app.route("/")
def index():
    return "Hello from Chat - with colorful bars & grid"


@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    user_text = data.get("question", "").strip()
    if not user_text:
        return jsonify({"answer": "何について知りたいですか？", "image": None})

    # 1) 複数条件を解析
    filter_dict, target_col = parse_conditions(user_text)
    # 2) データをフィルタ
    filtered_df = apply_filters(df, filter_dict)

    if len(filtered_df) == 0:
        return jsonify({"answer": "条件に合うデータがありませんでした。", "image": None})

    # 3) ターゲット列で集計＆グラフ
    text_msg, chart_b64 = get_distribution_and_chart(filtered_df, target_col)
    return jsonify({"answer": text_msg, "image": chart_b64})


@app.route("/chat")
def chat():
    return """
<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <title>アンケート分析チャット</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <!-- これらはあくまでHTML上のwebフォント; matplotlibには反映されない -->
  <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@100..900&display=swap" rel="stylesheet">
  <style>
    body {
      /* WebフォントをHTMLに適用 (matplotlib画像には適用されない) */
      font-family: 'Noto Sans JP', sans-serif; 
      margin: 20px;
    }
    h1 { font-size: 18px; }
    #chatArea {
      border: 1px solid #ccc;
      width: 80vw; height: 65vh;
      max-width: 1000px;
      margin-bottom: 10px;
      overflow-y: auto;
    }
    @media (min-width: 1024px) {
      #chatArea {
        width: 60vw; height: 70vh;
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
      cursor: pointer; /* クリック拡大 */
    }
    /* 拡大表示用モーダル */
    #imgModal {
      display: none;
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
  <h1>アンケート分析チャット Version：25/1/07</h1>
  <div id="chatArea"></div>
  <div>
    <input type="text" id="question" placeholder="質問を入力" style="width:70%;" />
    <button onclick="sendMessage()">送信</button>
  </div>

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

    function enlargeImage(img) {
      const modal = document.getElementById("imgModal");
      const modalContent = document.getElementById("imgModalContent");
      modalContent.src = img.src;
      modal.style.display = "block";
    }
    function closeModal() {
      document.getElementById("imgModal").style.display = "none";
    }
  </script>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
