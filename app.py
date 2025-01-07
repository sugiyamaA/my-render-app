import os
import re
import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from difflib import SequenceMatcher
from flask import Flask, request, jsonify

app = Flask(__name__)

#######################
# 1) フォント設定
#######################
# 例: ./font/NotoSansJP.ttf
local_font_path = os.path.join(os.path.dirname(__file__), "font", "NotoSansJP.ttf")
if os.path.exists(local_font_path):
    font_prop = fm.FontProperties(fname=local_font_path)
    plt.rcParams["font.family"] = font_prop.get_name()
    print("Using font from local file:", local_font_path)
else:
    plt.rcParams["font.family"] = "sans-serif"
    print("Warning: NotoSansJP.ttf not found. Fallback to sans-serif.")

#######################
# 2) CSV読み込み
#######################
CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "survey_data.csv")
if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH, encoding="utf-8")
    print("CSV loaded:", CSV_PATH)
else:
    # 簡易デモデータ
    df = pd.DataFrame({
        "安全装備パッケージ": ["STANDARD", "PREMIUM", "ADVANCE", "BASIC", "PREMIUM"],
        "荷台形状": ["ミキサ", "ダンプ", "温度管理車", "バン/ウィング", "ミキサ"],
        "稼働日数": ["5日", "2日以下", "7日", "3～4日", "6日"],
        "稼働時間": ["4～8時間", "8～12時間", "12時間以上", "8～12時間", "4時間未満"],
        "休憩時間": ["1～3時間", "5時間以上", "1時間未満", "3～5時間", "3～5時間"],
        "燃費(km/L)": ["10～15", "20以上", "15～20", "5～10", "10～15"],
    })

print("Columns in CSV:", df.columns.tolist())

#######################
# 3) 稼働日数の数値化 (例)
#######################
def parse_kadou_nissu(s):
    if not isinstance(s, str):
        return None
    # "2日以下" -> 2
    m_le = re.match(r"(\d+)日以下", s)
    if m_le:
        return float(m_le.group(1))
    # "3～4日" -> (3+4)/2=3.5
    m_range = re.match(r"(\d+)～(\d+)日", s)
    if m_range:
        start = float(m_range.group(1))
        end = float(m_range.group(2))
        return (start + end)/2
    # "5日", "7日"
    m_single = re.match(r"(\d+)日", s)
    if m_single:
        return float(m_single.group(1))
    return None

if "稼働日数" in df.columns:
    df["稼働日数_num"] = df["稼働日数"].apply(parse_kadou_nissu)

#######################
# ファジーマッチ
#######################
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

#######################
# ガイドの文面
#######################
def get_guide_message():
    guide = """【ガイド】
以下のように質問できます:
- 「荷台形状がミキサのデータ」
- 「稼働日数が5日以上のグラフ」
- 「稼働時間が8時間以上 の 安全装備パッケージ」
- 「休憩時間が5時間以上 の 燃費(km/L)グラフ作成」
- 「燃費(km/L)が10～15 の結果」

使用可能な列:
"""
    for c in df.columns:
        guide += f"- {c}\n"
    guide += "\n例: 「稼働時間が8時間以上」などと指定してください。"
    return guide

#######################
# 条件解析
#######################
def parse_conditions(user_text: str):
    """
    入力文からフィルタ条件とターゲット列を抽出する
    """
    # 助詞などを除去
    user_text = user_text.replace("のグラフを作成", "")
    user_text = user_text.replace("のグラフ作成", "")
    user_text = user_text.replace("のグラフ", "")
    user_text = user_text.replace("の結果", "")
    # まとめて正規表現で除去
    user_text = re.sub(r"[のでをにはがと]", " ", user_text)  
    user_text = re.sub(r"\s+", " ", user_text).strip()

    filter_dict = {}
    target_col = None

    # 正規表現: 日数 "(\d+)日以上" 等
    pattern_ge_day = re.compile(r"(\d+)日以上")
    pattern_le_day = re.compile(r"(\d+)日以下")
    pattern_eq_day = re.compile(r"(\d+)日")

    # 時間系: "(\d+)時間以上" など (ここでは細かいロジック省略)
    pattern_ge_hour = re.compile(r"(\d+)時間以上")
    pattern_eq_hour = re.compile(r"(\d+)時間")

    tokens = user_text.split()
    col_in_focus = None

    for t in tokens:
        c = find_best_column(t)
        if c:
            col_in_focus = c
            continue

        if col_in_focus:
            # 日数系
            m_ge_d = pattern_ge_day.search(t)
            m_le_d = pattern_le_day.search(t)
            m_eq_d = pattern_eq_day.search(t)

            # 時間系 (簡易)
            m_ge_h = pattern_ge_hour.search(t)
            m_eq_h = pattern_eq_hour.search(t)

            if m_ge_d:
                val = m_ge_d.group(1) + "日"
                filter_dict[col_in_focus] = (">=", val)
                col_in_focus = None
            elif m_le_d:
                val = m_le_d.group(1) + "日"
                filter_dict[col_in_focus] = ("<=", val)
                col_in_focus = None
            elif m_eq_d:
                val = m_eq_d.group(1) + "日"
                filter_dict[col_in_focus] = ("==", val)
                col_in_focus = None
            elif m_ge_h:
                val = m_ge_h.group(1) + "時間以上"
                filter_dict[col_in_focus] = (">=", val)
                col_in_focus = None
            elif m_eq_h:
                val = m_eq_h.group(1) + "時間"
                filter_dict[col_in_focus] = ("==", val)
                col_in_focus = None
            else:
                # 文字列
                filter_dict[col_in_focus] = ("==", t)
                col_in_focus = None

    # 文末付近にもう1回カラム名が出ればそれをターゲットに
    for t in reversed(tokens):
        c = find_best_column(t)
        if c:
            target_col = c
            break

    # target_col が None ならデフォルト1列目に
    if not target_col:
        target_col = df.columns[0]

    return filter_dict, target_col

#######################
# フィルタ適用
#######################
def apply_filters(df_in: pd.DataFrame, filter_dict: dict):
    filtered = df_in.copy()

    for col, (op, val) in filter_dict.items():
        if col not in filtered.columns:
            continue

        # 稼働日数_num で数値比較
        if col == "稼働日数" and "稼働日数_num" in filtered.columns:
            use_col = "稼働日数_num"
            m_num = re.search(r"(\d+)", val)
            if m_num:
                v = float(m_num.group(1))
                if op == ">=":
                    filtered = filtered[filtered[use_col] >= v]
                elif op == "<=":
                    filtered = filtered[use_col] <= v
                    filtered = filtered[filtered[use_col] <= v]
                elif op == "==":
                    filtered = filtered[filtered[use_col] == v]

        # 稼働時間などはここでは部分一致 (簡易実装)
        else:
            if op == "==":
                filtered = filtered[filtered[col].astype(str).str.contains(val)]
            elif op == ">=":
                filtered = filtered[filtered[col].astype(str).str.contains(val)]
            elif op == "<=":
                filtered = filtered[filtered[col].astype(str).str.contains(val)]

    return filtered

#######################
# グラフ生成
#######################
def get_distribution_and_chart(df_in, column_name):
    if column_name not in df_in.columns:
        return f"列 '{column_name}' は存在しません。", None

    series = df_in[column_name]
    if len(series) == 0:
        return f"列 '{column_name}' にデータがありません。", None

    if series.dtype in [int, float]:
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
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(counts)))
    ax.bar(range(len(counts)), counts.values, color=colors, edgecolor="white")
    ax.set_xticks(range(len(counts)))
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

#######################
# Flask Routes
#######################
@app.route("/")
def index():
    return "アンケート分析チャット: /chat へどうぞ"

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    user_text = data.get("question", "").strip()
    if not user_text:
        # 入力が空ならガイド返す
        return jsonify({"answer": "何について知りたいですか？\n" + get_guide_message(), "image": None})

    # (1) conditions 解析
    filter_dict, target_col = parse_conditions(user_text)

    # (2) フィルタ適用
    filtered_df = apply_filters(df, filter_dict)

    # もしフィルタ条件が全く0件 (filter_dictが空) なら、ユーザは列名を指定できなかったと判断
    if len(filter_dict) == 0:
        # さらに filtered_df が df 全体の件数と同じなら(まったく絞り込まれていない)
        # → たぶん曖昧な質問
        if len(filtered_df) == len(df):
            return jsonify({"answer": "どの列をどう絞り込むか認識できませんでした。\n" + get_guide_message(), 
                            "image": None})
        # もし偶然全体からなんらかのデータが消えた場合 (レアケース) もガイド
        if len(filtered_df) == 0:
            return jsonify({"answer": "条件に合うデータがありませんでした。\n" + get_guide_message(), 
                            "image": None})

    # (3) フィルタ後の件数をチェック
    if len(filtered_df) == 0:
        return jsonify({"answer": "条件に合うデータがありませんでした。\n" + get_guide_message(), "image": None})

    # (4) ターゲット列でグラフ
    if target_col not in df.columns:
        return jsonify({"answer": "どの列をグラフ化するか分かりませんでした。\n" + get_guide_message(), 
                        "image": None})
    text_msg, chart_b64 = get_distribution_and_chart(filtered_df, target_col)

    # ここまで来て filter_dict が空なら -> 何か曖昧かも？ → ただし 0件ではないならグラフは作る
    # → とりあえず「ガイド」は出さずに回答だけ
    #   (ユーザが「何も指定しなかったけど、とりあえず最初の列で回答」みたいなケースもある)
    return jsonify({"answer": text_msg, "image": chart_b64})

@app.route("/chat")
def chat():
    return """
<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <title>アンケート分析チャット ver:250108_2</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <style>
    body { font-family: sans-serif; margin: 20px; }
    h1 { font-size: 18px; }
    #chatArea {
      border: 1px solid #ccc;
      width: 80vw; height: 65vh;
      max-width: 1000px;
      margin-bottom: 10px;
      overflow-y: auto;
    }
    @media (min-width: 1024px) {
      #chatArea { width: 60vw; height: 70vh; }
    }
    .msg-user { text-align: right; margin: 5px; }
    .msg-ai { text-align: left; margin: 5px; }
    .bubble-user {
      display: inline-block; background-color: #DCF8C6;
      padding: 5px 10px; border-radius: 8px; max-width: 70%; word-wrap: break-word;
    }
    .bubble-ai {
      display: inline-block; background-color: #E4E6EB;
      padding: 5px 10px; border-radius: 8px; max-width: 70%; word-wrap: break-word;
    }
    img {
      max-width: 100%;
      display: block;
      margin-top: 5px;
      cursor: pointer;
    }
    #imgModal {
      display: none; position: fixed; z-index: 9999; left: 0; top: 0;
      width: 100%; height: 100%; background-color: rgba(0,0,0,0.8);
    }
    #imgModalContent {
      margin: 10% auto; display: block; max-width: 90%;
    }
  </style>
</head>
<body>
  <h1>アンケート分析チャット (改訂版)</h1>
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
            imageTag = '<img src="data:image/png;base64,' + msg.image + '" alt="chart" onclick="enlargeImage(this)" />';
          }
          const contentHtml = (msg.content || "").replace(/\\n/g, "<br/>");
          chatArea.innerHTML += `
            <div class="msg-ai">
              <div class="bubble-ai">
                ${contentHtml}
                ${imageTag}
              </div>
            </div>
          `;
        }
      });
      chatArea.scrollTop = chatArea.scrollHeight;
    }

    async function sendMessage() {
      const q = document.getElementById('question').value.trim();
      if (!q) return;
      messages.push({ role: "user", content: q });
      renderMessages();
      document.getElementById('question').value = "";

      const resp = await fetch("/ask", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ question: q })
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
