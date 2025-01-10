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

##################
# Flask App
##################
app = Flask(__name__)

##################
# 1) フォント設定
##################
local_font_path = os.path.join(os.path.dirname(__file__), "font", "NotoSansJP.ttf")
if os.path.exists(local_font_path):
    try:
        # font_managerを使って fontProp 作成
        font_prop = fm.FontProperties(fname=local_font_path)
        # キャッシュ再構築
        fm._rebuild()
        # rcParams にセット
        plt.rcParams["font.family"] = font_prop.get_name()
        print("Using local font:", font_prop.get_name())
    except Exception as e:
        print("Error loading font:", e)
        plt.rcParams["font.family"] = "sans-serif"
else:
    print("Warning: NotoSansJP.ttf not found. Fallback to sans-serif.")
    plt.rcParams["font.family"] = "sans-serif"

##################
# 2) CSV読み込み
##################
CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "survey_data.csv")
if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH, encoding="utf-8")
    print("CSV loaded:", CSV_PATH)
else:
    # デモデータ
    df = pd.DataFrame({
        "安全装備パッケージ": ["STANDARD", "PREMIUM", "ADVANCE", "BASIC", "PREMIUM"],
        "荷台形状": ["ミキサ", "ダンプ", "温度管理車", "バン/ウィング", "ミキサ"],
        "稼働日数": ["5日", "2日以下", "7日", "3～4日", "6日"],
    })
    print("CSV not found, using demo data.")

print("CSV columns:", df.columns.tolist())

##################
# 3) 稼働日数の数値化
##################
def parse_kadou_nissu(s):
    if not isinstance(s, str):
        return None
    m_le = re.match(r"(\d+)日以下", s)
    if m_le:
        return float(m_le.group(1))
    m_range = re.match(r"(\d+)～(\d+)日", s)
    if m_range:
        start = float(m_range.group(1))
        end = float(m_range.group(2))
        return (start + end)/2
    m_single = re.match(r"(\d+)日", s)
    if m_single:
        return float(m_single.group(1))
    return None

if "稼働日数" in df.columns:
    df["稼働日数_num"] = df["稼働日数"].apply(parse_kadou_nissu)

##################
# 4) ファジーマッチ
##################
def normalize_str(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[()\s（）]", "", s)
    return s

def calc_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_str(a), normalize_str(b)).ratio()

def find_best_column(user_text: str, threshold=0.5):
    """
    threshold=0.5 に設定。
    「あ」 -> 類似度が低すぎ -> None
    「荷台形状」 -> ある程度合致。
    """
    best_score = 0.0
    best_col = None
    for col in df.columns:
        score = calc_similarity(user_text, col)
        if score > best_score:
            best_score = score
            best_col = col
    # print(f"[DEBUG] {user_text} -> best:{best_col} score:{best_score}")
    if best_score < threshold:
        return None
    return best_col

##################
# 5) ガイドメッセージ
##################
def get_guide_message():
    guide_text = """【ガイド】
以下のように質問してみてください:
- 「安全装備パッケージがSTANDARD」
- 「荷台形状がミキサ」
- 「稼働日数が5日以上」
- 「荷台形状がダンプ の 安全装備パッケージ」
(複数条件は「の」で区切り)

使用可能な列:
"""
    for c in df.columns:
        guide_text += f"- {c}\n"
    return guide_text

##################
# 6) 条件解析
##################
def parse_conditions(user_text: str):
    # 助詞除去
    user_text = user_text.replace("のグラフ", "")
    user_text = re.sub(r"[のでをにはが]", " ", user_text)
    user_text = re.sub(r"\s+", " ", user_text).strip()

    filter_dict = {}
    target_col = None

    pat_ge = re.compile(r"(\d+)日以上")
    pat_le = re.compile(r"(\d+)日以下")
    pat_eq = re.compile(r"(\d+)日")

    tokens = user_text.split()
    col_in_focus = None

    for t in tokens:
        c = find_best_column(t)
        if c:
            col_in_focus = c
            continue

        if col_in_focus:
            m_ge = pat_ge.search(t)
            m_le = pat_le.search(t)
            m_eq_ = pat_eq.search(t)

            if m_ge:
                val = m_ge.group(1) + "日"
                filter_dict[col_in_focus] = (">=", val)
                col_in_focus = None
            elif m_le:
                val = m_le.group(1) + "日"
                filter_dict[col_in_focus] = ("<=", val)
                col_in_focus = None
            elif m_eq_:
                val = m_eq_.group(1) + "日"
                filter_dict[col_in_focus] = ("==", val)
                col_in_focus = None
            else:
                # 文字
                filter_dict[col_in_focus] = ("==", t)
                col_in_focus = None

    # 文末付近でターゲット列を探す
    for t in reversed(tokens):
        c = find_best_column(t)
        if c:
            target_col = c
            break

    if not target_col:
        target_col = df.columns[0]

    return filter_dict, target_col

##################
# 7) フィルタ適用
##################
def apply_filters(df_in: pd.DataFrame, filter_dict: dict):
    filtered = df_in.copy()

    for col, (op, val) in filter_dict.items():
        if col not in filtered.columns:
            continue

        if col == "稼働日数" and "稼働日数_num" in filtered.columns:
            # 数値比較
            m_num = re.search(r"(\d+)", val)
            if m_num:
                v = float(m_num.group(1))
                if op == ">=":
                    filtered = filtered[filtered["稼働日数_num"] >= v]
                elif op == "<=":
                    filtered = filtered[filtered["稼働日数_num"] <= v]
                elif op == "==":
                    filtered = filtered[filtered["稼働日数_num"] == v]
        else:
            # 文字列部分一致
            if op == "==":
                filtered = filtered[filtered[col].astype(str).str.contains(val)]
            elif op == ">=":
                filtered = filtered[filtered[col].astype(str).str.contains(val)]
            elif op == "<=":
                filtered = filtered[filtered[col].astype(str).str.contains(val)]

    return filtered

##################
# 8) グラフ生成
##################
def get_distribution_and_chart(df_in: pd.DataFrame, column_name: str):
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

    fig, ax = plt.subplots(figsize=(4,3))
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

##################
# 9) Flask ルート
##################
@app.route("/")
def index():
    return "Hello from Chat - see /chat"

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    user_text = data.get("question", "").strip()

    # 入力が空
    if not user_text:
        return jsonify({
            "answer": "何について知りたいですか？\n" + get_guide_message(),
            "image": None
        })

    filter_dict, target_col = parse_conditions(user_text)
    filtered_df = apply_filters(df, filter_dict)

    # フィルタ0個: target_colがあれば全データのグラフ
    if len(filter_dict) == 0:
        if target_col in df.columns:
            msg, chart = get_distribution_and_chart(df, target_col)
            return jsonify({"answer": "（列条件なし）\n" + msg, "image": chart})
        else:
            # どの列にも該当しない→ガイド
            return jsonify({
                "answer": "列を認識できませんでした。\n" + get_guide_message(),
                "image": None
            })

    # フィルタ後0件
    if len(filtered_df) == 0:
        return jsonify({
            "answer": "条件に合うデータがありませんでした。\n" + get_guide_message(),
            "image": None
        })

    # ターゲット列が実在しない
    if target_col not in df.columns:
        return jsonify({
            "answer": "グラフ化する列がわかりませんでした。\n" + get_guide_message(),
            "image": None
        })

    msg, chart = get_distribution_and_chart(filtered_df, target_col)
    return jsonify({"answer": msg, "image": chart})


##################
# 10) チャット画面
##################
@app.route("/chat")
def chat():
    """
    - 初回ロード時にガイドを自動送信する例
    """
    return """
<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <title>アンケート分析チャット</title>
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
      padding: 5px 10px; border-radius: 8px;
      max-width: 70%; word-wrap: break-word;
    }
    .bubble-ai {
      display: inline-block; background-color: #E4E6EB;
      padding: 5px 10px; border-radius: 8px;
      max-width: 70%; word-wrap: break-word;
    }
    img {
      max-width: 100%;
      display: block;
      margin-top: 5px;
      cursor: pointer;
    }
    #imgModal {
      display: none;
      position: fixed; z-index: 9999; left: 0; top: 0;
      width: 100%; height: 100%; background-color: rgba(0,0,0,0.8);
    }
    #imgModalContent {
      margin: 10% auto; display: block; max-width: 90%;
    }
  </style>
</head>
<body onload="initChat()">
  <h1>アンケート分析チャット ver:250108_4</h1>
  <div id="chatArea"></div>
  <div>
    <input type="text" id="question" placeholder="質問を入力" style="width:70%;" />
    <button onclick="sendMessage()">送信</button>
  </div>

  <!-- モーダル -->
  <div id="imgModal" onclick="closeModal()">
    <img id="imgModalContent">
  </div>

  <script>
    let messages = [];

    // 初期ガイドを表示
    function initChat() {
      const welcome = "(初期ガイド)\\n" + 
`""" + get_guide_message().replace("\n", "\\n") + """`;
      messages.push({
        role: "assistant",
        content: welcome,
        image: null
      });
      renderMessages();
    }

    function renderMessages() {
      const chatArea = document.getElementById('chatArea');
      chatArea.innerHTML = "";
      messages.forEach(msg => {
        if (msg.role === "user") {
          chatArea.innerHTML += `
            <div class="msg-user">
              <div class="bubble-user">${msg.content}</div>
            </div>`;
        } else {
          let imageTag = "";
          if (msg.image) {
            imageTag = '<img src="data:image/png;base64,' + msg.image + '" alt="chart" onclick="enlargeImage(this)" />';
          }
          let contentHtml = (msg.content || "").replace(/\\n/g, "<br/>");
          chatArea.innerHTML += `
            <div class="msg-ai">
              <div class="bubble-ai">
                ${contentHtml}
                ${imageTag}
              </div>
            </div>`;
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
        headers: { "Content-Type": "application/json" },
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
