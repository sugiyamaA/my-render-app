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
# 例: ./font/NotoSansJP.ttf をリポジトリに含める前提
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
    # デモ用ダミーデータ
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
# 3) 稼働日数など数値化
#######################
def parse_kadou_nissu(s):
    """
    例:
      "2日以下" -> 2
      "3～4日" -> 3.5
      "7日" -> 7
      "5日" -> 5
    """
    if not isinstance(s, str):
        return None
    # "2日以下"
    m_le = re.match(r"(\d+)日以下", s)
    if m_le:
        return float(m_le.group(1))
    # "3～4日"
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
# 3') 稼働時間をカテゴリ→数値レンジ化する例
#######################
# "4～8時間" -> (4,8)
# "8～12時間" -> (8,12)
# "12時間以上" -> (12, None)
# "4時間未満" -> (None,4)
# などにして比較できるようにする (あくまでサンプル)
def parse_kadou_jikan(s):
    """
    "4～8時間" -> (4,8)
    "12時間以上" -> (12, None)
    "4時間未満" -> (None,4)
    etc...
    """
    if not isinstance(s, str):
        return (None, None)
    # "(\d+)～(\d+)時間" のパターン
    m_range = re.match(r"(\d+)～(\d+)時間", s)
    if m_range:
        low = float(m_range.group(1))
        high = float(m_range.group(2))
        return (low, high)
    # "(\d+)時間以上"
    m_ge = re.match(r"(\d+)時間以上", s)
    if m_ge:
        low = float(m_ge.group(1))
        return (low, None)
    # "(\d+)時間未満"
    m_le = re.match(r"(\d+)時間未満", s)
    if m_le:
        high = float(m_le.group(1))
        return (None, high)
    return (None, None)

if "稼働時間" in df.columns:
    df["稼働時間_range"] = df["稼働時間"].apply(parse_kadou_jikan)
    # 数値で比較する場合は、(low, high)を保存し、フィルタロジックで活用

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
# 条件解析 (対話性・ガイド付きに強化)
#######################
def parse_conditions(user_text: str):
    """
    入力文からフィルタ条件とターゲット列を抽出する
    例: 「荷台形状のミキサ で 稼働日数が5日以上 の 安全装備パッケージ のグラフを作成」
    さらに「稼働時間が8時間以上」などを解釈する
    """
    # 事前に余計な助詞を除去
    user_text = user_text.replace("のグラフを作成", "")
    user_text = user_text.replace("のグラフ", "")
    user_text = re.sub(r"[のでをにはが]", " ", user_text)
    user_text = re.sub(r"\s+", " ", user_text).strip()

    filter_dict = {}
    target_col = None

    # 日数系
    pattern_ge_day = re.compile(r"(\d+)日以上")
    pattern_le_day = re.compile(r"(\d+)日以下")
    pattern_eq_day = re.compile(r"(\d+)日")

    # 時間系 (例: "8時間以上", "4～8時間" など)
    pattern_ge_hour = re.compile(r"(\d+)時間以上")
    pattern_le_hour = re.compile(r"(\d+)時間以下")  # あれば
    pattern_eq_hour = re.compile(r"(\d+)時間")

    tokens = user_text.split()
    col_in_focus = None

    for t in tokens:
        c = find_best_column(t)
        if c:
            col_in_focus = c
            continue

        if col_in_focus:
            # まず「日数系」
            m_ge_d = pattern_ge_day.search(t)
            m_le_d = pattern_le_day.search(t)
            m_eq_d = pattern_eq_day.search(t)

            # 時間系
            m_ge_h = pattern_ge_hour.search(t)
            m_le_h = pattern_le_hour.search(t)
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
                # 例: "8時間以上"
                # filter_dict["稼働時間"] = (">=", 8)
                val = m_ge_h.group(1) + "時間以上"
                filter_dict[col_in_focus] = (">=", val)
                col_in_focus = None
            elif m_le_h:
                val = m_le_h.group(1) + "時間以下"
                filter_dict[col_in_focus] = ("<=", val)
                col_in_focus = None
            elif m_eq_h:
                # "8時間" 等 → 現実的にはどう扱うか？
                val = m_eq_h.group(1) + "時間"
                filter_dict[col_in_focus] = ("==", val)
                col_in_focus = None
            else:
                # 単純文字列
                filter_dict[col_in_focus] = ("==", t)
                col_in_focus = None

    # 文末付近でカラムが再登場すれば target とみなす
    for t in reversed(tokens):
        c = find_best_column(t)
        if c:
            target_col = c
            break

    if not target_col:
        # デフォルト
        target_col = df.columns[0]

    return filter_dict, target_col

#######################
# フィルタ適用 (稼働時間8時間以上→8～12時間,12時間以上を拾う 等)
#######################
def apply_filters(df_in: pd.DataFrame, filter_dict: dict):
    filtered = df_in.copy()

    for col, (op, val) in filter_dict.items():
        if col not in filtered.columns:
            continue

        # (1) 稼働日数の数値比較 (既存)
        if col == "稼働日数" and "稼働日数_num" in filtered.columns:
            use_col = "稼働日数_num"
            m = re.search(r"(\d+)", val)
            if m:
                v = float(m.group(1))
                if op == ">=":
                    filtered = filtered[filtered[use_col] >= v]
                elif op == "<=":
                    filtered = filtered[use_col] <= v
                    filtered = filtered[filtered[use_col] <= v]
                elif op == "==":
                    filtered = filtered[filtered[use_col] == v]

        # (2) 稼働時間のカテゴリを文字列マッチ or 数値レンジ化
        elif col == "稼働時間" and "稼働時間_range" in filtered.columns:
            # "稼働時間_range" は (low, high) タプル
            # 例: (4,8), (8,12), (12,None), (None,4)
            # val が "8時間以上" のとき → low>=8 or low=8～, etc
            # ここでは、簡単に部分一致 + カテゴリをまとめて拾う
            if op == ">=":
                # 例: "8時間以上" が来たら「(8,12) or (12,None)」を拾う
                match_num = re.search(r"(\d+)", val)
                if match_num:
                    th = float(match_num.group(1))
                    # 稼働時間_range列を走査
                    def check_range(r):
                        low, high = r
                        # (low, None)→ 12時間以上
                        # (8,12) → 8～12
                        # ざっくり「区間が閾値以上かどうか」で判定
                        if low is not None and low >= th:
                            return True
                        if high is not None and high >= th and (low or 0) <= th:
                            return True
                        return False
                    filtered = filtered[filtered["稼働時間_range"].apply(check_range)]
            elif op == "<=":
                # 時間以下 (この例ではあまり使わないかも)
                match_num = re.search(r"(\d+)", val)
                if match_num:
                    th = float(match_num.group(1))
                    def check_range(r):
                        low, high = r
                        # "4時間以下" → (None,4) or (0,4)
                        if high is not None and high <= th:
                            return True
                        return False
                    filtered = filtered[filtered["稼働時間_range"].apply(check_range)]
            elif op == "==":
                # 例: "8時間"ちょうどは実際のデータが "8～12時間" か "4～8時間" か曖昧
                # とりあえず部分一致
                # もしくは区間に8が含まれるかどうか
                match_num = re.search(r"(\d+)", val)
                if match_num:
                    x = float(match_num.group(1))
                    def check_range(r):
                        low, high = r
                        if low is None: low = 0
                        if high is None: high = 999  # 仮
                        return (low <= x <= high)
                    filtered = filtered[filtered["稼働時間_range"].apply(check_range)]
            else:
                # 単純部分一致
                filtered = filtered[filtered[col].astype(str).str.contains(val)]

        else:
            # (3) 文字列カラムの単純フィルタ
            if op == "==":
                # 部分一致
                filtered = filtered[filtered[col].astype(str).str.contains(val)]
            elif op == ">=":
                # 例: "10～15"とか"20以上"とかあるかもしれないが、現状は部分一致サンプル
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
        return f"列 '{column_name}' についてデータがありません。", None

    # 数値かカテゴリか
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
# アシスト/ガイドメッセージ
#######################
def get_guide_message():
    guide = """質問が曖昧か、どのカラムにも該当しませんでした。
以下のサンプルを参考に質問してみてください:

- 「稼働日数が5日以上のデータを見せて」
- 「荷台形状がミキサ」
- 「稼働時間が8時間以上 の 稼働日数が5日以上 の 安全装備パッケージ」
- 「休憩時間が1時間未満 の グラフ」
- 「燃費(km/L)が15～20 の グラフを作成」

使用可能なカラム一覧:
"""
    for c in df.columns:
        guide += f"- {c}\n"
    return guide

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
        # 入力が空
        guide_msg = "何について知りたいですか？\n" + get_guide_message()
        return jsonify({"answer": guide_msg, "image": None})

    # 1) conditions を解析
    filter_dict, target_col = parse_conditions(user_text)

    # 2) フィルタ適用
    filtered_df = apply_filters(df, filter_dict)

    # (A) フィルタ結果が0件
    if len(filtered_df) == 0:
        guide_msg = f"条件に合うデータがありませんでした。\n\n{get_guide_message()}"
        return jsonify({"answer": guide_msg, "image": None})

    # (B) カラムが None などになる場合
    if target_col not in df.columns:
        guide_msg = f"対象のカラムが特定できませんでした。\n\n{get_guide_message()}"
        return jsonify({"answer": guide_msg, "image": None})

    # 3) 集計＆グラフ
    text_msg, chart_b64 = get_distribution_and_chart(filtered_df, target_col)

    # 入力が「意味不明」(threshold 以下) の場合はガイドを返すが、
    # ここでは parse_conditions内部で col_in_focus=None になるだけなので、
    # 一定以上の類似カラムが無いときに filter_dict は空になる。
    # filter_dictが空の場合もあり得るので、その時は「曖昧かもしれない」と出しておく
    if len(filter_dict) == 0:
        text_msg = "質問がやや曖昧の可能性があります。\n\n" + text_msg

    return jsonify({"answer": text_msg, "image": chart_b64})

@app.route("/chat")
def chat():
    return """
<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <title>アンケート分析チャット (ブラッシュアップ版)</title>
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
      #chatArea {
        width: 60vw; height: 70vh;
      }
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
  <h1>アンケート分析チャット ver:250108_1</h1>
  <div id="chatArea"></div>
  <div>
    <input type="text" id="question" placeholder="質問を入力して下さい" style="width:70%;" />
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
          // 改行を <br/> に変換
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
      // 一番下までスクロール
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

# メイン
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
