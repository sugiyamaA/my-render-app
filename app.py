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
    # encoding を明示
    df = pd.read_csv(CSV_PATH, encoding="utf-8")
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
    """
    df.columns の中で最も類似度が高い列名を返す。
    threshold未満なら None。
    """
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

# ===== 条件解析用の簡易パーサ =====
def parse_conditions(user_text: str):
    """
    ユーザテキストから
    - フィルタ条件: {列名: (op, value)} の辞書
    - 最終的に集計・可視化したい列 (target_col)
    を抽出する簡易実装の例。
    
    例: 「荷台形状のミキサで稼働日数が5日以上の安全装備パッケージのグラフ」
     -> {
          "Q2 (荷台形状)": ("==", "ミキサ"),
          "Q3 (稼働日数)": (">=", "5日")
        }, target_col = "Q1 (安全装備パッケージ)"
    """

    # まず全列名(日本語部分含む)をファジーマッチ用にリスト化
    col_candidates = [c for c in df.columns]

    filter_dict = {}
    target_col = None

    # 分かりやすくするため「○○以上」「○○以下」は正規表現で捉える例
    # 厳密な実装はデータの形式次第で要カスタム
    pattern_ge = re.compile(r"(\d+)日以上")
    pattern_le = re.compile(r"(\d+)日以下")
    pattern_eq_day = re.compile(r"(\d+)日")

    # テキストをスペース区切りでスキャン(超簡易)
    tokens = user_text.split()

    # すべてのトークンをざっくり見て、
    # 1) カラム名っぽいもの
    # 2) その後の条件(以上, 以下, 完全一致 など)
    # を推定する例
    col_in_focus = None
    for t in tokens:
        # まずはファジーマッチで列候補を探す
        col_found = find_best_column(t)
        if col_found:
            # 新規にカラムを認識したら、注目カラムにセット
            col_in_focus = col_found
            continue
        
        # カラムが確定している状態で、その条件を拾う
        if col_in_focus:
            # 例: 「ミキサ」「バン/ウィング」など文字列一致
            #     「5日以上」「3日以下」「5日」など
            m_ge = pattern_ge.search(t)
            m_le = pattern_le.search(t)
            m_eq = pattern_eq_day.search(t)

            if m_ge:
                val = m_ge.group(1) + "日"  # "5日" など
                filter_dict[col_in_focus] = (">=", val)
                col_in_focus = None
            elif m_le:
                val = m_le.group(1) + "日"
                filter_dict[col_in_focus] = ("<=", val)
                col_in_focus = None
            elif m_eq:
                # もし「◯日以上」「◯日以下」にマッチしなかったら
                # いったん「==」とみなす
                val = m_eq.group(1) + "日"
                filter_dict[col_in_focus] = ("==", val)
                col_in_focus = None
            else:
                # 日数以外(文字列)はそのまま "==" 扱いにする(ざっくり)
                # 例えば "ミキサ" "平ボディ" "BASIC" など
                filter_dict[col_in_focus] = ("==", t)
                col_in_focus = None

    # 最後に「○○のグラフ」っぽいところから target_col を推定
    # 今回は非常に大雑把に、文末付近に再度登場したカラム名をターゲットとみなす
    # 例: 「○○で ×× の グラフ」→ ×× をターゲット
    possible_target = None
    # 文末付近の単語からカラムっぽいものを探索
    for t in reversed(tokens):
        c = find_best_column(t)
        if c:
            possible_target = c
            break

    if possible_target is not None:
        target_col = possible_target
    else:
        # もし特定できなければ適当にQ1をデフォルトターゲットに
        target_col = "Q1 (安全装備パッケージ)"

    return filter_dict, target_col

def apply_filters(df: pd.DataFrame, filter_dict: dict):
    """
    parse_conditions で抽出したフィルタ辞書を使い、
    データフレームを絞り込む
    """
    filtered_df = df.copy()
    for col, (op, val) in filter_dict.items():
        # 対象カラムが文字列なら単純一致/含む など好きに
        # 例: 日数カラムが "5日", "3~4日" などの場合、
        #     "5日以上" -> "5日" or "6日" or "7日" ... のように本当は細かい変換要
        #     ここではサンプルとして "5日" を含むかどうかで判定してみる例
        if col not in filtered_df.columns:
            continue
        
        # 数値カラムの場合
        if filtered_df[col].dtype in [int, float]:
            # 数値として比較する例(必要に応じて変換等)
            try:
                val_num = float(re.findall(r"(\d+\.?\d*)", val)[0])
            except:
                val_num = None

            if val_num is not None:
                if op == ">=":
                    filtered_df = filtered_df[filtered_df[col] >= val_num]
                elif op == "<=":
                    filtered_df = filtered_df[filtered_df[col] <= val_num]
                elif op == "==":
                    filtered_df = filtered_df[filtered_df[col] == val_num]
        else:
            # 文字列カラムの場合
            if op == "==":
                # 例: 単純に "val を含む" として絞る
                #     （完全一致にしたいなら == val）
                filtered_df = filtered_df[filtered_df[col].astype(str).str.contains(val)]
            elif op == ">=":
                # "5日以上" のように文字列同士で大小比較は普通できないので、
                # 日数だけ特別処理など要実装
                # ここは簡易例なので「val を含む行とみなす」という雑実装
                filtered_df = filtered_df[filtered_df[col].astype(str).str.contains(val)]
            elif op == "<=":
                # 同様に雑実装
                filtered_df = filtered_df[~filtered_df[col].astype(str).str.contains(val)]

    return filtered_df

# ===== データ集計＋グラフ生成 =====
def get_distribution_and_chart(df: pd.DataFrame, column_name: str):
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
@app.route("/")
def index():
    return "Hello from Chat - with colorful bars & grid"

@app.route("/ask", methods=["POST"])
def ask():
    """
    POST { "question": "安全装備" } など
    1) ユーザ入力の解析（複数条件＆ターゲットカラム）
    2) データをフィルタ
    3) ターゲットカラムを集計＆グラフ
    4) なければ、曖昧メッセージ
    """
    data = request.json
    user_text = data.get("question", "").strip()

    if not user_text:
        return jsonify({"answer": "何について知りたいですか？", "image": None})

    # --- ここで複数条件を解析 ---
    filter_dict, target_col = parse_conditions(user_text)

    # --- フィルタ適用 ---
    filtered_df = apply_filters(df, filter_dict)

    if len(filtered_df) == 0:
        # フィルタの結果が0件ならメッセージ
        return jsonify({
            "answer": "条件に合うデータがありませんでした。",
            "image": None
        })

    # --- ターゲット列で集計＆グラフ生成 ---
    if target_col:
        text_msg, chart_b64 = get_distribution_and_chart(filtered_df, target_col)
        return jsonify({"answer": text_msg, "image": chart_b64})
    else:
        # ターゲット列が見つからなかった場合のフォールバック
        fallback_msg = """質問がわからなかったです。以下のどれかを含む文言でもう一度質問してください。
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
