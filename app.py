import os
import io
import base64
import json

from flask import Flask, request, jsonify, render_template_string
import openai  # azure-openai ではなく、場合により azure-openai ライブラリを使う場合も
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # サーバーサイドで描画するため
import matplotlib.pyplot as plt

# --------------------------------
# Flaskアプリ設定
# --------------------------------
app = Flask(__name__)

# ◆Azure OpenAI のエンドポイント・キー設定
#   実際には Render の環境変数などで設定するほうが安全です
#   Azureの場合、APIベースURLが通常のOpenAIと異なる
#   例: "https://<your-resource-name>.openai.azure.com/"
openai.api_type = "azure"
openai.api_base = os.environ.get("OPENAI_API_BASE", "https://<YOUR-AZURE-RESOURCE-NAME>.openai.azure.com/")
openai.api_version = "2023-07-01-preview"
openai.api_key = os.environ.get("OPENAI_API_KEY", "<YOUR-AZURE-OPENAI-KEY>")

# --------------------------------
# 簡易UI用のHTML（本格的なフロントは別ファイルでもOK）
# --------------------------------
# 画面の仕様：PCは幅800px中央、SPはフルサイズ、入力はフッター固定
# ここでは超簡易的にHTMLを返す例
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8" />
    <title>Azure OpenAI Chat (Render)</title>
    <style>
        body { margin:0; padding:0; font-family: sans-serif; }
        .container {
            max-width: 800px;
            margin: 0 auto;
            padding: 1em;
        }
        .chat-bubble {
            border: 1px solid #ccc;
            padding: 0.5em;
            margin: 0.5em 0;
            border-radius: 0.5em;
        }
        .user { background-color: #eef; }
        .assistant { background-color: #efe; }
        .footer-input {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background: #fafafa;
            border-top: 1px solid #ccc;
            padding: 0.5em;
        }
        .footer-input form {
            display: flex;
        }
        .footer-input input[type='text'] {
            flex: 1;
            padding: 0.5em;
            margin-right: 0.5em;
        }
    </style>
</head>
<body>
<div class="container">
    <h1>Azure OpenAI - アンケート結果可視化チャット</h1>
    <div id="chat-box"></div>
</div>
<div class="footer-input">
  <form id="chat-form">
    <input type="text" id="user-input" placeholder="ここに入力..." />
    <button type="submit">送信</button>
  </form>
</div>

<script>
const chatBox = document.getElementById("chat-box");
const chatForm = document.getElementById("chat-form");
const userInput = document.getElementById("user-input");

chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    let text = userInput.value.trim();
    if(!text) return;
    
    // 画面にユーザー投稿を表示
    appendMessage("user", text);
    userInput.value = "";
    
    // サーバーにPOST
    let res = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type":"application/json" },
        body: JSON.stringify({ user_input: text })
    });
    let data = await res.json();
    // data: { message, graph_base64? ... }
    
    appendMessage("assistant", data.message);
    
    if(data.graph_base64){
        let img = document.createElement("img");
        img.src = "data:image/png;base64," + data.graph_base64;
        img.style.maxWidth = "100%";
        chatBox.appendChild(img);
    }
});

// チャットメッセージを追加表示する
function appendMessage(role, content){
    let div = document.createElement("div");
    div.className = "chat-bubble " + role;
    div.textContent = content;
    chatBox.appendChild(div);
    // スクロール位置を下へ
    window.scrollTo(0, document.body.scrollHeight);
}
</script>
</body>
</html>
"""

@app.route('/')
def index():
    # 単純に上記HTMLテンプレートを返すだけ
    return render_template_string(HTML_TEMPLATE)


# --------------------------------
# システムプロンプト: "自然言語→フィルタ条件(JSON)を出力して"
# --------------------------------
SYSTEM_PROMPT = """
あなたは荷台形状や稼働時間などが書かれたCSVをフィルタする役割です。
ユーザーの要望を受け、以下の形式のJSONのみを出力してください。
出力以外の文章は書かないでください。

JSON形式:
{
  "conditions": [
    {
      "column": "稼働時間",
      "op": ">=",
      "value": 3
    },
    {
      "column": "荷台形状",
      "op": "IN",
      "value": ["ミキサ","ダンプ"]
    },
    ...
  ],
  "output": {
    "type": "bar_chart",    // bar_chart, or line_chart... (今回はbar_chart想定)
    "group_by": "安全装備パッケージ",
    "aggregate": "count",   // count, or percentage
    "title": "稼働時間3時間以上のミキサ・ダンプ"
  }
}

※ 使える op は次の通り: ==, !=, IN, >=, <=, BETWEEN
※ "value"は文字列 or 数値 or 配列
※ "group_by"にはグラフの横軸に使いたい列を指定
※ "aggregate"が "percentage"なら、各グループが全体に占める割合(%)を計算

"""


# --------------------------------
# /chat エンドポイント
# --------------------------------
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('user_input', '')

    # (1) ChatGPT(Azure OpenAI) へ問い合わせ - JSONを生成してもらう
    try:
        # ChatCompletion (Azure) は 'deployment_id' or 'engine' の指定が必要
        # 例: deployment_name = "gpt-35-model" (Azure で作成した名前)
        deployment_name = os.environ.get("OPENAI_DEPLOYMENT_NAME", "gpt-35-model")

        response = openai.ChatCompletion.create(
            engine=deployment_name,
            messages = [
                {"role":"system", "content": SYSTEM_PROMPT},
                {"role":"user",   "content": user_input}
            ],
            temperature=0.2,
            max_tokens=800,
        )
        ai_content = response['choices'][0]['message']['content']
        
    except Exception as e:
        return jsonify({
            "message": f"Azure OpenAIの呼び出しに失敗: {str(e)}"
        })

    # (2) OpenAIから返ってきた JSONパース
    #     失敗した場合はエラーメッセージを返す
    try:
        # Azure側で余計な文章が混じる可能性があるため、正規表現などで { ... } 部分だけ抜き取ることも検討。
        # ここでは一旦 json.loads() 試行
        parsed = json.loads(ai_content)
    except Exception as e:
        return jsonify({
            "message": f"ChatGPT応答のJSONパースに失敗: {str(e)}\n応答:\n{ai_content}"
        })
    
    # (3) CSVを読み込み
    df = pd.read_csv("data/survey_data.csv")

    # (4) CSV列のうち「稼働時間」や「休憩時間」「燃費(km/L)」などの文字列を、数値範囲にパースする準備
    #     ここでは例として「稼働時間」「休憩時間」は parse_hour_range() で範囲にするなど。
    #     本格的には複数列対応が必要。サンプルとして稼働時間だけパース例を作ります。
    def parse_hour_range(x):
        """
        '8～12時間' -> (8,12), '12時間以上' -> (12,999), '4時間未満' -> (0,4)等
        """
        x = str(x).strip()
        if '以上' in x:
            lower = int(x.replace('時間以上','').strip())
            return (lower, 9999)
        elif '未満' in x:
            upper = int(x.replace('時間未満','').strip())
            return (0, upper)
        elif '～' in x:
            # 例: "8～12時間"
            tmp = x.replace('時間','').split('～')
            lower = int(tmp[0])
            upper = int(tmp[1])
            return (lower, upper)
        else:
            # それ以外の形式: "3時間" などあれば適宜
            try:
                val = int(x.replace('時間',''))
                return (val, val)
            except:
                return (0,0)
    
    # DataFrameに範囲用カラムを追加
    df['稼働時間_range'] = df['稼働時間'].apply(parse_hour_range)
    
    # (5) フィルタ適用
    def row_matches_condition(row, cond):
        # cond は { "column":"稼働時間", "op":">=", "value":3 } など
        col = cond["column"]
        op  = cond["op"]
        val = cond["value"]

        # 特例: 稼働時間の場合は range と比較
        if col == "稼働時間":
            # row['稼働時間_range'] => (low, high)
            (low, high) = row['稼働時間_range']
            if op == ">=":
                return low >= val
            elif op == "<=":
                return high <= val
            elif op == "==":
                # ある範囲内に val が含まれている？ など運用次第
                return low <= val <= high
            elif op == "BETWEEN":
                # valが [v1, v2] だとして (low,high)全体がその範囲内かどうか？
                # ここは要件に合わせて実装
                if not isinstance(val, list) or len(val)!=2:
                    return False
                v1, v2 = val
                return (low >= v1) and (high <= v2)
            else:
                return False
        else:
            # それ以外の列
            cell_value = row[col]
            
            # 文字列 or 数値 どちらで比較するか？
            # CSVが文字列の場合が多いので、ひとまず文字列前提
            cell_value_str = str(cell_value).strip()
            
            # op に応じて判定
            if op == "==":
                # val が文字列の場合
                return cell_value_str == str(val)
            elif op == "!=":
                return cell_value_str != str(val)
            elif op == "IN":
                if not isinstance(val, list):
                    return False
                return cell_value_str in [str(v) for v in val]
            # >=, <= とかは、本来数値変換が必要
            # 例: 燃費(km/L)列とか
            # ここでは簡易的にやる
            elif op == ">=":
                try:
                    cell_num = float(cell_value_str.replace('～','').replace('km','').replace('日',''))
                    return cell_num >= float(val)
                except:
                    return False
            elif op == "<=":
                try:
                    cell_num = float(cell_value_str.replace('～','').replace('km','').replace('日',''))
                    return cell_num <= float(val)
                except:
                    return False
            elif op == "BETWEEN":
                # val = [v1,v2]
                try:
                    cell_num = float(cell_value_str.replace('～','').replace('km','').replace('日',''))
                    v1, v2 = val
                    return v1 <= cell_num <= v2
                except:
                    return False
            else:
                return False

    def match_all_conditions(row, conditions):
        # すべてAND判定 (複数conditions: "かつ")
        for c in conditions:
            if not row_matches_condition(row, c):
                return False
        return True

    conditions = parsed.get("conditions", [])
    df_filtered = df[df.apply(lambda r: match_all_conditions(r, conditions), axis=1)]

    # (6) グラフ用パラメータを取り出す
    output = parsed.get("output", {})
    chart_type = output.get("type", "bar_chart")
    group_by = output.get("group_by", "安全装備パッケージ")
    aggregate = output.get("aggregate", "count")  # count or percentage
    title = output.get("title", "フィルタ結果")

    # (7) 集計: group_by列ごとに件数orパーセント
    if len(df_filtered) == 0:
        return jsonify({
            "message": "条件に一致するデータがありませんでした。",
        })
    
    group_count = df_filtered.groupby(group_by).size().reset_index(name='count')
    if aggregate == "percentage":
        total_count = group_count['count'].sum()
        group_count['value'] = group_count['count'] / total_count * 100
    else:
        # count の場合
        group_count['value'] = group_count['count']

    # (8) グラフ描画
    plt.rcParams['font.family'] = 'IPAexGothic'  # 日本語フォント(環境に合わせて)
    fig, ax = plt.subplots(figsize=(6,4))
    
    # ブルー系グラデーション
    cmap = plt.get_cmap('Blues')  
    colors = [cmap(i/len(group_count)) for i in range(len(group_count))]
    
    ax.bar(group_count[group_by], group_count['value'], color=colors, edgecolor='black')
    ax.set_title(title)
    ax.set_xlabel(group_by)
    if aggregate == "percentage":
        ax.set_ylabel('割合(%)')
    else:
        ax.set_ylabel('件数')
    
    # 背景グリッド
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')

    # 画像をメモリに保存
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    graph_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return jsonify({
        "message": f"グラフを生成しました。(抽出件数: {len(df_filtered)})",
        "graph_base64": graph_base64
    })


# --------------------------------
# メイン
# --------------------------------
if __name__ == "__main__":
    # Render 上ではポートが固定になるので、$PORT を取得してlistenする
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
