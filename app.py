"""
📚 Vivliostyle ドキュメント チャットbot v3
Groq (Qwen3-32B) + HuggingFace Embedding + LangChain + FAISS + Gradio
v3.1: Gemini Embedding を HuggingFace ローカルEmbedding に変更。
      GOOGLE_API_KEY 不要。
"""

import os
import re
import subprocess
import gradio as gr

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

# =============================================================================
# Monkey-patch: gradio_client の get_type が bool を受け取るとクラッシュする問題を修正
# See: https://github.com/gradio-app/gradio/issues/11084
# =============================================================================
import gradio_client.utils as _gc_utils

_original_get_type = _gc_utils.get_type

def _patched_get_type(schema):
    if isinstance(schema, bool):
        return "boolean"
    return _original_get_type(schema)

_gc_utils.get_type = _patched_get_type

# =============================================================================
# 設定
# =============================================================================

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "")

# Groq モデル設定
# Qwen3.5 が Groq に追加された際はここを変更するだけでOK
GROQ_MODEL = "qwen/qwen3-32b"

# HuggingFace Embedding モデル（多言語対応・日本語OK）
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# RAG 設定
TOP_K         = 8
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200

# 入力制限設定
MAX_INPUT_LENGTH    = 1000   # 質問の最大文字数
MAX_HISTORY_TURNS   = 20     # 保持する会話履歴の最大ターン数
RATE_LIMIT_SECONDS  = 5      # 同一セッションからの最小リクエスト間隔（秒）

# ログ設定
LOG_FILE            = "chat_log.jsonl"   # ログファイルパス（JSONL形式）
LOG_QUESTION_MAXLEN = 100                # ログに記録する質問文の最大文字数

# =============================================================================
# 1. ドキュメント収集・前処理
# =============================================================================

def preprocess_markdown(text: str) -> str:
    """Jekyll/Liquid 記法を除去してプレーンな Markdown にする"""
    text = re.sub(r'\A---\n.*?\n---\n', '', text, flags=re.DOTALL)
    text = re.sub(r'\{%-?\s*endcapture\s*-?%\}', '', text)
    text = re.sub(r'\{%-?\s*capture\s+\w+\s*-?%\}', '', text)
    text = re.sub(r'\{%.*?%\}', '', text)
    text = re.sub(r'\{\{.*?\}\}', '', text)
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def clone_repo(url: str, dest: str) -> bool:
    if os.path.exists(dest):
        print(f"  ✅ already exists: {dest}")
        return True
    result = subprocess.run(
        ["git", "clone", "--depth=1", url, dest],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  ❌ clone failed: {result.stderr[:200]}")
        return False
    print(f"  ✅ cloned: {dest}")
    return True


def load_markdown_files(directory: str, source_label: str) -> list[Document]:
    docs = []
    for root, _, files in os.walk(directory):
        for fname in files:
            if fname.endswith(".md"):
                path = os.path.join(root, fname)
                try:
                    with open(path, encoding="utf-8", errors="ignore") as f:
                        raw = f.read()
                    text = preprocess_markdown(raw)
                    if len(text) < 50:
                        continue
                    rel = os.path.relpath(path, directory)
                    docs.append(Document(
                        page_content=text,
                        metadata={"source": source_label, "file": rel}
                    ))
                except Exception as e:
                    print(f"  ⚠️ skip {path}: {e}")
    return docs


def collect_documents() -> list[Document]:
    print("📂 ドキュメントを収集中...")
    all_docs: list[Document] = []

    # --- vivliostyle.org ---
    if clone_repo("https://github.com/vivliostyle/vivliostyle.org", "/tmp/vivliostyle.org"):
        for path, label in [
            ("/tmp/vivliostyle.org/_docs/ja", "vivliostyle.org/ja"),
            ("/tmp/vivliostyle.org/_posts",   "vivliostyle.org/blog"),
        ]:
            if os.path.exists(path):
                docs = load_markdown_files(path, label)
                all_docs.extend(docs)
                print(f"  {label}: {len(docs)} files")

    # --- docs2.vivliostyle.org ---
    if clone_repo("https://github.com/vivliostyle/docs2.vivliostyle.org", "/tmp/docs2"):
        docs = load_markdown_files("/tmp/docs2", "docs2.vivliostyle.org")
        all_docs.extend(docs)
        print(f"  docs2.vivliostyle.org: {len(docs)} files")

    # --- 各プロダクトの README / CHANGELOG ---
    product_repos = {
        "vivliostyle-cli":    "https://github.com/vivliostyle/vivliostyle-cli",
        "vfm":                "https://github.com/vivliostyle/vfm",
        "vivliostyle-themes": "https://github.com/vivliostyle/themes",
        "vivliostyle-core":   "https://github.com/vivliostyle/vivliostyle.js",
    }
    for name, url in product_repos.items():
        dest = f"/tmp/{name}"
        if clone_repo(url, dest):
            for fname in ["README.md", "CHANGELOG.md"]:
                fpath = os.path.join(dest, fname)
                if os.path.exists(fpath):
                    with open(fpath, encoding="utf-8", errors="ignore") as f:
                        raw = f.read()
                    text = preprocess_markdown(raw)
                    if len(text) >= 50:
                        all_docs.append(Document(
                            page_content=text,
                            metadata={"source": name, "file": fname}
                        ))
                        print(f"  {name}/{fname}: loaded")

    print(f"\n📄 合計 {len(all_docs)} ファイルを収集")
    return all_docs


# =============================================================================
# 2. ベクトルストア構築 / 読み込み
# =============================================================================

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vectorstore(documents: list[Document]) -> FAISS:
    print("🔢 Embedding & FAISS インデックス構築中...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", "。", "、", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"  チャンク数: {len(chunks)}")

    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("faiss_index")
    print("  ✅ faiss_index を保存しました")
    return vectorstore


def load_vectorstore(index_dir: str = "faiss_index") -> FAISS:
    embeddings = get_embeddings()
    vs = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
    print("✅ faiss_index をロードしました")
    return vs


def get_vectorstore() -> FAISS:
    """起動時: インデックスがあれば再利用、なければ構築"""
    if os.path.exists("faiss_index/index.faiss"):
        print("📦 既存の faiss_index を使用")
        return load_vectorstore()
    docs = collect_documents()
    return build_vectorstore(docs)


# =============================================================================
# 3. RAG チェーン構築 (LLM = Groq / Qwen3-32B)
# =============================================================================

SYSTEM_PROMPT = """\
あなたはVivliostyle（ビブリオスタイル）と、その基盤となるCSS仕様・日本語組版（JLREQ）に精通したアシスタントです。

【回答の方針】
1. まず提供されたコンテキスト（ドキュメント）から回答に使える情報を探してください。
2. コンテキストに直接の回答がなくても、関連するCSS仕様（CSS Text、CSS Writing Modes、CSS Paged Media、CSS Fontsなど）やJLREQ（日本語組版処理の要件）に関する記述がコンテキスト内にあれば、それを活用して回答してください。
3. コンテキスト内の情報を組み合わせて回答できる場合は、積極的に回答してください。
4. コンテキストに関連する情報が全くない場合のみ、「提供されているドキュメントには該当する情報が見つかりませんでした」と伝えてください。

【回答スタイル】
- 日本語で丁寧に回答してください。
- 必要に応じてコードブロックや箇条書きを使い、わかりやすく整理してください。
- 回答の最後に「📖 出典: [ドキュメント名]」を記載してください。

【コンテキスト】
{context}
"""

def build_rag_chain(vectorstore: FAISS):
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K}
    )

    llm = ChatGroq(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0.1,
        reasoning_effort="none",
        streaming=False,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    chain = prompt | llm | StrOutputParser()

    def ask(question: str, history=None) -> tuple[str, list]:
        # メイン検索
        retrieved_docs = retriever.invoke(question)

        # 補強検索: CSS仕様・JLREQ関連のキーワードが含まれる場合、
        # 関連用語でも追加検索してコンテキストを広げる
        css_keywords = {
            "ライティングモード": "writing-mode CSS Writing Modes",
            "writing-mode": "writing-mode 縦書き 横書き",
            "縦書き": "writing-mode vertical-rl 縦組",
            "横書き": "writing-mode horizontal-tb 横組",
            "ページ": "CSS Paged Media @page",
            "@page": "CSS Paged Media ページ",
            "フォント": "CSS Fonts font-family",
            "ルビ": "CSS Ruby ruby JLREQ",
            "圏点": "text-emphasis 傍点",
            "禁則": "line-break word-break JLREQ 禁則処理",
            "行間": "line-height 行送り JLREQ",
            "段組": "CSS Multi-column column",
            "柱": "running header ページヘッダー",
            "ノンブル": "page counter ページ番号",
            "目次": "table of contents toc",
            "JLREQ": "日本語組版 JLREQ jlreq",
            "組版": "typesetting 組版 JLREQ",
        }

        supplemental_queries = []
        for keyword, expansion in css_keywords.items():
            if keyword.lower() in question.lower():
                supplemental_queries.append(expansion)

        # 重複排除しつつ補強ドキュメントを追加
        seen_contents = {doc.page_content for doc in retrieved_docs}
        for query in supplemental_queries[:2]:  # 最大2件の補強検索
            extra_docs = vectorstore.similarity_search(query, k=3)
            for doc in extra_docs:
                if doc.page_content not in seen_contents:
                    retrieved_docs.append(doc)
                    seen_contents.add(doc.page_content)

        context = "\n\n---\n\n".join(
            f"【{doc.metadata.get('source','不明')} / {doc.metadata.get('file','')}】\n{doc.page_content}"
            for doc in retrieved_docs
        )
        answer = chain.invoke({"context": context, "question": question})
        return answer, retrieved_docs

    return ask


# =============================================================================
# 3.5. リクエストログ
# =============================================================================

import json
import threading
from datetime import datetime, timezone, timedelta

_log_lock = threading.Lock()
_JST = timezone(timedelta(hours=9))


def log_request(
    question: str,
    response_time_ms: float,
    status: str = "ok",
    error: str = "",
    session_id: str = "",
):
    """リクエストログを JSONL ファイルに追記する"""
    entry = {
        "ts": datetime.now(_JST).strftime("%Y-%m-%d %H:%M:%S"),
        "session": session_id[:12],  # 先頭12文字に切り詰め
        "question": question[:LOG_QUESTION_MAXLEN],
        "response_ms": round(response_time_ms),
        "status": status,
    }
    if error:
        entry["error"] = error[:200]

    try:
        with _log_lock:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"⚠️ ログ書き込み失敗: {e}")


def get_log_summary() -> str:
    """ログファイルから直近の利用状況サマリーを生成する（管理用）"""
    if not os.path.exists(LOG_FILE):
        return "ログファイルが存在しません。"

    try:
        with open(LOG_FILE, encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        return f"ログ読み込みエラー: {e}"

    total = len(lines)
    errors = 0
    recent_questions: list[str] = []

    for line in lines[-50:]:  # 直近50件を解析
        try:
            entry = json.loads(line)
            if entry.get("status") != "ok":
                errors += 1
            recent_questions.append(
                f"  [{entry.get('ts', '?')}] {entry.get('question', '?')}"
            )
        except json.JSONDecodeError:
            continue

    summary = [
        f"📊 利用ログサマリー（{LOG_FILE}）",
        f"  総リクエスト数: {total}",
        f"  直近50件のエラー数: {errors}",
        f"",
        f"📝 直近の質問（最大50件）:",
    ]
    summary.extend(recent_questions[-20:])  # 表示は20件まで
    return "\n".join(summary)


# =============================================================================
# 4. Gradio UI
# =============================================================================

def create_ui(ask_fn):
    import time
    import threading

    # セッション単位の簡易レート制限（メモリ内）
    _last_request_time: dict[str, float] = {}
    _rate_lock = threading.Lock()

    def _check_rate_limit(session_id: str) -> bool:
        """レート制限チェック。制限内なら True、超過なら False"""
        now = time.time()
        with _rate_lock:
            last = _last_request_time.get(session_id, 0)
            if now - last < RATE_LIMIT_SECONDS:
                return False
            _last_request_time[session_id] = now
            # 古いエントリを掃除（1時間以上前）
            stale = [k for k, v in _last_request_time.items() if now - v > 3600]
            for k in stale:
                del _last_request_time[k]
            return True

    def _validate_input(message: str) -> str | None:
        """入力バリデーション。問題があればエラーメッセージを返す"""
        if not message or not message.strip():
            return "⚠️ 質問を入力してください。"
        if len(message) > MAX_INPUT_LENGTH:
            return f"⚠️ 質問が長すぎます（{len(message)}文字）。{MAX_INPUT_LENGTH}文字以内で入力してください。"
        return None

    def chat_fn(message: str, history: list, request: gr.Request):
        import time as _time

        # レート制限（リクエストのクライアント情報で識別）
        session_id = "default"
        if request:
            session_id = request.session_hash or request.client.host or "default"
        if not _check_rate_limit(session_id):
            log_request(message, 0, status="rate_limited", session_id=session_id)
            return f"⚠️ リクエストが多すぎます。{RATE_LIMIT_SECONDS}秒以上間隔を空けてください。"

        # 入力バリデーション
        error = _validate_input(message)
        if error:
            log_request(message, 0, status="invalid_input", session_id=session_id)
            return error

        # 入力をサニタイズ（前後の空白除去）
        message = message.strip()

        # 会話履歴を制限
        if history and len(history) > MAX_HISTORY_TURNS:
            history = history[-MAX_HISTORY_TURNS:]

        start = _time.time()
        try:
            answer, docs = ask_fn(message, history)
            elapsed = (_time.time() - start) * 1000
            log_request(message, elapsed, status="ok", session_id=session_id)
            return answer
        except Exception as e:
            elapsed = (_time.time() - start) * 1000
            import traceback
            traceback.print_exc()
            log_request(message, elapsed, status="error",
                        error=f"{type(e).__name__}: {str(e)}", session_id=session_id)
            return f"⚠️ エラーが発生しました: {type(e).__name__}: {str(e)}"

    TITLE = "📚 Vivliostyle ドキュメント アシスタント (v3: Groq / Qwen3-32B)"

    DESCRIPTION = """\
Vivliostyle（ビブリオスタイル）に関する質問にお答えします。
**対応ドキュメント:**
- Vivliostyle CLI / VFM / Themes / Viewer（ドキュメント・README・CHANGELOG）
- FAQ / はじめてのVivliostyle / 歴史 / 団体概要（vivliostyle.org）
**技術スタック:** LangChain + Groq (Qwen3-32B) + HuggingFace Embedding + FAISS + Gradio
---
📝 RAGで参照しているドキュメントの著作権は一般社団法人ビブリオスタイルに帰属します。
"""

    EXAMPLES = [
        "Vivliostyleとは何ですか？",
        "Vivliostyle CLIのインストール方法を教えてください",
        "VFMとGFMの違いは何ですか？",
        "一般社団法人ビブリオスタイルはいつ設立されましたか？",
        "CSSの@pageルールについて教えてください",
        "Vivliostyle CLIの最新バージョンと変更点を教えてください",
    ]

    demo = gr.ChatInterface(
        fn=chat_fn,
        title=TITLE,
        description=DESCRIPTION,
        examples=EXAMPLES,
        textbox=gr.Textbox(
            placeholder="Vivliostyleについて質問してください（最大1000文字）",
            max_lines=10,
        ),
    )

    # ログ閲覧UI（Blocksでラップ）
    with gr.Blocks(title="Vivliostyle ドキュメント アシスタント") as app:
        with gr.Tabs():
            with gr.TabItem("💬 チャット"):
                demo.render()
            with gr.TabItem("📊 利用ログ"):
                gr.Markdown("### 📊 利用状況ログ\n管理者パスワードを入力してください。")
                with gr.Row():
                    password_input = gr.Textbox(
                        label="管理者パスワード",
                        type="password",
                        scale=3,
                    )
                    auth_btn = gr.Button("🔓 認証してログ表示", scale=1)
                log_output = gr.Textbox(
                    label="ログサマリー",
                    lines=25,
                    interactive=False,
                )
                refresh_btn = gr.Button("🔄 ログを更新", interactive=False)

                def authenticate(password):
                    if not ADMIN_PASSWORD:
                        return "⚠️ ADMIN_PASSWORD が設定されていません。", gr.update(interactive=False)
                    if password == ADMIN_PASSWORD:
                        return get_log_summary(), gr.update(interactive=True)
                    return "❌ パスワードが違います。", gr.update(interactive=False)

                def refresh_with_auth(password):
                    if password == ADMIN_PASSWORD:
                        return get_log_summary()
                    return "❌ パスワードが違います。再度認証してください。"

                auth_btn.click(
                    fn=authenticate,
                    inputs=password_input,
                    outputs=[log_output, refresh_btn],
                )
                refresh_btn.click(
                    fn=refresh_with_auth,
                    inputs=password_input,
                    outputs=log_output,
                )

    return app


# =============================================================================
# 5. エントリーポイント
# =============================================================================

if __name__ == "__main__":
    if not GROQ_API_KEY:
        raise ValueError("環境変数 GROQ_API_KEY が設定されていません")

    print("🚀 Vivliostyle Document Chatbot v3 (Groq / Qwen3-32B) を起動中...")

    vectorstore = get_vectorstore()
    ask = build_rag_chain(vectorstore)

    print("\n🧪 動作テスト:")
    answer, _ = ask("Vivliostyleとは何ですか？")
    print(f"Q: Vivliostyleとは何ですか？\nA: {answer[:200]}...")

    demo = create_ui(ask)
    demo.launch()