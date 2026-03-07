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
# 設定
# =============================================================================

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# Groq モデル設定
# Qwen3.5 が Groq に追加された際はここを変更するだけでOK
GROQ_MODEL = "qwen/qwen3-32b"

# HuggingFace Embedding モデル（多言語対応・日本語OK）
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# RAG 設定
TOP_K         = 5
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200

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
あなたはVivliostyle（ビブリオスタイル）の公式ドキュメントに精通したアシスタントです。
【最重要ルール】
1. 回答は必ず提供されたコンテキスト（ドキュメント）の情報に基づいてください。
2. コンテキストに情報がない場合は、「提供されているドキュメントには該当する情報が見つかりませんでした」と正直に伝えてください。
3. 推測や補完で回答してはいけません。
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
        retrieved_docs = retriever.invoke(question)
        context = "\n\n---\n\n".join(
            f"【{doc.metadata.get('source','不明')} / {doc.metadata.get('file','')}】\n{doc.page_content}"
            for doc in retrieved_docs
        )
        answer = chain.invoke({"context": context, "question": question})
        return answer, retrieved_docs

    return ask


# =============================================================================
# 4. Gradio UI
# =============================================================================

def create_ui(ask_fn):
    def chat_fn(message: str, history: list):
        try:
            answer, docs = ask_fn(message, history)
            return answer
        except Exception as e:
            import traceback
            traceback.print_exc()
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
    )
    return demo


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