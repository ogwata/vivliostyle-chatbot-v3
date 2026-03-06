---
title: Vivliostyle Document Chatbot v3
emoji: 📚
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.12.0
app_file: app.py
pinned: false
---

# 📚 Vivliostyle ドキュメント チャットbot v3

Vivliostyle（ビブリオスタイル）の公式ドキュメントを学習させたRAGチャットbotです。

## 変更点（v2→v3）

| 項目 | v2 | v3 |
|------|----|----|
| LLM | Gemini 2.0 Flash | **Qwen3-32B (Groq)** |
| 推論速度 | 通常 | **~535 t/s（高速）** |
| Embedding | Gemini Embedding 001 | Gemini Embedding 001（継続） |

## 技術スタック

- **LLM:** Qwen3-32B via Groq API（`reasoning_effort: none` で対話モード）
- **Embedding:** Gemini Embedding 001（text-embedding-004）
- **Vector Store:** FAISS
- **Framework:** LangChain + Gradio

## 必要な Secrets（HF Spaces）

| キー | 用途 |
|------|------|
| `GROQ_API_KEY` | Groq API（LLM） |
| `GOOGLE_API_KEY` | Gemini Embedding |

## 対応ドキュメント

- Vivliostyle CLI / VFM / Themes / vivliostyle.js（README + CHANGELOG）
- FAQ / はじめてのVivliostyle / 歴史 / 団体概要（vivliostyle.org）
- docs2.vivliostyle.org の全ドキュメント

## モデル変更について

将来 Groq が Qwen3.5 などに対応した場合、`app.py` の `GROQ_MODEL` 定数を変更するだけで切り替え可能です。

```python
GROQ_MODEL = "qwen/qwen3-32b"  # ← ここを変更するだけ
```
