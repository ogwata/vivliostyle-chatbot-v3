#!/usr/bin/env python3
"""
groq-model-watch: Groq がホストするモデル一覧 (GET /openai/v1/models) を取得し、
使用中モデル以外の Qwen 系モデルが追加されたら追跡 Issue で通知する。
使用中モデルが一覧から消えた場合（提供終了の可能性）も警告する。

- 設定: .github/groq-model-watch.config.json
- 使用中モデル: app.py の `GROQ_MODEL = "..."` を自動取得。
- 通知: ラベル "groq-model-watch" の Issue を 1 件だけ維持し、毎回本文を更新する。

標準ライブラリのみで動作する（GitHub Actions の python で実行）。
"""
import json
import os
import re
import datetime
import urllib.request
import urllib.error

ROOT = os.getcwd()
CONFIG_PATH = os.path.join(ROOT, ".github", "groq-model-watch.config.json")
LABEL = "groq-model-watch"
ISSUE_TITLE = "🤖 Groq モデル提供状況のチェック (groq-model-watch)"
GROQ_MODELS_URL = "https://api.groq.com/openai/v1/models"


# ----------------------------- 使用中モデルの取得 -----------------------------
GROQ_MODEL_RE = re.compile(r'GROQ_MODEL\s*=\s*["\']([^"\']+)["\']')


def read_model_in_use(fallback):
    """app.py の GROQ_MODEL = "..." を読む。取れなければ fallback。"""
    path = os.path.join(ROOT, "app.py")
    try:
        with open(path, encoding="utf-8") as f:
            m = GROQ_MODEL_RE.search(f.read())
            if m:
                return m.group(1)
    except Exception as e:
        print("WARN: app.py read failed:", e)
    return fallback


# ----------------------------- Groq モデル一覧の取得 -----------------------------
def fetch_groq_models(api_key):
    """Groq の models 一覧を list[dict] で返す。失敗時 None。"""
    req = urllib.request.Request(GROQ_MODELS_URL)
    req.add_header("Authorization", "Bearer " + api_key)
    req.add_header("Content-Type", "application/json")
    req.add_header("User-Agent", "groq-model-watch")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read().decode("utf-8", "replace"))
    except Exception as e:
        print("ERROR: Groq models fetch failed:", e)
        return None
    return data.get("data", [])


# ----------------------------- 集計 -----------------------------
def fmt_date(created):
    try:
        return datetime.datetime.utcfromtimestamp(int(created)).strftime("%Y-%m-%d")
    except Exception:
        return "—"


def build_report(models, in_use, cfg):
    patterns = [p.lower() for p in cfg.get("watch_patterns", ["qwen"])]
    acknowledged = set(cfg.get("acknowledged_models", []))

    def matches(mid):
        low = mid.lower()
        return any(p in low for p in patterns)

    active = [m for m in models if m.get("active", True)]
    active_ids = {m["id"] for m in active}

    watched = sorted(
        (m for m in active if matches(m["id"])),
        key=lambda m: m.get("created", 0),
        reverse=True,  # 新しい順
    )
    candidates = [
        m for m in watched
        if m["id"] != in_use and m["id"] not in acknowledged
    ]

    return {
        "in_use": in_use,
        "in_use_active": in_use in active_ids,
        "candidates": candidates,
        "watched": watched,
        "total_active": len(active),
        "has_alert": (in_use not in active_ids) or bool(candidates),
    }


def model_row(m):
    return (
        "`%s`" % m["id"],
        m.get("owned_by", "—"),
        fmt_date(m.get("created")),
    )


def md_table(rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def build_issue_body(rep):
    now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    parts = ["_最終チェック: %s（定期自動更新）_\n" % now]
    parts.append("使用中モデル: `%s`\n" % rep["in_use"])

    if not rep["in_use_active"]:
        parts.append(
            "## ⚠️ 使用中モデルが一覧にありません\n\n"
            "`%s` が Groq のアクティブ一覧に見つかりませんでした。**提供終了/名称変更の可能性**があります。"
            "Space が停止する前に、`app.py` の `GROQ_MODEL` を下の一覧から選び直してください。\n"
            % rep["in_use"])

    if rep["candidates"]:
        parts.append("## 🔔 使用中以外の候補モデルあり\n")
        parts.append(
            "Groq に、使用中ではない Qwen 系モデルがあります。"
            "アップグレード候補（例: Qwen3.5 系）かどうか確認してください。"
            "切り替えは `app.py` の `GROQ_MODEL` を変更するだけです。\n")
        parts.append(md_table(
            [model_row(m) for m in rep["candidates"]],
            ["モデル ID", "提供元", "登録日"]))
    elif rep["in_use_active"]:
        parts.append("## ✅ 新しい候補はありません（使用中モデルは提供継続中）\n")

    parts.append("\n## 参考: 現在の監視対象モデル一覧（新しい順）\n")
    if rep["watched"]:
        parts.append(md_table(
            [model_row(m) for m in rep["watched"]],
            ["モデル ID", "提供元", "登録日"]))
    else:
        parts.append("_監視パターンに一致するモデルは見つかりませんでした。_")

    parts.append(
        "\n---\n"
        "- 使用中モデルは `app.py` の `GROQ_MODEL` から自動取得しています。\n"
        "- 監視対象は `.github/groq-model-watch.config.json` の `watch_patterns` で調整できます。\n"
        "- 通知不要なモデルは同設定の `acknowledged_models` に ID を足すと候補から外れます。\n"
        "- Groq のアクティブモデル総数: %d。\n"
        "- この Issue は groq-model-watch が定期的に本文を更新します"
        "（クローズしても新たな検知があれば再オープンされます）。" % rep["total_active"])
    return "\n".join(parts)


# ----------------------------- GitHub API -----------------------------
API = "https://api.github.com"


def gh(method, path, token, payload=None):
    url = API + path
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", "Bearer " + token)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("User-Agent", "groq-model-watch")
    with urllib.request.urlopen(req, timeout=30) as r:
        body = r.read().decode("utf-8", "replace")
        return json.loads(body) if body else {}


def ensure_label(repo, token):
    try:
        gh("POST", "/repos/%s/labels" % repo, token,
           {"name": LABEL, "color": "1d76db",
            "description": "Groq hosted model availability watcher"})
    except urllib.error.HTTPError as e:
        if e.code != 422:  # 422 = 既に存在
            print("WARN: label create:", e)


def find_issue(repo, token):
    issues = gh("GET", "/repos/%s/issues?state=all&labels=%s&per_page=20" % (repo, LABEL), token)
    for it in issues:
        if it.get("title") == ISSUE_TITLE and "pull_request" not in it:
            return it
    return None


def main():
    token = os.environ["GITHUB_TOKEN"]
    repo = os.environ["GH_REPO"]
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_key:
        raise SystemExit("ERROR: GROQ_API_KEY is not set")

    cfg = json.load(open(CONFIG_PATH, encoding="utf-8"))
    in_use = read_model_in_use(cfg.get("model_in_use_fallback", ""))

    models = fetch_groq_models(groq_key)
    if models is None:
        raise SystemExit("ERROR: could not fetch Groq models; aborting without touching the Issue")

    rep = build_report(models, in_use, cfg)
    body = build_issue_body(rep)
    print("in_use=%s active=%s candidates=%d watched=%d total=%d"
          % (rep["in_use"], rep["in_use_active"], len(rep["candidates"]),
             len(rep["watched"]), rep["total_active"]))

    ensure_label(repo, token)
    existing = find_issue(repo, token)

    if existing:
        num = existing["number"]
        state = "open" if rep["has_alert"] else existing.get("state", "open")
        gh("PATCH", "/repos/%s/issues/%d" % (repo, num), token,
           {"body": body, "state": state})
        print("updated issue #%d (state=%s)" % (num, state))
    else:
        created = gh("POST", "/repos/%s/issues" % repo, token,
                     {"title": ISSUE_TITLE, "body": body, "labels": [LABEL]})
        print("created issue #%s" % created.get("number"))


if __name__ == "__main__":
    main()
