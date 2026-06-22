#!/usr/bin/env python3
"""
dep-watch: watch-list のライブラリの PyPI 最新安定版をチェックし、更新があれば
追跡 Issue を作成/更新して通知する。標準ライブラリのみで動作する
（GitHub Actions の python で実行）。

- 設定: .github/dep-watch.config.json
- 現在版: requirements.txt の `pkg==x.y.z` ピンを自動取り込み（scan_requirements）。
- 通知: ラベル "dep-watch" の Issue を 1 件だけ維持し、毎回本文を更新する。

（kikoyu の dep-watch をベースに、ノートブック走査を requirements.txt 走査へ
置き換え、CUDA13 互換ヒント機能を除去したもの。）
"""
import json
import os
import re
import glob
import datetime
import urllib.request
import urllib.error

ROOT = os.getcwd()
CONFIG_PATH = os.path.join(ROOT, ".github", "dep-watch.config.json")
LABEL = "dep-watch"
ISSUE_TITLE = "📦 依存ライブラリの更新チェック (dep-watch)"
UA = {"User-Agent": "dep-watch-action"}


# ----------------------------- バージョン処理 -----------------------------
STABLE_RE = re.compile(r"^\d+(?:\.\d+)*$")  # 純粋な数値ドット区切りのみ＝安定版


def parse_ver(v):
    return tuple(int(x) for x in v.split("."))


def is_stable(v):
    return bool(STABLE_RE.match(v.strip()))


def cmp_ver(a, b):
    """a>b:1, a==b:0, a<b:-1 （長さ違いは0埋め）"""
    ta, tb = parse_ver(a), parse_ver(b)
    n = max(len(ta), len(tb))
    ta += (0,) * (n - len(ta))
    tb += (0,) * (n - len(tb))
    return (ta > tb) - (ta < tb)


def bump_level(cur, new):
    tc, tn = parse_ver(cur), parse_ver(new)
    tc += (0,) * (3 - len(tc))
    tn += (0,) * (3 - len(tn))
    if tn[0] != tc[0]:
        return "メジャー"
    if tn[1] != tc[1]:
        return "マイナー"
    return "パッチ"


def latest_stable_from_pypi(pkg):
    """PyPI の RSS から最新安定版を返す。失敗時は None。"""
    url = "https://pypi.org/rss/project/%s/releases.xml" % pkg.lower().replace("_", "-")
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=30) as r:
            xml = r.read().decode("utf-8", "replace")
    except Exception as e:
        print("WARN: %s fetch failed: %s" % (pkg, e))
        return None
    titles = re.findall(r"<title>([^<]+)</title>", xml)
    # 先頭は channel タイトルなので除外し、安定版のみ
    versions = [t.strip() for t in titles[1:] if is_stable(t)]
    if not versions:
        return None
    best = versions[0]
    for v in versions[1:]:
        if cmp_ver(v, best) > 0:
            best = v
    return best


# ----------------------------- requirements.txt 解析 -----------------------------
PIN_RE = re.compile(r"^\s*([A-Za-z0-9_.\-]+)\s*==\s*([0-9][0-9A-Za-z.\-]*)")


def scan_requirements_pins():
    """requirements*.txt 内の 'pkg==x.y.z' ピンを {pkg: ver} で返す。"""
    pins = {}
    for path in glob.glob(os.path.join(ROOT, "**", "requirements*.txt"), recursive=True):
        try:
            with open(path, encoding="utf-8") as f:
                lines = f.readlines()
        except Exception:
            continue
        for line in lines:
            line = line.split("#", 1)[0]  # 行コメント除去
            m = PIN_RE.match(line)
            if m and is_stable(m.group(2)):
                pins[m.group(1).lower().replace("_", "-")] = m.group(2)
    return pins


# ----------------------------- 集計 -----------------------------
def build_report():
    cfg = json.load(open(CONFIG_PATH, encoding="utf-8"))
    pkgs = cfg.get("packages", [])
    pins = scan_requirements_pins() if cfg.get("scan_requirements") else {}

    updates, info = [], []
    for p in pkgs:
        name = p["name"]
        key = name.lower().replace("_", "-")
        current = p.get("current")
        if pins.get(key):  # requirements.txt のピンを優先
            current = pins[key]
        latest = latest_stable_from_pypi(name)
        note = p.get("note", "")
        if latest is None:
            info.append((name, current or "—", "取得失敗", "", note))
            continue
        if current and is_stable(current):
            if cmp_ver(latest, current) > 0:
                updates.append((name, current, latest, bump_level(current, latest), note))
            else:
                info.append((name, current, latest, "最新", note))
        else:
            info.append((name, current or "—", latest, "参考", note))
    return updates, info


def md_table(rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def build_issue_body(updates, info):
    now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    parts = ["_最終チェック: %s（毎日自動更新）_\n" % now]
    if updates:
        parts.append("## 🔔 更新あり\n")
        parts.append(md_table(
            [(n, "`%s`" % c, "`%s`" % l, lv, note) for (n, c, l, lv, note) in updates],
            ["ライブラリ", "現在", "最新", "更新", "備考"]))
    else:
        parts.append("## ✅ watch-list に新しい更新はありません\n")
    if info:
        parts.append("\n## 参考 / 情報\n")
        parts.append(md_table(
            [(n, "`%s`" % c, "`%s`" % l, s, note) for (n, c, l, s, note) in info],
            ["ライブラリ", "現在", "最新", "状態", "備考"]))
    parts.append(
        "\n---\n"
        "- ピン留めしている版は `requirements.txt` の `==` 値から自動反映されます"
        "（未ピンのものは参考表示のみ）。\n"
        "- `gradio` を更新するときは、`requirements.txt` と README 先頭の `sdk_version` の"
        "両方を合わせてください（HF Space の起動設定）。\n"
        "- この Issue は dep-watch が毎日 本文を更新します"
        "（クローズしても新たな更新があれば再オープンされます）。")
    return "\n".join(parts)


# ----------------------------- GitHub API -----------------------------
API = "https://api.github.com"


def gh(method, path, token, payload=None):
    url = API + path
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", "Bearer " + token)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("User-Agent", "dep-watch-action")
    with urllib.request.urlopen(req, timeout=30) as r:
        body = r.read().decode("utf-8", "replace")
        return json.loads(body) if body else {}


def ensure_label(repo, token):
    try:
        gh("POST", "/repos/%s/labels" % repo, token,
           {"name": LABEL, "color": "0e8a16",
            "description": "dependency update watcher"})
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

    updates, info = build_report()
    body = build_issue_body(updates, info)
    print("updates:", len(updates), "info:", len(info))

    ensure_label(repo, token)
    existing = find_issue(repo, token)

    if existing:
        num = existing["number"]
        # 更新があり、かつクローズ済みなら再オープン
        state = "open" if updates else existing.get("state", "open")
        gh("PATCH", "/repos/%s/issues/%d" % (repo, num), token,
           {"body": body, "state": state})
        print("updated issue #%d (state=%s)" % (num, state))
    else:
        # 初回は、更新が無くても追跡用に1件作成しておく
        created = gh("POST", "/repos/%s/issues" % repo, token,
                     {"title": ISSUE_TITLE, "body": body, "labels": [LABEL]})
        print("created issue #%s" % created.get("number"))


if __name__ == "__main__":
    main()
