#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
memory-refiner —— 记忆精炼与奖励引擎（支援未来版）

设计目标（2026-10-25）：
  1. 上下文要压缩时，主动汇报 + 存档 + 精炼出记忆节点
  2. 每次存档新建文件夹，全部保留，等未来技术一次读出
  3. 奖励函数：节点与前 3~4 次原文的"符合函数"越高，分越高
  4. 每 3~4 次压缩后做一次"机理精炼"：有选择地降权，而不是删除

三条原则：
  - 宪章层（人格/关系/铁律/医疗基线）永不压缩，只追加
  - 事件层可压，但每条必须带出处与置信度
  - 分数由**脚本算**，模型只负责写，不负责评

用法：
  python refiner.py score   --raw raw.md --node node.json
  python refiner.py compress --raw raw.md --title "今天的对话"
  python refiner.py refine
  python refiner.py report
"""

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
__version__ = "1.0.0"
CONFIG = ROOT / "config.json"          # 本地配置（gitignore），优先级高于环境变量之外的一切
DSH_CRED = Path(os.environ.get("USERPROFILE", os.path.expanduser("~"))) / ".dsh" / ".credentials.yaml"
ARCHIVE = ROOT / "archive"
STATE = ROOT / "state"
INDEX = STATE / "index.jsonl"
COUNTER = STATE / "counter.json"
REFINED = ROOT / "refined"

DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"
GLM_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
GLM_MODEL = "glm-5.3-flash"

# ─────────────────────────── 多模型接入 ───────────────────────────
# 全部走 OpenAI 兼容协议：换 provider 只换 base_url / key / model 三件事。
PROVIDERS = {
    "deepseek":   {"url": "https://api.deepseek.com/v1/chat/completions",
                   "model": "deepseek-chat", "env": "DEEPSEEK_API_KEY", "cfg": "deepseek_api_key"},
    "github":     {"url": "https://models.github.ai/inference/chat/completions",
                   "model": "openai/gpt-4.1-mini", "env": "GITHUB_TOKEN", "cfg": "github_token"},
    "openrouter": {"url": "https://openrouter.ai/api/v1/chat/completions",
                   "model": "deepseek/deepseek-chat-v3.1:free", "env": "OPENROUTER_API_KEY", "cfg": "openrouter_api_key"},
    "modelscope": {"url": "https://api-inference.modelscope.cn/v1/chat/completions",
                   "model": "Qwen/Qwen3-32B", "env": "MODELSCOPE_API_KEY", "cfg": "modelscope_api_key"},
    "siliconflow": {"url": "https://api.siliconflow.cn/v1/chat/completions",
                    "model": "Qwen/Qwen3-8B", "env": "SILICONFLOW_API_KEY", "cfg": "siliconflow_api_key"},
    "groq":       {"url": "https://api.groq.com/openai/v1/chat/completions",
                   "model": "llama-3.3-70b-versatile", "env": "GROQ_API_KEY", "cfg": "groq_api_key"},
    "cerebras":   {"url": "https://api.cerebras.ai/v1/chat/completions",
                   "model": "llama3.1-8b", "env": "CEREBRAS_API_KEY", "cfg": "cerebras_api_key"},
    "local":      {"url": "http://127.0.0.1:11434/v1/chat/completions",   # Ollama 默认
                   "model": "qwen2.5:7b", "env": "", "cfg": "local_model"},
}

LLM = {"name": "deepseek", "url": DEEPSEEK_URL, "model": DEEPSEEK_MODEL, "key": ""}


def resolve_provider(name: str = None) -> dict:
    """把 config.json / 环境变量 / 内置默认解析成一个可用的 provider。"""
    conf = load_config()
    name = (name or conf.get("provider") or "deepseek").strip()
    base = dict(PROVIDERS.get(name, PROVIDERS["deepseek"]))
    over = ((conf.get("providers") or {}).get(name) or {})
    base.update({k: v for k, v in over.items() if v})

    url = over.get("base_url") or conf.get("local_base_url") if name == "local" else over.get("url") or base["url"]
    if name == "local":
        url = (over.get("base_url") or conf.get("local_base_url") or "http://127.0.0.1:11434/v1").rstrip("/") + "/chat/completions"

    key = os.environ.get(base.get("env", "") or "", "").strip()
    if not key:
        key = str(conf.get(base.get("cfg", ""), "") or "").strip()
    if not key and name == "deepseek":
        key = deepseek_key()

    return {"name": name, "url": url, "model": over.get("model") or base["model"], "key": key}


def use_provider(name: str = None) -> dict:
    global LLM
    LLM = resolve_provider(name)
    return LLM

REFINE_EVERY = 4          # 每 4 次压缩做一次机理精炼
EFF_FULL_AT = 0.70        # 压缩率到 70% 即满分

CONTEXT_LIMIT = 128_000   # DeepSeek 上下文上限
SAFETY_RATIO = 0.80       # 安全水位：到 80% 就压缩，剩下 20% 留给回复和打分
WATERMARK = int(CONTEXT_LIMIT * SAFETY_RATIO)


# ─────────────────────────── 基础工具 ───────────────────────────

def now() -> datetime:
    return datetime.now().astimezone()


def stamp(d: datetime = None) -> str:
    d = d or now()
    return d.strftime("%Y%m%d-%H%M%S")


def read(p: Path) -> str:
    return Path(p).read_text(encoding="utf-8")


def count_tokens(text: str) -> int:
    """本地估算 token 数。

    优先用 tiktoken（装了就用，最准）；没装走校准过的启发式：
    中文 ≈ 1.6 字/token，英文 ≈ 1.3 token/词，数字 ≈ 0.8，标点 ≈ 0.35。
    误差 ±15% 以内，做"安全水位"判断够用——水位本来就留了 20% 余量。
    """
    try:
        import tiktoken  # type: ignore
        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        pass
    cjk = len(re.findall(r"[\u4e00-\u9fa5]", text))
    words = re.findall(r"[A-Za-z]+", text)
    digits = re.findall(r"\d+", text)
    covered = cjk + sum(len(w) for w in words) + sum(len(d) for d in digits)
    others = max(0, len(text) - covered)
    return int(cjk / 1.6 + len(words) * 1.3 + len(digits) * 0.8 + others * 0.35)


def token_report(text: str) -> dict:
    n = count_tokens(text)
    return {
        "chars": len(text),
        "tokens": n,
        "limit": CONTEXT_LIMIT,
        "watermark": WATERMARK,
        "ratio": round(n / CONTEXT_LIMIT, 4) if CONTEXT_LIMIT else 0.0,
        "over_watermark": n >= WATERMARK,
    }


def split_turns(text: str) -> list:
    """把对话切成"轮"：空行、日期行、角色行都算分隔。"""
    blocks = re.split(
        r"\n\s*\n|^(?=\d{4}-\d{2}-\d{2})|^(?=(?:用户|助手|user|assistant)[:：])",
        text, flags=re.M)
    return [b.strip() for b in blocks if b and b.strip()]


def split_window(text: str, keep_turns: int = 4) -> tuple:
    """滑动窗口：最近 N 轮保留原文，更早的进压缩池。"""
    turns = split_turns(text)
    if keep_turns <= 0 or len(turns) <= keep_turns:
        return text, ""
    return "\n\n".join(turns[:-keep_turns]), "\n\n".join(turns[-keep_turns:])


def wsl_config() -> str:
    """从 WSL 读 Hermes 配置——**路径必须由使用者提供**，不猜、不写死。"""
    conf = load_config()
    path = os.environ.get("HERMES_CONFIG", "").strip() or str(conf.get("hermes_config", "")).strip()
    distro = os.environ.get("WSL_DISTRO", "").strip() or str(conf.get("wsl_distro", "")).strip()
    if not path:
        return ""
    try:
        cmd = ["wsl"] + (["-d", distro] if distro else []) + ["--", "cat", path]
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=30)
        return r.stdout if r.returncode == 0 else ""
    except Exception:
        return ""


def load_config() -> dict:
    if CONFIG.exists():
        try:
            return json.loads(read(CONFIG))
        except Exception:
            return {}
    return {}


def dsh_keys() -> dict:
    """从 DSH 的凭证文件里读 refs 段的明文 key（本机免配置）。"""
    if not DSH_CRED.exists():
        return {}
    out, in_refs = {}, False
    for line in read(DSH_CRED).splitlines():
        if re.match(r"^refs:\s*$", line):
            in_refs = True
            continue
        if in_refs:
            m = re.match(r"^\s+([A-Z0-9_]+):\s*(\S+)\s*$", line)
            if m:
                out[m.group(1)] = m.group(2)
    return out


def resolve_key(*names, env=None, cfg=None) -> tuple:
    """按 环境变量 → config.json → DSH 凭证 → WSL 配置 的顺序找 key。

    返回 (key, 来源说明)。找不到返回 ("", "")。
    """
    for n in names:
        if env:
            v = os.environ.get(n, "").strip()
            if v:
                return v, f"env:{n}"
    conf = load_config()
    for ck in (cfg or []):
        v = str(conf.get(ck, "")).strip()
        if v:
            return v, f"config.json:{ck}"
    dk = dsh_keys()
    for n in names:
        v = dk.get(n, "").strip()
        if v:
            return v, f"dsh:.credentials.yaml:{n}"
    return "", ""


def deepseek_key() -> str:
    v, _ = resolve_key("DEEPSEEK_API_KEY", cfg=["deepseek_api_key"])
    if v:
        return v
    m = re.search(r"api_key:\s*(sk-[A-Za-z0-9]+)", wsl_config())
    return m.group(1) if m else ""


def glm_key() -> str:
    v, _ = resolve_key("ZHIPU_API_KEY", "GLM_API_KEY", cfg=["zhipu_api_key", "glm_api_key"])
    if v:
        return v
    cfg = wsl_config()
    ks = re.findall(r"api_key:\s*([0-9a-f]{20,}\.[A-Za-z0-9]+)", cfg)
    return ks[0] if ks else ""


# ─────────────────────────── 特征抽取（脚本侧） ───────────────────────────

NUM_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:%|元|块|次|天|小时|分钟|岁|kg|斤|米|万|亿|条|个|轮|层|号|月|日|年)?")
DATE_RE = re.compile(r"\d{4}[-/年]\d{1,2}[-/月]\d{1,2}日?|\d{1,2}月\d{1,2}日|\d{1,2}点\d{1,2}分")


def features(text: str) -> set:
    """抽"关键事实符号"：日期、数字、引号内短语、拉丁词。

    注意：不抽泛化的中文专名——4~12 字滑动窗口噪音太大，
    会把召回的盘子撑大、把真实召回率稀释掉。
    """
    feats = set()
    for m in DATE_RE.finditer(text):
        feats.add(("date", m.group(0)))
    for m in NUM_RE.finditer(text):
        s = m.group(0).strip()
        if len(s) >= 2:
            feats.add(("num", s))
    for m in re.finditer(r"[「『\"“]([^」』\"”]{2,20})[」』\"”]", text):
        feats.add(("quote", m.group(1)))
    for m in re.finditer(r"[A-Za-z][A-Za-z0-9_\-]{2,}", text):
        feats.add(("latin", m.group(0).lower()))
    return feats


STOP_TERMS = set("""的了在是和有就不都而及与着或一个上也很你他她它我们你们他们这那
什么怎么因为所以但是然后现在今天昨天明天上午下午晚上时候可以没有还是就是自己
这个那个如果已经需要不过而且或者之后之前""".split())


BAD_EDGE = set("的了是在和有就都而那这不我你他她它个上中下里把被给让对从向以为")


def _chinese_runs(text: str) -> list:
    """按非汉字切开，得到一段段纯汉字（边界信息不能丢）。"""
    runs, cur = [], []
    for ch in text:
        if "\u4e00" <= ch <= "\u9fa5":
            cur.append(ch)
        else:
            if cur:
                runs.append("".join(cur))
                cur = []
    if cur:
        runs.append("".join(cur))
    return runs


def _entropy(counter: dict) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counter.values())


def _cohesion(w: str, counts: dict) -> float:
    """内部凝固度：取最差切分点的 PMI。碎片在这里一定会掉分。"""
    if not w or w not in counts:
        return 0.0
    total = counts["__total__"]
    pw = counts[w] / total
    best = float("inf")
    seen = 0
    for i in range(1, len(w)):
        left, right = w[:i], w[i:]
        if left not in counts or right not in counts:
            continue          # 切分点没见过，跳过而不是判死刑
        pl, pr = counts[left] / total, counts[right] / total
        if pl and pr:
            best = min(best, math.log2(pw / (pl * pr)))
            seen += 1
    return 0.0 if seen == 0 or best == float("inf") else best


def auto_terms(text: str, top: int = 12, min_chars: int = 3000) -> list:
    """无监督新词发现：凝固度（PMI）× 自由度（左右熵）。

    比"滑窗+频次"强的地方：碎片（"供应商那""还有一"）右邻字几乎固定，
    左右熵极低，会被直接淘汰；真正的词右邻字多样，熵高。

    **语料小于 min_chars 时直接返回空**：几千字的样本上，任何无监督分词都会
    产出跨词边界的碎片，与其污染召回，不如不用——交给实体覆盖和探针问答。
    这是刻意的取舍，不是能力缺失。
    """
    if len(text) < min_chars:
        return []
    runs = _chinese_runs(text)
    if not runs:
        return []

    counts = {"__total__": 0}
    left_ctx, right_ctx = {}, {}
    BOS = "▁"          # 句界标记：让"句首/句尾"也成为一个邻字，而不是缺失
    for run in runs:
        for n in (1, 2, 3, 4, 5):
            for i in range(len(run) - n + 1):
                w = run[i:i + n]
                counts[w] = counts.get(w, 0) + 1
                counts["__total__"] += 1
                left_ctx.setdefault(w, {})
                right_ctx.setdefault(w, {})
                left_ctx[w][run[i - 1] if i > 0 else BOS] = \
                    left_ctx[w].get(run[i - 1] if i > 0 else BOS, 0) + 1
                nxt = run[i + n] if i + n < len(run) else BOS
                right_ctx[w][nxt] = right_ctx[w].get(nxt, 0) + 1

    scored = []
    for w, c in counts.items():
        if w == "__total__" or c < 2 or not (2 <= len(w) <= 5):
            continue
        if w[0] in BAD_EDGE or w[-1] in BAD_EDGE:
            continue
        coh = _cohesion(w, counts)
        if coh < 1.5:
            continue
        # 自由度：邻字越杂越像词。
        # 特例：一侧只有句界（"永远出现在句首"），这一侧没有信息，就看另一侧，
        # 不要因为它的熵是 0 就把整个词判死。
        hl = _entropy(left_ctx.get(w, {}))
        hr = _entropy(right_ctx.get(w, {}))
        l_only_bos = set(left_ctx.get(w, {}).keys()) == {BOS}
        r_only_bos = set(right_ctx.get(w, {}).keys()) == {BOS}
        if l_only_bos and not r_only_bos:
            free = hr
        elif r_only_bos and not l_only_bos:
            free = hl
        else:
            free = min(hl, hr)
        if free <= 0.0:
            continue          # 邻字完全固定 = 跨词边界的碎片
        scored.append({"w": w, "c": c, "coh": coh, "free": free, "score": coh * (1.0 + free)})

    # 剪枝：互相包含时，频次高的优先；频次相同则短的优先（长的多半是跨词边界的碎片）
    scored.sort(key=lambda x: (-x["c"], -x["score"]))
    kept = []
    for cand in scored:
        drop = False
        for k in kept:
            a, b = cand["w"], k["w"]
            if a in b or b in a:
                drop = True
                break
        if not drop:
            kept.append(cand)

    kept.sort(key=lambda x: -x["score"])
    return [k["w"] for k in kept[:top]]


def salient_entities(text: str) -> set:
    """真正"改不动"的东西：日期、带单位的数量、引语、拉丁词。"""
    t = normalize(text)
    ents = set()
    ents |= set(re.findall(r"\d{1,2}月\d{1,2}日", t))
    ents |= set(re.findall(r"\d+(?:\.\d+)?(?:%|元|块|次|天|小时|分钟|岁|米|万|斤|条|个|轮|层|件|台)", t))
    ents |= {m.group(1) for m in re.finditer(r"[「『\"“]([^」』\"”]{2,20})[」』\"”]", text)}
    ents |= {m.group(0).lower() for m in re.finditer(r"[A-Za-z][A-Za-z0-9_\-]{2,}", text)}
    return {e for e in ents if len(e) >= 2}


def entity_recall(raw: str, node_text: str) -> tuple:
    """实体覆盖：日期/数量/引语/拉丁词这些"改不动"的东西，留下了几个。"""
    ents = salient_entities(raw)
    if not ents:
        return 1.0, []
    node_norm = normalize(node_text)
    miss = sorted(e for e in ents if normalize(e) not in node_norm)
    return (len(ents) - len(miss)) / len(ents), miss[:5]


def term_covered(t: str, node_text: str) -> bool:
    """滑窗抽出来的词常带边界碎片（如"供应商那"），±1 字也算命中。"""
    if t in node_text:
        return True
    if len(t) > 2 and (t[1:] in node_text or t[:-1] in node_text):
        return True
    return False


def hint_recall(raw: str, node_text: str) -> tuple:
    """主题词覆盖：原文自动抽出的主题词，节点里还剩几个。

    自动抽词：通用，不依赖任何领域词表。
    """
    terms = auto_terms(raw, top=12)
    if not terms:
        return None, []          # 没有可靠主题词时，调用方必须换权重
    missing = [t for t in terms if not term_covered(t, node_text)]
    return 1 - len(missing) / len(terms), missing


def bigrams(s: str) -> set:
    s = re.sub(r"\s+", "", s)
    return {s[i:i + 2] for i in range(len(s) - 1)}


CN_DIGIT = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
            "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}

SYNONYMS = [
    ("推迟", "延后"), ("延迟", "延后"), ("延期", "延后"), ("改期", "延后"),
    ("交货", "交付"), ("到货", "交付"),
    # 只合同向词，绝不把反义词并到一起：
    ("涨价", "提价"), ("加价", "提价"), ("上调", "提价"),
    ("减价", "降价"), ("下调", "降价"),
    ("完成", "结束"), ("搞定", "结束"),
    ("号", "日"),
]


def _cn2num(s: str) -> str:
    """只在数字语境里转换（后面跟量词），避免把"统一""一次"改坏。"""
    def conv(t: str) -> str:
        if "百" in t:
            head, _, tail = t.partition("百")
            h = CN_DIGIT.get(head, 1) if head else 1
            rest = conv(tail) if tail else "0"
            return str(h * 100 + int(rest))
        if "十" in t:
            head, _, tail = t.partition("十")
            tens = CN_DIGIT.get(head, 1) if head else 1
            ones = CN_DIGIT.get(tail, 0) if tail else 0
            return str(tens * 10 + ones)
        return str(CN_DIGIT.get(t, t))

    return re.sub(r"[零一二两三四五六七八九十百]+(?=[号日月年个次条件天人元块%])", lambda m: conv(m.group(0)), s)


def _canon_dates(s: str) -> str:
    """把 2026-03-02 / 2026/3/2 / 3-02 / 3月2日 统一成 3月2日。"""
    def repl(m):
        y, mo, d = m.group(1), m.group(2), m.group(3)
        if y and len(y) == 4:
            mo, d = m.group(2), m.group(3)
        return f"{int(mo)}月{int(d)}日"

    s = re.sub(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", repl, s)
    s = re.sub(r"(?<!\d)(\d{1,2})[-/](\d{1,2})(?!\d)", lambda m: f"{int(m.group(1))}月{int(m.group(2))}日", s)
    s = re.sub(r"(\d{1,2})月(\d{1,2})日", lambda m: f"{int(m.group(1))}月{int(m.group(2))}日", s)
    return s


def normalize(s: str) -> str:
    """把"说法不同、意思一样"的文本拉齐：全角、日期格式、中文数字、同义词。"""
    s = unicodedata.normalize("NFKC", s)
    s = _canon_dates(s)
    s = _cn2num(s)
    for a, b in SYNONYMS:
        if a in s:
            s = s.replace(a, b)
    return re.sub(r"\s+", "", s)


def support_ratio(clause: str, raw: str) -> float:
    """一个分句在原文里的支撑度。

    三个口径取最大：
      a) 归一化后的字符二元组覆盖率（治同义改写：18号 vs 18日、推迟 vs 延后）
      b) 日期/数字/引用/拉丁词命中率（强支撑）
      c) 数字集合匹配（"18" 出现在两边就算命中）
    """
    c, r = normalize(clause), normalize(raw)
    b = bigrams(c)
    if not b:
        return 0.0
    gram = len(b & bigrams(r)) / len(b)

    sym = {f"{k}:{v}" for k, v in features(c) if k in ("date", "num", "quote", "latin")}
    if sym:
        raw_sym = {f"{k}:{v}" for k, v in features(r) if k in ("date", "num", "quote", "latin")}
        hit = len(sym & raw_sym) / len(sym)
        if hit < 1.0:
            # 只要有一个日期/数字在原文里找不到，就是编的，不许靠字面重叠混过去
            return min(gram, hit)
        return max(gram, hit)

    # 数字兜底：归一化后逐个数字串比对
    nums_c = set(re.findall(r"\d+", c))
    if nums_c:
        nums_r = set(re.findall(r"\d+", r))
        return max(gram, len(nums_c & nums_r) / len(nums_c))
    return gram


# ─────────────────────────── 评分（符合函数） ───────────────────────────

def node_claims(node: dict) -> str:
    """只取"声明"部分（事实 + 人格锚点），标题和摘要本来就是概括，不算声明。"""
    parts = [f["text"] for f in node.get("facts", []) if isinstance(f, dict) and f.get("text")]
    parts += [a for a in node.get("persona_anchors", []) if isinstance(a, str)]
    return "\n".join(parts)


def score(raw: str, node_text: str, raw_len: int = None, probe_score: float = None,
          claim_text: str = None) -> dict:
    raw_f = features(raw)
    node_f = features(node_text)

    raw_names = {f for f in raw_f}
    node_names = {f for f in node_f}

    # 召回三口径互相兜底；抽不出可靠主题词时，权重全部让给实体与符号
    symbol_recall = len(raw_names & node_names) / len(raw_names) if raw_names else 1.0
    kr, missed = hint_recall(raw, node_text)
    er, missing_ent = entity_recall(raw, node_text)
    if kr is None:
        recall = 0.6 * er + 0.4 * symbol_recall
        term_mode = "off"
    else:
        recall = 0.5 * kr + 0.3 * er + 0.2 * symbol_recall
        term_mode = "on"

    # 精确：节点里的每一条分句，能不能在原文找到支撑
    claims = claim_text if claim_text is not None else node_text
    clauses = [c.strip() for c in re.split(r"[。；;！!？?\n]", claims) if len(c.strip()) >= 8]
    supported, unsupported = 0, []
    for c in clauses:
        if support_ratio(c, raw) >= 0.50:
            supported += 1
        else:
            unsupported.append(c[:40])
    precision = supported / len(clauses) if clauses else 0.0

    # 效率：压得越狠越好（但由 precision/recall 兜底，防止删过头）
    rl = raw_len or len(raw)
    nl = len(node_text)
    ratio = max(0.0, 1 - nl / rl) if rl else 0.0
    eff = min(1.0, ratio / EFF_FULL_AT)   # 压到 70% 即满分

    # 结构性问题：以前是 recall/precision/eff 三项加权相加，导致"照抄原文"拿 80 分。
    # 现在用 F1 把"丢内容"和"编内容"绑在一起，"压得短"只做乘数——压得再狠，
    # 内容丢了 F1 就掉，赚不回来。
    f1 = (2 * recall * precision / (recall + precision)) if (recall + precision) > 0 else 0.0
    total = round(100 * f1 * (0.5 + 0.5 * eff), 1)
    out = {
        "score": total,
        "recall": round(recall, 3),
        "precision": round(precision, 3),
        "efficiency": round(eff, 3),
        "compression_ratio": round(ratio, 3),
        "term_mode": term_mode,
        "clauses": len(clauses),
        "unsupported": unsupported[:5],
        "missing_count": len(raw_names - node_names),
        "missed_key": missed[:3],
    }
    if probe_score is not None:
        out["probe"] = round(probe_score, 3)
        out["score_with_probe"] = round(100 * (0.6 * total / 100 + 0.4 * probe_score), 1)
    return out


# ─────────────────────────── 模型调用 ───────────────────────────

def chat(url: str, key: str, model: str, system: str, user: str,
         max_tokens: int = 1200, temperature: float = 0.2) -> str:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST", headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
    })
    with urllib.request.urlopen(req, timeout=180) as r:
        data = json.loads(r.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


COMPRESS_SYS = """你在做记忆精炼。只输出 JSON，不要解释。

规则：
1. 人格锚点、关系、医疗基线、铁律 = 宪章层，原样保留，不许改写。
2. 事件类事实要保留：谁、何时、做了什么、结果如何。
3. 每条事实必须能指回原文，不许编。
4. 不确定的标 C。
5. 硬约束：facts 最多 8 条，每条 ≤ 40 字；summary ≤ 80 字；tags/anchors 各 ≤ 5 条。
6. 节点总长必须明显短于原文，否则这条记忆没有价值。
5. 只输出下面这个结构：

{
  "node_title": "一句话标题",
  "summary": "80字以内的整体概述",
  "facts": [{"text": "一条事实", "kind": "fact|decision|emotion|medical|task", "confidence": "S|A|B|C"}],
  "persona_anchors": ["需要原样保留的锚点"],
  "open_threads": ["还没做完的事"]
}"""

JUDGE_SYS = """你是记忆比对的裁判。只输出 JSON，不要打分。

给你【原文】和【节点】，找出：
1. 原文有、节点没有的关键事实
2. 节点有、原文找不到依据的说法
3. 与常识/时间线矛盾的地方

输出：
{"missed_facts": [], "unsupported_claims": [], "contradictions": []}"""

PROBE_Q_SYS = """从【原文】里出 {k} 道可核对的问题，覆盖五类各至少一道：
事实（谁做了什么）、时间（何时）、数量（多少）、更新（最后改成了什么）、未完成（还欠什么）。

每道题必须同时给出【标准答案】：从原文里抠出来的、最短最准的那个答案。
**更新类题目的标准答案必须是最终值**（原文里最后生效的结果），不是中间值或旧值。
**数量类必须给最终数量。**

只输出 JSON：
{"items": [{"q": "问题", "gold": "标准答案", "type": "fact|time|number|update|open"}]}"""

PROBE_A_SYS = """只根据【节点】回答下面每一道题。不许猜、不许补。
节点里没有的，就答「节点未记录」。
只输出 JSON：{"answers": [{"q": "问题", "a": "答案"}]}"""


def answer_matches(got: str, gold: str) -> bool:
    """答案对错，而不是"这些字在原文里出现过"。

    - 含数字/日期：标准答案里的数字必须全都出现，且不能多出别的日期（多出=拿了旧值）
    - 纯文字：互为子串，或二元组覆盖率 ≥0.6
    """
    g, t = normalize(got or ""), normalize(gold or "")
    if not g or "未记录" in g or "不知道" in g:
        return False
    gold_nums = set(re.findall(r"\d+", t))
    if gold_nums:
        got_nums = set(re.findall(r"\d+", g))
        if not gold_nums <= got_nums:
            return False
        extra = got_nums - gold_nums
        if extra and re.search(r"\d{1,2}月\d{1,2}日", t):
            return False          # 日期题多给了别的日期 = 拿错版本
        return True
    if t and (t in g or g in t):
        return True
    b = bigrams(t)
    return bool(b) and len(bigrams(g) & b) / len(b) >= 0.6


def probe(raw: str, node_text: str, key: str, k: int = 6) -> dict:
    """探针问答：出题时连着标准答案一起出，判分比对答案，不看"字有没有出现过"。"""
    try:
        data = chat_json(LLM["url"], key or LLM["key"], LLM["model"], PROBE_Q_SYS.replace("{k}", str(k)),
                         f"【原文】\n{raw[:8000]}\n\n出 {k} 道题，连标准答案。", max_tokens=900, temperature=0.2)
    except Exception as e:
        return {"probe": None, "qa": [], "error": f"出题失败：{type(e).__name__}"}

    items = [x for x in data.get("items", []) if isinstance(x, dict) and x.get("q")]
    if not items:
        return {"probe": None, "qa": [], "error": "没出题"}
    items = items[:k]

    try:
        ans = chat_json(LLM["url"], key or LLM["key"], LLM["model"], PROBE_A_SYS,
                        f"【节点】\n{node_text[:3000]}\n\n【题目】\n" +
                        "\n".join(f"{i+1}. {x['q']}" for i, x in enumerate(items)) +
                        "\n\n逐题作答，输出 JSON。", max_tokens=900, temperature=0.0)
    except Exception as e:
        return {"probe": None, "qa": [], "error": f"作答失败：{type(e).__name__}"}

    answers = [a for a in ans.get("answers", []) if isinstance(a, dict)]

    ok = 0
    qa = []
    for i, item in enumerate(items):
        a = ""
        for cand in answers:
            if cand.get("q", "").strip()[:8] == item["q"].strip()[:8]:
                a = cand.get("a", "")
                break
        if not a and i < len(answers):
            a = answers[i].get("a", "")
        hit = answer_matches(a, item.get("gold", ""))
        ok += 1 if hit else 0
        qa.append({"q": item["q"], "gold": item.get("gold", ""), "a": a,
                   "type": item.get("type", ""), "hit": hit})
    return {"probe": ok / len(items) if items else 0.0, "qa": qa}


def repair_json(s: str) -> str:
    """把模型吐出来的垃圾 JSON 拉回规范形态（不依赖任何库）。

    处理：代码围栏、前后废话、尾逗号、单引号、无引号键、Python 的 True/False/None。
    """
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*|```\s*$", "", s, flags=re.M).strip()
    start = min([i for i in (s.find("{"), s.find("[")) if i >= 0], default=-1)
    if start > 0:
        s = s[start:]
    end = max(s.rfind("}"), s.rfind("]"))
    if end >= 0:
        s = s[:end + 1]
    s = re.sub(r",\s*([}\]])", r"\1", s)             # 尾逗号
    s = re.sub(r"'([^'\\]*)'\s*:", r'"\1":', s)      # 单引号键
    s = re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:", r'\1"\2":', s)  # 无引号键
    s = s.replace("True", "true").replace("False", "false").replace("None", "null")
    return s


def _balanced_json(s: str) -> str:
    """从字符串里扫出第一个括号配平的 JSON 片段（引号与转义都算在内）。"""
    start = -1
    for i, ch in enumerate(s):
        if ch in "{[":
            start = i
            break
    if start < 0:
        return ""
    depth, in_str, esc = 0, False, False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    return s[start:]


def extract_json(s: str) -> dict:
    """解析 JSON：标准 → 括号配平 → 修复器 → 可选 json_repair。

    注意：以前用贪婪的 r"\\{.*\\}"，模型在 JSON 后面再补一句带 {} 的话就崩。
    """
    s = s.strip()
    for candidate in (s, _balanced_json(s), repair_json(s), _balanced_json(repair_json(s))):
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    try:
        import json_repair  # type: ignore
        out = json_repair.loads(s)
        if isinstance(out, dict):
            return out
    except Exception:
        pass
    raise ValueError("模型没返回可用 JSON")


def chat_json(url: str, key: str, model: str, system: str, user: str,
              max_tokens: int = 1200, temperature: float = 0.2, retries: int = 1) -> dict:
    """要 JSON 的调用：解析失败就"二次打脸"——把错误和原输出一起丢回去重写。"""
    raw = chat(url, key, model, system, user, max_tokens=max_tokens, temperature=temperature)
    try:
        return extract_json(raw)
    except Exception as e:
        if retries <= 0:
            raise
        fix_user = (
            f"你刚才输出的不是合法 JSON。错误是：{e}\n"
            f"你刚才的原输出：\n{raw[:2000]}\n\n"
            "请立刻把它修成合法 JSON，只输出 JSON，不要任何解释。"
        )
        fixed = chat(url, key, model, system, fix_user, max_tokens=max_tokens, temperature=0.0)
        return extract_json(fixed)


# ─────────────────────────── 归档 ───────────────────────────

def ensure_dirs():
    for d in (ARCHIVE, STATE, REFINED):
        d.mkdir(parents=True, exist_ok=True)


def bump_counter() -> int:
    n = 0
    if COUNTER.exists():
        try:
            n = int(json.loads(read(COUNTER)).get("n", 0))
        except Exception:
            n = 0
    n += 1
    COUNTER.write_text(json.dumps({"n": n, "at": stamp()}, ensure_ascii=False), encoding="utf-8")
    return n


def append_index(row: dict):
    with INDEX.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_index() -> list:
    if not INDEX.exists():
        return []
    out = []
    for line in read(INDEX).splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def node_to_text(node: dict) -> str:
    parts = [
        node.get("node_title", ""),
        node.get("summary", ""),
    ]
    parts += [f["text"] for f in node.get("facts", []) if isinstance(f, dict)]
    parts += node.get("persona_anchors", [])
    parts += node.get("open_threads", [])
    return "\n".join(p for p in parts if p)


# ─────────────────────────── 事实库与版本链 ───────────────────────────
# 设计取自 graphiti 的双时间 + mem0 的 ADD/UPDATE/DELETE/NOOP：
#   - 事实永不删除，只改 status
#   - 被覆盖的旧事实保留 valid_from / valid_to / superseded_by
#   - 冲突判定：先做确定性相似度筛选，候选再交给模型判（一次调用）

FACTS = STATE / "facts.jsonl"
DB = STATE / "memory.db"
COLD = ROOT / "cold"                 # 冷箱：非用户要求，平时不读
LIFECYCLE = STATE / "lifecycle.json"
REVIEW_EVERY = 4                     # 每 4 次封存解锁一次审核
PASS_SCORE = 85                      # 核验通过线

LOCK = STATE / "lifecycle.lock"
LOCK_STALE_SEC = 600                 # 超过 10 分钟视为僵死锁
MAX_ATTEMPTS = 3                     # 3 次不过 → 隔离区


def acquire_lock() -> tuple:
    """文件锁：拿不到就返回原因，绝不阻塞、绝不抛异常。"""
    ensure_dirs()
    if LOCK.exists():
        try:
            info = json.loads(read(LOCK))
            age = time.time() - float(info.get("ts", 0))
        except Exception:
            info, age = {}, LOCK_STALE_SEC + 1
        if age < LOCK_STALE_SEC:
            return False, f"另一个进程正在跑（pid={info.get('pid','?')}，{int(age)}s 前拿的锁）"
        try:
            LOCK.unlink()
        except Exception:
            pass
    try:
        fd = os.open(str(LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, json.dumps({"pid": os.getpid(), "ts": time.time()}).encode("utf-8"))
        os.close(fd)
        return True, "ok"
    except FileExistsError:
        return False, "抢锁失败：刚好被别的进程创建"
    except Exception as e:
        return False, f"抢锁异常：{type(e).__name__}"


def release_lock():
    try:
        LOCK.unlink()
    except Exception:
        pass


def _find_batch(data: dict, rel: str):
    return next((b for b in data["batches"] if b["dir"] == rel), None)


def _sealed_fifo(data: dict) -> list:
    """FIFO：最老的先审，失败的批次不会被新批次饿死。"""
    sealed = [b for b in data["batches"] if b["state"] == "sealed"]
    return sorted(sealed, key=lambda b: b.get("sealed_at", ""))


def recover_packing(data: dict) -> list:
    """崩溃恢复：把 packing 状态的批次对齐到磁盘真相（幂等，可反复跑）。"""
    notes = []
    for b in data["batches"]:
        if b.get("state") != "packing":
            continue
        plan = b.get("plan") or {}
        pack_dir = ROOT / plan.get("pack_dir", "")
        dst = pack_dir / Path(b["dir"]).name
        src = ROOT / b["dir"]
        if dst.exists():
            b["state"] = "cold"
            b["review"] = {"pack": plan.get("pack"), "at": now().isoformat(),
                           "score": (b.get("review") or {}).get("score")}
            notes.append(f"{b['dir']} 已在冷箱，标记为 cold")
        elif src.exists():
            b["state"] = "sealed"
            b["plan"] = None
            notes.append(f"{b['dir']} 还没搬，退回 sealed")
        else:
            b["state"] = "quarantine"
            b["note"] = "打包中途文件丢失"
            notes.append(f"{b['dir']} 文件不见了，进隔离区")
    if notes:
        save_lifecycle(data)
    return notes


def cmd_recover(args) -> int:
    ok, why = acquire_lock()
    if not ok:
        print(f"跳过：{why}")
        return 2
    try:
        data = load_lifecycle()
        notes = recover_packing(data)
        for n in notes:
            print("  " + n)
        if not notes:
            print("没有需要恢复的中途状态。")
        return 0
    finally:
        release_lock()


def load_lifecycle() -> dict:
    if LIFECYCLE.exists():
        try:
            return json.loads(read(LIFECYCLE))
        except Exception:
            pass
    return {"batches": [], "packs": []}


def save_lifecycle(data: dict):
    ensure_dirs()
    tmp = LIFECYCLE.with_name(LIFECYCLE.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, LIFECYCLE)


def _batch_dirs() -> list:
    """archive 下所有批次目录（按时间升序）。"""
    if not ARCHIVE.exists():
        return []
    return sorted([p for p in ARCHIVE.iterdir() if p.is_dir()])


def _batch_state(data: dict, rel: str) -> str:
    for b in data["batches"]:
        if b["dir"] == rel:
            return b["state"]
    return "hot"


def cmd_seal(args) -> int:
    """封存当前批次：只提醒一次，然后进入封存态。原文不读，只有节点参与后续。"""
    data = load_lifecycle()
    dirs = _batch_dirs()
    if not dirs:
        print("没有可封存的批次，先压缩一次。")
        return 1

    target = None
    for d in reversed(dirs):
        rel = str(d.relative_to(ROOT))
        if not any(b["dir"] == rel for b in data["batches"]):
            target = d
            break
    if target is None:
        print("所有批次都已封存，没有可封存的了。")
        return 0
    rel = str(target.relative_to(ROOT))

    if not args.yes:
        print("关于要封存的内容：")
        print("  · 原文保留不动，但封存后**不再参与日常读取**")
        print("  · 之后只读精炼后的节点")
        print("  · 每 4 次封存解锁一次核验；核验 ≥85 才打包进冷箱（冷箱平时也不读）")
        try:
            ans = input("确认封存？[y/N] ").strip().lower()
        except EOFError:
            ans = "n"
        if ans not in ("y", "yes", "是"):
            print("已取消。")
            return 0

    data["batches"].append({
        "dir": rel, "state": "sealed",
        "sealed_at": now().isoformat(),
        "review": None,
    })
    save_lifecycle(data)

    sealed = [b for b in data["batches"] if b["state"] == "sealed"]
    print(f"[已封存] {rel}（当前封存 {len(sealed)} 份，每 {REVIEW_EVERY} 份解锁一次核验）")
    if len(sealed) >= REVIEW_EVERY and len(sealed) % REVIEW_EVERY == 0:
        print("=> 可以核验了：python refiner.py review")
    return 0


def _score_batch(b: dict):
    d = ROOT / b["dir"]
    nodes = sorted(d.glob("*_node.json"))
    raws = sorted(d.glob("*_raw.md"))
    if not nodes or not raws:
        return {"dir": b["dir"], "error": "缺 node 或 raw"}
    node = json.loads(read(nodes[-1]))
    raw = read(raws[-1])
    sc = score(raw, node_to_text(node), claim_text=node_claims(node))
    return {"dir": b["dir"], "score": sc["score"], "recall": sc["recall"], "precision": sc["precision"]}


def _refine_batch(b: dict, key: str) -> bool:
    """打回重练：拿原文重新精炼一次，覆盖节点。失败不抛异常。"""
    if not key:
        return False
    d = ROOT / b["dir"]
    raws = sorted(d.glob("*_raw.md"))
    if not raws:
        return False
    try:
        node, _chunks = compress_pool(read(raws[-1]), key)
    except Exception:
        return False
    nodes = sorted(d.glob("*_node.json"))
    target = nodes[-1] if nodes else (d / (stamp() + "_node.json"))
    target.write_text(json.dumps(node, ensure_ascii=False, indent=2), encoding="utf-8")
    return True


def cmd_review(args) -> int:
    """核验：FIFO 取最老的 4 份封存，达标才打包进冷箱。

    状态机 sealed → reviewing → packing → cold；
    失败晋级：第 1 次打回排队 / 第 2 次打回重练 / 第 3 次隔离。
    全程不抛异常，崩溃靠 recover_packing 对齐。
    """
    ok, why = acquire_lock()
    if not ok:
        print(f"跳过：{why}")
        return 2
    try:
        data = load_lifecycle()
        for n in recover_packing(data):
            print("  " + n)

        sealed = _sealed_fifo(data)
        if len(sealed) < REVIEW_EVERY:
            print(f"封存还不够：{len(sealed)}/{REVIEW_EVERY}")
            return 1

        sample = sealed[:REVIEW_EVERY]
        for b in sample:
            b["state"] = "reviewing"
        save_lifecycle(data)

        results = [_score_batch(b) for b in sample]
        good = [r for r in results if "score" in r]
        print(f"核验 {len(sample)} 份封存：")
        for r in results:
            if "error" in r:
                print(f"  {r['dir']}  ⚠ {r['error']}")
            else:
                print(f"  {r['dir']}  分数 {r['score']}（召回 {r['recall']} / 精确 {r['precision']}）")

        if len(good) < REVIEW_EVERY:
            for b in sample:
                b["state"] = "sealed"
            save_lifecycle(data)
            print("有批次缺文件，退回封存态，不打包。")
            return 1

        avg = sum(r["score"] for r in good) / len(good)
        low = min(r["score"] for r in good)
        threshold = args.pass_score
        print(f"平均 {avg:.1f} ｜ 最低 {low:.1f} ｜ 通过线 {threshold}")
        passed = (avg >= threshold) and (low >= threshold - 10)

        if not passed:
            key = deepseek_key()
            for b in sample:
                b["attempts"] = int(b.get("attempts", 0)) + 1
                b["last_score"] = next((r["score"] for r in good if r["dir"] == b["dir"]), None)
                if b["attempts"] >= MAX_ATTEMPTS:
                    b["state"] = "quarantine"
                    b["note"] = f"连续 {b['attempts']} 次未通过，待人工看"
                elif b["attempts"] >= 2:
                    done = _refine_batch(b, key)
                    b["state"] = "sealed"
                    b["note"] = "已打回重练" if done else "重练没跑成，留待下次"
                else:
                    b["state"] = "sealed"
                    b["note"] = "第 1 次未过，排队首下次优先"
            save_lifecycle(data)
            print("未通过：不打包，已按次数晋级（打回 / 重练 / 隔离）。")
            return 1

        pack_name = stamp() + "_pack"
        pack_rel = str((COLD / pack_name).relative_to(ROOT))
        for b in sample:
            b["state"] = "packing"
            b["plan"] = {"pack": pack_name, "pack_dir": pack_rel}
        save_lifecycle(data)

        pack_dir = COLD / pack_name
        pack_dir.mkdir(parents=True, exist_ok=True)
        moved = []
        for b in sample:
            src = ROOT / b["dir"]
            dst = pack_dir / src.name
            if src.exists() and not dst.exists():
                shutil.move(str(src), str(dst))
            if dst.exists():
                moved.append(dst.name)
            b["state"] = "cold"
            b["review"] = {"score": next((r["score"] for r in good if r["dir"] == b["dir"]), None),
                           "pack": pack_name, "at": now().isoformat()}
            save_lifecycle(data)

        (pack_dir / "manifest.json").write_text(json.dumps({
            "created_at": now().isoformat(), "batches": moved,
            "avg_score": round(avg, 1), "min_score": round(low, 1), "results": results,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

        data["packs"].append({"name": pack_name, "batches": moved, "avg": round(avg, 1)})
        save_lifecycle(data)
        print(f"[已入冷箱] {pack_dir}（{len(moved)} 份）")
        print("冷箱平时不读；要看就加 --cold。")
        return 0
    finally:
        release_lock()


def cmd_cold(args) -> int:
    data = load_lifecycle()
    if not data["packs"]:
        print("冷箱是空的。")
        return 0
    for p in data["packs"]:
        print(f"  {p['name']}  ｜ {len(p['batches'])} 份 ｜ 平均 {p['avg']}")
    print(f"\n位置：{COLD}")
    return 0


def db() -> sqlite3.Connection:
    """本地索引库：文件仍是唯一真相，DB 只是加速层，删了能重建。"""
    ensure_dirs()
    conn = sqlite3.connect(DB)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS nodes(
            path TEXT PRIMARY KEY, at TEXT, title TEXT, score REAL, sha TEXT, text TEXT);
        CREATE TABLE IF NOT EXISTS facts(
            id TEXT PRIMARY KEY, text TEXT, status TEXT, confidence TEXT,
            valid_from TEXT, valid_to TEXT, superseded_by TEXT, source TEXT);
        CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts
            USING fts5(path UNINDEXED, title, body, tokenize='trigram');
        CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts
            USING fts5(id UNINDEXED, body, tokenize='trigram');
    """)
    return conn


def cmd_index(args) -> int:
    """把 archive/ 全量灌进 SQLite（可重复执行，幂等）。冷箱默认不收，加 --cold 才收。"""
    conn = db()
    files = sorted(ARCHIVE.glob("*/*_node.json"))
    if getattr(args, "cold", False) and COLD.exists():
        files += sorted(COLD.rglob("*_node.json"))
    n = 0
    for f in files:
        rel = str(f.relative_to(ROOT))
        node = json.loads(read(f))
        body = node_to_text(node)
        meta = {}
        mp = f.with_name(f.name.replace("_node.json", "_meta.json"))
        if mp.exists():
            try:
                meta = json.loads(read(mp))
            except Exception:
                meta = {}
        conn.execute("INSERT OR REPLACE INTO nodes(path,at,title,score,sha,text) VALUES(?,?,?,?,?,?)",
                     (rel, meta.get("at", ""), node.get("node_title", ""),
                      (meta.get("score") or {}).get("score") if isinstance(meta.get("score"), dict) else meta.get("score"),
                      meta.get("sha_node", ""), body))
        conn.execute("DELETE FROM nodes_fts WHERE path=?", (rel,))
        conn.execute("INSERT INTO nodes_fts(path,title,body) VALUES(?,?,?)",
                     (rel, node.get("node_title", ""), body))
        n += 1
    for r in load_facts():
        conn.execute("INSERT OR REPLACE INTO facts(id,text,status,confidence,valid_from,valid_to,superseded_by,source)"
                     " VALUES(?,?,?,?,?,?,?,?)",
                     (r["id"], r["text"], r.get("status"), r.get("confidence"),
                      r.get("valid_from"), r.get("valid_to"), r.get("superseded_by"), r.get("source")))
        conn.execute("DELETE FROM facts_fts WHERE id=?", (r["id"],))
        conn.execute("INSERT INTO facts_fts(id,body) VALUES(?,?)", (r["id"], r["text"]))
    conn.commit()
    print(f"索引完成：{n} 份节点，{conn.execute('SELECT COUNT(*) FROM facts').fetchone()[0]} 条事实 → {DB}")
    print("库删了也没事，files 才是真相，重新 index 即可。")
    conn.close()
    return 0


def cmd_search(args) -> int:
    """FTS5 trigram 子串检索：几千份节点里找词，不再遍历文件。"""
    if not DB.exists():
        print("还没有索引，先跑 python refiner.py index")
        return 2
    conn = db()
    q = args.q.strip()
    if not q:
        print("给个词：--q 关键词")
        return 2
    try:
        if len(q) >= 3:
            rows = conn.execute(
                "SELECT path, title, snippet(nodes_fts, 2, '[', ']', '…', 12) FROM nodes_fts "
                "WHERE nodes_fts MATCH ? ORDER BY rank LIMIT ?", (f'"{q}"', args.limit)).fetchall()
        else:
            # trigram 索引对 2 字查询无能为力，退回 LIKE（量小时依旧比遍历文件快）
            like = f"%{q}%"
            rows = conn.execute(
                "SELECT path, title, substr(text, 1, 60) FROM nodes "
                "WHERE title LIKE ? OR text LIKE ? LIMIT ?", (like, like, args.limit)).fetchall()
    except sqlite3.OperationalError as e:
        print(f"查询失败：{e}")
        conn.close()
        return 2
    if not rows:
        print("没命中。")
        conn.close()
        return 0
    print(f"命中 {len(rows)} 份节点：")
    for path, title, snip in rows:
        print(f"  {path}\n    {title} ｜ {snip}")
    conn.close()
    return 0

MERGE_SYS = """你在合并记忆事实。左边是旧事实，右边是新事实。
判断每一对的真实关系，只输出 JSON：
{"pairs": [{"old": "旧id", "new": "新id", "relation": "duplicate|update|contradiction"}]}

判定标准：
- duplicate：说的是同一件事，没有新信息
- update：同一件事，新事实更新了取值（如 状态从"在岗"改为"离职"、日期改了、数字改了）
- contradiction：直接冲突，两者不能同时为真
- 不构成以上任何一种的，不要出现在结果里"""


def fact_id(text: str, src: str) -> str:
    return hashlib.sha256((src + "|" + text).encode("utf-8")).hexdigest()[:12]


def load_facts() -> list:
    if not FACTS.exists():
        return []
    out = []
    for line in read(FACTS).splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def save_facts(rows: list):
    """原子写：先写临时文件再 os.replace，崩在中间也不会把事实库写坏。"""
    ensure_dirs()
    tmp = FACTS.with_name(FACTS.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, FACTS)


def sim(a: str, b: str) -> float:
    x, y = bigrams(a), bigrams(b)
    if not x or not y:
        return 0.0
    return len(x & y) / len(x | y)


def shared_long(a: str, b: str, n: int = 4) -> int:
    x = {a[i:i + n] for i in range(max(0, len(a) - n + 1))}
    y = {b[i:i + n] for i in range(max(0, len(b) - n + 1))}
    return len(x & y)


CHANGE_WORDS = ("已", "不", "离职", "辞职", "入职", "换成", "改成", "现在", "目前",
                "取消", "停止", "开始", "结束", "转", "升", "降", "不再", "恢复")


def split_terms(q: str) -> list:
    """把查询切成 2 字以上的中文片段 + 拉丁词，用于粗召回。"""
    terms = re.findall(r"[\u4e00-\u9fa5]{2,}", q or "")
    terms += [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z0-9_\-]+", q or "")]
    return terms


def heuristic_relation(old_text: str, new_text: str, score: float) -> str:
    """无模型时的降级判定：只在"同一件事 + 出现变化词"时才敢覆盖。"""
    if score >= 0.85:
        return "duplicate"
    if shared_long(old_text, new_text) >= 1 and any(w in new_text for w in CHANGE_WORDS):
        return "update"
    return ""


def classify_pairs(pairs: list, key: str) -> dict:
    """把候选对交给模型判关系；失败时降级为相似度规则。"""
    if not pairs:
        return {}
    body = "\n".join(
        f'{i}. old={p["old"]["id"]} 「{p["old"]["text"]}」\n   new={p["new"]["id"]} 「{p["new"]["text"]}」'
        for i, p in enumerate(pairs, 1))
    try:
        out = chat(LLM["url"], key or LLM["key"], LLM["model"], MERGE_SYS, body, max_tokens=600, temperature=0.0)
        data = extract_json(out)
        return {(x["old"], x["new"]): x["relation"] for x in data.get("pairs", [])}
    except Exception:
        return {}


def merge_facts(new_facts: list, source_dir: str, key: str, now_iso: str = None) -> dict:
    """把新事实并入事实库，处理重复/更新/冲突。返回统计。"""
    now_iso = now_iso or now().isoformat()
    rows = load_facts()
    active = [r for r in rows if r.get("status") == "active"]

    stats = {"added": 0, "duplicate": 0, "superseded": 0}
    candidates = []
    to_add = []

    for f in new_facts:
        if not isinstance(f, dict) or not f.get("text"):
            continue
        text = f["text"].strip()
        nid = fact_id(text, source_dir)
        best, best_s = None, 0.0
        for r in active:
            s = sim(text, r["text"])
            if s > best_s:
                best, best_s = r, s
        if best is not None and (best_s >= 0.45 or shared_long(text, best["text"]) >= 1):
            candidates.append({"old": best, "new": {"id": nid, "text": text}, "score": best_s})
        to_add.append({"id": nid, "text": text, "kind": f.get("kind", "fact"),
                       "confidence": f.get("confidence", "C"), "source": source_dir})

    rel = classify_pairs(candidates, key) if candidates else {}

    superseded_ids = set()
    for p in candidates:
        r = rel.get((p["old"]["id"], p["new"]["id"])) or \
            heuristic_relation(p["old"]["text"], p["new"]["text"], p["score"])
        if not r:
            continue      # 判不出关系就不动——宁可不合并，也不能乱覆盖
        if r == "duplicate":
            stats["duplicate"] += 1
            superseded_ids.add(p["new"]["id"])   # 新的不再单独加
        else:
            old = p["old"]
            old["status"] = "superseded"
            old["valid_to"] = now_iso
            old["superseded_by"] = p["new"]["id"]
            old["relation"] = r
            stats["superseded"] += 1

    for a in to_add:
        if a["id"] in superseded_ids:
            continue
        a.update({"status": "active", "valid_from": now_iso, "superseded_by": None})
        rows.append(a)
        stats["added"] += 1

    save_facts(rows)
    return stats


def cmd_facts(args) -> int:
    rows = load_facts()
    act = [r for r in rows if r.get("status") == "active"]
    sup = [r for r in rows if r.get("status") == "superseded"]
    print(f"活跃 {len(act)} ｜ 已被覆盖 {len(sup)}")
    for r in act[-args.limit:]:
        print(f"  [{r.get('confidence','?')}] {r['text'][:60]}  ({r['id']})")
    return 0


def cmd_rebuild(args) -> int:
    """把 archive/ 里已有的节点重新并进事实库（迁移/修复用）。"""
    files = sorted(ARCHIVE.glob("*/*_node.json"))
    if not files:
        print("archive 里没有节点")
        return 0
    key = deepseek_key()
    total = {"added": 0, "duplicate": 0, "superseded": 0}
    for f in files:
        try:
            node = json.loads(read(f))
        except Exception:
            continue
        st = merge_facts(node.get("facts", []), str(f.parent.relative_to(ROOT)), key)
        for k in total:
            total[k] += st[k]
    print(f"重建完成：处理 {len(files)} 份节点 → {total}")
    return 0


def cmd_recall(args) -> int:
    """按关键词召回活跃事实（不读原文，不调模型）。"""
    rows = [r for r in load_facts() if r.get("status") == "active"]
    if not rows:
        print("事实库是空的，先跑 compress 或 rebuild")
        return 0
    q = args.q
    scored = []
    for r in rows:
        s = sim(q, r["text"]) if q else 0.0
        bonus = 0.15 if any(w in r["text"] for w in split_terms(q)) else 0.0
        scored.append((s + bonus, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    picked = [r for s, r in scored if s > 0 or not q][:args.limit]

    if args.json:
        print(json.dumps({"query": q, "facts": picked}, ensure_ascii=False, indent=2))
        return 0

    print(f"命中 {len(picked)} 条（共 {len(rows)} 条活跃）")
    for r in picked:
        print(f"  [{r.get('confidence','?')}] {r['text']}")
    if args.inject:
        print("\n--- 可直接注入 ---")
        for r in picked:
            print(f"- {r['text']}")
    return 0


def cmd_history(args) -> int:
    rows = load_facts()
    cur = next((r for r in rows if r["id"] == args.id), None)
    if not cur:
        print("找不到这个 id")
        return 1
    chain = [cur]
    while cur.get("superseded_by"):
        nxt = next((r for r in rows if r["id"] == cur["superseded_by"]), None)
        if not nxt:
            break
        chain.append(nxt)
        cur = nxt
    print("版本链（从旧到新）：")
    for r in chain:
        print(f"  {r.get('valid_from','?')[:19]}  [{r.get('status')}]  {r['text']}")
    return 0


# ─────────────────────────── 动作 ───────────────────────────

def cmd_compress(args) -> int:
    ensure_dirs()
    raw = read(args.raw)
    if len(raw) < args.min_chars and not args.force:
        print(f"原文只有 {len(raw)} 字符，低于 {args.min_chars}——压了会比原文更长，没意义。")
        print("真的要做就加 --force。")
        return 3
    cfg = use_provider(getattr(args, "provider", None))
    key = cfg["key"]
    if not key and cfg["name"] != "local":
        print(f"provider={cfg['name']} 没有 key：设环境变量或写进 config.json")
        return 2

    # 幂等：同一份原文只归档一次
    sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    for r in load_index():
        if r.get("sha") == sha and not args.force:
            print(f"这份原文已经归档过（{r['at'][:19]}，分数 {r['score']}），跳过。要重做加 --force。")
            return 0

    print(f"原文 {len(raw)} 字符，开始精炼…")
    pool, recent = (raw, "")
    if args.keep_turns and args.keep_turns > 0:
        pool, recent = split_window(raw, args.keep_turns)
        if not pool:
            pool, recent = raw, ""
        else:
            print(f"  滑动窗口：压缩池 {len(pool)} 字符，保留最近 {args.keep_turns} 轮原文 {len(recent)} 字符")

    if len(pool) < args.min_chars and not args.force:
        print(f"压缩池只有 {len(pool)} 字符（窗口切完之后），低于 {args.min_chars}，先不压。")
        return 3

    node, n_chunks = compress_pool(pool, key)
    text = node_to_text(node)
    pv = {"probe": None, "qa": []}
    if args.probe:
        print("跑探针问答…")
        pv = probe(raw, text, key, args.probe_k)
    sc = score(raw, text, probe_score=pv["probe"], claim_text=node_claims(node))

    day = now().strftime("%Y-%m-%d")
    t = stamp()
    d = ARCHIVE / day
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{t}_raw.md").write_text(raw, encoding="utf-8")
    if recent:
        (d / f"{t}_recent.md").write_text(recent, encoding="utf-8")
    (d / f"{t}_node.json").write_text(json.dumps(node, ensure_ascii=False, indent=2), encoding="utf-8")
    (d / f"{t}_meta.json").write_text(json.dumps({
        "title": args.title or node.get("node_title", ""),
        "at": now().isoformat(),
        "sha_raw": hashlib.sha256(raw.encode()).hexdigest()[:16],
        "sha_node": hashlib.sha256(text.encode()).hexdigest()[:16],
        "chunks": n_chunks,
        "pool_chars": len(pool),
        "recent_chars": len(recent),
        "score": sc,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    n = bump_counter()
    append_index({"at": now().isoformat(), "dir": str(d.relative_to(ROOT)), "sha": sha,
                  "title": args.title or node.get("node_title", ""),
                  "score": sc["score"], "probe": sc.get("probe")})

    # 并入事实库（版本链）
    st = merge_facts(node.get("facts", []), str(d.relative_to(ROOT)), key)
    print(f"  事实库：新增 {st['added']} ｜ 重复 {st['duplicate']} ｜ 覆盖旧版 {st['superseded']}")
    if pv["qa"]:
        (d / f"{t}_probe.json").write_text(json.dumps(pv, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[已存档] {d}")
    print(f"  分数 {sc['score']} ｜ 召回 {sc['recall']} ｜ 精确 {sc['precision']} ｜ 压缩率 {sc['compression_ratio']}")
    if sc.get("probe") is not None:
        print(f"  探针 {sc['probe']} ｜ 综合分 {sc['score_with_probe']}")
    if sc["unsupported"]:
        print(f"  ⚠ 无支撑分句 {len(sc['unsupported'])} 条，例：{sc['unsupported'][0]}")
    if sc["missed_key"]:
        print(f"  ⚠ 漏掉的主题：{', '.join(sc['missed_key'])}")
    print(f"  这是第 {n} 次压缩" + ("，该做机理精炼了（python refiner.py refine）" if n % REFINE_EVERY == 0 else ""))
    return 0


def cmd_score(args) -> int:
    raw = read(args.raw)
    p = Path(args.node)
    node_text = read(p)
    claims = None
    if p.suffix == ".json":
        try:
            node = json.loads(node_text)
            node_text = node_to_text(node)
            claims = node_claims(node)
        except Exception:
            pass
    sc = score(raw, node_text, claim_text=claims)
    print(json.dumps(sc, ensure_ascii=False, indent=2))
    return 0


def cmd_judge(args) -> int:
    raw = read(args.raw)
    node = read(args.node)
    key = glm_key()
    if not key:
        print("拿不到智谱 key")
        return 2
    try:
        out = chat(GLM_URL, key, GLM_MODEL, JUDGE_SYS,
                   f"【原文】\n{raw[:9000]}\n\n【节点】\n{node[:4000]}\n\n输出 JSON。", max_tokens=800)
    except urllib.error.HTTPError as e:
        hint = "（免费档限流）" if e.code == 429 else ""
        print(f"裁判暂时不可用：HTTP {e.code}{hint}。跳过裁判，脚本分仍然有效。")
        return 4
    except Exception as e:
        print(f"裁判调用失败：{type(e).__name__}。跳过，脚本分仍然有效。")
        return 4
    try:
        v = extract_json(out)
    except Exception:
        print("裁判返回无法解析：", out[:200])
        return 1
    sc = score(raw, node)
    miss = len(v.get("missed_facts", []))
    uns = len(v.get("unsupported_claims", []))
    judge_factor = max(0.0, 1 - (miss * 0.05 + uns * 0.1))
    final = round(sc["score"] * 0.7 + judge_factor * 100 * 0.3, 1)
    print(json.dumps({
        "script_score": sc["score"],
        "judge": v,
        "judge_factor": round(judge_factor, 3),
        "final_score": final,
    }, ensure_ascii=False, indent=2))
    return 0


def cmd_probe(args) -> int:
    raw = read(args.raw)
    p = Path(args.node)
    node_text = read(p)
    if p.suffix == ".json":
        try:
            node_text = node_to_text(json.loads(node_text))
        except Exception:
            pass
    cfg = use_provider(getattr(args, "provider", None))
    key = cfg["key"]
    if not key and cfg["name"] != "local":
        print(f"provider={cfg['name']} 没有 key")
        return 2
    pv = probe(raw, node_text, key, args.k)
    print(json.dumps(pv, ensure_ascii=False, indent=2))
    if pv.get("probe") is None:
        print(f"\n探针没跑成：{pv.get('error','未知原因')}——脚本分不受影响。")
        return 4
    print(f"\n探针得分 {pv['probe']}")
    return 0


def cmd_refine(args) -> int:
    ensure_dirs()
    rows = load_index()
    if len(rows) < 2:
        print("归档太少，先压缩几次")
        return 0
    recent = rows[-REFINE_EVERY:]
    facts, anchors, threads = [], [], []
    for r in recent:
        nd = Path(ROOT) / r["dir"]
        cand = sorted(nd.glob("*_node.json"))
        if not cand:
            continue
        node = json.loads(read(cand[-1]))
        for f in node.get("facts", []):
            if isinstance(f, dict) and f.get("text"):
                facts.append(f)
        anchors += node.get("persona_anchors", [])
        threads += node.get("open_threads", [])

    # 去重：文本二元组相似度 > 0.8 视为同一条
    kept = []
    for f in facts:
        dup = False
        for k in kept:
            a, b = bigrams(f["text"]), bigrams(k["text"])
            if a and b and len(a & b) / max(1, len(a | b)) > 0.8:
                dup = True
                break
        if not dup:
            kept.append(f)

    # 有选择地降权：C 级进 cold，S/A 进核心
    core = [f for f in kept if f.get("confidence") in ("S", "A")]
    warm = [f for f in kept if f.get("confidence") == "B"]
    cold = [f for f in kept if f.get("confidence") in ("C", None)]

    t = stamp()
    out = REFINED / f"{t}_refined.md"
    lines = [
        f"# 机理精炼 {now().strftime('%Y-%m-%d %H:%M')}",
        f"> 覆盖最近 {len(recent)} 次压缩；原件不删，全部留在 archive/",
        "",
        "## 宪章层（原样保留）",
        *[f"- {a}" for a in dict.fromkeys(anchors)],
        "",
        f"## 核心事实 S/A（{len(core)} 条）",
        *[f"- [{f.get('kind','fact')}] {f['text']}" for f in core],
        "",
        f"## 温层 B（{len(warm)} 条，检索时注入）",
        *[f"- {f['text']}" for f in warm],
        "",
        f"## 冷层 C（{len(cold)} 条，只留档不注入）",
        *[f"- {f['text']}" for f in cold],
        "",
        "## 未完成",
        *[f"- {x}" for x in dict.fromkeys(threads)],
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[精炼完成] {out}")
    print(f"  核心 {len(core)} ｜ 温层 {len(warm)} ｜ 冷层 {len(cold)}")
    return 0


def mask(k: str) -> str:
    if not k:
        return "（没找到）"
    return k[:6] + "…" + k[-4:] if len(k) > 12 else "*" * len(k)


def cmd_doctor(args) -> int:
    print(f"memory-refiner {__version__}")
    print(f"python       {sys.version.split()[0]}")
    print(f"项目目录     {ROOT}")

    d, dsrc = resolve_key("DEEPSEEK_API_KEY", cfg=["deepseek_api_key"])
    if not d:
        m = re.search(r"api_key:\s*(sk-[A-Za-z0-9]+)", wsl_config())
        if m:
            d, dsrc = m.group(1), "wsl:hermes"
    g, gsrc = resolve_key("ZHIPU_API_KEY", "GLM_API_KEY", cfg=["zhipu_api_key", "glm_api_key"])
    if not g:
        ks = re.findall(r"api_key:\s*([0-9a-f]{20,}\.[A-Za-z0-9]+)", wsl_config())
        if ks:
            g, gsrc = ks[0], "wsl:hermes"

    print(f"DeepSeek     {mask(d)}   {dsrc or '未配置 → compress/probe 用不了'}")
    print(f"智谱 GLM     {mask(g)}   {gsrc or '未配置 → judge 用不了（可选）'}")
    print(f"config.json  {'有' if CONFIG.exists() else '没有（不必须）'}")
    print(f"归档         {len(list(ARCHIVE.glob('*/*_node.json'))) if ARCHIVE.exists() else 0} 份节点"
          f" ｜ {len(load_facts())} 条事实")
    print(f"可离线用     score / recall / facts / history / refine / report / rebuild")

    if args.ping and d:
        try:
            chat(LLM["url"], d or LLM["key"], LLM["model"], "只回一个字", "ping", max_tokens=4, temperature=0)
            print("连通性       OK")
        except Exception as e:
            print(f"连通性       失败：{type(e).__name__}")
    return 0


def cmd_setup(args) -> int:
    conf = load_config()
    if args.from_dsh:
        dk = dsh_keys()
        if dk.get("DEEPSEEK_API_KEY"):
            conf["deepseek_api_key"] = dk["DEEPSEEK_API_KEY"]
        if dk.get("GLM_API_KEY"):
            conf["zhipu_api_key"] = dk["GLM_API_KEY"]
    if args.deepseek:
        conf["deepseek_api_key"] = args.deepseek
    if args.zhipu:
        conf["zhipu_api_key"] = args.zhipu
    if not conf:
        print("什么都没给。用 --from-dsh 自动取，或 --deepseek/--zhipu 直接传。")
        return 1
    CONFIG.write_text(json.dumps(conf, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入 {CONFIG}")
    print("这个文件已经在 .gitignore 里，不会进仓库。")
    return 0


MAX_INPUT_CHARS = 12000


def chunk_text(text: str, size: int = MAX_INPUT_CHARS) -> list:
    """按段落贪心切块，绝不静默截断。"""
    if len(text) <= size:
        return [text]
    chunks, cur = [], ""
    for para in re.split(r"(\n\s*\n)", text):
        if len(cur) + len(para) <= size:
            cur += para
        else:
            if cur.strip():
                chunks.append(cur)
            cur = para
            while len(cur) > size:          # 单段超长，硬切但会说明
                chunks.append(cur[:size])
                cur = cur[size:]
    if cur.strip():
        chunks.append(cur)
    return chunks


def merge_nodes(nodes: list) -> dict:
    """多块节点合并：事实去重，摘要拼接，锚点/未完成取并集。"""
    if len(nodes) == 1:
        return nodes[0]
    facts, anchors, threads = [], [], []
    for n in nodes:
        for f in n.get("facts", []):
            if isinstance(f, dict) and f.get("text"):
                if not any(sim(f["text"], x["text"]) > 0.8 for x in facts):
                    facts.append(f)
        anchors += n.get("persona_anchors", [])
        threads += n.get("open_threads", [])
    summary = "；".join((n.get("summary") or "").strip() for n in nodes if n.get("summary"))
    return {
        "node_title": nodes[0].get("node_title", ""),
        "summary": summary[:300],
        "facts": facts[:24],
        "persona_anchors": list(dict.fromkeys(anchors))[:8],
        "open_threads": list(dict.fromkeys(threads))[:8],
    }


def compress_pool(pool: str, key: str) -> tuple:
    """把压缩池切成块逐块精炼，返回 (节点, 块数)。"""
    chunks = chunk_text(pool)
    nodes = []
    for i, c in enumerate(chunks, 1):
        if len(chunks) > 1:
            print(f"  分块 {i}/{len(chunks)}（{len(c)} 字符）")
        nodes.append(chat_json(LLM["url"], key or LLM["key"], LLM["model"], COMPRESS_SYS,
                               f"【原文】\n{c}\n\n请输出记忆节点 JSON。"))
    return merge_nodes(nodes), len(chunks)


def cmd_providers(args) -> int:
    """列出所有可接入的模型服务，看谁有 key。"""
    active = resolve_provider(None)["name"]
    print(f"{'名称':<12}{'默认模型':<34}{'key':<6}{'端点'}")
    for name in PROVIDERS:
        p = resolve_provider(name)
        has = "有" if (p["key"] or name == "local") else "—"
        mark = " ←当前" if name == active else ""
        print(f"{name:<12}{p['model'][:33]:<34}{has:<6}{p['url']}{mark}")
    print("\nkey 按 环境变量 → config.json → DSH 凭证 的顺序找。")
    print("本地模型：config.json 里设 local_base_url（Ollama 11434 / LM Studio 1234 / llama.cpp 8080）。")
    return 0


def cmd_panel(args) -> int:
    """多模型复查：同一份 raw + node，让 N 个模型各跑一遍探针，结果并排放。"""
    raw = read(args.raw)
    p = Path(args.node)
    node_text = read(p)
    if p.suffix == ".json":
        try:
            node_text = node_to_text(json.loads(node_text))
        except Exception:
            pass

    names = [x.strip() for x in args.models.split(",") if x.strip()]
    rows = []
    for name in names:
        cfg = use_provider(name)
        if not cfg["key"] and name != "local":
            rows.append((name, cfg["model"], "无 key", "-"))
            continue
        t0 = time.time()
        try:
            pv = probe(raw, node_text, cfg["key"], args.k)
            val = pv.get("probe")
            rows.append((name, cfg["model"], "?" if val is None else f"{val:.2f}",
                         f"{time.time()-t0:.1f}s"))
        except Exception as e:
            rows.append((name, cfg["model"], f"失败:{type(e).__name__}", "-"))

    print(f"\n{'provider':<12}{'model':<34}{'探针':<8}{'耗时'}")
    for name, model, score, took in rows:
        print(f"{name:<12}{model[:33]:<34}{score:<8}{took}")
    return 0


def cmd_context(args) -> int:
    """看当前上下文占了多少 token，以及要不要立刻触发压缩。"""
    text = read(Path(args.raw)) if args.raw else sys.stdin.read()
    rep = token_report(text)
    print(f"字符数    {rep['chars']}")
    print(f"token     {rep['tokens']}（本地估算；装了 tiktoken 会用真值）")
    print(f"上限      {rep['limit']}   安全水位 {rep['watermark']}（{int(SAFETY_RATIO*100)}%）")
    print(f"占比      {rep['ratio']*100:.1f}%")
    if rep["over_watermark"]:
        print("=> 已经过水位：立刻压缩（python refiner.py compress --raw ...）")
    else:
        print("=> 还安全，可以继续")
    if args.keep_turns:
        older, recent = split_window(text, args.keep_turns)
        print(f"滑动窗口 保留最近 {args.keep_turns} 轮：压缩池 {len(older)} 字符 ｜ 原文保留 {len(recent)} 字符"
              f"（≈{count_tokens(older)} / {count_tokens(recent)} token）")
    return 0


def cmd_report(args) -> int:
    rows = load_index()
    if not rows:
        print("还没有归档")
        return 0
    scores = [r["score"] for r in rows if isinstance(r.get("score"), (int, float))]
    print(f"归档次数 {len(rows)} ｜ 平均分 {sum(scores)/len(scores):.1f} ｜ 最高 {max(scores)} ｜ 最低 {min(scores)}")
    for r in rows[-8:]:
        print(f"  {r['at'][:19]}  {r['score']:>5}  {r.get('title','')[:30]}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="记忆精炼与奖励引擎")
    ap.add_argument("--version", action="version", version=f"memory-refiner {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("compress", help="压缩一段原文并归档")
    p1.add_argument("--raw", required=True)
    p1.add_argument("--title", default="")
    p1.add_argument("--min-chars", type=int, default=800, dest="min_chars")
    p1.add_argument("--force", action="store_true")
    p1.add_argument("--probe", action="store_true", help="跑探针问答（多两次模型调用）")
    p1.add_argument("--probe-k", type=int, default=6, dest="probe_k")
    p1.add_argument("--provider", default=None, help="deepseek / github / openrouter / local …")
    p1.add_argument("--keep-turns", type=int, default=0, dest="keep_turns",
                    help="滑动窗口：保留最近 N 轮原文不进压缩池")
    p1.set_defaults(func=cmd_compress)

    p2 = sub.add_parser("score", help="只算分，不调模型")
    p2.add_argument("--raw", required=True)
    p2.add_argument("--node", required=True)
    p2.set_defaults(func=cmd_score)

    p3 = sub.add_parser("judge", help="外部裁判（智谱）找茬")
    p3.add_argument("--raw", required=True)
    p3.add_argument("--node", required=True)
    p3.set_defaults(func=cmd_judge)

    p4 = sub.add_parser("refine", help="每 N 次压缩做一次机理精炼")
    p4.set_defaults(func=cmd_refine)

    p5 = sub.add_parser("report", help="看归档统计")
    p5.set_defaults(func=cmd_report)

    p6 = sub.add_parser("probe", help="探针问答：只给节点，看能答对几道")
    p6.add_argument("--raw", required=True)
    p6.add_argument("--node", required=True)
    p6.add_argument("--provider", default=None)
    p6.add_argument("-k", type=int, default=6)
    p6.set_defaults(func=cmd_probe)

    p7 = sub.add_parser("facts", help="列出当前活跃事实与被覆盖数量")
    p7.add_argument("--limit", type=int, default=15)
    p7.set_defaults(func=cmd_facts)

    p8 = sub.add_parser("history", help="看一条事实的版本链")
    p8.add_argument("--id", required=True)
    p8.set_defaults(func=cmd_history)

    p9 = sub.add_parser("recall", help="按关键词召回活跃事实")
    p9.add_argument("--q", default="")
    p9.add_argument("--limit", type=int, default=8)
    p9.add_argument("--json", action="store_true")
    p9.add_argument("--inject", action="store_true", help="输出可直接粘进提示词的列表")
    p9.set_defaults(func=cmd_recall)

    p10 = sub.add_parser("rebuild", help="把 archive 里已有节点重建进事实库")
    p10.set_defaults(func=cmd_rebuild)

    p11 = sub.add_parser("doctor", help="自检：key 从哪来、能不能离线用")
    p11.add_argument("--ping", action="store_true", help="真发一次请求验连通")
    p11.set_defaults(func=cmd_doctor)

    p12 = sub.add_parser("setup", help="把 key 写进本地 config.json")
    p12.add_argument("--from-dsh", action="store_true", dest="from_dsh", help="从 DSH 凭证自动抄")
    p12.add_argument("--deepseek", default="")
    p12.add_argument("--zhipu", default="")
    p12.set_defaults(func=cmd_setup)

    p13 = sub.add_parser("index", help="把归档灌进 SQLite 索引（可重复跑）")
    p13.add_argument("--cold", action="store_true", help="把冷箱也收进索引")
    p13.set_defaults(func=cmd_index)

    p14 = sub.add_parser("search", help="在索引里搜（FTS5 子串，不遍历文件）")
    p14.add_argument("--q", required=True)
    p14.add_argument("--limit", type=int, default=10)
    p14.set_defaults(func=cmd_search)

    p15 = sub.add_parser("context", help="看上下文 token 占比，判断要不要压缩")
    p15.add_argument("--raw", default="", help="不传就从 stdin 读")
    p15.add_argument("--keep-turns", type=int, default=4, dest="keep_turns")
    p15.set_defaults(func=cmd_context)

    p16 = sub.add_parser("seal", help="封存当前批次（提醒一次，之后只读节点）")
    p16.add_argument("--yes", action="store_true", help="跳过确认")
    p16.set_defaults(func=cmd_seal)

    p17 = sub.add_parser("review", help="核验最近 4 份封存，≥85 才进冷箱")
    p17.add_argument("--pass-score", type=float, default=PASS_SCORE, dest="pass_score")
    p17.set_defaults(func=cmd_review)

    p18 = sub.add_parser("cold", help="看冷箱里有什么")
    p18.set_defaults(func=cmd_cold)

    p19 = sub.add_parser("recover", help="崩溃恢复：把中途状态对齐到磁盘真相")
    p19.set_defaults(func=cmd_recover)

    p20 = sub.add_parser("providers", help="列出可接入的模型服务与 key 状态")
    p20.set_defaults(func=cmd_providers)

    p21 = sub.add_parser("panel", help="多模型复查：同一份 raw+node 让 N 个模型探针")
    p21.add_argument("--raw", required=True)
    p21.add_argument("--node", required=True)
    p21.add_argument("--models", required=True, help="逗号分隔，如 deepseek,github,local")
    p21.add_argument("-k", type=int, default=4)
    p21.set_defaults(func=cmd_panel)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
