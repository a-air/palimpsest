# 发布清单（v0.1.0）

> 目标：把本地这个仓库安全地推上 GitHub，并且出门之前把该扫的都扫一遍。

---

## 一、出门前检查（跑一次就行）

```powershell
powershell -File scripts/preflight.ps1
```

它会做四件事：

1. 跑两套测试
2. 扫私人词（真实人名 / 机构 / 地点 / 医疗细节，词表在 `scripts/privacy_patterns.txt`，不入库）
3. 检查 `archive/ state/ refined/` 没有被 git 跟踪
4. 检查工作区是否干净、当前 commit 是什么

**任何一项红，都不许推。**

---

## 二、建仓库（只能你来）

仓库名候选：`palimpsest` / `memory-refiner` / `probe-memory`

**方式 A：网页（推荐，手机热点也行）**

1. 打开 https://github.com/new
2. Repository name 填选定的名字
3. **不要**勾选 "Add a README / .gitignore / license"（本地已经有了，勾了会冲突）
4. Public → Create repository

**方式 B：gh CLI（需要先登录）**

```powershell
gh auth login          # 浏览器登录一次
gh repo create <名字> --public --source . --remote origin --push
```

---

## 三、推送

**一条命令（推荐）**：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\publish.ps1 -Name <名字>
```

它会先跑 preflight，通过后设 remote、改名 main、推送。加 `-DryRun` 只演练不推。

**手动等价**：

```powershell
cd E:\望舒\projects\memory-refiner
git remote add origin git@github.com:a-air/<名字>.git
git branch -M main
git push -u origin main
```

SSH 已实测可用（`Hi a-air!`），22 与 443 都通。

---

## 四、推完之后（十分钟内做完）

1. **仓库描述**（About 右侧齿轮）：
   `记忆精炼 + 探针验证 + 事实版本链 · Memory refiner with probe-based verification`
2. **Topics**：`llm` `agent` `memory` `context-compression` `evaluation` `python`
3. **Release**：`v0.1.0`，标题 `v0.1.0 — 探针验证与版本链`
4. 把 README 里 `a-air/xxx` 的占位链接换成真实地址

---

## 五、回滚

推错了不要慌：

```powershell
# 删除远端仓库（网页 Settings → Danger Zone → Delete）
# 或只撤本地 remote
git remote remove origin
```

---

## 六、发布前必须回答的三个问题

- [ ] 仓库里有没有任何真实的人名、机构、病历、联系方式？→ `preflight.ps1` 会扫
- [ ] README 里的每个数字，是不是都能复现？→ 是（`tests/` 里就是那两组）
- [ ] 如果明天有人提 issue 说"这分数没意义"，我能不能回答？→ 能：分数是代理指标，探针才是真指标，README 里写着了
