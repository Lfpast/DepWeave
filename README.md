# DepWeave

DepWeave 用两层记忆辅助小模型在 Python 仓库里定位实体和核对跨文件依赖。研究问题、证据边界与实验设计见 [doc/v2_plan.md](doc/v2_plan.md)。

`globalMem/` 索引仓库快照，建立定义、调用、导入和继承关系，并通过 MCP 返回候选实体、关系源码位置与未解析项。`localMem/` 保留 PixelMem 的抽取、缓存和依赖推导能力；根目录的 `depweave/` 将局部证据与全局导航组合，在模型输入预算内选择证据。两套 `pixelmem` 包分别运行在本地评测进程和 MCP 子进程中。

## 运行评测

先安装与机器匹配的 PyTorch，再安装依赖。Qwen3-4B 权重和两个数据集需要预先放在本地；推理代码不会下载模型。

```bash
cd /home/jackson/python/DepWeave
python -m pip install -r requirements.txt
export QWEN3_4B_PATH=/path/to/local/Qwen3-4B
python -m experiments.exp56_repoqa_python_full --limit 1
python -m experiments.exp58_v5_depeval_full --limit 1
```

去掉 `--limit` 分别运行 RepoQA Python 全部 100 条函数定位样本和 DependEval Task 2 中 166 条 3–5 文件样本。主输入上限为 4096 token；可用 `--max-input-tokens 2048` 或 `8192` 做敏感性分析。结果写入 `results/qwen3-4b/`，其中有快照 ID、索引覆盖、已选证据、缺口、输入 token 和耗时。RepoQA 同时记录函数名准确率与完整实体 ID 准确率；DependEval 保留顺序精确匹配。`exp59`、`exp60` 是旧的局部 primitive dump 消融，不代表完整 DepWeave 评测。

RepoQA 的图覆盖数据集给出的仓库文件。DependEval 数据只提供每题的 3–5 个文件，因此该评测的全局图只覆盖题目切片；结果中的 `source_scope` 明确标记这一点，不能把它当成整仓库导航实验。

```bash
python -m unittest discover -s localMem/tests -v
python -m unittest discover -s tests -v
```

第二条测试会实际启动 globalMem MCP 服务，用确定性的假模型检查跨层调用。它验证接口和证据传递，不产生 Qwen 准确率。当前仓库还没有新方案的 Qwen3-4B 实测分数，也没有独立标注的跨文件关系真值；关系可靠性与论文增益仍需按计划评测。历史单层结果见 [localMem/README.md](localMem/README.md)，不能直接当作新方案结果。
