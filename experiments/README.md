# 实验

`experiments/` 只保存实验编排代码、配置和结果索引。核心优化器在
`SemCache/optimizer/`，共同 SQL 与数据在 `benchmarks/`。

## 目录

| 目录 | 内容 |
|---|---|
| `config/` | 实验、训练、评测和 Teacher 配置 |
| `semcache/` | SemCache 查询、采集、训练、参考答案和监控 |
| `system_comparison/` | PLOP、Sema、Nirvana、LOTUS、Abacus、MOAR 的对比调度 |
| `validation/` | 查询结构、绑定和运行时契约检查 |
| `results/` | 运行生成的报告、日志和模型检查点 |
| `core/` | 实验通用的路径、原子写入和进程工具 |

## 配置

实验配置位于 `config/system/`，按 `config/system/config.json` 的 `include`
组合。Teacher 模板位于 `config/teacher/teacher.example.json`；密钥只从环境变量读取。
核心搜索、采样和 value model 配置仍由 `SemCache/config/` 管理，两套配置不会自动合并。

## SemCache 训练

训练查询位于 `semcache/queries/`，源库位于 `semcache/data/`，参考答案位于
`semcache/reference_answers/`。训练流程是：收集查询计划与运行标签，合并校准统计，训练
value model，再用独立测试查询评测。监控脚本维护采集、训练和损失报告。

```sh
PYTHONPATH=SemCache/optimizer:experiments/semcache:. \
  SemCache/.venv-optimizer/bin/python \
  experiments/semcache/training/launcher.py
```

## 跨系统对比

当前对比使用 5 个数据集、20 条查询和 6 个系统。调度器、后端、执行器、模型传输、报告
和准确率代码均在 `system_comparison/` 中；结果写入 `results/system_comparison/`。
系统的原生优化器和模型适配保持在各自 `systems/` 目录。

```sh
PYTHONPATH=. SemCache/.venv-optimizer/bin/python \
  -m experiments.system_comparison --help
```

## 检查

结构审计不请求模型；它只核对查询字段、依赖和关系边界：

```sh
PYTHONPATH=. SemCache/.venv-optimizer/bin/python -m experiments.validation
```

训练、评测和对比结果均以运行目录为单位保存。结果中的执行明细和模型日志是实验产物，
不与源代码说明混写。

## 跨系统代码分区

跨系统调度代码位于 `system_comparison/`：`scheduling/` 管理任务，`data/` 管理输入，
`execution/` 负责 SQL 与结果捕获，`backends/` 适配各系统，`transport/` 记录模型请求，
`reporting/` 生成进度和计时，`accuracy/` 负责参考答案评分，`diagnostics/` 提供无模型检查。
系统原生优化器仍在 `systems/`，结果写入 `results/system_comparison/`。

系统执行和参考答案评分分开；固定字段先做多重集匹配，文本字段交 Teacher 判断，聚合值按
相对误差评分。错误、超时和不支持状态不会伪造成空结果。
