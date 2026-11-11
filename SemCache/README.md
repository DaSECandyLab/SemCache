# SemCache

SemCache 是面向前缀缓存感知语义查询处理的 DuckDB 研究系统。关系算子和语义
Filter、Projection、Join 在同一执行框架中运行；Python 优化器负责计划、采样、特征和
value model，C++ 扩展负责搜索与执行状态，SGLang 提供模型服务。

## 组件

| 路径 | 内容 |
|---|---|
| `semantic-duckdb/` | DuckDB 语义扩展、提示词、模型通信和运行时 |
| `beam-search-duckdb/` | Beam search、采样、统计和缓存历史扩展 |
| `optimizer/` | 计划状态、特征、采样、搜索和训练接口 |
| `statistics/` | bottom-k 前缀统计和跨字段摘要 |
| `config/` | 核心搜索、采样、模型和 Teacher 配置 |
| `../experiments/` | 训练、参考答案、跨系统对比和结果 |
| `../benchmarks/` | 共同 SQL、数据和数据库 |
| `../sglang/` | 模型服务启动和缓存补丁 |

## 构建

```sh
git submodule update --init --recursive
python3 -m venv .venv-tools
python3 -m venv .venv-optimizer
.venv-optimizer/bin/python -m pip install -r optimizer/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cpu
make release
```

`make release` 构建 DuckDB、语义扩展、beam-search 扩展和 Python 绑定；正常查询、采样和
训练使用 Release 构建。格式与静态检查：

```sh
make format-check
make optimizer-typecheck
git diff --check
```

## 运行

模型服务配置与端点见 `../sglang/README.md`。优化器入口、训练入口和评测入口见
`../experiments/README.md`；共同 SQL 结构见 `../benchmarks/README.md`。

前缀统计表示可复用潜力，不等同于在线 KV 命中。bottom-k 的 SHA-256 是统计排名算法的一部分，
与文件完整性校验无关。
