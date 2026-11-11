# Benchmark 数据、查询与数据库

本目录维护共同 benchmark 的活动 SQL、源数据和独立数据库。当前工作负载是 5 个数据集、20 条论文来源查询；六个系统各有一套翻译程序，共 120 个程序。查询和数据说明集中在本文件，避免多个子目录 README 产生重复定义。

## 当前查询集

| 数据集 | 查询 | 数量 |
|---|---|---:|
| Estate | Q5、Q9、Q10、Q11、Q12 | 5 |
| Movies/Nirvana | Q9、Q10、Q11、Q12 | 4 |
| Movies/Sema | Q10 | 1 |
| Appstore | Q7、Q8、Q9 | 3 |
| TPC-H | Q16、Q17、Q25、Q26、Q27、Q28、Q30 | 7 |

原测试集已删除 Estate Q7/Q8、Movies/Nirvana Q5/Q7/Q8 和 Movies/Sema Q13/Q14/Q15。Appstore Q7/Q8/Q9 与 Movies/Sema Q10 使用论文和技术报告附录中的条件；代码仓库同编号变体和 CoinMarketCap 查询不纳入。

原始查询位于 [queries/original/](queries/original/)，六个系统的翻译位于：

- [SemCache](queries/translated/semcache/)、[Sema](queries/translated/sema/)、[PLOP](queries/translated/plop/)：SQL；
- [Abacus](queries/translated/abacus/)、[Nirvana](queries/translated/nirvana/)、[Lotus](queries/translated/lotus/)：Python。

TPC-H 使用稳定键排序和当前 Q17 订单—客户关联修订。各系统只引用一次的 CTE 已内联；显式 MATERIALIZED 和多次引用的共享结果保留。数值 projection、确定性聚合、NULL、重复行、ORDER BY/LIMIT 和关系字段依赖按共同契约执行。

## 数据集

从 [Google Drive](https://drive.google.com/file/d/1jjHYS8-VLoP8TDs1XTEnV7sz8oD7gKUi/view) 下载并解压到项目根目录的 `benchmarks/datasets/`。数据集和生成的 `databases/` 不纳入 Git。

| 数据集 | 输入规模与位置 |
|---|---|
| Estate / Nirvana | 1,041 行，`datasets/estate/data/nirvana/`；Parquet 中的 image 列不发送给模型 |
| Movies / Nirvana | 249 行、22 列，`datasets/movies/data/nirvana/movie_data.csv` |
| Movies / Sema | 7 张表、合计 35,159 行，`datasets/movies/data/sema/` |
| Appstore / Sema | 2 张表、合计 75,126 行，`datasets/appstore/data/sema/` |
| TPC-H | SF=0.005，8 张表，`datasets/tpch/data/` |

两套 Movies 数据独立保存。Estate 保留完整文本和全部行；Appstore 保留稳定源行标识；TPC-H 不使用 LIMIT 截取输入。`datasets/paper-workloads/` 只保留来源数据说明和兼容路径。

## 独立数据库

六个系统各有五个数据库，位于 `databases/<system>/`：`semcache`、`sema`、`plop`、`abacus`、`nirvana`、`lotus`。数据库包含相同实体表和兼容的 `raw_<table>` 视图；SQL 系统使用 DuckDB，Python 系统使用其执行适配层。

| 数据库 | 实体表 | 总行数 |
|---|---:|---:|
| Estate | 1 | 1,041 |
| Movies/Nirvana | 1 | 249 |
| Movies/Sema | 7 | 35,159 |
| Appstore | 2 | 75,126 |
| TPC-H | 8 | 43,531 |

TPC-H 的公开 SF=0.01 Parquet 仅作为来源记录保留；当前对比输入使用 SF=0.005。数据库构建不调用模型，并核对 schema、NULL、重复行、行数和完整结果多重集。

## SemCache 使用

SemCache 训练 SQL 位于 `../experiments/semcache/queries/`，训练源库位于 `../experiments/semcache/data/`。测试执行直接读取 `queries/translated/semcache/` 中的 SQL；训练和评测不依赖本目录的历史 metadata 文件。

结构审计使用：

```bash
PYTHONPATH=. SemCache/.venv-optimizer/bin/python -m experiments.validation
```

该审计只检查 SQL 结构、字段和关系契约，不代表真实模型准确率。正式训练、参考答案和评测入口见 [experiments/README.md](../experiments/README.md)。

## 目录约定

新增或修改查询时，只维护对应 SQL、来源目录和本文件中的活动清单。数据库准备脚本位于 `databases/prepare.py`；实验运行结果位于 `../experiments/results/`，不放回 benchmark 源目录。
