# 数据快照

`candidates.csv` 和 `generation_times.csv` 是本地静态快照，运行分析不需要原始作业6目录。

## 文件与字段

* `candidates.csv`：816条脱敏候选，覆盖24个任务、20个组、两次历史运行、baseline／ordinary／diverse 三种策略。保留候选译文、术语规则通过状态、选择器检查与分数、独立评审分数及七项状态。`case_id`、`cluster_id` 和 `candidate_id` 均是新生成的编号；`index` 从0开始。
* `generation_times.csv`：与每条候选一一对应的816条生成请求记录。保留实测服务耗时、token 计数、结束原因和重试次数；不含请求／响应标识及时间戳。
* `review_batch.jsonl`：4条原文、8份候选译文，文件内仅含匿名复核编号、宽泛领域、原文和无标签译文。此文件不提供候选评分、自动判定、错误分类或选择排名。
* `astra_review.json`：主代理使用 GPT-6.2 Astra 对复核包逐条作出的主观判断和可核查理由，原样保留；该目的性样本不覆盖全候选池，也未并入自动统计。
* `review_index.json`：将复核包 A/B 编号映射到匿名任务与候选编号，不包含人工判断。
* `manifest.json`：数据来源、去标识映射和数量摘要。

数据由作业6固定测试集的 `results/analysis/candidate_metrics.csv`、`results/api_records/generation_*.json` 及 `data/dataset.json` 的四条复核样例派生。时间字段保留的是源记录中的单请求 `seconds`，不是并发队列时延或完整管线延迟。主快照未拷入题目原文、源文 URL、文档标题、提示词或 API 请求内容。复核文件保留少量源文是为了由 GPT-6.2 Astra 独立判断译文。

主快照继承作业6的自动筛选和自动评审标签；这些标签不是人工金标。Astra 复核仅覆盖4个目的性样例。所有历史候选在作业6报告和分析中已被查看，不能将本次回放称为事前独立测试。
