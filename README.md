## Adaptive Moderation Agent – Perception Stage

This project is a **learning-based moderation agent** that uses **multimodal LLM perception** and **engagement signals** to adaptively choose content intervention actions over time.

hezihang：目前这个代码就只是 文本数据->agent->关于文本有无conflict risk/harmful info等的state
跑法：python run_perception_on_jigsaw.py 然后就会根据dataset生成对应的state 存在某output.jsonl里

### Architecture Overview

- **Stage 1 – Perception (this module)**  
  - Input:
    - Post text
    - Optional media summary (images/video) and safety tags
    - Conversation context (thread summary, parent text)
    - Engagement signals (likes, comments, reports, sentiment, growth, etc.)
  - Output: `PerceptionState` (Python dataclass) with:
    - Content risk scores (toxicity, harassment, hate, self‑harm, sexual)
    - Conflict / escalation, ambiguity, uncertainty, disagreement
    - Engagement‑driven risk
    - Short reasons and a suggested moderation action

- **Stage 2 – Decision & Learning (to be added)**  
  - Will take `PerceptionState` as input and learn a policy over actions:
    - `do_nothing`, `downrank`, `add_friction`, `throttle`, etc.

### Quick Start

1. **Setup environment:**
   ```bash
   conda activate agents  # or your preferred environment
   pip install openai python-dotenv numpy matplotlib seaborn scikit-learn
   ```

2. **Configure API keys:**
   Create a `.env` file in the project root:
   ```env
   OPENAI_API_KEY=sk-...
   OPENAI_BASE_URL=https://tritonai-api.ucsd.edu
   OPENAI_MODEL=api-gpt-oss-120b
   ```

3. **Run perception on Jigsaw dataset:**
   ```bash
   python run_perception_on_jigsaw.py
   ```
   This generates `jigsaw_perception_output.jsonl` with perception states for each comment.

4. **Evaluate perception results:**
   ```bash
   python evaluate_perception.py
   ```
   This creates comparison plots in `evaluation_plots/`:
   - ROC curves for each label type
   - Precision-recall curves
   - Score distributions (toxic vs non-toxic)
   - Correlation heatmap between scores and labels
   - Confusion matrices at different thresholds
   
   Also prints summary metrics (AUC, precision, recall, F1) to the console.

### Files

- `perception.py` - Core perception module with `PerceptionAgent` and data structures
- `data_pipeline.py` - Dataset loading utilities for Jigsaw CSV files
- `run_perception_on_jigsaw.py` - Script to run perception over the dataset
- `evaluate_perception.py` - Evaluation script that generates comparison plots
- `jigsaw_perception_output.jsonl` - Output file with perception states (generated)
- `evaluation_plots/` - Directory with visualization plots (generated)

### Contextual Bandit (LinUCB) Demo

Use this to learn an action policy from context + reward instead of directly executing LLM `suggested_action`.

1. Generate perception states:
   ```bash
   python run_perception_on_jigsaw.py
   ```
2. Train/evaluate LinUCB:
   ```bash
   python roll_out_demo.py --data jigsaw_perception_output.jsonl --train-episodes 2000 --eval-episodes 500 --horizon 10 --plot-dir policy_evaluation_plots
   ```

The script reports mean episode return for:
- `always_do_nothing`
- `rule_policy`
- `always_throttle`
- `LinUCB (Greedy)` after online bandit updates

LinUCB context features include:
- environment observation from simulator (`obs`)
- one-hot encoding of `PerceptionState.suggested_action` as a prior
- bias term

It also saves policy evaluation plots (to `--plot-dir`):
- `roc_curves.png`
- `precision_recall_curves.png`
- `score_distributions.png`
- `correlation_heatmap.png`
- `confusion_matrices.png`



文字comments数据->llm给出评价->根据评价选择action->reward给score

文字comments数据（1 添加图片数据，添加帖子数据，现有的是comments数据 不成组）->llm给出评价->根据评价选择action（2 用什么react等method选action，现有是直接llm reason写action）->reward给score（3 找其他办法定义reward 现在是随便写的hardcode定义->（4 现在只算了reward 没有update policy，写一个根据reward update policy）

