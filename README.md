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

- **Stage 2 – Decision & Learning**
  - Takes `PerceptionState` as input and chooses among:
    - `do_nothing`, `downrank`, `add_friction`, `throttle`
  - Baseline now includes a **ReAct-based action chooser** in `action_chooser.py`
    that reasons over the structured state with a small local tool loop instead of
    directly trusting the `suggested_action` string emitted by Stage 1.

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

5. **Run the ReAct action chooser on saved perception output:**
   ```bash
   python run_action_chooser_on_output.py
   ```
   This generates `jigsaw_action_output.jsonl` with an additional `action_decision`
   field containing:
   - `action_id`
   - `action_name`
   - `reasoning`
   - `used_fallback`
   - `trace`

6. **Compare rollout policies in the simulator:**
   ```bash
   python roll_out_demo.py
   ```
   If API credentials are configured, the demo now includes a `ReAct Action Chooser`
   policy in addition to the fixed baseline policies.

### Files

- `perception.py` - Core perception module with `PerceptionAgent` and data structures
- `data_pipeline.py` - Dataset loading utilities for Jigsaw CSV files
- `run_perception_on_jigsaw.py` - Script to run perception over the dataset
- `action_chooser.py` - ReAct-based Stage 2 action selection module
- `run_action_chooser_on_output.py` - Script to attach ReAct actions to saved perception output
- `evaluate_perception.py` - Evaluation script that generates comparison plots
- `jigsaw_perception_output.jsonl` - Output file with perception states (generated)
- `jigsaw_action_output.jsonl` - Output file with ReAct action decisions (generated)
- `evaluation_plots/` - Directory with visualization plots (generated)



文字comments数据->llm给出评价->ReAct根据评价选择action->reward给score

文字comments数据（1 添加图片数据，添加帖子数据，现有的是comments数据 不成组）->llm给出评价->ReAct根据评价选择action（2 现在不是直接让第一阶段LLM写action，而是单独的action chooser）->reward给score（3 找其他办法定义reward 现在是随便写的hardcode定义->（4 现在只算了reward 没有update policy，写一个根据reward update policy）
