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

## Adaptive Moderation Agent – Thinking Method

### Overview

For each step, the agent will:
   - Constructs a thinking prompt from the current observation and memory.
   - Calls the LLM to 'think' about the situation and propose an action.
   - Parses the structured output and returns an integer action.

### Functions

#### Data Structures

- **`Step`** — Dataclass storing one timestep of episode memory: timestep index `t`, observation dict, chosen action, reward received, and environment info dict.

#### Prompt Construction

- **`obs_to_dict(obs)`** — Converts the raw numpy observation array into a named dictionary matching the environment's field order (`harm`, `conf`, `ambiguity`, `uncertainty`, `V`, `E_norm`, `R_norm`, `S`).

- **`build_user_prompt(obs_dict, history, cumulative_reward, t)`** — Assembles the per-step user message sent to the LLM. Includes the current observation values, cumulative reward, a sliding window of the last 5 step outcomes, and content-specific hints (e.g., flagging clearly benign or clearly harmful content).

#### Response Parsing

- **`parse_action(text)`** — Extracts the integer action (0–3) from the LLM's `<action>` XML tag. Falls back to scanning the last 100 characters for any digit 0–3 if the tag is malformed. Defaults to 0 (`do_nothing`) if parsing fails entirely.

- **`parse_thinking(text)`** — Extracts the chain-of-thought reasoning from the LLM's `<think>` XML tag for logging and inspection.

#### Client Setup

- **`get_client()`** — Builds an `AsyncOpenAI` client from environment variables (`OPENAI_API_KEY` or `MOD_AGENT_OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`). Defaults to the UCSD Triton AI proxy endpoint.

#### Policy Class

- **`ThinkingPolicy`** — The main policy class. Maintains episode memory (action history, cumulative reward, timestep counter) and exposes two calling conventions:
  - `__call__(obs)` — Synchronous interface matching the `policy(obs) → int` signature used by the environment rollout loop.
  - `act_async(obs)` — Native async interface for use inside async event loops.
  - `record_step(action, reward, info)` — Must be called after each `env.step()` to feed the outcome back into episode memory so the LLM can reason about trends.
  - `reset()` — Clears episode memory at the start of each new episode.

### Quick Start

#### 1. Environment Setup

Set up env vars in your environemnt:
```
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://tritonai-api.ucsd.edu
OPENAI_MODEL=api-gpt-oss-120b
```

#### 2. Run Directly
```bash
python thinking.py
```

This runs 3 episodes with balanced toxic/non-toxic sampling and prints the full thinking trace for each step.

#### 3. Use as a Module
```python
from sim.data import load_items
from sim.env import ModerationSimEnv
from thinking import ThinkingPolicy, run_llm_episode

# Load data and create environment
items = load_items("jigsaw_perception_output.jsonl")
env = ModerationSimEnv(items, T=T, seed=SEED, pos_frac=0.5)

# Create policy and run one episode
policy = ThinkingPolicy(temperature=0.2, verbose=True)
total_reward = run_llm_episode(env, policy, "Your agent")

# Or use in a custom loop
policy.reset()
obs = env.reset()
for t in range(env.T):
    action = policy(obs)
    obs, reward, done, info = env.step(action)
    policy.record_step(action, reward, info)
    if done:
        break
```

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
- `thinking.py` - Initial implementation of the thinking method for decision & learning stage 



文字comments数据->llm给出评价->ReAct根据评价选择action->reward给score

文字comments数据（1 添加图片数据，添加帖子数据，现有的是comments数据 不成组）->llm给出评价->ReAct根据评价选择action（2 现在不是直接让第一阶段LLM写action，而是单独的action chooser）->reward给score（3 找其他办法定义reward 现在是随便写的hardcode定义->（4 现在只算了reward 没有update policy，写一个根据reward update policy）
