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