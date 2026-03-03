## Adaptive Moderation Agent – Perception Stage

This project is a **learning-based moderation agent** that uses **multimodal LLM perception** and **engagement signals** to adaptively choose content intervention actions over time.

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

