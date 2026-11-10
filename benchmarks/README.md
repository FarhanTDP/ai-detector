# Detector benchmark

Why ScribeScan uses **desklib + tropa-mini with 500-word windows** instead of the original
`Hello-SimpleAI/chatgpt-detector-roberta` with 200-word windows.

## The problem

The original model is a 2023 RoBERTa trained on ChatGPT-3.5 answers (HC3). On AI-written
**stories** from current models, it often scored them under 20%, so the app labelled them
"Human-Written". It also failed the other way, flagging many genuinely human stories.

## Test set

| Set | Count | Source |
|---|---|---|
| AI | 10 | Written by a current frontier LLM (Claude): 8 stories/narratives in different genres, 1 essay, 1 blog post. One deliberately imitates casual Reddit-style writing. Committed in `data/ai/`. |
| Human | 60 | r/WritingPrompts stories (2018, before ChatGPT). Downloaded by `fetch_human_stories.py`; not committed. |

The **false-positive rate on human text** matters as much as catching AI. If an innocent
writer is shown "Possibly AI-Assisted", the detector has failed.

## Results

*Measured on an RTX 3050 8 GB, fp16. "Flagged" = shown with a percentage (score ≥ 20%)
instead of "Human-Written".*

### Step 1: 9 models, 30 human stories, app's original 200-word windows

| Model | AI caught | Human flagged | Notes |
|---|---|---|---|
| Hello-SimpleAI/chatgpt-detector-roberta *(original)* | 7/10 | **13/30** | Misses literary and letter-style AI text; flags many humans |
| fakespot-ai/roberta-base-ai-text-detection-v1 | 10/10 | 23/30 | Over-flags humans |
| Oxidane/tmr-ai-text-detector | 6/10 | 8/30 | |
| rasbt/ai-text-detector-modernbert | 0/10 | 0/30 | Unusable on short windows |
| tabularisai/ai-text-detection | 4/10 | 0/30 | Too conservative |
| AICodexLab/answerdotai-ModernBERT-base-ai-detector | 5/10 | 2/30 | |
| ShantanuT01/vanguard-ai-text-detector | 10/10 | 13/30 | |
| **desklib/ai-text-detector-v1.01** | **10/10** | 5/30 | Lowest AI score 0.74 |
| **wasitaigeneratedcom/tropa-mini-ai-text-detector** | **10/10** | 16/30 | Every AI score 1.00, but noisy on short human windows |

### Step 2: window size (desklib / tropa-mini)

| Window | desklib human flagged | tropa human flagged | desklib lowest AI score |
|---|---|---|---|
| 200 words | 5/30 | 16/30 | 0.74 |
| 350 words | 4/30 | 6/30 | 0.85 |
| 500 words | 4/30 | 3/30 | 0.96 |

Short windows give the models too little context, and noise in a single window drags the
average up. Keeping the original paragraph breaks inside windows made no measurable difference.

### Step 3: ensemble, 500-word windows, 60 human stories

| Setup | AI caught | Lowest AI score | Human flagged ≥ 20% | Human ≥ 50% |
|---|---|---|---|---|
| desklib alone | 10/10 | 0.96 | 6/60 | 2/60 |
| tropa-mini alone | 10/10 | 1.00 | 7/60 | 6/60 |
| average of both | 10/10 | 0.98 | 8/60 | 3/60 |
| **both must agree (min), the default** | **10/10** | **0.96** | **4/60 (7%)** | **1/60** |

The two models mostly make *different* mistakes on human text, so requiring both to agree
removes most false accusations while still catching every AI sample. Cost: about 2 GB VRAM
and about 0.4 s per scan.

## Caveats

- **Small set.** 10 AI texts from one model family is a sanity check, not a certification.
  Add your own files to `data/ai/` and `data/human/` and re-run.
- **Training-data overlap.** desklib and tropa were trained on RAID, which is why RAID wasn't
  used here; it would flatter them.
- **Remaining false positives** are polished, conventional human prose, the hardest case for every detector.
- **Model drift.** tropa-mini's authors report much lower recall on 2026 frontier models and
  "humanizer" tools. No open detector is reliable against deliberate evasion.

## Reproduce

```powershell
.\.venv\Scripts\Activate.ps1
python benchmarks\fetch_human_stories.py
python benchmarks\evaluate.py
# compare a single model:
$env:AIDET_MODELS = "Hello-SimpleAI/chatgpt-detector-roberta"; $env:AIDET_CHUNK_WORDS = 200; $env:AIDET_CHUNK_OVERLAP = 50
python benchmarks\evaluate.py
```
