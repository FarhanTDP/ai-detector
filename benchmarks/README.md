# Detector benchmark

Why ScribeScan uses **tropa-mini + desklib + vanguard with 500-word windows** instead of the original
`Hello-SimpleAI/chatgpt-detector-roberta` with 200-word windows. The setup was chosen in two rounds;
round 2 (below) supersedes the round-1 default.

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

## Round 2: wider test set, new ensemble rule

Round 1 only used AI text from one model and human fiction. Round 2 added:

- **8 more AI texts** written by a second, newer model (Claude Fable) in styles round 1 lacked: dialogue, product
  review, work email, technical explainer, humour, news report, memoir, sci-fi. 18 AI texts in total.
- **60 more human texts** from other domains: 30 IMDB reviews (2011) and 30 BBC news articles (XSum, 2010-2017).
  120 human texts in total.
- **36 half-and-half documents**: about 300 human words joined to about 300 AI words, in both orders.

### The round-1 default missed the new AI texts

| Rule (500-word windows) | AI detected | AI at 80%+ | Human flagged (of 120) | Human at 50%+ |
|---|---|---|---|---|
| min(desklib, tropa), the round-1 default | **14/18** | 14 | 4 | 1 |
| tropa-mini alone | 18/18 | 18 | 7 | 6 |
| desklib alone | 14/18 | 14 | 13 | 5 |
| vanguard alone | 18/18 | 18 | 63 | 44 |
| average of desklib and tropa | 18/18 | 14 | 12 | 3 |
| middle score of desklib, tropa, vanguard | 18/18 | 18 | 10 | 3 |
| **tropa, confirmed by desklib or vanguard** | **18/18** | **18** | **7** | **3** |

desklib scored four of the new, more natural-sounding AI texts (dialogue, humour, memoir, sci-fi) at 3-8%, so
"both must agree" cleared them. tropa-mini caught all 18 but is overconfident on some human fiction. Requiring
tropa to be **confirmed by at least one of the other two** keeps its recall and trims its confident mistakes.
vanguard is useless alone (it flags half of all human text) but works as a second opinion.

### Half-AI documents

Large windows blend the two halves, so the document often came out "Human-Written":

| Document score | Half-AI docs flagged (of 36) | Human flagged (of 120) |
|---|---|---|
| mean of 500-word windows only | 25 | 7 |
| mean of 150-word windows only | 36 | 32 |
| **500-word windows, plus the fine score when a 150+ word passage scores 80%+** | **30** | **8** |

### Final numbers through the app (`evaluate.py`)

| | Result |
|---|---|
| AI texts detected | **18/18** |
| Human texts flagged | **8/120 (7%)**: 8 of 60 stories, 0 of 30 reviews, 0 of 30 news articles |
| Human texts shown at 50%+ | 5/120 |
| Half-AI documents flagged | 30/36 |
| VRAM / scan time | about 3 GB / 1-2 s |

This is a deliberate trade against round 1: four more AI texts and five more half-AI documents are caught, at
the cost of four more human stories flagged. Set `AIDET_ENSEMBLE=min` and `AIDET_PARTIAL_RUN_WORDS=0` to get the
stricter round-1 behaviour back.

### Section scores

On the half-and-half documents that were flagged, sections from the human half averaged 14% and sections from
the AI half averaged 94% (150-word windows, overlap 100). Sections right at the join are the uncertain ones.

## Caveats

- **Small set.** 18 AI texts from two models of one family is a sanity check, not a certification.
  The ensemble rule was also chosen on this same set, so expect somewhat worse numbers on new text.
  Add your own files to `data/ai/` and `data/human/` and re-run.
- **Training-data overlap.** desklib and tropa were trained on RAID, which is why RAID wasn't
  used here; it would flatter them.
- **Remaining false positives** are polished, conventional human prose, the hardest case for every detector.
- **Model drift.** tropa-mini's authors report much lower recall on 2026 frontier models and
  "humanizer" tools. No open detector is reliable against deliberate evasion.

## Reproduce

```powershell
.\.venv\Scripts\Activate.ps1
python benchmarks\fetch_human_stories.py      # 120 human texts
python benchmarks\evaluate.py
# compare a single model:
$env:AIDET_MODELS = "Hello-SimpleAI/chatgpt-detector-roberta"; $env:AIDET_CHUNK_WORDS = 200; $env:AIDET_CHUNK_OVERLAP = 50
python benchmarks\evaluate.py
```
