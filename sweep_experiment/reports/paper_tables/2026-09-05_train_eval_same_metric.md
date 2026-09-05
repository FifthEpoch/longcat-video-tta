# Train on a quality signal, evaluate on the same family (2026-09-05)

Not a submit. Literature after the user asked whether
Hypothesis 2 (distill with official Dynamic Degree +
Imaging Quality, then report those) is metric hacking.
Canvas: `canvases/train-eval-same-metric.canvas.tsx`.

No GPU. Do not start 8-GPU DMD. Hypothesis 2 as “the
loss *is* the RAFT bit” stays not-a-title. This note
only answers: does the field already do train/eval on
the same quality family, and what version got into
CVPR / NeurIPS.

---

## The SOTA gap is real (us and them)

VBench added Dynamic Degree because the other temporal
dims (subject, flicker, smoothness) **reward a still**.
Huang et al., CVPR 2024: a static clip can look
consistent and smooth; Dyn is the counter-metric.

Our cite-128 (V2V, `metadata.csv`): Self Forcing
**32.8%** (42/128) dynamic, Rolling **28.9%** (37) —
the sink pays a Dyn tax to hold identity. Always-search
**50.8%** (65) at Imaging Quality 72.19. Freeze +
sharpen is the 30 s signature we already measured.

The field says the same thing on T2V MovieGen / VBench-Long:

- Relax Forcing cites Rolling Dyn **32.71** vs their **65.67**.
- Reward Forcing (CVPR 2026 Highlight): vanilla DMD cannot
  prioritize motion; LongLive is “high consistency, low
  dynamism”; their Re-DMD lifts long Dyn to **66.95**
  vs LongLive **35.54**.

So yes: using Dynamic Degree as the official motion
proxy, and treating low Dyn as the SOTA long-horizon
gap, is already the table those papers argue on.

---

## Two different self-references

| Pattern | Train signal | Eval signal | Reviewer temperature |
|---|---|---|---|
| **A. Related family** | A motion / preference / VLM score that is *not* the official classifier | Official VBench, including Dyn and IQ | Accepted. Reward Forcing, T2V-Turbo, VideoDPO |
| **B. Identical bit** | Official VBench Dynamic Degree (RAFT 0/1) or MUSIQ | The same numbers | Authors themselves show hacking (DOLLAR). This is H2 as written. |

The user asked for B. The published *motion* papers are
almost all A. B exists, and when it is Dyn itself the
paper usually treats it as a caution.

---

## Pattern A — accepted, same *family*

**Reward Forcing** (Lu et al., CVPR 2026 Highlight,
[2512.04678](https://arxiv.org/abs/2512.04678)).
Train: Re-DMD *weights* the DMD gradient by a
**VideoAlign** vision-language motion score. They do
not backprop the RAFT bit. Eval: VBench / VBench-Long
**Dynamic Degree** is the headline lift (66.95, +88%
vs LongLive). Also report subject, IQ drift, Qwen-VL,
user study. Closest cousin to “we wanted more motion
on the official table.”

**T2V-Turbo** (Li et al., NeurIPS 2024,
[2405.12467](https://arxiv.org/abs/2405.12467)).
Train: mix of differentiable **HPSv2.1** (image-text /
aesthetic) and **ViCLIP / InternVideo2** (video-text).
Eval: full **VBench** (Aesthetic and Imaging Quality
are in the same family as HPS; Semantic Score is in
the same family as CLIP). They add a human study
because the table is the reward’s cousin.

**T2V-Turbo-v2** ([2410.05677](https://arxiv.org/abs/2410.05677)).
Train: same reward mix plus **motion guidance** pulled
from the training videos into the teacher ODE.
Eval: they explicitly check “motion-related metrics
from VBench and T2V-CompBench.” SOTA VBench total 85.13.

**VideoDPO** (Liu et al., CVPR 2025,
[2412.14167](https://arxiv.org/abs/2412.14167)).
Train: **OmniScore**, which they say is built from the
VBench taxonomy (intra-frame quality, inter-frame
consistency, text alignment). Eval: **VBench total**
is the primary table, plus HPS(V) and PickScore as
held-out preference models. This is the most
self-referential A that still got in: the train
taxonomy *is* the eval taxonomy, but the train
classifier is not the official RAFT/MUSIQ code.

**VADER** (Prabhudesai et al.,
[2407.08737](https://arxiv.org/abs/2407.08737)).
Train: backprop **HPS / PickScore / aesthetics /
VideoMAE**. Eval: those same reward curves, plus a
Mechanical Turk pairwise study. They plot the train
reward as the result.

---

## Pattern B — identical metric, older and riskier

**DOLLAR** (Zhang et al.,
[2412.15689](https://arxiv.org/abs/2412.15689)).
They *did* latent-reward fine-tune on **VBench
Dynamic Degree** and then report Dyn. Dyn rose to
**0.97**. They write that samples grow a **“noise
flow”** and Imaging Quality dies. That is the paper
that already ran H2-as-written and called it a
trade-off, not a win. Same signature as our mixctx
twitch (Dyn 8/8, flicker 0.978).

**SCST** (Rennie et al., CVPR 2017). Image captioning.
Train: **CIDEr**. Eval: CIDEr (and BLEU / METEOR).
The field lived on this for years. Later people
complained about n-gram gaming and moved to learned
rewards / human. The accepted pattern was “optimize
the community table + show the other automatic
scores.”

**DDPO** (Black et al., 2023) and the image RLHF
line. Train: CLIP or LAION aesthetic. Eval: the same
rewards, plus human. Reviewers now expect the human
column because Goodhart on CLIP is known.

**FID checkpoint pick.** Almost every GAN / diffusion
paper selects the checkpoint by FID and reports FID.
Nobody pretends that is a different metric. The
defense is a second number (IS, precision/recall,
human).

---

## What this means for Hypothesis 2

The self-referential *shape* is not automatically a
reject. Reward Forcing is a Highlight for “train a
motion signal, report official Dyn.” VideoDPO is CVPR
for “train a VBench-shaped score, report VBench.”

The version a reviewer will call hacking is **DOLLAR /
H2-as-written**: the loss *is* the RAFT 0/1 (or MUSIQ),
and the table *is* that bit. The authors of that
experiment already showed the cheap Pareto (noise flow).
We already showed it at test (twitch).

If Hypothesis 2 is kept at all, the published analog
is Reward Forcing’s split: **train VideoAlign (or
another motion judge that is not RAFT), evaluate
official Dyn + IQ + subject + flicker + a human or
Qwen-VL column.** That is occupied as a T2V student
paper unless the label or the protocol (V2V leftover)
is actually different. It is not “put VBench in the
loss.”

---

## Do not

Launch 8-GPU DMD. Remake cite-128. Treat official
RAFT Dynamic Degree as a differentiable reward.
Reopen mix / FIFO / leftover ρ.
