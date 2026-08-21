# Sampling-space reading of the nine TTA/TTC ideas (2026-08-21)

**Status:** analysis only. Do not submit until `v2v_panda_lineage_8v`
(16140808–816) finishes.

**Lock we score against:** Wan 1.3B + Self-Forcing causal DMD; V2V 30 s
from a real prefix; **sampling-space controller, no weight TTA / no
LoRA-at-test-time.** Contribution is a drift-gated, GT-free intervention
on the *trajectory*, not on θ.

**What we already know (do not reopen):**
- Unconditional prefix-match search (`seed_bon`) is an identity damper
  at N=32: subject +0.039, tail motion −8.8%, Dyn 0.
- Two prefix populations: *live* (0007, prefix_motion 0.070 → notta
  collapsed, search recovered) vs *still* (0002/0003, prefix ~0.0008 →
  notta invented motion, search damped).
- CFG and FlowMatch `shift` do **not** move pixels on this DMD student.
- Replay-sink without a rerope kernel is a no-op.
- `late_bon` (search only at the horizon) missed the recoveries in
  chunk 0.
- `quiet_bon` inverted the live/still gate and lost 19%.

So any proposal that (a) updates weights, (b) two-sided-matches motion
to a still prefix, or (c) assumes a live CFG/shift/per-frame-noise knob
on *this* student, is not in our problem space until a probe shows the
knob moves pixels.

Ratings are **research value for *our* V2V sampling-space paper**, not
intrinsic quality of the idea in general.

| # | Idea | Fit? | Rating | One-line |
|---|---|---|---:|---|
| 1 | Pseudo-future validation | **Yes**, if δ is a seed/policy not θ | **9** | Best missing gate. `live_bon` is a cheap proxy. |
| 2 | Self-rollout TTA | **No as written** (weights). Sampling rewrite ≈ #1 | **3** | Principle already in AR chunk search; optimizing θ is locked out. |
| 3 | Noise-calibrated TTC | **Yes**, trigger not weights | **7** | Excellent fit *if* ε-stats are non-degenerate on 4-step DMD. Probe first. |
| 4 | Adaptive rolling schedule | **Not on this student.** Maybe RF host | **4** | Same class as the dead shift/CFG probe. Do not fake DFoT. |
| 5 | Prefix-anchored path correction | **Yes**, with appearance ≠ motion | **8** | We already ran the naive version and it froze Dyn. The *split* is the paper residue. |
| 6 | Lookahead branch-and-correct | Weak on 4-step DMD | **4** | Collapses to seed BoN / CachedSearch. Wait on Rolling Forcing. |
| 7 | Trust-region TTA | **No** (weights). Soft reject is #1+#5 | **3** | Locked. Sampling analogue = reject a pick that damps motion. |
| 8 | Horizon-dependent δ | **Contradicted** by `late_bon` | **2** | Recoveries live in chunk 0, not at the far horizon. |
| 9 | Hybrid router | **Yes as framing**, TTA arm out | **8** | Paper system: {none, search, latent-correct, RF-schedule} gated by #1/#3. |

---

## 1. Pseudo-future validation — 9/10, **do this**

As written, δ is an adaptation *parameter*. That is TTA and out of lock.

**Sampling-space rewrite (this is the method):** split the 9-latent
prefix into A = first 6 latents (~1.4 s) and B = last 3 (~0.7 s).
For each candidate intervention (seed, history, “do nothing”),
condition on A, generate B, score against the *real* held-out B.
Keep the intervention only if it beats notta on B by γ; otherwise
generate the true 30 s tail with notta.

Why it fits:
- We already have per-video heterogeneity. A universal search lost at
  N=32. A gate is the controller.
- The validation set is GT-free for the *unseen* future. The paper can
  still say “no GT of the continuation.”
- Self-Forcing’s lesson (train/test mismatch) is applied at *selection*
  time: we score a real continuation of this video, not prefix
  reconstruction.
- `live_bon` (in flight, 16140808) is the cheap motion-threshold
  cousin. Pseudo-future is the same idea with a task-matched score.

Main risk: **B is short.** Matching 0.7 s may not predict 30 s. That is
an empirical question, not a conceptual hole. If N=8 pseudo-future
picks disagree with 30 s tail ranking, the method dies honestly.

Do **not** use this to choose LoRA / LR / steps. Those arms are TTA.

---

## 2. Self-rollout TTA — 3/10 in our space

The hypothesis is right and we already live it: every later V2V chunk
conditions on generated history, not GT. That is why teacher-forced
prefix matching is the wrong objective.

The *method* as written optimizes δ on that rollout. That is test-time
training. Locked out.

Sampling rewrite (stopgrad previous latents, pick a seed that reduces
error on B under self-history) **is idea 1 with AR generation of B**.
On a 3-latent B there is almost no “later tokens condition on earlier
ˆx.” So #2 does not add a distinct experiment unless we lengthen the
prefix (e.g. 21-latent / 5 s prefix, split 12+9). That would be a new
protocol, not a drop-in on the current 9-latent bake-off.

Do not backprop through the DMD student at test time.

---

## 3. Noise-calibrated TTC (TANGO / Pathwise) — 7/10, **probe then maybe**

This is the cleanest *sampling-space* TTC in the list. No weights.
U_t is a trigger; Δz / ε_init is the correction. Pathwise TTC’s
“correct the stochastic state, not θ” is exactly our lock.

Caveats that are ours, not TANGO’s:
- **4-step DMD.** Predicted noise on a distilled student is often not
  N(0,I). CFG was already dead. If U_t is constant across chunks, the
  trigger is as useless as shift.
- Correcting ε_init **is seed search**. We already have k=4. The new
  scientific object is the *gate* U_t, not another picker.
- Must log ε mean/var on notta N=8 (bake-off mp4s are not enough —
  need a denoise hook). One cheap probe, same 2 videos as the shift
  probe. If U_t does not rise on 0002-style freeze vs 0007-style live
  collapse, stop.

If the probe moves, combine with #1: search only when U_t is high
*and* pseudo-future says search wins.

---

## 4. Adaptive rolling noise schedule — 4/10 here, higher on RF

Rolling Diffusion / Diffusion Forcing need **independent per-frame
noise at inference**. History Guidance needed DFoT training. Our
student is Self-Forcing DMD with a single shared schedule. Shift and
CFG did not move pixels. A sample-specific ρ on *this* checkpoint is
likely another no-op.

**Exception:** Rolling Forcing (job 16140815) *was* trained with
staggered window noise. If that host keeps motion at 30 s, then a
prefix-dependent ρ is a real method — on *their* sampler, not ours.

Do not implement mixed-noise windows on vanilla SF until a 2-video
probe shows pixel change (same bar as the shift probe).

---

## 5. Prefix-anchored latent path correction — 8/10 for the split

Naive version is `seed_bon`: pull the continuation toward the prefix
in a two-sided sum that includes motion. N=32: identity up, Dyn dead.
History Guidance’s vanilla HG did the same thing (consistency↑,
dynamics↓).

The **critical modification in the proposal is the actual method**:
anchor appearance / identity / geometry; do **not** match motion to
the prefix; at most extrapolate motion when the prefix is live.

That is already our locked sentence: *never two-sided-match motion to
a still reference.* Sampling-space form, no gradients required:

- Score = appearance/color/contrast/seam only (drop `|Δmotion|` from
  the sum, or hinge it one-sided: penalize collapse when prefix is
  live, ignore extras when prefix is still).
- `hinge_bon` was a half-step (motion hinge too loose). Redo the
  verifier, don’t add ∇_z.

Gradient-on-z_t through a decoded ˜x_0 is allowed *as sampling-space
TTC*, but it is expensive, 4-step, and will re-freeze if the loss
still contains motion. Discrete seed search with the split score is
the cheaper test of the same hypothesis.

This is the strongest *paper-residue* idea even if live_bon fails:
the N=32 VBench already says prefix-match search is identity control.

---

## 6. Lookahead latent branch-and-correct — 4/10 on this student

Video-T1 / latent beam search / FIFO lookahead need either many
denoising steps or a FIFO/rolling sampler.

We have 4 DMD steps. L-step lookahead is almost a full chunk denoise.
That is `seed_bon` / `cached_bon`, which we already know is an
identity damper when un-gated. CachedSearch ranking survived; quality
did not beat the full k=4 pick.

On Rolling Forcing (in flight) lookahead inside the rolling window is
a different algorithm. Park it until 16140815 says the host moves.

---

## 7. Trust-region TTA — 3/10 as written

Parameter-space. Locked out. The DAS / FIFO “don’t leave the
pretrained distribution” moral is real, and we already saw
unconstrained search leave it (freeze).

Sampling analogue, worth a one-line picker rule not a new job:
accept a searched chunk only if its motion is not far *below* notta
cand0. That is a trust region on the *trajectory*, not on θ. It is
also mostly redundant with #1 and #5.

---

## 8. Horizon-dependent adaptation — 2/10, **empirically wrong here**

The Rolling Diffusion story (more uncertainty farther out → more
intervention later) is the opposite of our V2V traces. `late_bon`
skipped chunk 0, where 0007-style recoveries live, and died.
`good_backtrack` on a dead tail could not unstick.

Near-horizon is *more* constrained by the real prefix, which is
exactly when a sampling intervention can recover collapse. Far
horizon is where the model has already frozen; searching there
rewinds a corpse.

A horizon *gate* we would believe: search chunk 0 if the prefix is
live; never start searching after the tail has died. That is not δ_h
on weights.

---

## 9. Hybrid intervention router — 8/10 as the system, not a first GPU

This is the paper’s controller narrative, not a new primitive.

Allowed arms in **our** A:
- A0 none (`notta`)
- A2 latent / seed TTC (`live_bon`, appearance-only pick)
- A3 branch search (k=4, already have)
- A4 rolling schedule **only if** RF host is live

Forbidden: A1 parameter TTA.

Features q(x) we can actually compute without new weights:
prefix_motion, pseudo-future gain (#1), U_t (#3, after probe).

`live_bon` is a 1-feature router. #9 becomes interesting **after** #1
and #5 have N=8 numbers, not before. Do not train a ΔQ model on N=8.

---

## Combined formulation that is actually ours

Not “Pseudo-Future Adaptive TTA.” That name smuggles weights.

**Pseudo-future gated sampling control:**

```
observed prefix
  → split A | B
  → score {notta, seed, appearance-only pick} on generating B
  → if no candidate beats notta on B: emit notta for the 30 s tail
  → else continue with that pick, never matching motion to a still prefix
```

Optional later: U_t as a cheaper trigger than generating B; RF
schedule as an extra arm if 16140815 keeps motion.

This uses Self-Forcing (rollout mismatch), Rolling Diffusion (regime
dependence), TANGO/Pathwise (sampling-state correction), and Video-T1
(search) without touching θ.

## Do not spend GPUs on yet

- Anything with δ in parameter space, LoRA, last-layer adapter, D_f(θ+δ,θ).
- Rolling ρ on vanilla SF without a 2-clip pixel probe.
- Horizon-increasing adaptation (`late_bon` already falsified).
- Full router, lookahead beam, or ∇_z Pathwise until lineage N=8 is in.

Wait for 16140808 (`live_bon`) — it is the cheap ablation of #1’s gate.
If it already separates 0007 from 0002, pseudo-future is the next
precision step, not a new family.
