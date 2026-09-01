# Non-linear Rolling Forcing timestep list + student cost (2026-09-01)

Not a submit. After leftover ρ **NO** and the schedule-neighbors
note. Official Rolling Forcing (Liu et al., ICLR 2026) trains
**T=5** on `[1000, 800, 600, 400, 200]` — **linear in t**.
We run the checkpoint’s `denoising_step_list` (4-step family
on our host). Print that list from one sidecar before any
new arm.

Leftover ρ is **not** this experiment. ρ scaled *how much*
Gaussian was injected on later blocks. The timestep list
tells the DiT *which noise level it is at* (adaLN /
`c_noise`). Same window, different knob. ρ already killed
Imaging Quality under captions.

## Two lists to smoke (inference, existing student)

Keep T equal to the live list. Do not add steps (that
widens the window and is a different student). Host =
caption Rolling Forcing first 8. Cite vs that host.

If live list is the official five-step
`[1000, 800, 600, 400, 200]`:

| Arm | List | Shape |
|---|---|---|
| native | 1000, 800, 600, 400, 200 | linear (already on disk) |
| linger-high | 1000, 920, 800, 520, 200 | stay noisy, dump at the end |
| dump-early | 1000, 520, 360, 260, 200 | jump down, linger near-clean |

If live list is four-step Self Forcing-style
`[1000, 750, 500, 250]`:

| Arm | List | Shape |
|---|---|---|
| native | 1000, 750, 500, 250 | linear |
| linger-high | 1000, 875, 650, 250 | stay noisy, dump at the end |
| dump-early | 1000, 500, 350, 250 | jump down, linger near-clean |

N=8, `metadata_csv`, k=1, no WAVE=3. Same bars as leftover:
tail vs host; Imaging Quality not worse by ≥1.0; Subject
Consistency not worse by ≥0.02. Expect the leftover letter
(pixels move, Imaging Quality dies) until proven otherwise.
`sf_roll` already showed student and sampler are a pair.

Do **not** launch until the user says GO. Do not remake
cite-128.

## Why we parked “needs a student”

Not because 3,000 Distribution Matching Distillation (DMD)
steps are “training Wan from scratch.” Official Rolling
Forcing: **3,000 steps, batch 8, 27 latent frames, one
machine with 8 GPUs**, after a reused causal Ordinary
Differential Equation (ODE) init (`ode_init.pt`). No video
dataset. Their own limitation: the window plus DMD is
**memory-heavy**; backpropagating every window Out-Of-Memory
(OOM)s even on 80 GB. Public train recipe also downloads
Wan **14B** as the teacher score.

Relative costs:

| Job | Order of magnitude |
|---|---|
| Leftover generate N=8, one H200 | **9–11 minutes** |
| Official Rolling Forcing student | **~1 day on 8 GPUs** (their number: 3k steps) |
| Wan 1.3B pretrain | weeks / many nodes |

So: cheap versus pretrain. **Not** cheap versus a test-time
arm. We also do not have that 8-GPU DMD stack wired on this
account, and a new student with a new list is the Stream
Forcing / Ms. Forcing paper class — not test-time adaptation
on Self Forcing.

If linger-high / dump-early **fail** the N=8 Imaging Quality
bar (likely), the literature answer is a short DMD on that
list. That is a **go/no-go** after the smoke, not the first
GPU. Paper lock stays: no Test-Time Training (TTC), no
Image-to-Video (I2V) scale-up, no remake of cite-128.
