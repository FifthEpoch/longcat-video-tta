# Beat Rolling Forcing by intervening like Rolling Forcing (2026-08-30)

Not a submit. Seed search is off the table as the next method.
Mid-chunk rewrite, guidance / shift, and weight updates stay closed.

## How Rolling Forcing actually differs from Self Forcing

Self Forcing finishes a ~5 s chunk (four denoise steps), **locks**
it, and treats those frames as clean history. A freeze or morph
inside the chunk is already committed.

Rolling Forcing keeps several short blocks in one **window** at
staggered noise. The newest block is almost pure noise. Older
blocks in the window are only partly cleaned and can still change.
Attention inside the window is bidirectional, so the future that
is still being drawn can fix the present. A block **locks only
when it leaves the window**. They also trained an attention sink
so the first frames stay available.

That is why Rolling is **45 s** (one continuous roll) and why tail
motion is higher: local error gets a few more looks before commit.
It is also why official Dynamic Degree is **lower** than Self
Forcing on caption 128 (28.9% vs 32.8%). The same “revise + sink”
that protects identity damps the “is this clip dynamic” call.

Subject 0.685 / IQ 71.52 / tail +33% vs Self Forcing 0.666 / 72.07.
Beating Rolling on our table means: keep the cheap roll and the
subject / tail, and pick up the few Dynamic clips Self Forcing
still wins — without twitch.

## What we already tried that sits on this lever (do not redo)

| Intervention | What it was | Outcome |
|---|---|---|
| Crossed host (`sf_roll` / `rf_chunk`) | SF weights + Rolling window, or the reverse | Tail ~0.028, Dyn% 59–75, flicker ~0.972. **Twitch.** Student and sampler are a pair. |
| Global ρ on injected noise | More / less noise on later blocks | Knob **lives**. IQ −1.4 to −3.8. Closed. |
| Extra attention sink | Pin first frames harder | Tail +24%. Subject −0.016, flicker 0.977. No scale. |
| VAE recache | Re-encode recent latents | +7% tail grain. Dyn still 0. Not living motion. |
| Rewind a sick window | Reroll last 21 latents | +8% tail. HOLD, small. |
| Pseudo-future on Rolling | Prefix hold-out gate | Dead (18/32 exact host). |
| Intra-chunk mix / redraw | Rewrite after the picture exists | Identity tax. Closed. |

LongLive as a third host did **not** recover Dynamic Degree.
Search-on-Rolling (look / sick / always) is still seed search.

## The hook that is actually Rolling-shaped

Not “try four seeds.” Not “mix the last denoise step.”

**When a block is about to leave the window**, decide how the
*next* window treats history and new noise. That is the same
place Rolling already differs from Self Forcing.

Three ideas, cheapest first. None is implemented tonight.

### 1. Context noise on committed history (first)

The Rolling loop already writes the leaving block into KV at
`context_noise` (today **0**). Self-Forcing++’s point: if later
windows attend to *too-clean* history, they copy the last still.
A little noise on that write is “don’t overfit the last frame.”
It is not a pixel rewrite and not a second seed.

Hypothesis: Rolling’s missing Dynamic clips are over-clean KV,
not a missing search. Cost ≈ native Rolling.

### 2. Online next-block noise (not leftover ρ)

Leftover ρ applied one schedule to the **whole video** from the
prefix. That taxed stills Rolling already won.

Different: keep the native schedule. Only if the *just-locked*
block’s latent travel died, scale **the next injected block**
up. One local bump, then back to native. IQ tax should stay
local if the gate is rare.

### 3. Soften the sink (opposite of `rf_sink`)

Extra sink moved pixels the wrong way (identity / flicker).
Native Rolling already sinks. Softening how hard later windows
attend to old clean frames is the motion-side cousin. Risk is
the twitch we saw when student and sampler were mismatched.
Probe after (1), not instead of it.

## What this is not

- Pathwise TTC’s “pull toward frame 0” — we already froze motion
  that way. Their useful sentence is: intervene on **sampling
  state** at low noise. On Rolling, low noise **is** window exit.
- TANGO’s LoRA. Their useful sentence is: predicted noise should
  look healthy. That can *gate* (1) or (2), not replace them.
- `sf_roll`’s Dyn 59%. That is flicker, not a quality win.
- Another k=4 picker on either host.

## After jobs already in flight

Cite-128 Pseudo vs Rolling still decides whether seed search
matches Rolling **quality**. Separate question. The next
**method** if we want to beat Rolling at Rolling-like cost is
(1) or (2) on the Rolling host, not a cheaper tree on Self
Forcing.
