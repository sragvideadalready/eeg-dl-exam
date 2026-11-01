# ANSWERS

## Status note (read first)

PhysioNet downloads ran at roughly 4-7 minutes per subject and repeatedly failed and retried (screenshots in `evidence/`). This cost a large part of my time, so I could not use the imagery runs (6, 10, 14) end to end. (My device is not gpu equipped)

All accuracy numbers below were therefore measured on the **executed-movement runs 5, 9, 13**, the data the starter used. I did identify that `RUNS = [5, 9, 13]` is wrong for an imagery task (the correct setting is `[6, 10, 14]`), but I could not measure its effect. Part 2 was **not run**. Anything labelled "expected" is a prediction, not a result.

---

## Part 1

### Task 1: Bugs

Baseline (buggy pipeline, runs 5/9/13, seeds 0/1/2): 0.667, 0.750, 0.786, **mean 0.734** (n_test = 84).
After the train.py fixes below (seed 0 only): **0.583** (val 0.643). One seed and about 84 test trials
means roughly +/-5% sampling noise, so per-bug numbers show direction only.

| # | File | What's wrong | Expected direction | Measured (seed 0) |
|---|------|--------------|--------------------|-------------------|
| 1 | train.py | Split was a random trial split, not by subject, although the docstring says "unseen subjects" (`groups` was never used). Same-person trials share a signature the net can memorise. | accuracy up | 0.667 -> 0.631 (different test set, within noise) |
| 2 | train.py | z-score mean/std computed over all data including test, before splitting. | slightly up | 0.631 -> 0.631 |
| 3 | train.py | Best checkpoint chosen on test accuracy; validation set was built but never used. | up (max of ~30 noisy estimates) | 0.631 -> 0.583 |
| 4 | erd.py | Baseline window (0, 0.5) s is after the cue, not rest. The cue-evoked transient (~0.25 s) sits inside it and biases the reference. | no effect on accuracy, distorts the ERD plot | 0.583 -> 0.583  |
| 5 | data.py | `RUNS = [5, 9, 13]` are executed movement; the task is imagery (6, 10, 14). Executed movement gives stronger ERD plus EMG and proprioceptive signal. | up | Not measured (imagery data could not be downloaded in time) |

Notes on rigor:
- The final number, 0.583 (49/84), has a one-sided binomial p of about 0.08, so it is not clearly above chance.
- Validation (0.643) exceeds test (0.583). That gap is the optimism of selecting a checkpoint on a small validation set.
- Each fix is its own commit (see `git log`).

### Task 2: ERD

(Plot made on runs 5/9/13, executed movement, not imagery.)

All three sites show a sharp transient about 0.25 s after the cue in both conditions. This is a cue-evoked,phase-locked response, not mu ERS. After about 0.5 s, mu power falls by roughly 15-35% in both conditions at C3,Cz and C4, and the curves are noisy (10 subjects, about 21 trials per class per subject, unsmoothed). Thehands-vs-feet differences are small relative to that noise: hands shows slightly deeper late ERD at C3 and C4,which fits hand areas desynchronising during hand movement. It does not show a clean foot-specific effect at Cz.


### Task 3: What is the model using?

Frontal group: `Fp1, Fpz, Fp2, AF7, AF3, AFz, AF4, AF8` (8 electrodes), runs 5/9/13, 3 seeds, each with
different held-out subjects.

| Channels | Seed 0 | Seed 1 | Seed 2 | Mean |
|----------|--------|--------|--------|------|
| frontal (8 ch) | 0.643 | 0.643 | 0.643 | **0.643** |
| all (64 ch) | 0.583 | not run | not run | - |

The frontal-only model reaches 64.3% on all three seeds (54/84 correct, one-sided binomial p of about 0.006 each).
On seed 0 the held-out subjects are identical for both runs (subjects 3 and 8), and frontal (0.643) is atleast as good as all channels (0.583). These electrodes sit far from motor cortex, so the class information they carry must be non-brain: most likely **EOG**, meaning vertical eye movements or saccades toward a target at the top of the screen (fists) vs the bottom (feet).
### Task 4: Short answers

1. **58% of 45 trials.** That is 26/45. A one-sided binomial test against 50% gives p of about 0.19, so it is not distinguishable from chance. The standard error is about 7.5 points (95% CI roughly 44-73%). You would need about 29/45 (64%) to reach p < 0.05.
2. **Depthwise layer.** For each temporal filter (a frequency band), it learns D spatial filters, weight vectors across the 64 electrodes. These are like CSP or beamformer-style spatial filters. They combine channels to emphasise a source pattern (for example vertex vs lateral motor areas) and suppress common noise.
3. **Stopping the Task 3 shortcut.** Experiment: counterbalance or randomise the cue position so that on-screen
   location is independent of class (or use an auditory cue). Analysis: high-pass at about 8 Hz (EOG is mostly
   below about 4 Hz), remove EOG with ICA or regression, and/or exclude frontal channels and the first moments
   after the cue.

---

## Part 2

**Not run.** I ran out of time after the data download problems described above, so there are no Part 2 results,
no best/worst across seeds, and no figure. I would rather say so than report numbers I don't have.

**Hypothesis (written before coding):** Fists-vs-feet is a spatial pattern of band power (hand area vs vertex), so
electrodes, not time samples, are the natural tokens. The vanilla model squeezes all 64 electrodes into one vector
per timestep, losing electrode identity, and must learn temporal locality and power from 481 tokens. Electrode
tokens built from learned band-power features plus a 3D-geometry positional encoding should shrink attention from
482^2 to 65^2 and put it on the axis carrying the signal. I expect it to beat vanilla and have lower seed variance,
but not necessarily beat EEGNet at 8 subjects.

**Design I intended:** Each electrode is one token. A temporal filter bank shared across electrodes is followed
by squaring, pooling and a log, so the token is band power over 6 time bins. A small MLP of the normalised 3D
electrode coordinates is added, then a CLS token and a 2-layer transformer. Squaring computes the ERD quantity
explicitly instead of making attention discover power from raw samples. The geometry encoding lets attention
distinguish "vertex" from "hand area", which a left/right prior cannot. Planned ablations: learned electrode ids
vs none. Planned harness fix: report final-epoch accuracy instead of the peak test accuracy (the starter harness
reports peak test accuracy, the same flaw as Part 1 bug 3), with the same seeds for all models.

**What I would check next:** run the baselines and my model on 5 or more seeds (two test subjects gives about
+/-5% noise); plot where the CLS token attends on the scalp; run the ablations to test whether geometry beats
learned ids; and ask whether the idea would still help with 80 subjects, where a vanilla transformer has more
data to learn electrode structure itself.
PART 2

