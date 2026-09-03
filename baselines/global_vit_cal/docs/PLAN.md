# Implementation plan

## Scope

Build a clean SALT-VI global CAL package without modifying historical
experiments. No real training is part of this task.

## Stages

1. Freeze the method contract and implementation notes.
2. Implement fixed-position token confidence dropout and its low-reliability
   guard.
3. Implement a single-branch ViT and expose the layer-2 CLS-to-patch attention
   needed by the geometric loss.
4. Port exact CAL and the successful soft ellipse loss/warm-up.
5. Implement Qwen ROI boards, atomic validation, eight independent draws per
   region, complete-link clustering, and 64-draw joint-world compression.
6. Add a backend-independent Qwen pipeline and track-anchor selection.
7. Run unit tests and a CPU dry run on the 3090 server.

## Integration rule

Confidence dropout masks patches as attention keys/values and zeros their
states, but does not physically shorten the global sequence. This preserves
the fixed positional geometry used by the attention prior.

## Stage-B addition

8. Add a model-independent Stage-B hard-mining schedule and a separate offline
   Qwen-world text adapter.

The Qwen-derived Stage-B profile is a separate code path.

## Completion status

All seven stages are implemented. The remote baseline package passes 19
deterministic tests, the fixed configuration verifier, full Python bytecode
compilation, and a CPU forward/backward dry run. No training or checkpoint
generation was performed.
