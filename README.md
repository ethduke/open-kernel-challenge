# open-kernel-challenge

My final kernel files from the kernel track of the Open Kernel Challenge (Muse Spark Open), as scored on the boards when kernel submissions closed on Nov 5, 2026.

## The kernel track

Each problem is one part of the Muse Spark Open model. You submit one Python file per problem, which builds and runs its own CUDA kernels. It runs on a single H200 and is timed against SGLang's own kernel for the same work. The score is the speed-up over SGLang. There are five kernel boards, and the overall rank follows the whole decode layer.

## Results

Best scored run on each board. Ranks were read on Nov 5, 2026 and are **provisional**: final standings come from the organizers' reruns.

| Board | File | Speed-up vs SGLang | Provisional rank | Job |
|---|---|---|---|---|
| Decode layer | [`kernels/muse_decode_layer.py`](kernels/muse_decode_layer.py) | 2.297× | 115 of 1,299 | 92b2961dcb7d4888 |
| Attention decode | [`kernels/muse_attention_decode.py`](kernels/muse_attention_decode.py) | 2.731× | 85 of 1,379 | 609f3fb0074244eb |
| MoE (W4A16 routed experts) | [`kernels/muse_moe_w4a16.py`](kernels/muse_moe_w4a16.py) | 2.400× | 205 of 1,425 | fa71127eb0e844da |
| BF16 projections | [`kernels/muse_bf16_proj.py`](kernels/muse_bf16_proj.py) | 1.596× | 83 of 1,746 | 384dce8f584741df |
| Prefill layer | [`kernels/muse_prefill_layer.py`](kernels/muse_prefill_layer.py) | 1.922× | 102 of 898 | 4e0002f8678540f2 |
| **Overall** | | | **115 of 2,014** | |

A single run carries about ±1.5% of noise. On the layer board that is many places, so read the ranks loosely.

The files here differ from the scored files in comments only: internal job ids, experiment tags and references to private notes and scripts were removed. No code changed.

## What these are built on

Almost all of the code is public Apache-2.0 work by others. All five files start from [keell0renz/muse-open-kernel-challenge](https://github.com/keell0renz/muse-open-kernel-challenge) at commit 8e30e1b. That repository combines kernels from vjeux, alygg77, chenwainuo, ZhimaoL, Madhavan113, Matsvei, NathenFernandes (MIT) and others. Each file keeps those authors' attribution lines. See [NOTICE](NOTICE) for the full list and licences.

Switching from my own kernels to this code gave most of my speed-up. My own decode layer was at 1.67×; on keell0renz's code with my changes it reached 2.23×, and later 2.297×.

## What I changed

My changes are marked `ethduke` / `Change by ethduke` in each file's header and code. Most are small, and the kernels themselves are mostly unchanged.

- **Decode layer**
  - Build choices and size gates: attention is built the way the attention board builds it; MoE uses the MID build at T = 5..8; qkv at T = 1 uses the narrow direct-escape sk2 build.
  - A range fallback: for sizes the tuner never saw, the qkv, o and unpooler picks follow the same bucket rule the file already uses elsewhere, once prepare's self-test passes.
  - The T 16 / 32 graft choice is pinned on, because its in-prepare timing flipped between judge runs.
  - q_len-8 QK^T is done in FP16 accumulators, with P codes made by bit manipulation (as ZhimaoL's kernels do).
  - SwiGLU uses the GPU's fast tanh.
  - The rq GEMMs join the launch chain by programmatic dependent launch.
  - Direct-read escapes on the unfused sk2 kernels (qkv, o, unpooler): helper warps read x from global memory instead of waiting on the ring hand-back.
  - These last two changes came from two agents in a fork-and-flush round and were merged into this file (see [METHOD.md](METHOD.md)).
- **Attention decode**
  - The q_len-8 QK^T uses FP16 accumulators.
  - ZhimaoL's q_len-8 verify kernels are built with his piecewise-linear (Schraudolph) exp2.
  - Full-layer verify stays on the vjeux kernels, which measured faster with the FP16 QK^T.
- **MoE**
  - SwiGLU uses the fast tanh.
  - The dedicated batch-8 build also serves 5–7 tokens, but only when prepare's self-test shows it matches the base dispatch there.
- **BF16 projections**
  - prepare() self-tests every scored cell's kernel pick against an FP64 reference, using normal inputs and 30× outlier inputs at the judge's tolerances. A pick that fails falls back to the default, so a bad build on one node costs speed in that cell instead of failing the run.
- **Prefill layer**
  - ZhimaoL's prefill FMHA is back on his piecewise-linear exp2. I had built it with exact exp2 until the Nov 3 rules.
  - kernel_fn binds the device's primary CUDA context when it is called from a fresh thread.
  - FA4-style lazy rescale: the softmax keeps a stale row maximum until a score exceeds it by 1.5 (log2) instead of 0.75.

## Numerics

The organizers' kernel-board rules of Nov 3, 2026 allow approximate math accurate to BF16 precision. That includes an approximate exp2 for attention's P stored in FP8 E4M3, and FP16 accumulation in attention. These files use both, so they are **not exact**. They pass the judge's correctness tests at its tolerances. The files are also tuned to the challenge's fixed shapes and an H200. They include fallbacks for other sizes, but they are not general-purpose kernels.

## How I measured

A single board score was too noisy to judge changes under about 1%. So candidates were timed against their parent as paired, same-input runs on the judge, in the same check. Each harness was first validated with A/A runs (the parent against itself) and same-code twin builds, to measure its noise floor. A change was kept only if it gained in two separate reads.

## Licence

Apache License 2.0, see [LICENSE](LICENSE). Third-party notices, including the MIT and BSD-3-Clause parts, are in [NOTICE](NOTICE) and in the files themselves.
