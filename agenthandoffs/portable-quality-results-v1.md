# Frozen twelve-case functional screen — completed result v1

The unchanged scorer returned **inconclusive-baseline (exit1)**: native CUDA
10/12 and portable AMD/Vulkan10/12, failing the same cases9 (reasoning) and12
(code). It recorded identical output-token sequences on all12 cases. Both
runtime arms completed12 requests and exited0, with no reported guard failure.
The scorer's exit1 is the expected baseline-miss verdict, not a runtime failure
or reason to repeat the suite. This is not broad model-quality equivalence.

Primary evidence read through the Slack connector:
[Hermes final receipt1793158976.227939](https://zutfen.slack.com/archives/C0C5MK91H3L/p1793158976227939).
This coding task did not execute either model or inspect private raw outputs.
No decoded answers or raw logits are published here.

| Item | Retained identity / observation |
| --- | --- |
| Handoff source | `fca0331fa44049acdfb4f21318e3241913bb2951`, portable-quality-hermes-v2.md |
| Portable binary source | PR43 `ac85ff3fdf0834c13437b69b1568d8704e9b0717` |
| Portable binary SHA256 | `cb71b55bd7309320a81446c117728e493ea1d19c8f1ec0d54c7ef2836c8734c3` |
| Native binary SHA256 | `931569c089a4f04114395a428c85f85227142c733f4fa0511243d05f97b46b89` |
| Existing frozen suite/scorer quality.py SHA256 | `01a1a6c4dc6c1a5f94f005316e34b7983699362fb5f3a4edb3811597892e38f1` |
| Actual private score path on02 | `/home/hermes/portable-quality-ac85ff3-v2/score.json` |
| Score SHA256 | `3154b628313c48b0786c363236454b91be99da09ca94dc25c032731571772c15` |
| Owned native worker cleanup | exited0, loopback port17431 closed |
| Owned portable cleanup | no owned process remains |

The score's actual path above differs from the artifact-volume path proposed
in the handoff; record the actual receipt, not the planned location. No other
unreported run/log path is inferred from the recipe.

**Original whole-model numerical parity remains FAILED.** The position37
relative logit L2 was0.0717160202 and KL(reference|candidate)0.0001161040;
matching tokens on this functional screen does not waive those frozen limits.
The separate physical HC probe demonstrated a CPU activation-quantization
contribution at that projection, not the cause of every model-level difference.
PR43 separately proved41 portable logit rows and stdout byte-identical with
expert reuse; that is portable optimization equivalence, not CUDA parity.

Clocks were unchanged and reported low. These timings are not full-speed AMD
performance evidence. No performance improvement or broad quality pass follows.

The initial incomplete source export remains retained. Its failed preparation
traceback existed in the tool output but **no separate host log was saved**.
Successful recovery has a retained log and exit receipt. Do not claim complete
host-local logging of the failed preparation or erase that failure history.
No retry for bad answers, tolerance/gold change or new model run is proposed.
This is an additive documentation checkpoint; PR42 remains draft and unmerged.
