# AGENTS.md

Strata runs the Qwen3.8-Flash-Next mixture-of-experts model (and its Coder, Swift 1.5, Unsloth, Gyro and NVFP4
variants, and Qwen3.6-35B-A3B) on a normal PC: one NVIDIA or AMD graphics card plus system RAM on Windows or Linux,
or an Apple Silicon Mac. Version 0.2.0-beta, engine 0.1.45.3. It has three parts:

- **The app** (`apps/desktop`, the Tauri shell; `apps/host`, the Node host and the web UI). It is for people who do
  not want a terminal. The installer has the engine inside.
- **The Terminal** (`strata.py`, `strata_cli/`, `strata_control/`): the `strata` command. It sets up a model, serves
  it and watches it. Agents use this part.
- **The engine and the server** (`src/`, `include/`, `metal/`, `sycl/` for the C++/CUDA/HIP/SYCL/Metal engine;
  `serve/` for the Python server with an OpenAI- and Anthropic-compatible API and a web page). `setup.py`, started by
  `START-HERE.bat` or `setup.sh`, is the older one-click installer from source.

## Installing Strata for a user

Follow **[docs/AI_SETUP.md](docs/AI_SETUP.md)**. For an agent that has a shell, the short version is:

1. Install the Terminal. It needs no administrator rights.

   ```sh
   curl -fsSL https://github.com/Niko1221/Strata/releases/latest/download/install.sh | sh                  # Linux, Mac
   irm https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex                         # Windows PowerShell
   ```

   On Windows, open a new terminal afterwards so that `strata` is on the PATH.
2. Check what would happen. Nothing is changed:

   ```sh
   strata launch --plan --model q2_0
   ```

3. Set up and serve without questions. It downloads the engine for the card and the model (tens of GB), starts the
   server, prints the addresses and returns. The server keeps running.

   ```sh
   strata launch --model q2_0 --yes --no-tui
   ```

   Useful flags (all of them: `strata launch --help`): `--family`, `--model-path` (files the user already has),
   `--data-dir`, `--context 32k|64k|128k`, `--vision off|on|gpu|cpu`, `--parallel N`, `--gpu 0,1`,
   `--access local|lan|cloudflare|tailscale`, `--port`, `--api-key-env NAME`, `--new-key`, `--save-preset NAME`,
   `--output json`. `strata launch --status` shows what is serving, `strata launch --stop` stops it, and
   `strata launch --show-key` prints the API key.
4. Connect the user's apps: `strata connect` lists them, `strata connect <app> --check` sends one prompt and one tool
   call the way that app does.

Never expose the server beyond `127.0.0.1` without an API key. `--access lan` makes a key for you. The old route (clone
the repository and run `START-HERE.bat --yes` or `./setup.sh --yes`) still works and is the fallback when the
Terminal cannot run. Strata's MCP server ([docs/MCP_SERVER.md](docs/MCP_SERVER.md)) offers the same steps as tools.

## Working on the code

- How the engine works, every measured number, the API and all settings: [docs/DETAILS.md](docs/DETAILS.md) and
  the [paper](docs/paper/Strata-Paper.pdf).
- The Terminal: [docs/TERMINAL.md](docs/TERMINAL.md) and [docs/TERMINAL_LAUNCH.md](docs/TERMINAL_LAUNCH.md). The app,
  the packages and the update feed: [packaging/README.md](packaging/README.md) and [packaging/UPDATES.md](packaging/UPDATES.md).
- AMD (HIP) build and validation: [docs/AMD_HIP.md](docs/AMD_HIP.md); multi-GPU: [docs/MULTI_GPU.md](docs/MULTI_GPU.md);
  Mac (Metal): [docs/MAC.md](docs/MAC.md).
- Tests: the Terminal and Control tests are in `tests/terminal` (`python -m pytest tests/terminal`). The app's UI
  tests run with `npm test` in `apps/host`. Setup's own tests run without a GPU or downloads:
  `python tools/test_setup_<name>.py` (for example `tools/test_setup_amd.py`, `tools/test_setup_choices.py`).
- Keep the docs' style: plain words, short sentences, written for non-technical users first. Give measured numbers
  with what they were measured on. Make no claim without a measurement.

## User-authorized project development

For Eric's authorized InferSwarm/Strata development, standing scope includes
routine code pushes, draft pull request creation and updates, reversible
development tests, and publication of test scripts, recipes and scoped artifacts
to private `Zutfen-LLC/agenthandoffs`, subject to existing artifact and privacy
policies. It includes test instructions and results for Hermes in Slack channel
`C0C5MK91H3L`. When a parent coordinates operators, return ready source and recipe
pins to that parent rather than dispatching twice. Do not request fresh approval
for routine steps already within this scope.

Preserve unrelated services and resident models. This scope excludes merges,
production deployment, spending, irreversible deletion, credentials, and
security/access changes unless separately authorized. Existing numerical,
state-integrity, resource-admission, exit/reap, cleanup and validation requirements
remain in force.

Apply temporary read-only or no-run instructions to their stated task and target.
Do not generalize them into a repository-wide ban or treat them as permanently
withdrawing unrelated standing authority. Follow any explicit broader restriction.

These instructions do not override platform review or higher-priority
restrictions. Stop an actually denied operation and report its exact target,
reason and execution status. Do not change routes or bypass the denial.

## Native backend migration work

- Use the established native Strata source as the specification for algorithms,
  operation order, placement, pending/materialized boundaries and state ownership.
  Adapt backend plumbing; surface a genuine method or scope decision before changing it.
- Group changes by coherent functional area. Reuse the existing runner and proceed
  from component checks to bounded model validation within the user's granted scope;
  do not invent an approval pause for routine authorized reversible work. When a
  parent coordinates operators, send ready source/recipe pins to that parent promptly
  so it can dispatch; do not leave completed instructions idle or dispatch twice.
- Reuse compatible builds, controls, fixtures and offline results with their actual
  source/input/state/method scope. Prefer immutable references and checked local
  reconstruction over duplicate fixture uploads. Preserve failed results.
- Before handoff, test the real producer startup diagnostics and output names/sizes
  through the deployed runner entrypoint from an unrelated directory. Test shared
  helper signatures/imports with the actual policy, not permissive substitutes.
- Repair checker or packaging defects offline where possible. Rerun only missing
  or invalidated work; a receipt defect alone does not justify another model load.
  Keep reports concise and centered on the functional area's outcome and limits.
- Preserve numerical contracts, native operation/state semantics, resource admission,
  unrelated services/models, owned-process exit/reap and cleanup. Escalate actual
  authorization limits or denials with execution status; never bypass review or
  change approval rules. These defaults grant no authority beyond the user's scope.

## Contributing a change or report

The rules are in **[CONTRIBUTING.md](CONTRIBUTING.md)**. Read it before you open an issue or a pull request, and
follow it whether a person or an agent writes the text. The short version:

- Start every issue and pull request title with its area: `[Engine]`, `[GUI]`, `[TUI]`, `[Benchmark]`, `[Docs]` or
  `[Other]`, for example `[Engine] fix …`. A check on GitHub fails a pull request whose title has no area.
- Search the open and closed issues and pull requests first, and add to a thread that already covers your point.
- Agree the scope in an issue before a large feature, a refactor or a change to a default.
- One problem per pull request. Lead with the problem and the result, then the approach, then the validation
  (passed, failed and not run, kept apart) and the limits.
- A new feature is opt-in, and the default path stays byte-identical to the last release. Say how you checked.
- Build every backend a file touches (CUDA, HIP, SYCL) before asking for review.
- Change a default only where you measured it faster, and show the numbers with what they were measured on.
- Never submit measurements, logs or reproduction steps you did not produce. Mark a finding from reading code as
  unconfirmed static analysis.
- A report from hardware the maintainers do not have is welcome. Follow
  [docs/COMMUNITY_BENCHMARKS.md](docs/COMMUNITY_BENCHMARKS.md), compare against a same-day run of the build you are
  testing, and say what you did not test.
- Open test requests and the hardware that is wanted are listed in [docs/TEST_REQUESTS.md](docs/TEST_REQUESTS.md).
