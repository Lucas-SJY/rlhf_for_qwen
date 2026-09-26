# Agentic RL Frameworks: Accepted Environment Standards

Snapshot 2026-09-26, checked against each repo's source (commits in [Sources](#sources)). vLLM is not compared: it is an inference engine and defines no environment standard. The fourth framework is rLLM.

Legend: ✅ first-party · ⚠️ partial (recipe, legacy API, or eval-only) · ❌ not supported

## Harbor rollouts

- **2 / 4** collect rollouts through Harbor's own runtime (`Trial`): SkyRL and rLLM.
- **3 / 4** train on Harbor-format tasks with first-party code: SkyRL, rLLM, and prime-rl. verl has no Harbor support in core.

Notes:

- SkyRL: `HarborGenerator` runs Harbor trials. Token-level training needs `rollout_details`, which only Harbor's `terminus-2` / `computer-1` agents emit.
- rLLM: Harbor's task format is rLLM's native `Task` format; `HarborRuntime` runs full Harbor trials, and its gateway captures tokens for any agent. It pins the old `harbor==0.3.0` (latest is 0.23.0).
- prime-rl: verifiers `HarborTaskset` loads Harbor tasks but runs them on verifiers' own runtimes and harnesses.
- verl: `verl-recipe/tmax` reuses the Harbor task format with its own Modal runtime; uni-agent's Harbor adapter is eval-only.

## Core interfaces

| Framework | Native environment interface | Sandbox / runtime |
|---|---|---|
| SkyRL | SkyRL-Gym `Env` (`init/step/close`), or any harness behind `GeneratorInterface.generate()` | In-process; Harbor providers on the Harbor path |
| rLLM | AgentFlow (`@rllm.rollout`) + Evaluator (`@rllm.evaluator`); Harbor-compatible `Task` | Docker / Daytona / Modal / local |
| verl | `AgentLoopBase.run()` with `BaseTool` tools | None built in |
| prime-rl | verifiers v1 `Taskset` + `Harness` + `Runtime` | subprocess / Docker / Prime / Modal / Apptainer |

## Compatibility matrix: environment standard × framework

| Environment standard or source | SkyRL | rLLM | verl | prime-rl |
|---|:--:|:--:|:--:|:--:|
| Gym-style `reset/step` loop | ✅ SkyRL-Gym (`init/step/close`) | ⚠️ legacy `BaseEnv` | ❌ | ❌ |
| Harbor task format | ✅ | ✅ native | ⚠️ `recipe/tmax` | ✅ |
| Harbor runtime (`Trial`) | ✅ | ✅ | ❌ (uni-agent: eval-only) | ❌ |
| verifiers / Environments Hub | ⚠️ legacy v0 API | ⚠️ legacy v0 API (example) | ❌ | ✅ native (v1) |
| OpenEnv (Meta PyTorch) | ✅ example integration | ❌ | ❌ | ✅ `OpenEnvTaskset` |
| NeMo Gym (NVIDIA) | ❌ | ❌ | ⚠️ `recipe/nemo_gym` | ✅ `NeMoGymTaskset` |
| Atropos (Nous Research) | ❌ | ❌ | ⚠️ `recipe/atropos` | ❌ |
| OpenReward | ✅ example integration | ❌ | ❌ | ❌ |
| TextArena | ❌ | ❌ | ❌ | ✅ taskset |
| AWS Bedrock AgentCore | ❌ | ✅ remote runtime | ❌ | ❌ |
| Tinker API server (run tinker-cookbook envs on your own GPUs) | ✅ SkyRL implements the Tinker API | — (uses Tinker as a hosted training backend) | ⚠️ `recipe/verl_tinker` | ❌ |
| MCP tools | ❌ | ✅ `MCPTool` | ❌ (removed "for now") | ✅ toolsets served as MCP |
| CLI agent harnesses (Claude Code, Codex, mini-swe-agent, ...) | ⚠️ via Harbor agents or custom generators | ✅ 10+ built-in harnesses | ⚠️ via uni-agent | ✅ built-in harnesses |
| Agent SDKs (LangGraph, OpenAI Agents SDK, ...) | ❌ (write a custom generator) | ✅ cookbook: LangGraph, OpenAI Agents SDK, smolagents, Strands | ⚠️ `recipe/langgraph_agent` | ❌ (write a custom harness) |

SkyRL and rLLM integrate verifiers through its legacy v0 API, which verifiers `main` has removed.

## Sources

SkyRL `6efa810` · rLLM `3b40c37` · verl `6093e00` (verl-recipe `242c73f`, uni-agent `8cbdea0`) · prime-rl `b944873` (verifiers `e1fdcd4`, prime-envs `342d602`) · Harbor `d10ac31`
