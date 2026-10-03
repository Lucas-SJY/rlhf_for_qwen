# Build context: this directory (GRPO/). ../run.sh builds it.
#
# Base: verl's own CI image for the verl 0.8 line -- CUDA 13.0 (with forward-compatibility
# libraries), Python 3.12, torch 2.11.0, vLLM 0.20.2, transformers 5.3.0, flash-attn 2.8.3
# prebuilt, verl's Python dependencies installed. verl 0.8.0's CI ran on this family
# (vllm020). Building flash-attn from source instead takes hours even on a fast amd64
# host and is not practical under emulation on a laptop.
ARG BASE_IMAGE=verlai/verl:vllm020.dev2
FROM ${BASE_IMAGE}

ARG VERL_VERSION=0.8.0
# rLLM main as of 2026-09-05; rLLM has no PyPI release, so pin the commit.
ARG RLLM_COMMIT=3b40c37cf6a262cf4d28cc987ebe4f4cf797956c
# Imports the training stack at build time so a missing dependency fails here, not after
# a GPU job has been scheduled. Set to 0 if an emulated (--platform linux/amd64) build
# cannot run torch.
ARG IMPORT_CHECK=1

ENV PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PYTHONPATH=/workspace/src \
    CUDA_COMPAT_AUTO=1

WORKDIR /workspace

# Freeze the GPU stack that ships with the image, so nothing installed below can swap it.
RUN pip freeze \
      | grep -iE '^(torch|torchvision|torchaudio|vllm|transformers|flash[-_]attn|flashinfer[-_]python|triton|numpy|xformers)==' \
      > /tmp/pins.txt \
    && cat /tmp/pins.txt

# verl 0.8.0 without dependency resolution: its metadata still caps vllm at 0.12 and numpy
# below 2, neither of which matches the image. Its real dependencies are already present;
# the second line only fills gaps, constrained to the frozen GPU stack.
RUN pip install --no-deps "verl==${VERL_VERSION}" \
    && pip install -c /tmp/pins.txt accelerate codetiming datasets dill hydra-core pandas peft \
         "pyarrow>=19.0.0" pybind11 pylatexenc "ray[default]>=2.41.0" torchdata \
         "tensordict>=0.8.0,<=0.10.0,!=0.9.0" wandb tensorboard packaging cachetools einops pillow

# rLLM without dependency resolution: its [verl] extra pins vllm==0.22.1 and its base
# dependencies pull tinker-cookbook, which would reinstall torch. Added below is what the
# unified trainer's workflow path imports.
RUN pip install --no-deps "rllm @ git+https://github.com/rllm-org/rllm.git@${RLLM_COMMIT}" \
    && pip install -c /tmp/pins.txt click rich pydantic PyYAML openai tqdm requests httpx \
         sympy "antlr4-python3-runtime==4.9.3" aiosqlite fastapi uvicorn

# Hugging Face math_verify for the answer check (0.5 = 1/2 = \frac{1}{2}). No dependency
# resolution: its default antlr4 extra would replace the 4.9.3 that hydra/omegaconf need,
# and latex2sympy2_extended accepts 4.9.3 as well. sympy is already installed.
RUN pip install --no-deps "math-verify==0.9.0" "latex2sympy2_extended==1.11.0"

COPY src/ src/
# The task set is a few MB, so it is baked into the image like the SFT data was.
COPY data/ data/

RUN if [ "${IMPORT_CHECK}" = "1" ]; then \
      python3 -c "import importlib.resources as r; \
assert (r.files('rllm.trainer.config') / 'unified.yaml').is_file(), 'rLLM configs missing'; \
assert (r.files('verl.trainer.config') / 'ppo_trainer.yaml').is_file(), 'verl configs missing'; \
import verl, peft; \
from rllm.trainer import AgentTrainer; \
from labelcot.workflow import LabeledCoTWorkflow; \
from labelcot.reward import MATH_VERIFY_AVAILABLE, answer_is_correct; \
assert MATH_VERIFY_AVAILABLE, 'math_verify missing'; \
assert answer_is_correct(r'so \\boxed{\\frac{1}{2}}', '0.5'), 'math_verify check failed'; \
print('import check ok, verl', verl.__version__)"; \
    fi

RUN chmod +x src/entrypoint.sh
CMD ["bash", "/workspace/src/entrypoint.sh"]
