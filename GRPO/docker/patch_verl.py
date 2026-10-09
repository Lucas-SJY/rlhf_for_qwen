"""Build-time fixes to the installed verl 0.8.0; the Dockerfile runs this once.

They are applied to verl's source rather than at runtime (labelcot/patches.py) because
the code runs in verl's GPU worker processes, which never import labelcot.

FSDP2 CPU offload and checkpoints. FSDPEngine.save_checkpoint moves the module to the GPU
when its parameters are on the CPU (load_fsdp_model_to_gpu). Under FSDP2's
CPUOffloadPolicy (FSDP_CPU_OFFLOAD=true) the parameters always live on the CPU and the
policy owns their placement; moving the module leaves it half on each device, and
module.state_dict() then fails with "Attempted to set the storage of a tensor on device
"cpu" to a storage on different device "cuda:0"". verl already skips that move in
get_per_tensor_param for this case (verl #5995); the patch applies the same rule to
save_checkpoint. load_checkpoint only moves the module when param_offload is on, which
CPU offload turns off, so it needs no change.

Each patch checks that the code it replaces is there exactly once, so a different verl
version fails the build instead of running unpatched.
"""

import importlib.util
import sys
from pathlib import Path

# find_spec on the top-level package locates it without importing it (and torch).
VERL_DIR = Path(importlib.util.find_spec("verl").submodule_search_locations[0])

PATCHES = [
    (
        "workers/engine/fsdp/transformer_impl.py",
        """        origin_module_device = next(self.module.parameters()).device.type
        if self._is_offload_param or origin_module_device == "cpu":
            load_fsdp_model_to_gpu(self.module)
""",
        """        origin_module_device = next(self.module.parameters()).device.type
        # Patched by GRPO/docker/patch_verl.py: FSDP2 CPUOffloadPolicy owns CPU<->GPU
        # placement, and moving the module here breaks state_dict() below (cf. #5995).
        if not self._uses_fsdp2_cpu_offload_policy and (self._is_offload_param or origin_module_device == "cpu"):
            load_fsdp_model_to_gpu(self.module)
""",
    ),
]


def main() -> None:
    for rel, old, new in PATCHES:
        path = VERL_DIR / rel
        text = path.read_text()
        if new in text:
            print(f"already patched: {path}")
            continue
        count = text.count(old)
        if count != 1:
            sys.exit(f"{path}: expected the code to patch exactly once, found it {count} times")
        path.write_text(text.replace(old, new))
        print(f"patched: {path}")


if __name__ == "__main__":
    main()
