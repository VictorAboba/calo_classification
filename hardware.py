"""Hardware auto-detection and hardware-aware training settings.

Detects the available accelerator (NVIDIA CUDA, AMD ROCm, Intel XPU, Apple MPS
or plain CPU) and derives a `TrainingProfile` from it: mixed precision, TF32,
torch.compile, fused AdamW and DataLoader settings. Every automatic choice can
be overridden in the `hardware` section of `experiment_config.yaml`.

CLI:
    uv run python hardware.py                            # show detected hardware and profile
    uv run python hardware.py --write-accelerate-config  # write accelerate_config.yaml for this machine
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import shutil
from dataclasses import dataclass, field

import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration

from constants import ACCELERATE_CONFIG_FILE, EXPERIMENT_CONFIG_FILE
from data_schemes import HardwareConfig

# AMD architectures with native bf16 matrix instructions:
# MI100 (gfx908), MI200 (gfx90a), MI300/MI350 (gfx94x/gfx95x), RDNA3/RDNA4 (gfx11xx/gfx12xx)
_AMD_BF16_ARCH_PREFIXES = ("gfx908", "gfx90a", "gfx94", "gfx95", "gfx11", "gfx12")


@dataclass(frozen=True)
class HardwareInfo:
    backend: str  # cuda | rocm | xpu | mps | cpu
    vendor: str  # nvidia | amd | intel | apple | cpu
    device_name: str
    device_count: int
    total_memory_gb: float | None
    arch: str | None  # e.g. "sm_86" for NVIDIA, "gfx942" for AMD
    compute_capability: tuple[int, int] | None  # NVIDIA only
    bf16_supported: bool  # native (not emulated) bf16
    tf32_supported: bool
    triton_available: bool
    os_name: str
    cpu_name: str
    cpu_count: int
    torch_version: str
    notes: tuple[str, ...] = ()

    @property
    def device_type(self) -> str:
        """torch device type; ROCm GPUs are exposed through the "cuda" device type."""
        return "cuda" if self.backend in ("cuda", "rocm") else self.backend

    @property
    def is_gpu(self) -> bool:
        return self.backend != "cpu"


@dataclass(frozen=True)
class TrainingProfile:
    device_type: str
    mixed_precision: str  # no | fp16 | bf16
    tf32: bool
    compile: bool
    fused_optimizer: bool
    num_workers: int
    pin_memory: bool
    prefetch_factor: int
    notes: tuple[str, ...] = field(default=())

    def dataloader_kwargs(self) -> dict:
        kwargs: dict = {"num_workers": self.num_workers, "pin_memory": self.pin_memory}
        if self.num_workers > 0:
            # workers survive between epochs: no re-spawn cost (expensive with "spawn" on Windows/macOS)
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = self.prefetch_factor
        return kwargs


# --------------------------------------------------------------------------- detection


def _cpu_capabilities() -> dict:
    try:
        return torch.cpu.get_capabilities()
    except Exception:
        return {}


def _local_device_index(device_count: int) -> int:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    return local_rank if local_rank < device_count else 0


def _common_fields() -> dict:
    caps = _cpu_capabilities()
    return {
        "triton_available": importlib.util.find_spec("triton") is not None,
        "os_name": platform.system(),
        "cpu_name": caps.get("cpu_name") or platform.processor() or platform.machine(),
        "cpu_count": os.process_cpu_count() or 1,
        "torch_version": torch.__version__,
    }


def _detect_cpu() -> HardwareInfo:
    caps = _cpu_capabilities()
    notes = []
    if torch.version.cuda is None and torch.version.hip is None:
        if shutil.which("nvidia-smi"):
            notes.append(
                "nvidia-smi found, but the installed torch is a CPU-only build: install a CUDA build of torch (see README)"
            )
        elif shutil.which("rocm-smi") or shutil.which("amd-smi"):
            notes.append(
                "ROCm tools found, but the installed torch is a CPU-only build: install a ROCm build of torch (see README)"
            )
    common = _common_fields()
    return HardwareInfo(
        backend="cpu",
        vendor="cpu",
        device_name=common["cpu_name"],
        device_count=1,
        total_memory_gb=None,
        arch=caps.get("architecture"),
        compute_capability=None,
        # without AVX512-BF16 / AMX the CPU emulates bf16 and autocast becomes slower than fp32
        bf16_supported=bool(caps.get("avx512_bf16") or caps.get("amx_bf16")),
        tf32_supported=False,
        notes=tuple(notes),
        **common,
    )


def _detect_cuda_or_rocm() -> HardwareInfo:
    count = torch.cuda.device_count()
    props = torch.cuda.get_device_properties(_local_device_index(count))
    memory = round(props.total_memory / 1024**3, 1)

    if torch.version.hip:  # AMD ROCm build of torch
        arch = (getattr(props, "gcnArchName", "") or "").split(":")[0] or None
        if arch:
            bf16 = arch.startswith(_AMD_BF16_ARCH_PREFIXES)
        else:
            bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
        return HardwareInfo(
            backend="rocm",
            vendor="amd",
            device_name=props.name,
            device_count=count,
            total_memory_gb=memory,
            arch=arch,
            compute_capability=None,
            bf16_supported=bf16,
            tf32_supported=False,
            **_common_fields(),
        )

    cc = (props.major, props.minor)
    return HardwareInfo(
        backend="cuda",
        vendor="nvidia",
        device_name=props.name,
        device_count=count,
        total_memory_gb=memory,
        arch=f"sm_{props.major}{props.minor}",
        compute_capability=cc,
        bf16_supported=cc >= (8, 0),  # Ampere and newer
        tf32_supported=cc >= (8, 0),
        **_common_fields(),
    )


def _detect_xpu() -> HardwareInfo:
    count = torch.xpu.device_count()
    props = torch.xpu.get_device_properties(_local_device_index(count))
    return HardwareInfo(
        backend="xpu",
        vendor="intel",
        device_name=props.name,
        device_count=count,
        total_memory_gb=round(props.total_memory / 1024**3, 1),
        arch=None,
        compute_capability=None,
        bf16_supported=torch.xpu.is_bf16_supported(including_emulation=False),
        tf32_supported=False,
        **_common_fields(),
    )


def _detect_mps() -> HardwareInfo:
    try:
        memory = round(torch.mps.recommended_max_memory() / 1024**3, 1)
    except Exception:
        memory = None
    mac_version = platform.mac_ver()[0] or "0"
    return HardwareInfo(
        backend="mps",
        vendor="apple",
        device_name=f"Apple GPU (MPS, macOS {mac_version})",
        device_count=1,
        total_memory_gb=memory,
        arch=platform.machine(),
        compute_capability=None,
        bf16_supported=int(mac_version.split(".")[0]) >= 14,
        tf32_supported=False,
        **_common_fields(),
    )


def detect_hardware(force_cpu: bool = False) -> HardwareInfo:
    """Detect the accelerator torch can use (priority: CUDA/ROCm > XPU > MPS > CPU)."""
    if force_cpu:
        return _detect_cpu()
    if torch.cuda.is_available():
        return _detect_cuda_or_rocm()
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return _detect_xpu()
    if torch.backends.mps.is_available():
        return _detect_mps()
    return _detect_cpu()


# --------------------------------------------------------------------------- profile


def _auto_mixed_precision(info: HardwareInfo) -> str:
    if info.backend == "cuda":
        if info.bf16_supported:
            return "bf16"
        # Volta/Turing have fp16 tensor cores; older GPUs gain little from fp16
        return "fp16" if info.compute_capability >= (7, 0) else "no"  # type: ignore[operator]
    if info.backend in ("rocm", "xpu"):
        return "bf16" if info.bf16_supported else "fp16"
    if info.backend == "mps":
        # Apple GPUs run fp32 at the same ALU rate as fp16 and MPS autocast is still immature
        return "no"
    return "bf16" if info.bf16_supported else "no"


def _auto_compile(info: HardwareInfo) -> tuple[bool, str | None]:
    if info.backend in ("cpu", "mps"):
        return False, f"torch.compile is off by default on {info.backend}"
    if not info.triton_available:
        return False, "torch.compile is off: triton is not installed"
    if info.backend == "cuda" and info.compute_capability < (7, 0):  # type: ignore[operator]
        return False, "torch.compile is off: triton needs compute capability >= 7.0"
    return True, None


def _fused_adamw_supported(device_type: str) -> bool:
    try:
        from torch.utils._foreach_utils import _get_fused_kernels_supported_devices

        return device_type in _get_fused_kernels_supported_devices()
    except ImportError:
        return device_type == "cuda"


def _auto_num_workers(info: HardwareInfo) -> int:
    processes = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    cpus_per_process = max(1, info.cpu_count // processes)
    if info.backend == "cpu":
        # the model itself runs on the CPU cores, keep most of them for torch intra-op threads
        return min(2, cpus_per_process // 4)
    # leave one core for the training process itself. Collate is light (binning + a 2x22x96 resize),
    # 4 workers keep a GPU busy; train and val workers are persistent, so up to 2x this many processes live at once
    return max(0, min(4, cpus_per_process - 1))


def resolve_profile(info: HardwareInfo, config: HardwareConfig) -> TrainingProfile:
    notes = list(info.notes)

    mixed_precision = (
        _auto_mixed_precision(info)
        if config.mixed_precision == "auto"
        else config.mixed_precision
    )

    if config.compile == "auto":
        use_compile, reason = _auto_compile(info)
        if reason:
            notes.append(reason)
    else:
        use_compile = config.compile

    tf32 = info.tf32_supported if config.tf32 == "auto" else config.tf32
    num_workers = (
        _auto_num_workers(info) if config.num_workers == "auto" else config.num_workers
    )
    pin_memory = info.device_type == "cuda"

    return TrainingProfile(
        device_type=info.device_type,
        mixed_precision=mixed_precision,
        tf32=tf32,
        compile=use_compile,
        fused_optimizer=_fused_adamw_supported(info.device_type),
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=4 if info.is_gpu else 2,
        notes=tuple(notes),
    )


def apply_torch_settings(profile: TrainingProfile) -> None:
    """Process-wide torch flags; call before building the model."""
    # TF32 tensor cores for fp32 matmuls: all of them with mixed_precision=no, the ones outside autocast otherwise
    torch.set_float32_matmul_precision("high" if profile.tf32 else "highest")


def build_accelerator(profile: TrainingProfile) -> Accelerator:
    """Accelerator configured from the profile.

    Explicit arguments take precedence over `mixed_precision` / `dynamo_config`
    coming from `accelerate launch --config_file ...`.
    """
    return Accelerator(
        cpu=profile.device_type == "cpu",
        mixed_precision=profile.mixed_precision,
        dynamo_backend="inductor" if profile.compile else "no",
        # async host->device copies, effective together with pin_memory
        dataloader_config=DataLoaderConfiguration(non_blocking=profile.pin_memory),
    )


def _is_main_process() -> bool:
    return int(os.environ.get("RANK", 0)) == 0


def format_report(info: HardwareInfo, profile: TrainingProfile) -> str:
    on_off = {True: "on", False: "off"}
    device = (
        f"{info.device_name} x{info.device_count}" if info.is_gpu else info.device_name
    )
    if info.total_memory_gb is not None:
        device += f", {info.total_memory_gb} GB"
    if info.arch and info.is_gpu:
        device += f", {info.arch}"

    lines = [
        f"{'=' * 20} Hardware {'=' * 20}",
        f"backend         : {info.backend} ({info.vendor})",
        f"device          : {device}",
        f"cpu             : {info.cpu_name}, {info.cpu_count} threads",
        f"system          : {info.os_name}, torch {info.torch_version}",
        f"{'-' * 16} Training profile {'-' * 16}",
        f"mixed precision : {profile.mixed_precision}",
        f"tf32 matmul     : {on_off[profile.tf32]}",
        f"torch.compile   : {on_off[profile.compile]}",
        f"fused AdamW     : {on_off[profile.fused_optimizer]}",
        f"dataloader      : num_workers={profile.num_workers}, pin_memory={on_off[profile.pin_memory]}",
    ]
    lines += [f"note            : {note}" for note in profile.notes]
    lines.append("=" * 50)
    return "\n".join(lines)


def setup_hardware(
    hardware_config: dict | None = None, verbose: bool = True
) -> tuple[HardwareInfo, TrainingProfile]:
    """Detect hardware, resolve the training profile and apply torch flags.

    `hardware_config` is the `hardware` section of experiment_config.yaml.
    """
    config = HardwareConfig.model_validate(hardware_config or {})
    info = detect_hardware(force_cpu=config.device == "cpu")
    profile = resolve_profile(info, config)
    apply_torch_settings(profile)
    if verbose and _is_main_process():
        print(format_report(info, profile), flush=True)
    return info, profile


# --------------------------------------------------------------------------- accelerate config


def accelerate_config_for(info: HardwareInfo, profile: TrainingProfile) -> dict:
    """`accelerate launch` config: one process per GPU, settings matching the profile."""
    multi_device = info.device_count > 1 and info.backend in ("cuda", "rocm", "xpu")
    if multi_device:
        distributed_type = "MULTI_XPU" if info.backend == "xpu" else "MULTI_GPU"
    else:
        distributed_type = "NO"

    config = {
        "compute_environment": "LOCAL_MACHINE",
        "debug": False,
        "distributed_type": distributed_type,
        "downcast_bf16": "no",
        # NUMA pinning in accelerate relies on NVML, i.e. NVIDIA on Linux
        "enable_cpu_affinity": info.vendor == "nvidia" and info.os_name == "Linux",
        "gpu_ids": "all",
        "machine_rank": 0,
        "main_training_function": "main",
        "mixed_precision": profile.mixed_precision,
        "num_machines": 1,
        "num_processes": info.device_count if multi_device else 1,
        "rdzv_backend": "static",
        "same_network": True,
        "tpu_env": [],
        "tpu_use_cluster": False,
        "tpu_use_sudo": False,
        "use_cpu": info.backend == "cpu",
    }
    if profile.compile:
        config["dynamo_config"] = {"dynamo_backend": "INDUCTOR"}
    return config


def write_accelerate_config(
    path: str, info: HardwareInfo, profile: TrainingProfile
) -> None:
    header = (
        f"# Generated by `python hardware.py --write-accelerate-config` for {info.device_name} "
        f"x{info.device_count} ({info.backend}).\n"
        "# mixed_precision / dynamo_config are informational: the training scripts take them\n"
        "# from the `hardware` section of experiment_config.yaml.\n"
    )
    with open(path, "w") as file:
        file.write(header)
        yaml.safe_dump(accelerate_config_for(info, profile), file, sort_keys=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect hardware and show the training profile derived from it"
    )
    parser.add_argument(
        "--write-accelerate-config",
        nargs="?",
        const=ACCELERATE_CONFIG_FILE,
        metavar="PATH",
        help=f"write an accelerate config for this machine (default: {ACCELERATE_CONFIG_FILE})",
    )
    args = parser.parse_args()

    with open(EXPERIMENT_CONFIG_FILE, "r") as file:
        experiment_config = yaml.load(file, yaml.FullLoader)
    info, profile = setup_hardware(experiment_config.get("hardware"))

    if args.write_accelerate_config:
        write_accelerate_config(args.write_accelerate_config, info, profile)
        print(f"Accelerate config written to {args.write_accelerate_config}")


if __name__ == "__main__":
    main()
