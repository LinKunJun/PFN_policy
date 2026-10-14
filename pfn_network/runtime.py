"""Explicit CPU/MUSA selection; no silent CPU fallback or CUDA alias patching."""
import importlib.metadata
import platform
import torch

def resolve_device(name="musa"):
    kind = str(name).split(":", 1)[0]
    if kind == "musa":
        try:
            import torch_musa  # Registers the MUSA backend.
        except ImportError as error:
            raise RuntimeError("torch_musa is unavailable. Use the supplied MUSA base image or explicitly --device cpu.") from error
        if not torch.musa.is_available():
            raise RuntimeError("MUSA is not available: check device exposure, driver and image libraries.")
        device = torch.device(name)
        index = device.index if device.index is not None else 0
        if not 0 <= index < torch.musa.device_count():
            raise ValueError(f"MUSA device index {index} is not visible")
        torch.musa.set_device(index)
        return torch.device(f"musa:{index}")
    if kind != "cpu":
        raise ValueError("Supported devices: cpu, musa, musa:N")
    return torch.device(name)

def synchronize(device):
    device = torch.device(device)
    if device.type == "musa":
        torch.musa.synchronize(device)

def action_generator():
    # A portable state format also avoids relying on backend-specific multinomial RNG.
    return torch.Generator(device="cpu")

def sample_action(probabilities, generator=None):
    if generator is not None and generator.device.type != "cpu":
        raise ValueError("Policy sampling requires a CPU generator")
    return torch.multinomial(probabilities.detach().cpu(), 1, generator=generator).to(probabilities.device)

def runtime_info(device):
    device = resolve_device(device)
    names = ["torch", "numpy", "scipy", "scikit-learn", "pandas"]
    if device.type == "musa":
        names.append("torch_musa")
    return {"python": platform.python_version(), "platform": platform.platform(),
            "device": str(device), "versions": {n: importlib.metadata.version(n) for n in names},
            "gpu": torch.musa.get_device_name(device.index) if device.type == "musa" else None,
            "visible_devices": torch.musa.device_count() if device.type == "musa" else 0,
            "torch_threads": torch.get_num_threads(), "action_rng_backend": "cpu",
            "attention": "explicit_matmul_softmax_fp32", "gcn": "auto_dense_small_graphs_edge_large_graphs"}
