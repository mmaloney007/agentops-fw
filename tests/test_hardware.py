"""Tests for hardware detection utilities."""
import pytest


def test_hardware_detection():
    """Test detect_hardware returns valid HardwareInfo."""
    from agent_stable_slo.utils.hardware import detect_hardware, HardwareInfo

    hw = detect_hardware()
    assert isinstance(hw, HardwareInfo)
    assert hw.kind in ("nvidia_4090", "nvidia_cuda", "apple_mps", "cpu_only")
    assert hw.device in ("cuda:0", "mps", "cpu")
    assert hw.backend in ("cuda", "metal", "cpu")


def test_hardware_summary():
    """Test HardwareInfo.summary() produces readable string."""
    from agent_stable_slo.utils.hardware import detect_hardware

    hw = detect_hardware()
    summary = hw.summary()
    assert isinstance(summary, str)
    assert hw.kind in summary
    assert hw.device in summary


def test_hardware_as_dict():
    """Test HardwareInfo.as_dict() is JSON-serializable."""
    import json
    from agent_stable_slo.utils.hardware import detect_hardware

    hw = detect_hardware()
    d = hw.as_dict()
    assert isinstance(d, dict)
    # Should be JSON serializable
    json_str = json.dumps(d)
    assert len(json_str) > 0


def test_recommended_defaults():
    """Test recommended_defaults returns valid config."""
    from agent_stable_slo.utils.hardware import detect_hardware, recommended_defaults

    hw = detect_hardware()
    cfg = recommended_defaults(hw)

    assert "device" in cfg
    assert "backend" in cfg
    assert "torch_dtype" in cfg
    assert "load_in_4bit" in cfg

    # torch_dtype should be valid
    assert cfg["torch_dtype"] in ("bfloat16", "float16", "float32")


@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(),
    reason="CUDA not available"
)
def test_nvidia_4090_detection():
    """Test 4090 is properly detected when present."""
    from agent_stable_slo.utils.hardware import detect_hardware

    hw = detect_hardware()

    # If we have a 4090, verify it's detected
    import torch
    gpu_name = torch.cuda.get_device_name(0).lower()
    if "4090" in gpu_name:
        assert hw.kind == "nvidia_4090"
        assert hw.supports_bf16 is True
        assert hw.capability == "8.9"
        assert hw.total_mem_gb is not None
        assert hw.total_mem_gb > 20  # 4090 has ~24GB
