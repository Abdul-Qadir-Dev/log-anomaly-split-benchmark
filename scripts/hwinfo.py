"""Detect hardware and write results/hardware.json."""
import json
import multiprocessing
import os
import platform
import subprocess

import torch


def get_ram_gb():
    if platform.system() == "Windows":
        out = subprocess.check_output(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory",
            ]
        ).decode(errors="ignore")
        for line in out.splitlines():
            line = line.strip()
            if line.isdigit():
                return round(int(line) / (1024**3), 2)
    return None


def main():
    info = {
        "platform": platform.platform(),
        "cpu_count_logical": multiprocessing.cpu_count(),
        "ram_gb": get_ram_gb(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_vram_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
        )
        info["hardware_tier"] = "gpu_present" if info["gpu_vram_gb"] >= 6 else "no_gpu"
    else:
        info["gpu_name"] = None
        info["gpu_vram_gb"] = None
        info["hardware_tier"] = (
            "no_gpu_high_ram" if (info["ram_gb"] or 0) >= 16 else "no_gpu_low_ram"
        )

    os.makedirs("results", exist_ok=True)
    with open("results/hardware.json", "w") as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
