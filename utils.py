import subprocess

def get_gpu_stats():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        used, total, util = result.stdout.strip().split(", ")
        return {"vram_used_mb": int(used), "vram_total_mb": int(total), "gpu_util_percent": int(util)}
    except:
        return None