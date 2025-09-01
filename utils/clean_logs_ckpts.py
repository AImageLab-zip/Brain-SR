import os
import re

def cleanup_checkpoints(logs_dir: str):
    # regex for model_xxxx.pth
    ckpt_pattern = re.compile(r"model_(\d+)\.pth")
    ema_ckpt_pattern = re.compile(r"ema_model_(\d+)\.pth")

    for root, dirs, files in os.walk(logs_dir):
        if os.path.basename(root) == "ckpts" or os.path.basename(root) == "ema_ckpts":
            print(f"\nChecking folder: {root}")

            is_ema = os.path.basename(root) == "ema_ckpts"
            pattern = ema_ckpt_pattern if is_ema else ckpt_pattern
            
            # collect all checkpoints
            checkpoints = []
            for f in files:
                match = pattern.match(f)
                if match:
                    step = int(match.group(1))
                    checkpoints.append((step, f))
            
            if not checkpoints:
                continue

            # find latest checkpoint
            latest_step, latest_file = max(checkpoints, key=lambda x: x[0])

            # files to keep: latest OR steps ending with 0000
            keep = {latest_file} | {f for step, f in checkpoints if step % 10000 == 0}

            # remove others
            for step, f in checkpoints:
                if f not in keep:
                    path = os.path.join(root, f)
                    print(f"Deleting {path}")
                    os.remove(path)

            print(f"Kept: {sorted(list(keep))}")

if __name__ == "__main__":
    logs_folder = "/homes/gcasari/bigbrain/work_data/logs"
    cleanup_checkpoints(logs_folder)