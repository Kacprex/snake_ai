import subprocess
import requests
import time
import json
import os

# --- Persistent Save Directory ---
SAVE_DIR = "/content/drive/MyDrive/snake_ai"   # adjust if not using Colab
os.makedirs(SAVE_DIR, exist_ok=True)

SUMMARY_PATH = os.path.join(SAVE_DIR, "training_summary.json")
LOG_PATH = os.path.join(SAVE_DIR, "trainloop.log")

WEBHOOK_URL = "https://discord.com/api/webhooks/1423813556146536480/4MLDRwpT56xKoOBJ0CejUGnkTkVoR4E3rMGorAoqdHtAgLpmBobMR7JZrOvLU3d8Z78E"

last_eval_reward = None  

def pick_color(eval_reward):
    global last_eval_reward
    if last_eval_reward is None:
        last_eval_reward = eval_reward
        return 0xFFFF00
    buffer = 0.5
    color = 0xFFFF00
    if eval_reward > last_eval_reward + buffer:
        color = 0x00FF00
    elif eval_reward < last_eval_reward - buffer:
        color = 0xFF0000
    last_eval_reward = eval_reward
    return color

def send_progress(run_num, total_runs):
    if not os.path.exists(SUMMARY_PATH):
        print("⚠️ No summary found, skipping report")
        return

    with open(SUMMARY_PATH, "r") as f:
        summary = json.load(f)

    eval_r, eval_len, eval_app, eval_ep = None, None, None, None
    if summary.get("eval_last"):
        eval_ep, eval_r, eval_len, eval_app = summary["eval_last"]

    color = pick_color(eval_r if eval_r is not None else 0)

    embed = {
        "title": f"🐍 Snake AI Training Update (Run {run_num}/{total_runs})",
        "color": color,
        "fields": [
            {
                "name": "📊 Training (last 50 eps)",
                "value": (
                    f"🏆 Avg Reward: {summary['avg_reward']:.2f}\n"
                    f"🐍 Avg Length: {summary['avg_length']:.2f}\n"
                    f"🍎 Avg Apples: {summary['avg_apples']:.2f}\n"
                    f"🎲 Epsilon: {summary['epsilon']:.3f}"
                ),
                "inline": False
            }
        ]
    }

    if eval_r is not None:
        embed["fields"].append({
            "name": f"🎯 Greedy Eval (ε=0) after {eval_ep} eps",
            "value": (
                f"🏆 Reward: {eval_r:.2f}\n"
                f"🐍 Length: {eval_len:.2f}\n"
                f"🍎 Apples: {eval_app:.2f}"
            ),
            "inline": False
        })

    payload = {"embeds": [embed]}

    try:
        requests.post(WEBHOOK_URL, json=payload, timeout=10)
        print("✅ Sent progress report to Discord")
    except Exception as e:
        print("⚠️ Discord message failed:", e)

    # Append log file
    with open(LOG_PATH, "a") as logf:
        logf.write(json.dumps(summary) + "\n")

def run_training(runs, report_interval):
    for i in range(1, runs+1):
        print(f"\n🚀 Starting training run {i}/{runs}...\n")
        subprocess.run(["python", "ai.py"], check=True)
        print(f"✅ Finished run {i}/{runs}")

        if i % report_interval == 0:
            send_progress(i, runs)

        time.sleep(5)

if __name__ == "__main__":
    try:
        runs = int(input("👉 How many training runs do you want? "))
    except ValueError:
        runs = 20
    try:
        report_interval = int(input("👉 Send Discord updates every how many runs? "))
    except ValueError:
        report_interval = 5
    run_training(runs, report_interval)
