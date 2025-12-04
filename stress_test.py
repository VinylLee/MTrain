import subprocess, json, csv, os, glob

models = [
    '/home/ubuntu/MTrain/experiments/rte_lr1.0_phi-4_labeled_data.json_20251117_140739/final_model',
    '/home/ubuntu/MTrain/MTrainNLI/experiments-old/rte_lr0.8_phi-4_20250725_142152/augment_training_best',
    '/home/ubuntu/MTrain/experiments/sick_lr1.0_phi-4_labeled_data.json_20251117_140813/final_model',
    '/home/ubuntu/MTrain/experiments-copy/sick_lr1.0_phi-4_augmented_data.json_20250728_134744/augment_training_best',
    '/home/ubuntu/MTrain/MTrainNLI/nli_saved_model/snli/bert-base-uncased/0.0037_False_phi-4_2025-11-06 14:38:41_0.7520_0.7497',
    '/home/ubuntu/MTrain/experiments/snli_lr0.0037_phi-4_augmented_data_三合一_0.8.json_20251105_153237/augment_training_best',
    '/home/ubuntu/MTrain/MTrainNLI/nli_saved_model/mnlim/bert-base-uncased/0.0051_False_phi-4_2025-07-01 14:37:23_0.6239_0.6242',
    '/home/ubuntu/MTrain/experiments/mnlim_lr0.0051_phi-4_augmented_data_三合一.json_20251106_153720/augment_training_best',
]

stress_dir = "/home/ubuntu/MTrain/MTrainNLI/nli_dataset_cache/stress_tests"
test_files = sorted(glob.glob(f"{stress_dir}/*.json"))

os.makedirs("results", exist_ok=True)
outfile = "results/all_results.csv"

with open(outfile, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["model", "test_file", "accuracy"])

    for m in models:
        for t in test_files:
            print(f"\n>>> Running\nModel: {m}\nTest: {t}\n")

            cmd = [
                "python", "new_test.py", "--model_path", m, "--test_data", t
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            output = result.stdout + result.stderr

            # 解析 accuracy
            acc = "N/A"
            for line in output.splitlines():
                if "acc" in line.lower():
                    import re
                    found = re.findall(r"[0-9]+\.[0-9]+", line)
                    if found:
                        acc = found[0]

            print(f"Accuracy = {acc}")

            writer.writerow([m, t, acc])

print("\nAll done! Results saved to results/all_results.csv")
