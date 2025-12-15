import os
import subprocess

# ===========================
# 配置参数
# ===========================
start_epoch = 1
end_epoch = 16
weight_template = r"D:\person\Low-Light\WDCFNet\weights\train\epoch_{}.pth"
eval_file = "eval.py"
log_file = "results_logVV.txt"  # 最终结果保存文件

# ===========================
# 更新 eval.py 中 LIME 分支的权重路径
# ===========================
def update_weight_path(new_path):
    """只修改 LIME 分支内的 weight_path，不改其他分支"""
    lines = []
    inside_lime_block = False

    with open(eval_file, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()

            # 进入 LIME 分支
            if stripped.startswith("elif ep.VV"):
                inside_lime_block = True
            # 离开 LIME 分支（下一个 elif 或 else）
            elif inside_lime_block and (stripped.startswith("elif ") or stripped.startswith("else:")):
                inside_lime_block = False

            # 在 LIME 分支内发现 weight_path 行 → 替换
            if inside_lime_block and "weight_path" in stripped:
                indent = line[:len(line) - len(line.lstrip())]
                line = f"{indent}weight_path = r'{new_path}'\n"

            lines.append(line)

    with open(eval_file, "w", encoding="utf-8") as f:
        f.writelines(lines)

# ===========================
# 清空日志文件
# ===========================
with open(log_file, "w", encoding="utf-8") as f:
    f.write(f"===== Low-Light Enhancement Log =====\n\n")  # 可自定义标题

# ===========================
# 批处理循环
# ===========================
for epoch in range(start_epoch, end_epoch + 1):
    new_weight_path = weight_template.format(epoch)
    print(f"\n====== Updating to epoch {epoch} ======")

    # 更新权重路径
    update_weight_path(new_weight_path)
    print(f"[INFO] Updated weight_path → {new_weight_path}")

    # 运行 eval.py
    print("\n[INFO] Running: python eval.py --VV")
    subprocess.run(["python", "eval.py", "--VV"])

    # 运行 measure_niqe_bris.py 并捕获输出
    print("\n[INFO] Running: python measure_niqe_bris.py --VV")
    result = subprocess.run(
        ["python", "measure_niqe_bris.py", "--VV"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    # 将原始输出写入日志文件
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"===== Epoch {epoch} =====\n")
        f.write(result.stdout)
        f.write("\n\n")  # 每轮空行分隔

    print(f"[LOG] Epoch {epoch} 输出已写入 {log_file}")

# ===========================
# 全部完成
# ===========================
print("\n===== ALL DONE =====")
print(f"结果已保存到 {log_file}")
