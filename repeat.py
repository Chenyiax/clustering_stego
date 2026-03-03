import subprocess
import time
import datetime

# 设置目标文件名
target_file = "example_classifier_cv.py"  # 替换为实际的py文件路径
log_file = "output_log.txt"  # 日志文件名
repeat_count = 8  # 重复次数
interval = 1  # 每次执行的时间间隔（秒）

# 打开日志文件
with open(log_file, "a") as log:
    for i in range(repeat_count):
        try:
            # 获取当前时间
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log.write(f"\n--- Execution {i + 1} at {current_time} ---\n")
            print(f"\n--- Execution {i + 1} at {current_time} ---\n")
            # 执行目标Python文件
            result = subprocess.run(
                ["python", target_file],
                text=True,  # 确保输出是字符串格式
                capture_output=True  # 捕获标准输出和错误
            )

            # 记录输出
            log.write(result.stdout)
            if result.stderr:
                log.write("\n--- Errors ---\n")
                log.write(result.stderr)

            log.write("\n--- End of Execution ---\n")
            log.flush()  # 确保实时写入日志文件

        except Exception as e:
            log.write(f"\nError during execution: {str(e)}\n")

        # 等待指定的时间间隔
        if i < repeat_count - 1:
            time.sleep(interval)
