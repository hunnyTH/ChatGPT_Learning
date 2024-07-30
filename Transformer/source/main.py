from data_tool import *
from model import *
from evaluate import *
import subprocess


def get_gpu_info():
    # 查看GPU情况
    try:
        # 使用subprocess调用nvidia-smi命令
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                                capture_output=True, text=True)
        output = result.stdout.strip().split('\\n')
        # 提取GPU信息
        gpu_info = {}
        for line in output:
            if line:
                name, memory = line.split(', ')
                gpu_info[name] = memory
                print(f"GPU型号: {name}, 内存: {memory}")
        print(f"GPU数量: {len(gpu_info)}")
    except Exception as e:
        print('Error:', str(e))

if __name__=="__main__":
    print("----------GPU信息查看----------")
    get_gpu_info()