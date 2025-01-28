import os
from llmrca.utils import global_logger, global_config, get_project_root, get_process_pid, get_container_pid, write_dict_to_csv
import json
import sqlite3
import pandas as pd
from datetime import timedelta, datetime
import random
import torch
import numpy as np
import re
import warnings
import subprocess
import time
import signal
from multiprocessing import Process
import sys
from llmrca.fault_injection.main_fi import *

servicevLLM_command_keywords_list = ["vllm.entrypoints.openai.api_server", global_config["PORT_LLM"]]
serviceRAG_command_keywords_list = ["server_rag.py"]
container_name_or_id = "qdrant"

# 先修改.env
project_path = get_project_root()
# 启动服务，发送请求，收集数据
server_allstart_path = os.path.join(project_path, "llmrca/request_server/server_allstart.py")
server_rag_path = os.path.join(project_path, "llmrca/request_server/server_rag.py")
metrics_system_all_path = os.path.join(project_path, "llmrca/request_server/metrics_system_all.py")
request_simulate_path = os.path.join(project_path, "llmrca/request_server/request_simulate.py")
# 数据处理
data_filter_path = os.path.join(project_path, "llmrca/data_process/data_filter.py")
data_extractor_path = os.path.join(project_path, "llmrca/data_process/data_extractor.py")

metrics_system_all_process = None
request_simulate_process = None


# 设置ctrl c退出
def signal_handler(sig, frame):
    global metrics_system_all_process, request_simulate_process
    global_logger.info("Ctrl+C detected, terminating processes...")
    if request_simulate_process:
        request_simulate_process.terminate()
        request_simulate_process.wait()
        global_logger.info("request_simulate_process terminated.")
    if metrics_system_all_process:
        metrics_system_all_process.terminate()
        metrics_system_all_process.wait()
        global_logger.info("metrics_system_all_process terminated.")
    global_logger.info("All processes terminated. Exiting...")
    exit(0)


def run_script(script_path):
    try:
        global_logger.info(f"正在执行脚本: {script_path}")
        process = subprocess.Popen(["python", script_path])
        return process
    except subprocess.CalledProcessError as e:
        global_logger.error(f"脚本执行出错: {e} - 脚本: {script_path}")
        return None


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    # 收集指标，发送请求
    metrics_system_all_process = run_script(metrics_system_all_path)
    request_simulate_process = run_script(request_simulate_path)
    # 故障注入
    time.sleep(1)
    if global_config["FAILURE_LABEL"] == "1.1":
        fi_process_cpu_usgae(command_keywords_list=serviceRAG_command_keywords_list, fi_time=720, usage=5)
    if global_config["FAILURE_LABEL"] == "1.2":
        fi_process_cpu_usgae(command_keywords_list=serviceRAG_command_keywords_list, fi_time=720, usage=8)
    if global_config["FAILURE_LABEL"] == "1.3":
        fi_process_cpu_usgae(command_keywords_list=serviceRAG_command_keywords_list, fi_time=720, usage=10)
    if global_config["FAILURE_LABEL"] == "1.4":
        fi_process_cpu_usgae(command_keywords_list=serviceRAG_command_keywords_list, fi_time=720, usage=12)
    if global_config["FAILURE_LABEL"] == "1.5":
        fi_process_cpu_usgae(command_keywords_list=serviceRAG_command_keywords_list, fi_time=720, usage=15)
    if global_config["FAILURE_LABEL"] == "2.1":
        fi_process_cpu_usgae(command_keywords_list=servicevLLM_command_keywords_list, fi_time=720, usage=5)
    if global_config["FAILURE_LABEL"] == "2.2":
        fi_process_cpu_usgae(command_keywords_list=servicevLLM_command_keywords_list, fi_time=720, usage=8)
    if global_config["FAILURE_LABEL"] == "2.3":
        fi_process_cpu_usgae(command_keywords_list=servicevLLM_command_keywords_list, fi_time=720, usage=10)
    if global_config["FAILURE_LABEL"] == "2.4":
        print("#########hello#######")
        fi_process_cpu_usgae(command_keywords_list=servicevLLM_command_keywords_list, fi_time=720, usage=12)
    if global_config["FAILURE_LABEL"] == "2.5":
        fi_process_cpu_usgae(command_keywords_list=servicevLLM_command_keywords_list, fi_time=720, usage=15)
    if global_config["FAILURE_LABEL"] == "3.1":
        fi_gpu_graph_clock(gpu_id=3, fi_time=720, clock=200)
    if global_config["FAILURE_LABEL"] == "3.2":
        fi_gpu_graph_clock(gpu_id=3, fi_time=720, clock=400)
    if global_config["FAILURE_LABEL"] == "3.3":
        fi_gpu_graph_clock(gpu_id=3, fi_time=720, clock=600)
    if global_config["FAILURE_LABEL"] == "3.4":
        fi_gpu_graph_clock(gpu_id=3, fi_time=720, clock=800)
    if global_config["FAILURE_LABEL"] == "4.1":
        fi_gpu_graph_clock(gpu_id=4, fi_time=720, clock=200)
    if global_config["FAILURE_LABEL"] == "4.2":
        fi_gpu_graph_clock(gpu_id=4, fi_time=720, clock=400)
    if global_config["FAILURE_LABEL"] == "4.3":
        fi_gpu_graph_clock(gpu_id=4, fi_time=720, clock=600)
    if global_config["FAILURE_LABEL"] == "4.4":
        fi_gpu_graph_clock(gpu_id=4, fi_time=720, clock=800)
    if global_config["FAILURE_LABEL"] == "5.1":
        fi_gpu_graph_clock(gpu_id=5, fi_time=720, clock=200)
    if global_config["FAILURE_LABEL"] == "5.2":
        fi_gpu_graph_clock(gpu_id=5, fi_time=720, clock=400)
    if global_config["FAILURE_LABEL"] == "5.3":
        fi_gpu_graph_clock(gpu_id=5, fi_time=720, clock=600)
    if global_config["FAILURE_LABEL"] == "5.4":
        fi_gpu_graph_clock(gpu_id=5, fi_time=720, clock=800)
        
    request_simulate_process.wait()
    global_logger.info("request_simulate_process 执行完毕")
    if metrics_system_all_process:
        metrics_system_all_process.terminate()
        metrics_system_all_process.wait()
        global_logger.info("metrics_system_all_process 进程终止")
    global_logger.info("所有进程已终止。")
