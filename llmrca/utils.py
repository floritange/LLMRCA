import os
import logging
from dotenv import dotenv_values
import psutil
import time, requests
import sys
import subprocess
import pandas as pd
import shutil
from datetime import datetime

def get_project_root():
    """Get the root directory path of the project."""
    # Absolute path of the current file
    current_file_path = os.path.abspath(__file__)
    # Find the root directory of the project
    project_root = os.path.dirname(os.path.dirname(current_file_path))
    return project_root


# Read the .env global config variables
global_config = dotenv_values(get_project_root() + "/.env")
global_logger = None
project_root = get_project_root()

def setup_logging(project_name=global_config["PROJECTNAME"]):
    """Configure the global logger to output to both console and log files."""
    # Create logger
    global_logger = logging.getLogger(project_name)
    global_logger.setLevel(logging.DEBUG)  # Set log level

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)  # Console log level

    # Create file handler
    log_file_path = os.path.join(get_project_root(), "logs/project_running.log")
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)  # File log level

    # Create formatter
    # [%(filename)s:%(lineno)d]
    formatter = logging.Formatter("%(asctime)s - %(name)s - [%(filename)s:%(lineno)d] - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers to logger
    global_logger.addHandler(console_handler)
    global_logger.addHandler(file_handler)

    return global_logger


# Configure the global logger
global_logger = setup_logging()


def setting_logger(logger_name=None, logfile_name="default.log"):
    # Configure the logger based on the logger name and file path, default to the root logger
    logger_path = os.path.join(get_project_root(), f"logs/{logfile_name}")
    os.makedirs(os.path.dirname(logger_path), exist_ok=True)
    log_formatter = logging.Formatter("%(asctime)s - %(name)s - [%(filename)s:%(lineno)d] - %(levelname)s - %(message)s")
    # Create the root logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # Configure console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)

    # Configure file handler
    file_handler = logging.FileHandler(logger_path)
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)


def wait_for_health_check(url, headers=None, timeout=60):
    """Wait for the server health check to pass."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(5)
    return False


def get_process_pid(command_keywords_list):
    for proc in psutil.process_iter(["pid", "cmdline"]):
        cmdline = proc.info["cmdline"]
        if cmdline:  # Check if cmdline is not empty
            cmdline_str = " ".join(cmdline)  # Join cmdline list into string
            # Check if cmdline contains all the keywords
            if all(keyword in cmdline_str for keyword in command_keywords_list):
                # Check if it's directly run by python, not through /bin/sh -c
                if "/bin/sh" not in cmdline_str:
                    return proc.info["pid"]
    return None  # Return None if no matching process is found


def get_container_pid(container_name_or_id):
    """Get the main process PID of a container by its name or ID."""
    try:
        # Get the container's main process PID using docker inspect
        pid = subprocess.check_output(["docker", "inspect", "--format", "'{{.State.Pid}}'", container_name_or_id])
        # Clean output and convert to integer
        return int(pid.strip().decode("utf-8").replace("'", ""))
    except subprocess.CalledProcessError as e:
        global_logger.error(f"Error getting PID for container {container_name_or_id}: {e}")
        return None


def write_dict_to_csv(data, filename):
    # Append dictionary data as a pandas dataframe
    df = pd.DataFrame(data)
    if os.path.exists(filename):
        existing_df = pd.read_csv(filename)
        existing_columns = existing_df.columns.tolist()
        for col in existing_columns:
            if col not in df.columns:
                df[col] = None
        for col in df.columns:
            if col not in existing_columns:
                existing_df[col] = None
        # Merge and write data according to the new column order
        combined_df = pd.concat([existing_df, df], ignore_index=True)
        # Overwrite data while keeping header consistent
        combined_df.to_csv(filename, mode="w", header=True, index=False)
    else:
        # Write data directly if file does not exist
        df.to_csv(filename, mode="w", header=True, index=False)


def clear_folders(directory_path):
    # Clear all subfolders and files inside the folder
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)
        else:
            pass

def timestamp_to_unix(timestamp_str):
    return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S,%f").timestamp()
