import os
from llmrca.utils import global_logger, get_project_root, global_config, project_root
import json
import sqlite3
import pandas as pd
from datetime import timedelta, datetime
import random
import torch
import numpy as np
import re
from statistics import mean
from sklearn.preprocessing import MinMaxScaler
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import jensenshannon
from dtaidistance import dtw
from torch_geometric.nn import GCNConv, GraphSAGE, GAE
from torch_geometric.data import Data, DataLoader
import torch
import torch.nn.functional as F
import joblib
from llmrca.rca_algorithm.dnn_models_b import GraphAnomalyDetectionModel
import sys
import warnings
import logging
from torch_geometric.utils import dense_to_sparse

# warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric.deprecation")
# logging.basicConfig(
#     level=logging.DEBUG,  # 日志级别
#     format="%(asctime)s - %(name)s - [%(filename)s:%(lineno)d] - %(levelname)s - %(message)s",
#     handlers=[logging.StreamHandler()],  # 输出到控制台
# )


seed_id = 10

pd.options.mode.chained_assignment = None
# 设置随机数种子
random.seed(seed_id)
torch.manual_seed(seed_id)
np.random.seed(seed_id)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed_id)
    torch.cuda.manual_seed_all(seed_id)

metric_name_vllm_gpu = f"{global_config['DEVICE_LLM']}"
metric_name_embedding_gpu = f"{global_config['DEVICE_EMBEDDING'].split(':')[1]}"
metric_name_reranker_gpu = f"{global_config['DEVICE_RERANKER'].split(':')[1]}"

# 读取标签键值对
failure_lable_df = pd.read_csv(project_root + "/llmrca/rca_algorithm/failure_label.csv")
failure_lable_dict = dict(zip(failure_lable_df["failure_label"], failure_lable_df["metric_name"]))
# print(failure_lable_dict)
quality_label_list = [9.1, 10.1, 11.1, 12.1, 13.1, 13.2, 14.1, 14.2, 14.3, 14.4, 15.1, 16.1, 16.2, 16.3]

# 所有列
data_columns_df = pd.read_csv(project_root + "/llmrca/rca_algorithm/data_columns.csv")
all_columns = data_columns_df.columns
metric_qdrant_nogpu_columns = [col for col in all_columns if col.startswith("metric@qdrant") and all(x not in col for x in ["gpu", "disk_", "mem_"])]
metric_rag_nogpu_columns = [col for col in all_columns if col.startswith("metric@rag") and all(x not in col for x in ["gpu", "disk_", "mem_"])]
metric_rag_gpu_embedding_columns = [
    col for col in all_columns if col.startswith("metric@rag") and f"gpu_{metric_name_embedding_gpu}" in col and all(x not in col for x in ["gpu_fan_speed", "gpu_memory", "gpu_utilization_percent", "gpu_temperature_C", "gpu_sm", "gpu_power"])
]
metric_rag_gpu_reranker_columns = [
    col for col in all_columns if col.startswith("metric@rag") and f"gpu_{metric_name_reranker_gpu}" in col and all(x not in col for x in ["gpu_fan_speed", "gpu_memory", "gpu_utilization_percent", "gpu_temperature_C", "gpu_sm", "gpu_power"])
]

metric_vllm_nogpu_columns = [col for col in all_columns if col.startswith("metric@vllm") and all(x not in col for x in ["gpu", "disk_", "mem_"])]
metric_vllm_gpu_columns = [
    col for col in all_columns if col.startswith("metric@vllm") and f"gpu_{metric_name_vllm_gpu}" in col and all(x not in col for x in ["gpu_fan_speed", "gpu_memory", "gpu_utilization_percent", "gpu_temperature_C", "gpu_sm", "gpu_power"])
]


log_rag_columns = [col for col in all_columns if col.startswith("log@rag") and all(x not in col for x in ["log@rag@httpcore.connection", "log@rag@urllib3.connectionpool", "log@rag@httpx"])]
log_vllm_columns = [col for col in all_columns if col.startswith("log@vllm") and all(x not in col for x in ["log@vllm@vllm.engine.metrics - [metrics.py:295] - INFO"])]

# X_node
# w     34 35 25
# w/o ML 16 17 7
# w/o M 28 29 19
# w/o L 22 23 13
# 4

# quality
# xgboost: 1,2,3
# bins:1
# performance_graph_data -> quality_graph_data
# jump quality_label_list

# metric_qdrant_nogpu_columns=[]
# metric_rag_nogpu_columns=[]
# metric_rag_gpu_embedding_columns=[]
# metric_rag_gpu_reranker_columns=[]
# metric_vllm_nogpu_columns=[]
# metric_vllm_gpu_columns=[]

# log_rag_columns=[]
# log_vllm_columns=[]

# trace@=[]
# performance_graph_data = {
#     "query>duration": ["retrieve>duration", "reranking>duration", "llm>duration"] + log_rag_columns,
#     "retrieve>duration": ["embedding>duration"] + metric_qdrant_nogpu_columns + metric_rag_nogpu_columns,
#     "llm>duration": ["llm_request>duration"] + metric_rag_nogpu_columns,
#     "llm_request>duration": ["llm_scheduler>duration", "llm_generate>duration"] + log_vllm_columns,
#     "embedding>duration": metric_rag_nogpu_columns + metric_rag_gpu_embedding_columns,
#     "llm_scheduler>duration": metric_vllm_nogpu_columns,
#     "llm_generate>duration": metric_vllm_nogpu_columns + metric_vllm_gpu_columns,
#     "reranking>duration": metric_rag_nogpu_columns + metric_rag_gpu_reranker_columns,
#     # "metric@rag@cpu_usage_percent": ["metric@vllm@cpu_usage_percent"],
#     # "metric@vllm@gpu_5_gpu_utilization_percent": ["metric@vllm@gpu_5_gpu_graph_clock"],
#     # "metric@vllm@gpu_5_gpu_sm_activity_percent": ["metric@vllm@gpu_5_gpu_graph_clock"],
#     # "metric@vllm@gpu_5_gpu_power_W": ["metric@vllm@gpu_5_gpu_graph_clock"],
#     # "metric@rag@gpu_3_gpu_memory_used_MB": ["metric@rag@gpu_3_gpu_memory_process_used_MB"],
#     # "metric@rag@gpu_3_gpu_memory_process_used_MB": ["trace@embedding@embedding_model_name"],
#     # "metric@rag@gpu_4_gpu_memory_used_MB": ["metric@rag@gpu_4_gpu_memory_process_used_MB"],
#     # "metric@rag@gpu_4_gpu_memory_process_used_MB": ["trace@reranking@rerank_model_name"],
#     # # "metric@rag@gpu_5_gpu_memory_used_MB": ["metric@rag@gpu_5_gpu_memory_process_u
#     # "metric@rag@gpu_3_gpu_sm_activity_percent": ["metric@rag@gpu_3_gpu_graph_clock"],
#     # "metric@rag@gpu_3_gpu_power_W": ["metric@rag@gpu_3_gpu_graph_clock"],
#     # "metric@rag@gpu_4_gpu_sm_activity_percent": ["metric@rag@gpu_4_gpu_graph_clock"],
#     # "metric@rag@gpu_4_gpu_power_W": ["metric@rag@gpu_4_gpu_graph_clock"],
# }


performance_graph_data = {
    "query>duration": ["retrieve>duration", "reranking>duration", "llm>duration"] + log_rag_columns,
    "retrieve>duration": ["embedding>duration", "trace@retrieve@documents_contents_sum", "trace@retrieve@retrieve_num", "quality@record@quality_kb"] + metric_qdrant_nogpu_columns + metric_rag_nogpu_columns,
    # "trace@retrieve@documents_contents_sum": ["trace@retrieve@retrieve_num", "quality@record@quality_kb", "trace@embedding@embedding_model_name"],
    "llm>duration": ["llm_request>duration"] + metric_rag_nogpu_columns,
    "llm_request>duration": ["llm_scheduler>duration", "llm_generate>duration"] + log_vllm_columns,
    "embedding>duration": ["trace@query@input_value_sum", "trace@embedding@embedding_model_name", "trace@embedding@embedding_vector"] + metric_rag_nogpu_columns + metric_rag_gpu_embedding_columns,
    "llm_scheduler>duration": metric_vllm_nogpu_columns,
    "llm_generate>duration": [
        "quality@record@quality_prompt",
        "trace@query@llm_token_count_prompt",
        "trace@query@llm_token_count_completion",
        "trace@llm_request@llm_model_name",
        "trace@llm_request@parameters_max_tokens",
        "trace@llm_request@parameters_temperature",
    ]
    + metric_vllm_nogpu_columns
    + metric_vllm_gpu_columns,
    # "trace@query@llm_token_count_completion": ["trace@llm_request@parameters_max_tokens"],
    "reranking>duration": ["trace@retrieve@retrieve_num", "trace@retrieve@documents_contents_sum", "trace@reranking@output_documents_contents_sum", "trace@reranking@rerank_model_name", "trace@reranking@top_k"]
    + metric_rag_nogpu_columns
    + metric_rag_gpu_reranker_columns,
    # "trace@reranking@output_documents_contents_sum": ["trace@reranking@top_k"],
    # "trace@reranking@output_documents_contents_sum": ["trace@reranking@top_k", "trace@retrieve@retrieve_num", "trace@embedding@embedding_model_name", "quality@record@quality_kb", "trace@llm_request@llm_model_name"],
    # "metric@rag@cpu_usage_percent": ["metric@vllm@cpu_usage_percent"],
    # "metric@vllm@gpu_5_gpu_utilization_percent": ["metric@vllm@gpu_5_gpu_graph_clock"],
    # "metric@vllm@gpu_5_gpu_sm_activity_percent": ["metric@vllm@gpu_5_gpu_graph_clock"],
    # "metric@vllm@gpu_5_gpu_power_W": ["metric@vllm@gpu_5_gpu_graph_clock"],
    # "metric@rag@gpu_3_gpu_memory_used_MB": ["metric@rag@gpu_3_gpu_memory_process_used_MB"],
    # "metric@rag@gpu_3_gpu_memory_process_used_MB": ["trace@embedding@embedding_model_name"],
    # "metric@rag@gpu_4_gpu_memory_used_MB": ["metric@rag@gpu_4_gpu_memory_process_used_MB"],
    # "metric@rag@gpu_4_gpu_memory_process_used_MB": ["trace@reranking@rerank_model_name"],
    # # "metric@rag@gpu_5_gpu_memory_used_MB": ["metric@rag@gpu_5_gpu_memory_process_u
    # "trace@query@llm_token_count_prompt": ["trace@llm_request@llm_model_name","trace@reranking@top_k"],
    # "trace@query@llm_token_count_prompt": ["trace@reranking@output_documents_contents_sum", "quality@record@quality_prompt"],
    # "metric@rag@gpu_3_gpu_sm_activity_percent": ["metric@rag@gpu_3_gpu_graph_clock"],
    # "metric@rag@gpu_3_gpu_power_W": ["metric@rag@gpu_3_gpu_graph_clock"],
    # "metric@rag@gpu_4_gpu_sm_activity_percent": ["metric@rag@gpu_4_gpu_graph_clock"],
    # "metric@rag@gpu_4_gpu_power_W": ["metric@rag@gpu_4_gpu_graph_clock"],
}

quality_graph_data = {
    "quality@answer_correctness": [
        "trace@reranking@output_documents_scores_top3mean",
        "quality@record@quality_prompt",
        "trace@llm_request@llm_model_name",
        "trace@llm_request@parameters_max_tokens",
        "trace@llm_request@parameters_best_of",
        "trace@llm_request@parameters_n",
        "trace@llm_request@parameters_temperature",
        "trace@llm_request@parameters_top_p",
    ],
    "trace@reranking@output_documents_scores_top3mean": ["trace@retrieve@documents_scores_top3mean", "trace@reranking@rerank_model_name", "trace@reranking@top_k"],
    "trace@retrieve@documents_scores_top3mean": ["trace@embedding@embedding_model_name", "trace@retrieve@retrieve_num", "quality@record@quality_kb"],
}

add_performance_graph_data = {
    "trace@reranking@output_documents_contents_sum": ["trace@reranking@top_k", "trace@retrieve@retrieve_num", "trace@embedding@embedding_model_name", "quality@record@quality_kb", "trace@llm_request@llm_model_name"],
    "trace@query@llm_token_count_prompt": ["trace@reranking@top_k", "trace@retrieve@retrieve_num", "trace@embedding@embedding_model_name", "quality@record@quality_kb", "trace@llm_request@llm_model_name"],
    "trace@retrieve@documents_contents_sum": ["trace@retrieve@retrieve_num"],
}


def filter_dataframe(df, graph_data, column_name="query>duration", lower_percentile=5, upper_percentile=95):
    lower_bound = df[column_name].quantile(lower_percentile / 100)
    upper_bound = df[column_name].quantile(upper_percentile / 100)
    filtered_df = df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]
 
    relevant_columns = set(graph_data.keys())
    for children in graph_data.values():
        relevant_columns.update(children)
    filtered_columns = [col for col in df.columns if col in relevant_columns]
    sorted_columns = sorted(filtered_columns)
    return filtered_df[sorted_columns]


def build_graph(graph_data, df):
    G = nx.DiGraph()
    for parent, children in graph_data.items():
        if parent not in df.columns:
            continue
        for child in children:
            if child in df.columns:
                G.add_edge(parent, child)
    return G


def build_edge_index(graph_data, df_columns):
    edges = []
    for parent, children in graph_data.items():
        if parent in df_columns:
            parent_index = df_columns.get_loc(parent)
            for child in children:
                if child in df_columns:
                    child_index = df_columns.get_loc(child)
                    edges.append((parent_index, child_index))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index


def create_graph_from_df(df, graph_structure):
    """根据邻接矩阵构建图并按照图的结构处理DataFrame"""
    # 收集所有节点，包括那些只作为依赖项出现的节点
    all_nodes = set(graph_structure.keys())
    for dependencies in graph_structure.values():
        all_nodes.update(dependencies)

    # 创建节点到索引的映射
    node_names = sorted(all_nodes)  # 排序以确保顺序一致
    node_to_index = {node: idx for idx, node in enumerate(node_names)}
    num_nodes = len(node_names)
    # for id, name in enumerate(node_names):
    #     print(id, name)

    # 构建邻接矩阵
    A = np.zeros((num_nodes, num_nodes))
    for node_i, dependencies in graph_structure.items():
        for node_j in dependencies:
            if node_j in node_to_index:
                A[node_to_index[node_i], node_to_index[node_j]] = 1

    # 生成邻接矩阵的边索引
    edge_index = dense_to_sparse(torch.tensor(A, dtype=torch.float))[0]

    # 逐行处理 DataFrame，生成图数据
    data_list = []
    for _, row in df.iterrows():
        # 确保 DataFrame 中的列名与 node_names 一致
        X = np.array([row.get(node_name, 0) for node_name in node_names], dtype=np.float32).reshape(num_nodes, 1)

        # 创建图数据
        data = Data(x=torch.tensor(X, dtype=torch.float), edge_index=edge_index)
        data_list.append(data)

    return data_list, node_names


def classify_nodes_with_duration(performance_graph_data):
    """
    将 performance_graph_data 中的节点分为 component 节点（名称中包含 '>duration'）和叶子节点。
    """
    # 所有父节点集合
    parent_nodes = set(performance_graph_data.keys())
    # 所有子节点集合
    child_nodes = set(node for children in performance_graph_data.values() for node in children)
    # 所有节点集合
    all_nodes = parent_nodes.union(child_nodes)
    # Component 节点：名称中包含 '>duration'
    component_nodes = {node for node in all_nodes if ">duration" in node}
    # 叶子节点 = 所有节点 - 非叶子节点（被依赖的节点 + 父节点）
    leaf_nodes = all_nodes - parent_nodes
    # 再从叶子节点中过滤掉包含 '>duration' 的节点
    leaf_nodes = {node for node in leaf_nodes if ">duration" not in node}
    return component_nodes, leaf_nodes


def find_parent_components_with_scores(nonservice_names_and_scores_all, performance_graph_data, sorted_names_and_scores_all):
    """
    找到 nonservice_names_and_scores_all 中的节点的父 component 节点（名称包含 '>duration'）。
    父节点按 score 从高到低排序，并去重，仅保留一个父节点。
    """
    # 构建反向索引（子节点 -> 父节点）
    reverse_graph = {}
    for parent, children in performance_graph_data.items():
        for child in children:
            if child not in reverse_graph:
                reverse_graph[child] = []
            reverse_graph[child].append(parent)

    # 构建结果列表，查找每个节点的父 component 节点
    parent_components_with_scores = []

    for sublist, original_sublist in zip(nonservice_names_and_scores_all, sorted_names_and_scores_all):
        sublist_with_parents = []
        for name, score in sublist:
            # 找到当前节点的父节点
            if name in reverse_graph:
                # 筛选出父节点中包含 '>duration' 的节点，排除 'query>duration'
                parent_components = [parent for parent in reverse_graph[name] if ">duration" in parent and "query>duration" not in parent]
                parent_scores = []
                for parent in parent_components:
                    # 获取父节点的分数，从 sorted_names_and_scores_all 中匹配
                    parent_score = next((s for s, n in original_sublist if n == parent), None)
                    if parent_score is not None:
                        parent_scores.append((parent, parent_score))

                # 对父节点按分数从高到低排序
                parent_scores = sorted(parent_scores, key=lambda x: x[1], reverse=True)

                # 追加到子列表
                sublist_with_parents.extend(parent_scores)

        # 按分数排序，去重，仅保留按顺序第一个父节点
        unique_parents = []
        seen_parents = set()
        for parent, score in sublist_with_parents:
            if parent not in seen_parents:
                unique_parents.append((parent, score))
                seen_parents.add(parent)

        parent_components_with_scores.append(unique_parents)

    return parent_components_with_scores


def hit_ratio_at_k(actual_root_causes, predicted_top_k, k):
    hr_k = []
    for actual_roots, predicted_roots in zip(actual_root_causes, predicted_top_k):
        # 检查预测的前 k 个根因中是否包含实际根因
        hit = any(root in predicted_roots[:k] for root in actual_roots)
        hr_k.append(hit)
    return np.mean(hr_k)


def ndcg_at_k(actual_root_causes, predicted_top_k, k):
    ndcg_k = []
    for actual_roots, predicted_roots in zip(actual_root_causes, predicted_top_k):
        # 计算 DCG@k
        dcg = 0
        for i, predicted_root in enumerate(predicted_roots[:k]):
            rel = 1 if predicted_root in actual_roots else 0
            dcg += (2**rel - 1) / np.log2(i + 2)  # i+2 for 1-based index

        # 计算 IDCG@k (理想情况下的 DCG@k)
        idcg = 0
        ideal_relevance = [1 if root in actual_roots else 0 for root in predicted_roots[:k]]
        ideal_relevance.sort(reverse=True)
        for i, rel in enumerate(ideal_relevance):
            idcg += (2**rel - 1) / np.log2(i + 2)  # i+2 for 1-based index

        # 计算 NDCG@k
        ndcg_k.append(dcg / idcg if idcg > 0 else 0)

    return np.mean(ndcg_k)


def write_results_to_csv(service_labels, service_lists, nonservice_labels, nonservice_lists, ks=[1, 3, 5], filename="evaluation_results.csv"):
    results = []
    for k in ks:
        hr_k = hit_ratio_at_k(service_labels, service_lists, k)
        ndcg_k = ndcg_at_k(service_labels, service_lists, k)
        nonservicehr_k = hit_ratio_at_k(nonservice_labels, nonservice_lists, k)
        nonservicendcg_k = ndcg_at_k(nonservice_labels, nonservice_lists, k)
        # 保存结果
        # results.append([k, hr_k, ndcg_k, nonservicehr_k, nonservicendcg_k])
        results.append([k, hr_k * 100, ndcg_k * 100, nonservicehr_k * 100, nonservicendcg_k * 100])


    # 使用 pandas DataFrame 存储结果
    df = pd.DataFrame(results, columns=["k", "HR@k", "NDCG@k", "nonserviceHR@k", "nonserviceNDCG@k"])
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")


def sub_main(train_path, predict_path):
    label_match = re.search(r"(\d+\.\d+)_", predict_path)
    label_str = label_match.group(1)
    with open(f"{project_root}/llmrca/rca_algorithm/failure_label.json", "r") as f:
        failure_label = json.load(f)
    service_label = failure_label[label_str]["service"]
    nonservice_label = failure_label[label_str]["nonservice"]

    train_df = pd.read_csv(os.path.join(train_path, "data_final.csv"))
    predict_df = pd.read_csv(os.path.join(predict_path, "data_final.csv"))
    performance_graph_data=quality_graph_data

    train_df = filter_dataframe(train_df, performance_graph_data)
    predict_df = filter_dataframe(predict_df, performance_graph_data)
    # train_df = train_df.iloc[5:90, :]
    # predict_df = predict_df.iloc[2:10, :]

    train_df = train_df
    predict_df = predict_df.iloc[2:10, :]
    # 5：90，3:4
    # 5:90, 2:10

    train_data, node_names = create_graph_from_df(train_df, performance_graph_data)
    predict_data, _ = create_graph_from_df(predict_df, performance_graph_data)

    # X_node1_id = node_names.index("trace@query@llm_token_count_completion")
    # X_node2_id = node_names.index("trace@query@llm_token_count_prompt")
    # y_id = node_names.index("query>duration")
    # logging.info(f"node_ids:{X_node1_id} {X_node2_id} {y_id}")

    model = GraphAnomalyDetectionModel(
        num_nodes=train_data[0].x.shape[0],  # 你应该传入图的节点数
        num_features=train_data[0].x.shape[1],  # 你可以用 DataFrame 的列数来决定每个节点的特征维度
        num_bins=4,  # 根据需求设置聚类数量
        latent_dim=16,  # 潜在空间维度
        encoder_hidden_channels=32,
        decoder_hidden_channels=32,
        epochs=100,  # 设置训练的epoch数量
        model_dir="models",  # 模型保存路径
        num_layers=4,
        dropout_prob=0.01,
        lr=1e-3,
    )
    model.node_recon_errors = {}  # 初始化节点重构误差字典
    model.feature_recon_errors = {}  # 初始化特征重构误差字典

    model_path = project_root + "/llmrca/rca_algorithm/models/encoder.pth"
    if os.path.exists(model_path):
        pass
    else:
        # 训练模型
        model.fit(train_data)
        model.save_models()
    # 预测模型
    reconstructions = model.predict(predict_data)
    test_node_z_scores, test_feature_z_scores = model.compute_anomaly_scores(predict_data, reconstructions)

    sorted_names_all = []
    sorted_names_and_scores_all = []

    for z_scores in test_node_z_scores:
        node_score_pairs = [(score, node_names[idx]) for idx, score in enumerate(z_scores)]
        sorted_node_score_pairs = sorted(node_score_pairs, key=lambda x: x[0], reverse=True)
        sorted_node_names = [node_name for _, node_name in sorted_node_score_pairs]
        sorted_names_all.append(sorted_node_names)
        sorted_names_and_scores_all.append(sorted_node_score_pairs)

    # service_names_all = [[name.replace(">duration", "") for name in sublist if ">duration" in name] for sublist in sorted_names_all]
    # nonservice_names_all = [[name for name in sublist if ">duration" not in name] for sublist in sorted_names_all]

    service_names_and_scores_all = [[(name.replace(">duration", ""), score) for score, name in sublist if ">duration" in name] for sublist in sorted_names_and_scores_all]
    nonservice_names_and_scores_all = [[(name, score) for score, name in sublist if ">duration" not in name] for sublist in sorted_names_and_scores_all]

    # new ======
    new_performance_graph_data = {**performance_graph_data, **add_performance_graph_data}
    component_nodes, leaf_nodes = classify_nodes_with_duration(new_performance_graph_data)
    nonservice_names_and_scores_all = [[(name, score) for score, name in sublist if name in leaf_nodes] for sublist in sorted_names_and_scores_all]
    # 找到对应的父节点 component 节点
    service_names_and_scores_all = find_parent_components_with_scores(nonservice_names_and_scores_all, performance_graph_data, sorted_names_and_scores_all)
    # =========

    service_names_all = [[name.replace(">duration", "") for name, _ in sublist if name != "query"] for sublist in service_names_and_scores_all]
    nonservice_names_all = [[name for name, _ in sublist] for sublist in nonservice_names_and_scores_all]

    service_lists_sub = service_names_all
    nonservice_lists_sub = nonservice_names_all
    service_labels_sub = [service_label] * len(service_lists_sub)
    nonservice_labels_sub = [nonservice_label] * len(service_lists_sub)

    logging.info(f"service_lists_sub:{service_names_and_scores_all[0]}")
    logging.info(f"nonservice_lists_sub:{nonservice_names_and_scores_all[0][:5]}")
    # logging.info(f"service_labels_sub:{service_labels_sub[0]}")
    # logging.info(f"nonservice_labels_sub:{nonservice_labels_sub[0]}")

    return service_lists_sub, nonservice_lists_sub, service_labels_sub, nonservice_labels_sub


if __name__ == "__main__":
    jump_folder = ["7.1", "7.2", "7.3","9.2"]
    model_dir_path = project_root + "/llmrca/rca_algorithm/models"
    for filename in os.listdir(model_dir_path):
        file_path = model_dir_path + "/" + filename
        os.remove(file_path)
        logging.info(f"remove {filename}")

    # /Users/daylight/Desktop/macos/1Code/BiYe/AllResearch/LLMRCA/data/raw_data_all/RGB/RGB_uni_request/final_data
    data_final_dir = project_root + "/data/raw_data_all/RGB/RGB_uni_request/final_data"
    folder_names = [f for f in os.listdir(data_final_dir) if os.path.isdir(os.path.join(data_final_dir, f)) and f != "temp"]
    folder_names.sort()
    train_path = os.path.join(data_final_dir, folder_names[0])

    service_lists = []
    nonservice_lists = []
    service_labels = []
    nonservice_labels = []

    for folder_name in folder_names[1:]:
        predict_path = data_final_dir + "/" + folder_name
        label_match = re.search(r"(\d+\.\d+)_", predict_path)
        label_str = label_match.group(1)
        if label_str in jump_folder:
            continue

        if float(label_match.group(1)) not in quality_label_list:
            continue

        # service_lists_sub, nonservice_lists_sub, service_labels_sub, nonservice_labels_sub = sub_main(train_path, predict_path)

        logging.info(f"{folder_name}")
        # try:
        service_lists_sub, nonservice_lists_sub, service_labels_sub, nonservice_labels_sub = sub_main(train_path, predict_path)
        # except Exception as e:
        #     logging.error(e)
        #     pass

        service_lists.extend(service_lists_sub)
        nonservice_lists.extend(nonservice_lists_sub)
        service_labels.extend(service_labels_sub)
        nonservice_labels.extend(nonservice_labels_sub)
    write_results_to_csv(service_labels, service_lists, nonservice_labels, nonservice_lists)
