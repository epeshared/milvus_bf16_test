#!/usr/bin/env python
from pymilvus import MilvusClient, DataType
import numpy as np
from multiprocessing import Pool
import time
import argparse
import concurrent.futures
import os
import sys
import tensorflow as tf
 
# N = int(10000000)  # Number of base vectors
N = 2_50_000  # 1M Number of base vectors
# N = int(10)  # Number of base vectors
#B = 1  # Number of query vectors
K = 100  # Number of similar vectors to search for
# K = 1 # Number of similar vectors to search for
R = 1 #30   # Number of rounds
vector_dim = 1024
# vector_dim = 10
train_seed=2024  # Random seed of training dataset
test_seed=4048   # Random seed of test dataset
insert_batch_size=16000 # batch size to insert data into milvus collection # per PRC max = 64 MB
# batch_size=2
##collection_name="test_collection_bf16_new"
collection_name="mil_vectors_bf16_25w_2"
search_ef=100

def drop_caches():
    with open('/proc/sys/vm/drop_caches', 'w') as f:
        f.write('3\n')

def human_readable_number(value):
# Define the units and their corresponding labels
    units = ['', 'K', 'M', 'B', "T"]
# Find the appropriate unit index based on the magnitude of the number
    unit_index = 0
    while abs(value) >= 1000 and unit_index < len(units) - 1:
        value /= 1000.0
        unit_index += 1
    # Format the number with the corresponding unit label
    return f"{value:.1f}{units[unit_index]}"

def benchmark(batchsize):
    print("Database records:", human_readable_number(N))
    print("Query batch size:", batchsize)
    print("Top K value:", K)


#    target_vectors = [np.random.rand(1024).tolist() for _ in range(batchsize)]
    rng = np.random.default_rng(seed=test_seed)
    #target_vectors = [(rng.random(1024)*2-1).tolist() for _ in range(batchsize)]
    target_vectors = [(rng.random(1024).astype(np.float16)*2-1) for _ in range(batchsize)]
    target_vectors = tf.cast(target_vectors, dtype=tf.bfloat16).numpy()

#     print(target_vectors)

#    drop_caches()
 
    client = MilvusClient(uri="http://localhost:19530")

    print("Begin to load collection ...")
    client.load_collection( collection_name=collection_name )

#    print("Warmup ...")
#    for _ in range(10):
        # Perform the similarity search and record the time taken
#        client.search(
#            collection_name=collection_name, 
#            data=target_vectors,
#            limit=K,
#            search_params={"metric_type": "COSINE" } 
#            search_params={"metric_type": "IP", "params": {} } ,
#            output_fields=["vector"]
#            search_params={"metric_type": "COSINE", "params": { "ef": search_ef}} 
#        )

    print("done.")

    latencies = []  # List to store latencies for each round
    for _ in range(R):
        # Perform the similarity search and record the time taken
        start_time = time.time()

        res = client.search(
            collection_name=collection_name, 
            data=target_vectors,
            anns_field="vector",
            limit=K,
#            search_params={"metric_type": "COSINE", "params": {} }  
            search_params={"metric_type": "IP", "params": {} }  
#            expr=None,
#	    consistency_level="Eventually"
#            search_params={"metric_type": "COSINE", "params": { "ef": search_ef}} 
        )

        end_time = time.time()

        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)

    avgLatency = np.mean(latencies)
    average_throughput = batchsize*N / (avgLatency / 1000)  # In records per second

    print("="*30)
    print("Average Latency (ms): {:.3f}".format(avgLatency))
    print("Average Throughput (records per second): ", int(average_throughput), "\n")

 
def main():

    parser = argparse.ArgumentParser(description='Milvus KNN/ANN benchmark tool')
    subparsers = parser.add_subparsers(dest='command')

    # Run subcommand
    run_parser = subparsers.add_parser('run', help='Run data')
    run_parser.add_argument('--batchsize', type=int, default=1, help='Batch size')
    args = parser.parse_args()

    benchmark(args.batchsize)


if __name__ == "__main__":
    main()
