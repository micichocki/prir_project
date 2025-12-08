import os
import subprocess
import re
import matplotlib.pyplot as plt
import shutil
import random
import datetime
import argparse
from collections import Counter

# --- Configuration ---
PROJECT_DIR = "."
BUILD_CMD = [
    "nvcc -c cuda_analyzer.cu -o cuda_analyzer.o",
    "mpic++ -fopenmp -std=c++17 -O2 -o log_analyzer_hybrid log_analyzer_hybrid.cpp Serializer.cpp cuda_analyzer.o -L/usr/local/cuda/lib64 -lcudart"
]

DATA_DIR_BASE = "bench_data"
RESULTS_DIR = "plots"
DEFAULT_PHRASES = ["ERROR", "WARNING", "INFO", "DEBUG", "CRITICAL"]

# --- Helper Functions ---

def compile_project():
    print("--- Compiling Project ---")
    for cmd in BUILD_CMD:
        print(f"Executing: {cmd}")
        ret = subprocess.run(cmd, shell=True, cwd=PROJECT_DIR)
        if ret.returncode != 0:
            print(f"Error compiling with command: {cmd}")
            return False
    return True

def generate_data(target_size_mb, dir_name):
    print(f"--- Generating {target_size_mb} MB of data in {dir_name} ---")
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name)

    target_bytes = target_size_mb * 1024 * 1024
    current_bytes = 0
    file_idx = 0
    
    while current_bytes < target_bytes:
        chunk_size = min(10 * 1024 * 1024, target_bytes - current_bytes)
        filename = os.path.join(dir_name, f"log_{file_idx}.log")
        
        with open(filename, "w") as f:
            lines = []
            bytes_written = 0
            while bytes_written < chunk_size:
                now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                level = random.choice(DEFAULT_PHRASES + ["TRACE", "VERBOSE"])
                # Add some random words to make Top-N interesting
                words = ["connection", "timeout", "database", "query", "failed", "success", "user", "login", "api", "latency"]
                extra = " ".join(random.choices(words, k=3))
                msg = f"{now} [{level}] {extra} id={random.randint(0,1000)}\n"
                lines.append(msg)
                bytes_written += len(msg)
                
                if len(lines) > 1000:
                    f.writelines(lines)
                    lines = []
            if lines:
                f.writelines(lines)
        
        current_bytes += chunk_size
        file_idx += 1

def parse_output(output_str):
    time_match = re.search(r"Time:\s+([\d\.]+)\s+s", output_str)
    throughput_match = re.search(r"Throughput:\s+([\d\.]+)\s+GB/s", output_str)
    
    time_val = float(time_match.group(1)) if time_match else None
    throughput_val = float(throughput_match.group(1)) if throughput_match else None
    return time_val, throughput_val

def run_benchmark(executable="./log_analyzer_hybrid", data_dir="bench_data", 
                  np=1, omp_threads=1, omp_schedule="dynamic", 
                  use_gpu=False, block_size=256,
                  phrases=DEFAULT_PHRASES):
    
    cmd = [
        "mpirun", "-np", str(np),
        "-x", f"OMP_NUM_THREADS={omp_threads}",
        "-x", f"OMP_SCHEDULE={omp_schedule}",
        executable, data_dir
    ]
    if use_gpu:
        cmd.append("--gpu")
        cmd.extend(["--block-size", str(block_size)])
    
    cmd.extend(phrases)
    
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_DIR)
        if result.returncode != 0:
            print("Error running command:")
            print(result.stderr)
            return None, None
        return parse_output(result.stdout)
    except Exception as e:
        print(f"Exception running benchmark: {e}")
        return None, None

# --- Plotting Functions ---

def plot_metric(x_values, y_values_dict, x_label, y_label, title, filename):
    plt.figure(figsize=(10, 6))
    for label, y_values in y_values_dict.items():
        plt.plot(x_values, y_values, marker='o', label=label)
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.close()

# --- Scenarios ---

def scenario_threads_scaling(base_data_dir):
    print("\n--- Scenario: Thread Scaling (GPU, Fixed Processes=2, Fixed Data) ---")
    threads = [1, 2, 4, 8, 16]
    modes = ["static", "dynamic"]
    results_time = {m: [] for m in modes}
    results_throughput = {m: [] for m in modes}
    
    t_seq, _ = run_benchmark(data_dir=base_data_dir, np=1, omp_threads=1, use_gpu=False)
    
    for mode in modes:
        for th in threads:
            t, throu = run_benchmark(data_dir=base_data_dir, np=2, omp_threads=th, omp_schedule=mode, use_gpu=True)
            if t:
                results_time[mode].append(t)
                results_throughput[mode].append(throu)
            else:
                results_time[mode].append(0)
                results_throughput[mode].append(0)
    
    plot_metric(threads, results_time, "Threads per Process", "Time (s)", "Thread Scaling Time (GPU, np=2)", "threads_time.png")
    plot_metric(threads, results_throughput, "Threads per Process", "Throughput (GB/s)", "Thread Scaling Throughput (GPU, np=2)", "threads_throughput.png")

def scenario_process_scaling(base_data_dir):
    print("\n--- Scenario: Process Scaling (GPU, Fixed Threads=4) ---")
    procs = [1, 2, 4, 8]
    threads = 4
    results_throughput = {"GPU": []}
    
    for p in procs:
        t, throu = run_benchmark(data_dir=base_data_dir, np=p, omp_threads=threads, omp_schedule="dynamic", use_gpu=True)
        if throu:
            results_throughput["GPU"].append(throu)
        else:
             results_throughput["GPU"].append(0)

    plot_metric(procs, results_throughput, "MPI Processes", "Throughput (GB/s)", "Process Scaling Throughput (GPU, th=4)", "procs_throughput.png")

def scenario_block_size(base_data_dir):
    print("\n--- Scenario: CUDA Block Size (128, 256, 512) ---")
    block_sizes = [128, 256, 512]
    times = []
    
    for bs in block_sizes:
        t, throu = run_benchmark(data_dir=base_data_dir, np=2, omp_threads=4, omp_schedule="dynamic", use_gpu=True, block_size=bs)
        times.append(t if t else 0)
        
    plt.figure()
    plt.bar([str(b) for b in block_sizes], times)
    plt.xlabel("Block Size")
    plt.ylabel("Time (s)")
    plt.title("Performance vs CUDA Block Size")
    plt.savefig(os.path.join(RESULTS_DIR, "block_size_time.png"))
    plt.close()

def scenario_cpu_vs_gpu_data_scaling():
    print("\n--- Scenario: CPU vs GPU Data Scaling ---")
    sizes_mb = [100, 500] 
    results_throughput = {"CPU": [], "GPU": []}
    
    for size in sizes_mb:
        dir_name = os.path.join(PROJECT_DIR, f"bench_data_{size}MB")
        generate_data(size, dir_name)
        
        _, throu_cpu = run_benchmark(data_dir=dir_name, np=2, omp_threads=4, use_gpu=False)
        _, throu_gpu = run_benchmark(data_dir=dir_name, np=2, omp_threads=4, use_gpu=True, block_size=256)
        
        results_throughput["CPU"].append(throu_cpu if throu_cpu else 0)
        results_throughput["GPU"].append(throu_gpu if throu_gpu else 0)
        
        shutil.rmtree(dir_name)
        
    plot_metric(sizes_mb, results_throughput, "Data Size (MB)", "Throughput (GB/s)", "CPU vs GPU Throughput Scaling", "cpu_gpu_scaling_throughput.png")

def analyze_top_n_words(dir_name, n=10):
    print(f"\n--- Analyzing Top-{n} Words ---")
    cnt = Counter()
    # Analyze sample files (up to 5)
    files = [os.path.join(dir_name, f) for f in os.listdir(dir_name) if f.endswith(".log")][:5]
    
    for fpath in files:
        with open(fpath, "r") as f:
            for line in f:
                # Basic tokenization
                words = re.findall(r'\b\w+\b', line.lower())
                cnt.update(words)
    
    top_n = cnt.most_common(n)
    print(f"Top {n} words: {top_n}")
    
    words, counts = zip(*top_n)
    plt.figure(figsize=(10, 6))
    plt.bar(words, counts)
    plt.title(f"Top {n} Most Frequent Words")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(RESULTS_DIR, "top_n_words.png"))
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", action="store_true", help="Compile project before running")
    parser.add_argument("--test", action="store_true", help="Run a quick smoke test")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    args = parser.parse_args()

    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    if args.compile:
        if not compile_project():
            return

    if args.test:
        dir_name = "test_data"
        generate_data(10, dir_name)
        print("Running CPU Test...")
        run_benchmark(data_dir=dir_name, np=2, omp_threads=2, use_gpu=False)
        shutil.rmtree(dir_name)

    if args.all:
        bench_data_dir = os.path.join(PROJECT_DIR, "bench_data_main")
        generate_data(500, bench_data_dir) 

        scenario_threads_scaling(bench_data_dir)
        scenario_process_scaling(bench_data_dir)
        scenario_block_size(bench_data_dir)
        analyze_top_n_words(bench_data_dir)
        
        shutil.rmtree(bench_data_dir)
        
        scenario_cpu_vs_gpu_data_scaling()
        
        print(f"\nBenchmarks completed. Results saved in {RESULTS_DIR}")

if __name__ == "__main__":
    main()
