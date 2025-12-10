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
# CMake build commands (Not used in manual compile mode but kept for ref)
BUILD_CMD = [
    "mkdir build",
    "cd build && cmake ..",
    "cd build && cmake --build . --config Release"
]

DATA_DIR_BASE = "bench_data"
RESULTS_DIR = "plots_bak"
DEFAULT_PHRASES = ["ERROR", "WARNING", "INFO", "DEBUG", "CRITICAL"]

# --- Helper Functions ---

def generate_data(target_size_mb, dir_name):
    print(f"--- Checking data in {dir_name} ---")
    if os.path.exists(dir_name) and os.listdir(dir_name):
        print(f"Data directory {dir_name} already exists and is not empty. Skipping generation.")
        return
    
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name)
    print(f"--- Generating ~{target_size_mb} MB of data in {dir_name} ---")

    source_log = "docker_full.log"
    use_real_data = os.path.exists(source_log)
    
    if use_real_data:
        src_size = os.path.getsize(source_log)
        # Calculate copies needed.
        num_copies = int((target_size_mb * 1024 * 1024) / src_size)
        if num_copies < 1: num_copies = 1
        
        print(f"Using '{source_log}' (Size: {src_size/1024/1024:.2f} MB). Creating {num_copies} copies.")
        
        for i in range(num_copies):
            dst = os.path.join(dir_name, f"log_{i}.log")
            shutil.copy(source_log, dst)
            
    else:
        print(f"'{source_log}' not found. Falling back to synthetic generation.")
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

def compile_project():
    print("--- Compiling Project Manually (Bypassing CMake) ---")
    
    # 1. Compile CUDA kernel
    print("[1/2] Compiling CUDA kernel...")
    
    # Try standard compile first
    # Use explicit cl.exe path to avoid PATH issues
    cl_path = r"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64\cl.exe"
    
    cmd_cuda = f'nvcc -ccbin "{cl_path}" -Xcompiler "/MD" -c cuda_analyzer.cu -o cuda_analyzer.obj'
    print(f"Executing: {cmd_cuda}")
    ret = subprocess.run(cmd_cuda, shell=True, cwd=PROJECT_DIR)
    
    if ret.returncode != 0:
        print("Standard CUDA compilation failed. Retrying with conservative logic...")
        # Retry with no optimizations and explicit CCBIN
        cmd_cuda_safe = f'nvcc -ccbin "{cl_path}" -Xcompiler "/MD" -O0 -arch=sm_86 -c cuda_analyzer.cu -o cuda_analyzer.obj'
        print(f"Executing: {cmd_cuda_safe}")
        ret = subprocess.run(cmd_cuda_safe, shell=True, cwd=PROJECT_DIR)
        
        if ret.returncode != 0:
            print("Error compiling CUDA kernel even with explicit compiler path.")
            return False
        
    # 2. Compile and Link Host Code (MSVC)
    print("[2/2] Compiling Host Code and Linking...")
    
    # Try to find MPI paths
    mpi_inc = r"C:\Program Files (x86)\Microsoft SDKs\MPI\Include"
    mpi_lib = r"C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64"
    
    if not os.path.exists(mpi_inc):
        # Fallback check
        mpi_inc = r"C:\Program Files\Microsoft MPI\Bin\..\Include"
    
    # Link with CUDA (Required)
    cuda_lib_dir = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\lib\x64"
    
    # Explicitly add MSVC and Windows SDK x64 lib paths
    msvc_lib = r"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC\14.29.30133\lib\x64"
    sdk_um_lib = r"C:\Program Files (x86)\Windows Kits\10\lib\10.0.19041.0\um\x64"
    sdk_ucrt_lib = r"C:\Program Files (x86)\Windows Kits\10\lib\10.0.19041.0\ucrt\x64"
    
    # REQUIRED INCLUDE PATHS (Fixes 'iostream' not found)
    msvc_inc = r"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC\14.29.30133\include"
    sdk_ucrt_inc = r"C:\Program Files (x86)\Windows Kits\10\Include\10.0.19041.0\ucrt"
    sdk_um_inc = r"C:\Program Files (x86)\Windows Kits\10\Include\10.0.19041.0\um"
    sdk_shared_inc = r"C:\Program Files (x86)\Windows Kits\10\Include\10.0.19041.0\shared"
    
    cmd_host = (
        f'"{cl_path}" /EHsc /std:c++17 /openmp /DUSE_CUDA /MD '
        f'/I"{mpi_inc}" /I"{msvc_inc}" /I"{sdk_ucrt_inc}" /I"{sdk_um_inc}" /I"{sdk_shared_inc}" '
        f'log_analyzer_hybrid.cpp Serializer.cpp cuda_analyzer.obj '
        f'/link /LIBPATH:"{mpi_lib}" /LIBPATH:"{cuda_lib_dir}" '
        f'/LIBPATH:"{msvc_lib}" /LIBPATH:"{sdk_um_lib}" /LIBPATH:"{sdk_ucrt_lib}" '
        f'msmpi.lib cudart.lib kernel32.lib user32.lib uuid.lib '
        f'/out:log_analyzer_hybrid.exe'
    )
    
    print(f"Executing: {cmd_host}")
    ret = subprocess.run(cmd_host, shell=True, cwd=PROJECT_DIR)
    if ret.returncode != 0:
        print("Error linking project (Host).")
        return False

    print("Compilation successful.")
    return True

def parse_output(output_str):
    time_match = re.search(r"Time:\s+([\d\.]+)\s+s", output_str)
    throughput_match = re.search(r"Throughput:\s+([\d\.]+)\s+GB/s", output_str)
    
    time_val = float(time_match.group(1)) if time_match else None
    throughput_val = float(throughput_match.group(1)) if throughput_match else None
    return time_val, throughput_val

def run_benchmark(executable="./log_analyzer_hybrid", data_dir="bench_data", 
                  np=1, omp_threads=1, omp_schedule="dynamic", 
                  use_gpu=False, block_size=256,
                  phrases=DEFAULT_PHRASES,
                  extra_args=None):
    
    # Detect MPI runner
    mpi_runner = shutil.which("mpirun") or shutil.which("mpiexec")
    if not mpi_runner:
        print("Error: 'mpirun' or 'mpiexec' not found in PATH. Please install MPI (e.g., MS-MPI on Windows) and ensure it's in your PATH.")
        return None, None

    # Handle Executable Path on Windows
    if os.name == 'nt':
        if not executable.endswith('.exe'):
            executable += '.exe'
        if not os.path.exists(executable):
             print(f"Warning: {executable} not found.")
            
    # Normalize path - USE ABSOLUTE PATH for MPI safety on Windows
    executable = os.path.abspath(executable)

    cmd = [
        mpi_runner, "-np", str(np),
        "-env", "OMP_NUM_THREADS", str(omp_threads),
        "-env", "OMP_SCHEDULE", str(omp_schedule),
        executable, data_dir
    ]
    if use_gpu:
        cmd.append("--gpu")
        cmd.extend(["--block-size", str(block_size)])
    
    if extra_args:
        cmd.extend(extra_args)
    
    cmd.extend(phrases)
    
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_DIR)
        if result.returncode != 0:
            print(f"Error running command (Return Code: {result.returncode}):")
            print("--- STDOUT ---")
            print(result.stdout)
            print("--- STDERR ---")
            print(result.stderr)
            print("----------------")
            return None, None

        t, throu = parse_output(result.stdout)
        if t is None or throu is None:
             print(f"WARNING: Parse failed for command: {' '.join(cmd)}")
             print("--- STDOUT ---")
             print(result.stdout)
             print("--- STDERR ---")
             print(result.stderr)
             print("----------------")
        return t, throu
    except FileNotFoundError:
             print(f"Error: The executable '{executable}' (or the MPI runner) was not found.")
             return None, None
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
    
    print("Running baseline CPU (np=1, th=4)...")
    t_seq, _ = run_benchmark(data_dir=base_data_dir, np=1, omp_threads=4, use_gpu=False)
    
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
    print("\n--- Scenario: CPU vs GPU Data Scaling (Granular 10-100 files) ---")
    
    
    # 100 files * 14.8MB = ~1480MB
    base_target_mb = 1500 
    dir_name = os.path.join(PROJECT_DIR, "bench_data_scaling")
    generate_data(base_target_mb, dir_name)
    
    # 2. Loop from 10 to 100 files
    file_counts = list(range(10, 101, 10)) # 10, 20, ..., 100
    results_throughput = {"CPU": [], "GPU": []}
    
    for count in file_counts:
        print(f"Benchmarking with {count} files...")
        
        # CPU Run
        _, throu_cpu = run_benchmark(
            data_dir=dir_name, 
            np=2, 
            omp_threads=4, 
            use_gpu=False,
            extra_args=["--limit-files", str(count)]
        )
        
        # GPU Run
        _, throu_gpu = run_benchmark(
            data_dir=dir_name, 
            np=2, 
            omp_threads=4, 
            use_gpu=True, 
            block_size=256,
            extra_args=["--limit-files", str(count)]
        )
        
        results_throughput["CPU"].append(throu_cpu if throu_cpu else 0)
        results_throughput["GPU"].append(throu_gpu if throu_gpu else 0)
        
    plot_metric(file_counts, results_throughput, "Number of Files (approx 15MB each)", "Throughput (GB/s)", "CPU vs GPU Throughput Scaling", "cpu_gpu_scaling_throughput.png")

def analyze_top_n_words(dir_name, n=10):
    print(f"\n--- Analyzing Top-{n} Words ---")
    cnt = Counter()
    # Analyze sample files (up to 5)
    files = [os.path.join(dir_name, f) for f in os.listdir(dir_name) if f.endswith(".log")][:5]
    
    for fpath in files:
        with open(fpath, "r") as f:
            for line in f:
                # Basic tokenization - ignore purely numeric tokens (timestamps/ids)
                words = re.findall(r'\b[a-zA-Z_]+\b', line.lower())
                cnt.update(words)
    
    top_n = cnt.most_common(n)
    print(f"Top {n} words: {top_n}")
    
    if top_n:
        words, counts = zip(*top_n)
        plt.figure(figsize=(10, 6))
        plt.bar(words, counts)
        plt.title(f"Top {n} Most Frequent Words")
        plt.ylabel("Frequency")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
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
        t, throu = run_benchmark(data_dir=dir_name, np=2, omp_threads=2, use_gpu=False)
        print(f"Test Result -> Time: {t} s, Throughput: {throu} GB/s")
        # shutil.rmtree(dir_name)

    if args.all:
        bench_data_dir = os.path.join(PROJECT_DIR, "bench_data_main")
        # Main benchmark now uses 1500MB (1.5GB) as requested
        generate_data(1500, bench_data_dir) 

        scenario_threads_scaling(bench_data_dir)
        scenario_process_scaling(bench_data_dir)
        scenario_block_size(bench_data_dir)
        analyze_top_n_words(bench_data_dir)
        
        # shutil.rmtree(bench_data_dir)
        
        scenario_cpu_vs_gpu_data_scaling()
        
        print(f"\nBenchmarks completed. Results saved in {RESULTS_DIR}")

if __name__ == "__main__":
    main()
