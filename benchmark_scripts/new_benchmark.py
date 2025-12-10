import os
import subprocess
import re
import matplotlib.pyplot as plt
import shutil
import time
import pandas as pd

# --- KONFIGURACJA ---
SOURCE_LOG = "docker_full.log"
DATA_DIR = "../bench_data_final"
RESULTS_DIR = "../plots"  # Zmieniłem na lokalny folder plots, dostosuj jeśli trzeba "../plots"
SEQ_EXE = "./log_analyzer_seq"
HYBRID_EXE = "./log_analyzer_hybrid"

# Domyślne ustawienia
DEFAULT_MPI_PROCS = 2
DEFAULT_OMP_THREADS = 4
DEFAULT_BLOCK_SIZE = 256
PHRASES = ["ERROR", "WARNING", "INFO"]

# --- PRZYGOTOWANIE DANYCH ---
def prepare_data():
    """Tworzy katalog z 10 kopiami logów, aby uzyskać ~1.5GB danych."""
    print(f"--- Preparing Data in {DATA_DIR} ---")

    if os.path.exists(DATA_DIR):
        print(f"Directory {DATA_DIR} exists. Checking content...")
        if len(os.listdir(DATA_DIR)) >= 10:
            print("Data seems ready. Skipping generation.")
            return

    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)
    os.makedirs(DATA_DIR)

    # Szukanie pliku źródłowego
    src = SOURCE_LOG
    if not os.path.exists(src):
        # Sprawdź w log_dir jeśli nie ma w root
        src = os.path.join("log_dir", SOURCE_LOG)
        if not os.path.exists(src):
            print(f"ERROR: Could not find {SOURCE_LOG} in root or log_dir.")
            exit(1)

    print(f"Using source: {src}")
    for i in range(10):
        dst = os.path.join(DATA_DIR, f"log_copy_{i}.log")
        shutil.copy(src, dst)
    print(f"Created 10 copies in {DATA_DIR}")

# --- PARSOWANIE WYNIKÓW ---
def parse_output(output_str):
    """Wyciąga Time i Throughput z wyjścia programu C++."""
    time_match = re.search(r"Time:\s+([\d\.]+)\s+s", output_str)
    throughput_match = re.search(r"Throughput:\s+([\d\.]+)\s+GB/s", output_str)

    t_val = float(time_match.group(1)) if time_match else None
    th_val = float(throughput_match.group(1)) if throughput_match else None
    return t_val, th_val

# --- URUCHAMIANIE ---
def run_benchmark(mode, data_path, np=1, threads=1, gpu=False, block_size=256, limit_files=0):
    """
    mode: 'seq' lub 'hybrid'
    """
    cmd = []

    if mode == 'seq':
        cmd = [SEQ_EXE, data_path]
        if limit_files > 0:
            cmd.extend(["--limit-files", str(limit_files)])
        cmd.extend(PHRASES)

    elif mode == 'hybrid':
        # Wykrywanie mpirun
        mpi_cmd = "mpirun" # Zakładam Linux/WSL. Na Windows może być "mpiexec"

        cmd = [mpi_cmd, "-np", str(np), "-x", f"OMP_NUM_THREADS={threads}", HYBRID_EXE, data_path]

        if gpu:
            cmd.append("--gpu")
            cmd.extend(["--block-size", str(block_size)])

        if limit_files > 0:
            cmd.extend(["--limit-files", str(limit_files)])

        cmd.extend(PHRASES)

    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"ERROR: Command failed.\nStderr: {result.stderr}")
            return None, None
        return parse_output(result.stdout)
    except FileNotFoundError:
        print(f"ERROR: Executable not found for command: {cmd}")
        exit(1)

# --- GENEROWANIE WYKRESÓW ---
def save_plot(x, ys, labels, xlabel, ylabel, title, filename):
    plt.figure(figsize=(10, 6))
    for y, label in zip(ys, labels):
        plt.plot(x, y, marker='o', label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.close()
    print(f"Saved plot: {filename}")

def save_bar(x, y, xlabel, ylabel, title, filename):
    plt.figure(figsize=(10, 6))
    x_str = [str(i) for i in x]
    plt.bar(x_str, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.close()
    print(f"Saved plot: {filename}")

# ==============================================================================
# SCENARIUSZE TESTOWE
# ==============================================================================

def main():
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    prepare_data()

    # Obliczamy rozmiar danych raz, przyda się do tabeli
    sample_file = os.path.join(DATA_DIR, "log_copy_0.log")
    single_file_size_mb = os.path.getsize(sample_file) / (1024*1024)
    total_data_size_gb = (10 * single_file_size_mb) / 1024.0

    print("\n=== 1. Calculating Baseline (Sequential) ===")
    t_seq, _ = run_benchmark('seq', DATA_DIR)
    if t_seq is None:
        print("CRITICAL: Failed to get sequential baseline.")
        return
    print(f"Baseline Sequential Time (T_seq): {t_seq:.4f} s")

    # ---------------------------------------------------------
    # SCENARIUSZ 1: Skalowalność OpenMP
    # ---------------------------------------------------------
    print("\n=== 2. OpenMP Scaling (Speedup & Efficiency) ===")
    threads_list = [1, 2, 4, 8, 16]
    omp_times = []

    for th in threads_list:
        t, _ = run_benchmark('hybrid', DATA_DIR, np=DEFAULT_MPI_PROCS, threads=th, gpu=True)
        omp_times.append(t if t else float('inf'))

    omp_speedup = [t_seq / t for t in omp_times]
    omp_efficiency = [s / th for s, th in zip(omp_speedup, threads_list)]

    save_plot(threads_list, [omp_speedup], ["OpenMP Speedup"],
              "Number of Threads", "Speedup (S)", "OpenMP Speedup (S = T_seq / T_omp)",
              "omp_speedup.png")

    save_plot(threads_list, [omp_efficiency], ["OpenMP Efficiency"],
              "Number of Threads", "Efficiency (E)", "OpenMP Efficiency (E = S / Threads)",
              "omp_efficiency.png")

    # ---------------------------------------------------------
    # SCENARIUSZ 2: Skalowalność MPI
    # ---------------------------------------------------------
    print("\n=== 3. MPI Scaling (Time, Speedup, Efficiency) ===")
    procs_list = [1, 2, 4, 8]
    mpi_times = []

    for np in procs_list:
        t, _ = run_benchmark('hybrid', DATA_DIR, np=np, threads=DEFAULT_OMP_THREADS, gpu=True)
        mpi_times.append(t if t else float('inf'))

    mpi_speedup = [t_seq / t for t in mpi_times]
    mpi_efficiency = [s / np for s, np in zip(mpi_speedup, procs_list)]

    save_plot(procs_list, [mpi_times], ["Execution Time"],
              "Number of Processes", "Time (s)", "MPI Execution Time vs Processes",
              "mpi_time.png")

    save_plot(procs_list, [mpi_speedup], ["MPI Speedup"],
              "Number of Processes", "Speedup (S)", "MPI Speedup (S = T_seq / T_mpi)",
              "mpi_speedup.png")

    save_plot(procs_list, [mpi_efficiency], ["MPI Efficiency"],
              "Number of Processes", "Efficiency (E)", "MPI Efficiency (E = S / Processes)",
              "mpi_efficiency.png")

    # ---------------------------------------------------------
    # SCENARIUSZ 3: CUDA Optimization
    # ---------------------------------------------------------
    print("\n=== 4. CUDA Optimization (Throughput) ===")
    block_sizes = [128, 256, 512]
    cuda_throughputs = []

    for bs in block_sizes:
        _, th = run_benchmark('hybrid', DATA_DIR, np=DEFAULT_MPI_PROCS, threads=DEFAULT_OMP_THREADS, gpu=True, block_size=bs)
        cuda_throughputs.append(th if th else 0)

    save_bar(block_sizes, cuda_throughputs,
             "Block Size", "Throughput (GB/s)", "CUDA Throughput vs Block Size",
             "cuda_throughput_block.png")

    # ---------------------------------------------------------
    # SCENARIUSZ 4: CPU vs GPU Time Scaling
    # ---------------------------------------------------------
    print("\n=== 5. CPU vs GPU Time Scaling ===")
    file_limits = [2, 4, 6, 8, 10]
    data_sizes_mb = [n * single_file_size_mb for n in file_limits]

    cpu_times = []
    gpu_times = []

    for limit in file_limits:
        t_cpu, _ = run_benchmark('hybrid', DATA_DIR, np=DEFAULT_MPI_PROCS, threads=DEFAULT_OMP_THREADS, gpu=False, limit_files=limit)
        cpu_times.append(t_cpu if t_cpu else 0)

        t_gpu, _ = run_benchmark('hybrid', DATA_DIR, np=DEFAULT_MPI_PROCS, threads=DEFAULT_OMP_THREADS, gpu=True, limit_files=limit)
        gpu_times.append(t_gpu if t_gpu else 0)

    save_plot(data_sizes_mb, [cpu_times, gpu_times], ["CPU Hybrid", "GPU Hybrid"],
              "Data Size (MB)", "Time (s)", "Execution Time: CPU vs GPU",
              "cpu_gpu_time_scaling.png")

    # ---------------------------------------------------------
    # TABELA PORÓWNAWCZA (ROZSZERZONA)
    # ---------------------------------------------------------
    print("\n=== 6. Generating EXTENDED Comparative Table ===")

    # Definicja scenariuszy do tabeli
    scenarios = [
        {"name": "Sequential (CPU)",       "type": "seq",    "np": 1, "th": 1, "gpu": False},
        {"name": "Pure MPI (CPU)",         "type": "hybrid", "np": 4, "th": 1, "gpu": False},
        {"name": "Pure OpenMP (CPU)",      "type": "hybrid", "np": 1, "th": 4, "gpu": False},
        {"name": "Hybrid CPU (Balanced)",  "type": "hybrid", "np": 2, "th": 4, "gpu": False},
        {"name": "Hybrid GPU (Single Node)","type": "hybrid", "np": 1, "th": 4, "gpu": True},
        {"name": "Hybrid GPU (Balanced)",  "type": "hybrid", "np": 2, "th": 4, "gpu": True},
        {"name": "Hybrid GPU (Max MPI)",   "type": "hybrid", "np": 4, "th": 1, "gpu": True},
    ]

    table_data = {
        "Configuration": [],
        "Details": [],
        "Time (s)": [],
        "Speedup": [],
        "Throughput (GB/s)": []
    }

    for s in scenarios:
        print(f"Benchmarking Table Case: {s['name']}...")

        if s['type'] == 'seq':
            # Używamy wcześniej policzonego baseline
            t = t_seq
            th = total_data_size_gb / t if t > 0 else 0
        else:
            t, th = run_benchmark('hybrid', DATA_DIR, np=s['np'], threads=s['th'], gpu=s['gpu'])

        # Zabezpieczenie przed None
        if t is None: t = 0
        if th is None: th = 0

        speedup = t_seq / t if t > 0 else 0.0

        details = f"{s['np']} Proc, {s['th']} Threads"
        if s['gpu']: details += " + CUDA"

        table_data["Configuration"].append(s['name'])
        table_data["Details"].append(details)
        table_data["Time (s)"].append(f"{t:.4f}")
        table_data["Speedup"].append(f"{speedup:.2f}x")
        table_data["Throughput (GB/s)"].append(f"{th:.3f}")

    df = pd.DataFrame(table_data)

    # Rysowanie tabeli
    fig, ax = plt.subplots(figsize=(12, 4)) # Nieco szersza tabela
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)

    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8) # Wyskalowanie dla czytelności

    plt.title("Extended Performance Comparison Summary")
    plt.savefig(os.path.join(RESULTS_DIR, "comparison_table.png"), bbox_inches='tight')
    plt.close()

    print("\n--- Final Table Data ---")
    print(df)
    print(f"\nAll charts generated in '{RESULTS_DIR}'.")

if __name__ == "__main__":
    main()