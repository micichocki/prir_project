#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <vector>
#include <regex>
#include <set>
#include <filesystem>
#include <algorithm>
#include <sstream>
#include <cstring>
#include <mpi.h>
#include <omp.h>
#include "Serializer.h"


// Uruchomienie
//  nvcc -c cuda_analyzer.cu -o cuda_analyzer.o
// mpic++ -fopenmp -std=c++17 -O2 -DUSE_CUDA -o log_analyzer_hybrid log_analyzer_hybrid.cpp Serializer.cpp cuda_analyzer.o -L/usr/local/cuda/lib64 -lcudart
// export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
// Zliczanie na CPU: mpirun -np 2 -x OMP_NUM_THREADS=4 ./log_analyzer_hybrid log_dir "ERROR" "WARNING"
// Zliczanie na GPU: mpirun -np 2 -x OMP_NUM_THREADS=4 ./log_analyzer_hybrid log_dir --gpu "ERROR" "WARNING"


namespace fs = std::filesystem;

const std::vector<std::string> DEFAULT_PHRASES = {"ERROR", "WARNING", "INFO"};

enum MsgTag {
    TAG_COUNTS = 1,
    TAG_PERHOUR_SIZE = 2,
    TAG_PERHOUR_DATA = 3,
    TAG_MATCHES_SIZE = 4,
    TAG_MATCHES_DATA = 5
};

// deklaracja CUDA
#ifdef USE_CUDA
extern "C" bool run_cuda_analysis_raw(const char* file_content, size_t file_size,
                                      const char* flat_phrases, int flat_phrases_size,
                                      const int* offsets,
                                      const int* lengths,
                                      int num_keywords,
                                      int* out_results,
                                      int blockSize);

bool run_cuda_analysis(const char* file_content, size_t file_size,
                       const std::vector<std::string>& phrases,
                       std::vector<int>& results,
                       int blockSize)
{
    std::string flat_phrases;
    std::vector<int> offsets;
    std::vector<int> lengths;

    for (const auto& p : phrases) {
        offsets.push_back(flat_phrases.size());
        lengths.push_back(p.size());
        flat_phrases += p;
    }

    results.resize(phrases.size());

    return run_cuda_analysis_raw(file_content, file_size,
                                 flat_phrases.data(), flat_phrases.size(),
                                 offsets.data(), lengths.data(),
                                 phrases.size(),
                                 results.data(),
                                 blockSize);
}
#endif
void analyze_content_for_time_and_matches(const std::string& content_str,
                                          const std::vector<std::string>& phrases,
                                          std::map<std::string, std::map<std::string, int>>& per_hour,
                                          std::map<std::string, std::vector<std::string>>& matches)
{
    std::stringstream ss(content_str);
    std::string line;
    static const std::regex ts_re(R"((\d{4}-\d{2}-\d{2} \d{2}):\d{2}:\d{2})");
    std::smatch m;

    while (std::getline(ss, line)) {
        std::string hour = "unknown";
        if (std::regex_search(line, m, ts_re) && m.size() > 1) {
            hour = m[1].str();
        }
        for (size_t i = 0; i < phrases.size(); ++i) {
            if (line.find(phrases[i]) != std::string::npos) {
                // Zliczanie per hour i zapisywanie linii do matches
                per_hour[phrases[i]][hour]++;
                matches[phrases[i]].push_back(line);
            }
        }
    }
}

// Funkcja główna do przetwarzania OpenMP
void run_openmp_analysis(const std::vector<std::string>& my_files,
                         const std::vector<std::string>& phrases,
                         std::vector<int>& local_total_counts,
                         std::map<std::string, std::map<std::string, int>>& local_per_hour,
                         std::map<std::string, std::vector<std::string>>& local_matches,
                         bool use_gpu_for_counting,
                         int cuda_block_size,
                         int world_rank)
{
    #pragma omp parallel
    {
        std::vector<int> thread_counts(phrases.size(), 0);
        std::map<std::string, std::map<std::string, int>> thread_per_hour;
        std::map<std::string, std::vector<std::string>> thread_matches;

        #pragma omp for schedule(runtime)
        for (int i = 0; i < (int)my_files.size(); ++i) {
            std::string current_file = my_files[i];

            // Wczytanie pliku do bufora (konieczne dla CUDA i dla pomiaru I/O)
            std::ifstream f(current_file, std::ios::binary | std::ios::ate);
            if (!f) {
                #pragma omp critical
                std::cerr << "[Rank " << world_rank << "] Failed to open " << current_file << "\n";
                continue;
            }
            size_t size = f.tellg();
            f.seekg(0, std::ios::beg);
            std::vector<char> buffer(size);
            f.read(buffer.data(), size);
            std::string content_str(buffer.data(), size);

            std::vector<int> current_file_counts(phrases.size(), 0);

            if (use_gpu_for_counting) {
                // A. TRYB GPU: Zliczanie na GPU
                if (!run_cuda_analysis(buffer.data(), size, phrases, current_file_counts, cuda_block_size)) {
                    #pragma omp critical
                    std::cerr << "[Rank " << world_rank << "] CUDA failure for " << current_file << "\n";
                    continue;
                }
            } else {
                // B. TRYB CPU (SZYBKI ALGORYTM): Zliczanie na CPU
                for(size_t j=0; j<phrases.size(); ++j) {
                    size_t pos = content_str.find(phrases[j], 0);
                    while(pos != std::string::npos) {
                        current_file_counts[j]++;
                        pos = content_str.find(phrases[j], pos + 1);
                    }
                }
            }

            // Analiza statystyk godzinowych i filtrowania (Zawsze na CPU)
            analyze_content_for_time_and_matches(content_str, phrases, thread_per_hour, thread_matches);

            // Dodanie wyników zliczania (z GPU lub CPU) do akumulatora wątkowego
            for (size_t j = 0; j < phrases.size(); ++j) {
                thread_counts[j] += current_file_counts[j];
            }
        }

        // Redukcja wyników wątkowych do lokalnych akumulatorów procesu
        #pragma omp critical
        {
            for (size_t i = 0; i < phrases.size(); ++i) {
                local_total_counts[i] += thread_counts[i];
            }
            // Mergowanie map per_hour
            for (const auto& [ph, hour_map] : thread_per_hour) {
                for (const auto& [h, c] : hour_map) {
                    local_per_hour[ph][h] += c;
                }
            }
            for (const auto& [ph, lines] : thread_matches) {
                local_matches[ph].insert(local_matches[ph].end(), lines.begin(), lines.end());
            }
        }
    }
}


int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (provided < MPI_THREAD_FUNNELED && world_rank == 0) {
        std::cerr << "Warning: MPI implementation does not support MPI_THREAD_FUNNELED.\n";
    }

    std::vector<std::string> phrases;
    std::vector<std::string> all_files;
    double start_time, end_time;
    bool use_gpu_flag = false;
    int cuda_block_size = 256;

    // Master (rank 0) setup
    if (world_rank == 0) {
        // Wczytywanie argumentów i plików
        if (argc < 2) {
            std::cerr << "Usage: " << argv[0] << " <dir> [--gpu] [phrase1 phrase2 ...]\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Sprawdzenie flagi GPU i zebranie fraz
        int file_limit = 0;

        for (int i = 1; i < argc; ++i) {
            if (std::string(argv[i]) == "--gpu") {
                use_gpu_flag = true;
            } else if (std::string(argv[i]) == "--block-size" && i + 1 < argc) {
                cuda_block_size = std::stoi(argv[++i]);
            } else if (std::string(argv[i]) == "--limit-files" && i + 1 < argc) {
                file_limit = std::stoi(argv[++i]);
            } else if (i != 1) { // Argumenty od 2 w górę (po katalogu) są frazami, chyba że to flaga
                phrases.emplace_back(argv[i]);
            }
        }
        if (phrases.empty()) {
            phrases = DEFAULT_PHRASES;
        }

        fs::path dir = argv[1];
        if (!fs::exists(dir) || !fs::is_directory(dir)) {
            std::cerr << "Not a directory: " << argv[1] << "\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Wczytanie listy plików
        for (const auto& entry : fs::directory_iterator(dir)) {
            if (fs::is_regular_file(entry.path()))
                all_files.push_back(entry.path().string());
        }
        
        // Sortowanie plików dla spójności
        std::sort(all_files.begin(), all_files.end());

		// Zastosowanie limitu plików, jeśli podano
        if (file_limit > 0 && (size_t)file_limit < all_files.size()) {
            all_files.resize(file_limit);
        }

        std::cout << "Hybrid Analyzer started.\n";
        std::cout << "Mode: " << (use_gpu_flag ? "GPU Accelerated" : "Pure CPU (Fast Search)") << "\n";
        std::cout << "MPI Processes: " << world_size << "\n";
        std::cout << "OpenMP max threads per process: " << omp_get_max_threads() << "\n";
        std::cout << "Files found: " << all_files.size() << "\n";
    }

    // Broadcast zmiennych konfiguracyjnych
    MPI_Bcast(&use_gpu_flag, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cuda_block_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Broadcast fraz
    int phrase_count = (world_rank == 0) ? phrases.size() : 0;
    MPI_Bcast(&phrase_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (world_rank != 0) phrases.resize(phrase_count);
    for (int i = 0; i < phrase_count; ++i) {
        int len = (world_rank == 0) ? phrases[i].size() : 0;
        MPI_Bcast(&len, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (world_rank != 0) phrases[i].resize(len);
        MPI_Bcast(phrases[i].data(), len, MPI_CHAR, 0, MPI_COMM_WORLD);
    }

    // Broadcast listy plików
    std::vector<char> file_list_buf;
    int file_list_size = 0;
    if (world_rank == 0) {
        file_list_buf = Serializer::serialize_string_list(all_files);
        file_list_size = file_list_buf.size();
    }
    MPI_Bcast(&file_list_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (world_rank != 0) file_list_buf.resize(file_list_size);
    MPI_Bcast(file_list_buf.data(), file_list_size, MPI_CHAR, 0, MPI_COMM_WORLD);
    if (world_rank != 0) {
        all_files = Serializer::deserialize_string_list(file_list_buf);
    }

    // Synchronizacja i rozpoczęcie pomiaru czasu
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    // Podział plików między procesy MPI
    std::vector<std::string> my_files;
    for (size_t i = 0; i < all_files.size(); ++i) {
        if (i % world_size == (size_t)world_rank) {
            my_files.push_back(all_files[i]);
        }
    }

    // Lokalne akumulatory wyników
    std::vector<int> local_total_counts(phrases.size(), 0);
    std::map<std::string, std::map<std::string, int>> local_per_hour;
    std::map<std::string, std::vector<std::string>> local_matches;

    // Uruchomienie analizy OpenMP/GPU
    run_openmp_analysis(my_files, phrases,
                        local_total_counts, local_per_hour, local_matches,
                        use_gpu_flag, cuda_block_size, world_rank);

    // Redukcja wyników zliczania do mastera
    std::vector<int> global_counts(phrases.size(), 0);
    MPI_Reduce(local_total_counts.data(), global_counts.data(), phrases.size(), MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Redukcja wyników per-hour i matches do mastera
    std::map<std::string, std::map<std::string, int>> global_per_hour;
    std::map<std::string, std::vector<std::string>> global_matches;

    // Kod redukcji map
    if (world_rank == 0) {
        global_per_hour = local_per_hour;
        global_matches = local_matches;

        for (int src = 1; src < world_size; ++src) {
            int size;
            MPI_Recv(&size, 1, MPI_INT, src, TAG_PERHOUR_SIZE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (size > 0) {
                std::vector<char> buf(size);
                MPI_Recv(buf.data(), size, MPI_CHAR, src, TAG_PERHOUR_DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                auto received = Serializer::deserialize_per_hour(buf);
                for (const auto& [ph, h_map] : received) {
                    for (const auto& [h, c] : h_map) global_per_hour[ph][h] += c;
                }
            }
            MPI_Recv(&size, 1, MPI_INT, src, TAG_MATCHES_SIZE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (size > 0) {
                std::vector<char> buf(size);
                MPI_Recv(buf.data(), size, MPI_CHAR, src, TAG_MATCHES_DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                auto received = Serializer::deserialize_matches(buf);
                for (const auto& [ph, lines] : received) {
                    global_matches[ph].insert(global_matches[ph].end(), lines.begin(), lines.end());
                }
            }
        }
    } else {
        auto buf_ph = Serializer::serialize_per_hour(local_per_hour);
        int size_ph = buf_ph.size();
        MPI_Send(&size_ph, 1, MPI_INT, 0, TAG_PERHOUR_SIZE, MPI_COMM_WORLD);
        if (size_ph > 0) MPI_Send(buf_ph.data(), size_ph, MPI_CHAR, 0, TAG_PERHOUR_DATA, MPI_COMM_WORLD);

        auto buf_m = Serializer::serialize_matches(local_matches);
        int size_m = buf_m.size();
        MPI_Send(&size_m, 1, MPI_INT, 0, TAG_MATCHES_SIZE, MPI_COMM_WORLD);
        if (size_m > 0) MPI_Send(buf_m.data(), size_m, MPI_CHAR, 0, TAG_MATCHES_DATA, MPI_COMM_WORLD);
    }

    end_time = MPI_Wtime();

    if (world_rank == 0) {
        // Zapis wyników do plików
        fs::create_directories("output");
        std::ofstream out("output/matches.txt");
        for (const auto& ph : phrases) {
            out << "=== " << ph << " ===\n";
            if (global_matches.count(ph)) {
                for (const auto& l : global_matches[ph]) out << l << "\n";
            }
            out << "\n";
        }

        std::cout << "Total Counts:\n";
        for (size_t i = 0; i < phrases.size(); ++i) {
            std::cout << phrases[i] << ": " << global_counts[i] << "\n";
        }

        std::cout << "\nPer-Hour Statistics:\n";
        std::set<std::string> all_hours;
        for (const auto& pair : global_per_hour) {
            for (const auto& h_pair : pair.second) all_hours.insert(h_pair.first);
        }
        for (const auto& h : all_hours) {
            std::cout << h;
            for (const auto& ph : phrases) {
                std::cout << " " << ph << ":" << global_per_hour[ph][h];
            }
            std::cout << "\n";
        }

        // Raport wydajności
        double elapsed = end_time - start_time;
        unsigned long long total_bytes = 0;
        for (const auto& f : all_files) {
            std::error_code ec;
            total_bytes += fs::file_size(f, ec);
        }
        double gb = total_bytes / (1024.0 * 1024.0 * 1024.0);


        std::cout << "Time: " << elapsed << " s\n";
        std::cout << "Total Size: " << gb << " GB\n";
        std::cout << "Throughput: " << (elapsed > 0 ? gb / elapsed : 0.0) << " GB/s\n";
    }

    MPI_Finalize();
    return 0;
}