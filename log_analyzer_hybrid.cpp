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

// running:
// nvcc -c cuda_analyzer.cu -o cuda_analyzer.o
// mpic++ -fopenmp -std=c++17 -O2 -o log_analyzer_hybrid log_analyzer_hybrid.cpp Serializer.cpp cuda_analyzer.o -L/usr/local/cuda/lib64 -lcudart
// export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
// mpirun -np 2 -x OMP_NUM_THREADS=4 ./log_analyzer_hybrid log_dir "ERROR" "INFO" "WARNING"

namespace fs = std::filesystem;

const std::vector<std::string> DEFAULT_PHRASES = {"ERROR", "WARNING", "INFO"};

enum MsgTag {
    TAG_COUNTS = 1,
    TAG_PERHOUR_SIZE = 2,
    TAG_PERHOUR_DATA = 3,
    TAG_MATCHES_SIZE = 4,
    TAG_MATCHES_DATA = 5
};

extern "C" bool run_cuda_analysis(const char* file_content, size_t file_size,
                                  const std::vector<std::string>& phrases,
                                  std::vector<int>& results);

bool analyze_file(const std::string& filepath, const std::vector<std::string>& phrases, 
                  std::vector<int>& counts, 
                  std::map<std::string, std::map<std::string, int>>& per_hour,
                  std::map<std::string, std::vector<std::string>>& matches) 
{
    std::ifstream in(filepath);
    if (!in) return false;

    std::string line;
    static const std::regex ts_re(R"((\d{4}-\d{2}-\d{2} \d{2}):\d{2}:\d{2})");
    std::smatch m;


    while (std::getline(in, line)) {
        std::string hour = "unknown";
        if (std::regex_search(line, m, ts_re) && m.size() > 1) {
            hour = m[1].str();
        }

        for (size_t i = 0; i < phrases.size(); ++i) {
            if (line.find(phrases[i]) != std::string::npos) {
                counts[i]++; // Tylko inkrementacja
                per_hour[phrases[i]][hour]++;
                matches[phrases[i]].push_back(line);
            }
        }
    }
    return true;
}

int main(int argc, char** argv) {
    // MPI with thread support
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // warning if thread support is not sufficient
    if (provided < MPI_THREAD_FUNNELED && world_rank == 0) {
        std::cerr << "Warning: MPI implementation does not support MPI_THREAD_FUNNELED.\n";
    }

    std::vector<std::string> phrases;
    std::vector<std::string> all_files;
    double start_time, end_time;

    // master (rank 0) setup
    if (world_rank == 0) {
            // --- 1. CZĘŚĆ PRZYWRÓCONA: Wczytywanie argumentów i plików ---
            if (argc < 2) {
                std::cerr << "Usage: " << argv[0] << " <dir> [phrase1 phrase2 ...]\n";
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            fs::path dir = argv[1];
            if (!fs::exists(dir) || !fs::is_directory(dir)) {
                std::cerr << "Not a directory: " << argv[1] << "\n";
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            // Wczytanie fraz z argumentów
            if (argc >= 3) {
                for (int i = 2; i < argc; ++i) phrases.emplace_back(argv[i]);
            } else {
                phrases = DEFAULT_PHRASES;
            }

            // Wczytanie listy plików
            for (const auto& entry : fs::directory_iterator(dir)) {
                if (fs::is_regular_file(entry.path()))
                    all_files.push_back(entry.path().string());
            }

            std::cout << "Hybrid Analyzer started.\n";
            std::cout << "MPI Processes: " << world_size << "\n";
            std::cout << "OpenMP max threads per process: " << omp_get_max_threads() << "\n";
            std::cout << "Files found: " << all_files.size() << "\n";

            // --- 2. CZĘŚĆ CUDA: Benchmark dla jednego pliku ---
            std::cout << "\n--- CUDA vs CPU Performance Benchmark ---\n";
            if (!all_files.empty()) {
                std::string test_file = all_files[0];

                std::ifstream f(test_file, std::ios::binary | std::ios::ate);
                if (f) {
                    size_t size = f.tellg();
                    f.seekg(0, std::ios::beg);
                    std::vector<char> buffer(size);
                    if (f.read(buffer.data(), size)) {

                        // CUDA
                        std::vector<int> gpu_results;
                        double t1 = MPI_Wtime();
                        run_cuda_analysis(buffer.data(), size, phrases, gpu_results);
                        double t2 = MPI_Wtime();
                        double cuda_time = t2 - t1;

                        // CPU (proste porównanie)
                        std::vector<int> cpu_results(phrases.size(), 0);
                        t1 = MPI_Wtime();
                        std::string content_str(buffer.data(), size);
                        for(size_t i=0; i<phrases.size(); ++i) {
                            size_t pos = content_str.find(phrases[i], 0);
                            while(pos != std::string::npos) {
                                cpu_results[i]++;
                                pos = content_str.find(phrases[i], pos + 1);
                            }
                        }
                        t2 = MPI_Wtime();
                        double cpu_time = t2 - t1;

                        // Wyniki benchmarku
                        std::cout << "File: " << test_file << " (" << size / (1024.0*1024.0) << " MB)\n";
                        std::cout << "CUDA Time: " << cuda_time << " s\n";
                        std::cout << "CPU Time:  " << cpu_time << " s\n";
                        if(cuda_time > 0) {
                            std::cout << "Speedup:   " << cpu_time / cuda_time << "x\n";
                            double gb_processed = size / (1024.0 * 1024.0 * 1024.0);
                            std::cout << "CUDA Throughput: " << gb_processed / cuda_time << " GB/s\n";
                        }

                        // Weryfikacja
                        if(!phrases.empty()) {
                             std::cout << "Verification (Count of '" << phrases[0] << "'): GPU="
                                       << gpu_results[0] << ", CPU=" << cpu_results[0] << "\n";
                        }
                    }
                }
            }
        }

    // broadcast configuration
    // broadcast phrases
    int phrase_count = (world_rank == 0) ? phrases.size() : 0;
    MPI_Bcast(&phrase_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (world_rank != 0) phrases.resize(phrase_count);

    for (int i = 0; i < phrase_count; ++i) {
        int len = (world_rank == 0) ? phrases[i].size() : 0;
        MPI_Bcast(&len, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (world_rank != 0) phrases[i].resize(len);
        MPI_Bcast(phrases[i].data(), len, MPI_CHAR, 0, MPI_COMM_WORLD);
    }

    // broadcast file list (serialization -> broadcast -> deserialization)
    // all nodes know about all files even if they don't share the same filesystem view
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

    // synchronization and starting the timer
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    // static load balancing
    // every rank picks files where (index % world_size == world_rank)
    std::vector<std::string> my_files;
    for (size_t i = 0; i < all_files.size(); ++i) {
        if (i % world_size == (size_t)world_rank) {
            my_files.push_back(all_files[i]);
        }
    }

    // local accumulators
    std::vector<int> local_total_counts(phrases.size(), 0);
    std::map<std::string, std::map<std::string, int>> local_per_hour;
    std::map<std::string, std::vector<std::string>> local_matches;

    // OpenMP parallelization
    #pragma omp parallel
    {
        // thread-local accumulators to minimize critical section usage
        std::vector<int> thread_counts(phrases.size(), 0);
        std::map<std::string, std::map<std::string, int>> thread_per_hour;
        std::map<std::string, std::vector<std::string>> thread_matches;

        // dynamic scheduling allows threads to pick up new files as they finish
        #pragma omp for schedule(dynamic)
        for (int i = 0; i < (int)my_files.size(); ++i) {
            // Analyze the file using thread-local variables
            if (!analyze_file(my_files[i], phrases, thread_counts, thread_per_hour, thread_matches)) {
                #pragma omp critical
                std::cerr << "[Rank " << world_rank << "] Failed to open " << my_files[i] << "\n";
            }
        }

        // reducing thread-local results to process-local results
        #pragma omp critical
        {
            for (size_t i = 0; i < phrases.size(); ++i) {
                local_total_counts[i] += thread_counts[i];
            }
            
            // merging maps
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

    // reduction
    // total counts
    std::vector<int> global_counts(phrases.size(), 0);
    MPI_Reduce(local_total_counts.data(), global_counts.data(), phrases.size(), MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // gathering maps/vectors to master
    // MPI_Reduce doesn't support maps, we send 
    // serialized data sent to Rank 0
    std::map<std::string, std::map<std::string, int>> global_per_hour;
    std::map<std::string, std::vector<std::string>> global_matches;

    if (world_rank == 0) {
        // init with master's own data
        global_per_hour = local_per_hour;
        global_matches = local_matches;

        // receiving from others
        for (int src = 1; src < world_size; ++src) {
            int size;
            MPI_Recv(&size, 1, MPI_INT, src, TAG_PERHOUR_SIZE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (size > 0) {
                std::vector<char> buf(size);
                MPI_Recv(buf.data(), size, MPI_CHAR, src, TAG_PERHOUR_DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                auto received = Serializer::deserialize_per_hour(buf);
                // merging
                for (const auto& [ph, h_map] : received) {
                    for (const auto& [h, c] : h_map) global_per_hour[ph][h] += c;
                }
            }

            // receiving matches
            MPI_Recv(&size, 1, MPI_INT, src, TAG_MATCHES_SIZE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (size > 0) {
                std::vector<char> buf(size);
                MPI_Recv(buf.data(), size, MPI_CHAR, src, TAG_MATCHES_DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                auto received = Serializer::deserialize_matches(buf);
                // merging
                for (const auto& [ph, lines] : received) {
                    global_matches[ph].insert(global_matches[ph].end(), lines.begin(), lines.end());
                }
            }
        }
    } else {
        // workers send data to Master
        auto buf_ph = Serializer::serialize_per_hour(local_per_hour);
        int size_ph = buf_ph.size();
        MPI_Send(&size_ph, 1, MPI_INT, 0, TAG_PERHOUR_SIZE, MPI_COMM_WORLD);
        if (size_ph > 0) MPI_Send(buf_ph.data(), size_ph, MPI_CHAR, 0, TAG_PERHOUR_DATA, MPI_COMM_WORLD);

        // matches
        auto buf_m = Serializer::serialize_matches(local_matches);
        int size_m = buf_m.size();
        MPI_Send(&size_m, 1, MPI_INT, 0, TAG_MATCHES_SIZE, MPI_COMM_WORLD);
        if (size_m > 0) MPI_Send(buf_m.data(), size_m, MPI_CHAR, 0, TAG_MATCHES_DATA, MPI_COMM_WORLD);
    }

    end_time = MPI_Wtime();

    if (world_rank == 0) {
        // writing to file
        fs::create_directories("output");
        std::ofstream out("output/matches.txt");
        for (const auto& ph : phrases) {
            out << "=== " << ph << " ===\n";
            if (global_matches.count(ph)) {
                for (const auto& l : global_matches[ph]) out << l << "\n";
            }
            out << "\n";
        }

        // stats
        std::cout << "Total Counts:\n";
        for (size_t i = 0; i < phrases.size(); ++i) {
            std::cout << phrases[i] << ": " << global_counts[i] << "\n";
        }

        std::cout << "\nPer-Hour Statistics:\n";
        // collecting all hours
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

        // measuring performance
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