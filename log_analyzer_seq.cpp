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
#include <chrono>

namespace fs = std::filesystem;

const std::vector<std::string> DEFAULT_PHRASES = {"ERROR", "WARNING", "INFO"};

// Delaracja CUDA
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

// Funckja analizująca zawartość pliku pod kątem znaczników czasu i dopasowań
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
                per_hour[phrases[i]][hour]++;
                matches[phrases[i]].push_back(line);
            }
        }
    }
}

// Główna funkcja analizy sekwencyjnej
void run_sequential_analysis(const std::vector<std::string>& files,
                             const std::vector<std::string>& phrases,
                             std::vector<int>& total_counts,
                             std::map<std::string, std::map<std::string, int>>& per_hour,
                             std::map<std::string, std::vector<std::string>>& matches,
                             bool use_gpu_for_counting,
                             int cuda_block_size)
{
    // Inicjalizacja liczników
    total_counts.assign(phrases.size(), 0);

    // Pętla sekwencyjna po wszystkich plikach
    for (const auto& current_file : files) {

        // Wczytanie pliku
        std::ifstream f(current_file, std::ios::binary | std::ios::ate);
        if (!f) {
            std::cerr << "Failed to open " << current_file << "\n";
            continue;
        }
        size_t size = f.tellg();
        f.seekg(0, std::ios::beg);
        std::vector<char> buffer(size);
        f.read(buffer.data(), size);
        std::string content_str(buffer.data(), size);

        std::vector<int> current_file_counts(phrases.size(), 0);

        // Zliczanie (GPU lub CPU)
        if (use_gpu_for_counting) {
            #ifdef USE_CUDA
            if (!run_cuda_analysis(buffer.data(), size, phrases, current_file_counts, cuda_block_size)) {
                std::cerr << "CUDA failure for " << current_file << "\n";
                continue;
            }
            #else
            std::cerr << "CUDA not compiled in!\n";
            #endif
        } else {
            // CPU Search
            for(size_t j=0; j<phrases.size(); ++j) {
                size_t pos = content_str.find(phrases[j], 0);
                while(pos != std::string::npos) {
                    current_file_counts[j]++;
                    pos = content_str.find(phrases[j], pos + 1);
                }
            }
        }

        // Analiza dokładna (zawsze CPU)
        analyze_content_for_time_and_matches(content_str, phrases, per_hour, matches);

        // Agregacja wyników
        for (size_t j = 0; j < phrases.size(); ++j) {
            total_counts[j] += current_file_counts[j];
        }
    }
}

int main(int argc, char** argv) {

    std::vector<std::string> phrases;
    std::vector<std::string> all_files;
    bool use_gpu_flag = false;
    int cuda_block_size = 256;

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <dir> [--gpu] [phrase1 phrase2 ...]\n";
        return 1;
    }

    int file_limit = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--gpu") {
            use_gpu_flag = true;
        } else if (std::string(argv[i]) == "--block-size" && i + 1 < argc) {
            cuda_block_size = std::stoi(argv[++i]);
        } else if (std::string(argv[i]) == "--limit-files" && i + 1 < argc) {
            file_limit = std::stoi(argv[++i]);
        } else if (i != 1) {
            phrases.emplace_back(argv[i]);
        }
    }
    if (phrases.empty()) {
        phrases = DEFAULT_PHRASES;
    }

    fs::path dir = argv[1];
    if (!fs::exists(dir) || !fs::is_directory(dir)) {
        std::cerr << "Not a directory: " << argv[1] << "\n";
        return 1;
    }

    for (const auto& entry : fs::directory_iterator(dir)) {
        if (fs::is_regular_file(entry.path()))
            all_files.push_back(entry.path().string());
    }
    std::sort(all_files.begin(), all_files.end());

    if (file_limit > 0 && (size_t)file_limit < all_files.size()) {
        all_files.resize(file_limit);
    }

    std::cout << "Sequential Analyzer started.\n";
    std::cout << "Mode: " << (use_gpu_flag ? "GPU Accelerated (Single Thread)" : "Pure CPU (std::string::find)") << "\n";
    std::cout << "Files found: " << all_files.size() << "\n";

    // Start Zegara
    auto start_time = std::chrono::high_resolution_clock::now();

    // Wyniki globalne
    std::vector<int> global_counts;
    std::map<std::string, std::map<std::string, int>> global_per_hour;
    std::map<std::string, std::vector<std::string>> global_matches;

    // Uruchomienie
    run_sequential_analysis(all_files, phrases, global_counts, global_per_hour, global_matches, use_gpu_flag, cuda_block_size);

    // Stop Zegara
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    // Raport
    fs::create_directories("output");
    std::ofstream out("output/matches_seq.txt");
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

    // Statystyki
    unsigned long long total_bytes = 0;
    for (const auto& f : all_files) {
        std::error_code ec;
        total_bytes += fs::file_size(f, ec);
    }
    double gb = total_bytes / (1024.0 * 1024.0 * 1024.0);

    std::cout << "Time: " << elapsed.count() << " s\n";
    std::cout << "Total Size: " << gb << " GB\n";
    std::cout << "Throughput: " << (elapsed.count() > 0 ? gb / elapsed.count() : 0.0) << " GB/s\n";

    return 0;
}