#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <vector>
#include <regex>
#include <set>
#include <filesystem>
#include <algorithm>
#include <omp.h>

// Usage:
//g++ -fopenmp -std=c++17 -O2 -o log_analyzer_openmp log_analyzer_openmp.cpp
// ./log_analyzer_openmp logs_folder <dir> [phrase1 phrase2 ...]

namespace fs = std::filesystem;

constexpr int NUM_THREADS = 4;
const std::vector<std::string> DEFAULT_PHRASES = {"ERROR", "WARNING", "INFO"};

bool analyze_events_per_hour(
    const char* filename,
    const std::vector<std::string>& phrases,
    std::map<std::string, std::map<std::string,int>>& per_hour_counts)
{
    std::ifstream in(filename);
    if (!in) return false;

    std::string line;
    std::regex ts_re(R"((\d{4}-\d{2}-\d{2} \d{2}):\d{2}:\d{2})");
    std::smatch m;

    while (std::getline(in, line)) {
        std::string hour = "unknown";
        if (std::regex_search(line, m, ts_re) && m.size() > 1) {
            hour = m[1].str();
        }
        for (const auto& ph : phrases) {
            if (line.find(ph) != std::string::npos) ++per_hour_counts[ph][hour];
        }
    }
    return true;
}

bool collect_matching_lines(const char* filename, const std::vector<std::string>& phrases, std::map<std::string, std::vector<std::string>>& results) {
    std::ifstream in(filename);
    if (!in) return false;

    std::string line;
    while (std::getline(in, line)) {
        for (const auto& ph : phrases) {
            if (line.find(ph) != std::string::npos) {
                results[ph].push_back(line);
            }
        }
    }
    return true;
}

bool analyze_file(const char* filename, const std::vector<std::string>& phrases, std::vector<int>& counts) {
    std::ifstream in(filename);
    if (!in) return false;

    std::string line;
    counts.assign(phrases.size(), 0);

    while (std::getline(in, line)) {
        for (size_t i = 0; i < phrases.size(); ++i) {
            if (line.find(phrases[i]) != std::string::npos)
                ++counts[i];
        }
    }
    return true;
}

void display_per_hour(const std::vector<std::string>& phrases, const std::map<std::string, std::map<std::string,int>>& per_hour_counts)
{
    std::cout << "Per-hour counts:\n";
    std::set<std::string> hours;
    for (const auto& ph : phrases) {
        auto it = per_hour_counts.find(ph);
        if (it == per_hour_counts.end()) continue;
        for (const auto& kv : it->second) hours.insert(kv.first);
    }

    for (const auto& hour : hours) {
        std::cout << hour;
        for (const auto& ph : phrases) {
            int cnt = 0;
            auto itp = per_hour_counts.find(ph);
            if (itp != per_hour_counts.end()) {
                auto ith = itp->second.find(hour);
                if (ith != itp->second.end()) cnt = ith->second;
            }
            std::cout << " " << ph << ":" << cnt;
        }
        std::cout << '\n';
    }
}

int main(int argc, char** argv) {

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <dir> [phrase1 phrase2 ...]\n";
        return 1;
    }

    fs::path dir = argv[1];
    if (!fs::exists(dir) || !fs::is_directory(dir)) {
        std::cerr << "Not a directory: " << argv[1] << "\n";
        return 1;
    }

    std::vector<std::string> phrases;
    if (argc >= 3) {
        for (int i = 2; i < argc; ++i) phrases.emplace_back(argv[i]);
    } else {
        phrases = DEFAULT_PHRASES;
    }

    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (fs::is_regular_file(entry.path()))
            files.push_back(entry.path());
    }

    std::vector<int> total_counts(phrases.size(), 0);
    std::map<std::string, std::map<std::string,int>> per_hour_counts;
    std::map<std::string, std::vector<std::string>> all_matches;

    omp_set_num_threads(NUM_THREADS);

    #pragma omp parallel reduction(+: total_counts[:phrases.size()])
    {
        std::map<std::string, std::map<std::string,int>> local_per_hour;
        std::map<std::string, std::vector<std::string>> local_matches;

        #pragma omp for schedule(dynamic)
        for (int i = 0; i < (int)files.size(); ++i) {
            std::string filepath = files[i].string();

            std::vector<int> counts;
            if (!analyze_file(filepath.c_str(), phrases, counts)) {
                #pragma omp critical
                std::cerr << "Cannot open " << filepath << "\n";
                continue;
            }

            for (size_t j = 0; j < phrases.size(); ++j)
                total_counts[j] += counts[j];

            analyze_events_per_hour(filepath.c_str(), phrases, local_per_hour);
            collect_matching_lines(filepath.c_str(), phrases, local_matches);
        }

        #pragma omp critical
        {
            for (const auto& ph_pair : local_per_hour) {
                for (const auto& hour_pair : ph_pair.second)
                    per_hour_counts[ph_pair.first][hour_pair.first] += hour_pair.second;
            }

            for (const auto& kv : local_matches) {
                auto& vec = all_matches[kv.first];
                vec.insert(vec.end(), kv.second.begin(), kv.second.end());
            }
        }
    }

    fs::create_directories("output");
    fs::path outpath = "output/matches.txt";
    std::ofstream outfs(outpath);
    if (!outfs) {
        std::cerr << "Cannot write to " << outpath.string() << "\n";
    } else {
        for (const auto& ph : phrases) {
            outfs << "=== " << ph << " ===\n";
            auto it = all_matches.find(ph);
            if (it != all_matches.end()) {
                for (const auto& l : it->second) outfs << l << "\n";
            }
            outfs << "\n";
        }
    }

    std::cout << "Total counts:\n";
    for (size_t i = 0; i < phrases.size(); ++i) {
        std::cout << phrases[i] << ": " << total_counts[i] << '\n';
    }

    display_per_hour(phrases, per_hour_counts);

    return 0;
}