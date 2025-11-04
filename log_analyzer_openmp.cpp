#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <regex>
#include <set>

//   Usage: g++ -fopenmp -o log_analyzer_openmp log_analyzer_openmp.cpp
//   ./log_analyzer_openmp test.log

constexpr int NUM_THREADS = 4;
constexpr const char ERROR_STR[] = "ERROR";
constexpr const char WARNING_STR[] = "WARNING";
constexpr const char INFO_STR[] = "INFO";

bool analyze_events_per_hour(
    const char* filename,
    std::map<std::string,int>& errors_per_hour,
    std::map<std::string,int>& warnings_per_hour,
    std::map<std::string,int>& infos_per_hour)
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
        if (line.find(ERROR_STR) != std::string::npos) ++errors_per_hour[hour];
        if (line.find(WARNING_STR) != std::string::npos) ++warnings_per_hour[hour];
        if (line.find(INFO_STR) != std::string::npos) ++infos_per_hour[hour];
    }
    return true;
}

bool analyze_file(const char* filename, int& error_count, int& warning_count, int& info_count) {
    std::ifstream in(filename);
    if (!in) return false;

    std::string line;
    error_count = warning_count = info_count = 0;

    while (std::getline(in, line)) {
        if (line.find(ERROR_STR) != std::string::npos) ++error_count;
        if (line.find(WARNING_STR) != std::string::npos) ++warning_count;
        if (line.find(INFO_STR) != std::string::npos) ++info_count;
    }
    return true;
}

void display_errors_per_hour(
    const std::map<std::string,int>& errors_per_hour,
    const std::map<std::string,int>& warnings_per_hour,
    const std::map<std::string,int>& infos_per_hour)
{
    std::cout << "Per-hour counts:\n";
    std::set<std::string> hours;
    for (const auto& kv : errors_per_hour) hours.insert(kv.first);
    for (const auto& kv : warnings_per_hour) hours.insert(kv.first);
    for (const auto& kv : infos_per_hour) hours.insert(kv.first);

    for (const auto& hour : hours) {
        int e = 0, w = 0, i = 0;
        auto ite = errors_per_hour.find(hour);
        if (ite != errors_per_hour.end()) e = ite->second;
        auto itw = warnings_per_hour.find(hour);
        if (itw != warnings_per_hour.end()) w = itw->second;
        auto iti = infos_per_hour.find(hour);
        if (iti != infos_per_hour.end()) i = iti->second;
        std::cout << hour << " E:" << e << " W:" << w << " I:" << i << '\n';
    }
}

int main(int argc, char** argv) {

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <file>\n";
        return 1;
    }

    const char* filename = argv[1];
    int error_count = 0;
    int warning_count = 0;
    int info_count = 0;

    if (!analyze_file(filename, error_count, warning_count, info_count)) {
        std::cerr << "Cannot open " << filename << "\n";
        return 1;
    }

    std::cout << error_count << '\n';
    std::cout << warning_count << '\n';
    std::cout << info_count << '\n';


    std::map<std::string,int> errors_per_hour;
    std::map<std::string,int> warnings_per_hour;
    std::map<std::string,int> infos_per_hour;

    if (!analyze_events_per_hour(filename, errors_per_hour, warnings_per_hour, infos_per_hour)) {
        std::cerr << "Cannot open " << filename << "\n";
        return 1;
    }

    display_errors_per_hour(errors_per_hour, warnings_per_hour, infos_per_hour);


    return 0;

}