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

// Usage:
// mpic++ -std=c++17 -O2 -o log_analyzer_mpi log_analyzer_mpi.cpp
// mpirun -np 4 ./log_analyzer_mpi <dir> [phrase1 phrase2 ...]

namespace fs = std::filesystem;

const std::vector<std::string> DEFAULT_PHRASES = {"ERROR", "WARNING", "INFO"};

// MPI Tags for different message types
enum MsgTag
{
    TAG_FILENAME = 0,
    TAG_COUNTS = 1,
    TAG_PERHOUR_SIZE = 2,
    TAG_PERHOUR_DATA = 3,
    TAG_MATCHES_SIZE = 4,
    TAG_MATCHES_DATA = 5,
    TAG_TERMINATE = 6
};

// Serialization helpers for complex data structures
class Serializer
{
public:
    static std::vector<char> serialize_counts(const std::vector<int> &counts)
    {
        std::vector<char> buffer(counts.size() * sizeof(int));
        std::memcpy(buffer.data(), counts.data(), buffer.size());
        return buffer;
    }

    static std::vector<int> deserialize_counts(const std::vector<char> &buffer, size_t count)
    {
        std::vector<int> counts(count);
        std::memcpy(counts.data(), buffer.data(), count * sizeof(int));
        return counts;
    }

    static std::vector<char> serialize_per_hour(
        const std::map<std::string, std::map<std::string, int>> &data)
    {
        std::ostringstream oss;
        oss << data.size() << '\n';
        for (const auto &[phrase, hour_map] : data)
        {
            oss << phrase << '\n'
                << hour_map.size() << '\n';
            for (const auto &[hour, count] : hour_map)
            {
                oss << hour << '\n'
                    << count << '\n';
            }
        }
        std::string str = oss.str();
        return std::vector<char>(str.begin(), str.end());
    }

    static std::map<std::string, std::map<std::string, int>> deserialize_per_hour(
        const std::vector<char> &buffer)
    {
        std::map<std::string, std::map<std::string, int>> result;
        std::string str(buffer.begin(), buffer.end());
        std::istringstream iss(str);

        size_t phrase_count;
        iss >> phrase_count;
        iss.ignore();

        for (size_t i = 0; i < phrase_count; ++i)
        {
            std::string phrase;
            std::getline(iss, phrase);

            size_t hour_count;
            iss >> hour_count;
            iss.ignore();

            for (size_t j = 0; j < hour_count; ++j)
            {
                std::string hour;
                std::getline(iss, hour);
                int count;
                iss >> count;
                iss.ignore();
                result[phrase][hour] = count;
            }
        }
        return result;
    }

    static std::vector<char> serialize_matches(
        const std::map<std::string, std::vector<std::string>> &data)
    {
        std::ostringstream oss;
        oss << data.size() << '\n';
        for (const auto &[phrase, lines] : data)
        {
            oss << phrase << '\n'
                << lines.size() << '\n';
            for (const auto &line : lines)
            {
                oss << line.size() << '\n'
                    << line;
            }
        }
        std::string str = oss.str();
        return std::vector<char>(str.begin(), str.end());
    }

    static std::map<std::string, std::vector<std::string>> deserialize_matches(
        const std::vector<char> &buffer)
    {
        std::map<std::string, std::vector<std::string>> result;
        std::string str(buffer.begin(), buffer.end());
        std::istringstream iss(str);

        size_t phrase_count;
        iss >> phrase_count;
        iss.ignore();

        for (size_t i = 0; i < phrase_count; ++i)
        {
            std::string phrase;
            std::getline(iss, phrase);

            size_t line_count;
            iss >> line_count;
            iss.ignore();

            for (size_t j = 0; j < line_count; ++j)
            {
                size_t line_len;
                iss >> line_len;
                iss.ignore();

                std::string line(line_len, '\0');
                iss.read(&line[0], line_len);
                result[phrase].push_back(line);
            }
        }
        return result;
    }
};

// Analysis functions (same as OpenMP version)
bool analyze_events_per_hour(
    const char *filename,
    const std::vector<std::string> &phrases,
    std::map<std::string, std::map<std::string, int>> &per_hour_counts)
{
    std::ifstream in(filename);
    if (!in)
        return false;

    std::string line;
    std::regex ts_re(R"((\d{4}-\d{2}-\d{2} \d{2}):\d{2}:\d{2})");
    std::smatch m;

    while (std::getline(in, line))
    {
        std::string hour = "unknown";
        if (std::regex_search(line, m, ts_re) && m.size() > 1)
        {
            hour = m[1].str();
        }
        for (const auto &ph : phrases)
        {
            if (line.find(ph) != std::string::npos)
                ++per_hour_counts[ph][hour];
        }
    }
    return true;
}

bool collect_matching_lines(
    const char *filename,
    const std::vector<std::string> &phrases,
    std::map<std::string, std::vector<std::string>> &results)
{
    std::ifstream in(filename);
    if (!in)
        return false;

    std::string line;
    while (std::getline(in, line))
    {
        for (const auto &ph : phrases)
        {
            if (line.find(ph) != std::string::npos)
            {
                results[ph].push_back(line);
            }
        }
    }
    return true;
}

bool analyze_file(
    const char *filename,
    const std::vector<std::string> &phrases,
    std::vector<int> &counts)
{
    std::ifstream in(filename);
    if (!in)
        return false;

    std::string line;
    counts.assign(phrases.size(), 0);

    while (std::getline(in, line))
    {
        for (size_t i = 0; i < phrases.size(); ++i)
        {
            if (line.find(phrases[i]) != std::string::npos)
                ++counts[i];
        }
    }
    return true;
}

void display_per_hour(
    const std::vector<std::string> &phrases,
    const std::map<std::string, std::map<std::string, int>> &per_hour_counts)
{
    std::cout << "Per-hour counts:\n";
    std::set<std::string> hours;
    for (const auto &ph : phrases)
    {
        auto it = per_hour_counts.find(ph);
        if (it == per_hour_counts.end())
            continue;
        for (const auto &kv : it->second)
            hours.insert(kv.first);
    }

    for (const auto &hour : hours)
    {
        std::cout << hour;
        for (const auto &ph : phrases)
        {
            int cnt = 0;
            auto itp = per_hour_counts.find(ph);
            if (itp != per_hour_counts.end())
            {
                auto ith = itp->second.find(hour);
                if (ith != itp->second.end())
                    cnt = ith->second;
            }
            std::cout << " " << ph << ":" << cnt;
        }
        std::cout << '\n';
    }
}

// Worker process function
void worker_process(const std::vector<std::string> &phrases)
{
    while (true)
    {
        MPI_Status status;

        // Probe for message to get size
        MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        if (status.MPI_TAG == TAG_TERMINATE)
        {
            break;
        }

        if (status.MPI_TAG == TAG_FILENAME)
        {
            int filename_len;
            MPI_Get_count(&status, MPI_CHAR, &filename_len);

            std::vector<char> filename_buf(filename_len);
            MPI_Recv(filename_buf.data(), filename_len, MPI_CHAR, 0, TAG_FILENAME,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            std::string filepath(filename_buf.begin(), filename_buf.end());

            // Analyze the file
            std::vector<int> counts;
            std::map<std::string, std::map<std::string, int>> per_hour;
            std::map<std::string, std::vector<std::string>> matches;

            if (analyze_file(filepath.c_str(), phrases, counts))
            {
                analyze_events_per_hour(filepath.c_str(), phrases, per_hour);
                collect_matching_lines(filepath.c_str(), phrases, matches);
            }

            // Send results back to master
            // 1. Send counts
            auto counts_buf = Serializer::serialize_counts(counts);
            int counts_size = counts_buf.size();
            MPI_Send(&counts_size, 1, MPI_INT, 0, TAG_COUNTS, MPI_COMM_WORLD);
            MPI_Send(counts_buf.data(), counts_size, MPI_CHAR, 0, TAG_COUNTS,
                     MPI_COMM_WORLD);

            // 2. Send per-hour data
            auto perhour_buf = Serializer::serialize_per_hour(per_hour);
            int perhour_size = perhour_buf.size();
            MPI_Send(&perhour_size, 1, MPI_INT, 0, TAG_PERHOUR_SIZE, MPI_COMM_WORLD);
            if (perhour_size > 0)
            {
                MPI_Send(perhour_buf.data(), perhour_size, MPI_CHAR, 0,
                         TAG_PERHOUR_DATA, MPI_COMM_WORLD);
            }

            // 3. Send matches
            auto matches_buf = Serializer::serialize_matches(matches);
            int matches_size = matches_buf.size();
            MPI_Send(&matches_size, 1, MPI_INT, 0, TAG_MATCHES_SIZE, MPI_COMM_WORLD);
            if (matches_size > 0)
            {
                MPI_Send(matches_buf.data(), matches_size, MPI_CHAR, 0,
                         TAG_MATCHES_DATA, MPI_COMM_WORLD);
            }
        }
    }
}

// Master process function
void master_process(
    const std::vector<fs::path> &files,
    const std::vector<std::string> &phrases,
    int world_size)
{
    std::vector<int> total_counts(phrases.size(), 0);
    std::map<std::string, std::map<std::string, int>> per_hour_counts;
    std::map<std::string, std::vector<std::string>> all_matches;

    size_t next_file = 0;
    int active_workers = 0;

    // Initial distribution - send one file to each worker
    for (int rank = 1; rank < world_size && next_file < files.size(); ++rank)
    {
        std::string filepath = files[next_file].string();
        MPI_Send(filepath.c_str(), filepath.size(), MPI_CHAR, rank,
                 TAG_FILENAME, MPI_COMM_WORLD);
        next_file++;
        active_workers++;
    }

    // Collect results and distribute remaining work
    while (active_workers > 0)
    {
        MPI_Status status;

        // Wait for any worker to send counts
        int counts_size;
        MPI_Recv(&counts_size, 1, MPI_INT, MPI_ANY_SOURCE, TAG_COUNTS,
                 MPI_COMM_WORLD, &status);
        int worker_rank = status.MPI_SOURCE;

        std::vector<char> counts_buf(counts_size);
        MPI_Recv(counts_buf.data(), counts_size, MPI_CHAR, worker_rank, TAG_COUNTS,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        auto counts = Serializer::deserialize_counts(counts_buf, phrases.size());
        for (size_t i = 0; i < phrases.size(); ++i)
        {
            total_counts[i] += counts[i];
        }

        // Receive per-hour data
        int perhour_size;
        MPI_Recv(&perhour_size, 1, MPI_INT, worker_rank, TAG_PERHOUR_SIZE,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (perhour_size > 0)
        {
            std::vector<char> perhour_buf(perhour_size);
            MPI_Recv(perhour_buf.data(), perhour_size, MPI_CHAR, worker_rank,
                     TAG_PERHOUR_DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            auto perhour = Serializer::deserialize_per_hour(perhour_buf);
            for (const auto &[phrase, hour_map] : perhour)
            {
                for (const auto &[hour, count] : hour_map)
                {
                    per_hour_counts[phrase][hour] += count;
                }
            }
        }

        // Receive matches
        int matches_size;
        MPI_Recv(&matches_size, 1, MPI_INT, worker_rank, TAG_MATCHES_SIZE,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (matches_size > 0)
        {
            std::vector<char> matches_buf(matches_size);
            MPI_Recv(matches_buf.data(), matches_size, MPI_CHAR, worker_rank,
                     TAG_MATCHES_DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            auto matches = Serializer::deserialize_matches(matches_buf);
            for (const auto &[phrase, lines] : matches)
            {
                auto &vec = all_matches[phrase];
                vec.insert(vec.end(), lines.begin(), lines.end());
            }
        }

        // Send next file or termination signal
        if (next_file < files.size())
        {
            std::string filepath = files[next_file].string();
            MPI_Send(filepath.c_str(), filepath.size(), MPI_CHAR, worker_rank,
                     TAG_FILENAME, MPI_COMM_WORLD);
            next_file++;
        }
        else
        {
            // No more files, signal this worker to terminate
            MPI_Send(nullptr, 0, MPI_CHAR, worker_rank, TAG_TERMINATE, MPI_COMM_WORLD);
            active_workers--;
        }
    }

    // Output results
    fs::create_directories("output");
    fs::path outpath = "output/matches.txt";
    std::ofstream outfs(outpath);
    if (!outfs)
    {
        std::cerr << "Cannot write to " << outpath.string() << "\n";
    }
    else
    {
        for (const auto &ph : phrases)
        {
            outfs << "=== " << ph << " ===\n";
            auto it = all_matches.find(ph);
            if (it != all_matches.end())
            {
                for (const auto &l : it->second)
                    outfs << l << "\n";
            }
            outfs << "\n";
        }
    }

    std::cout << "Total counts:\n";
    for (size_t i = 0; i < phrases.size(); ++i)
    {
        std::cout << phrases[i] << ": " << total_counts[i] << '\n';
    }

    display_per_hour(phrases, per_hour_counts);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_size < 2)
    {
        if (world_rank == 0)
        {
            std::cerr << "This program requires at least 2 processes (1 master + 1 worker)\n";
        }
        MPI_Finalize();
        return 1;
    }

    // Parse command line arguments
    std::vector<std::string> phrases;
    std::vector<fs::path> files;

    if (world_rank == 0)
    {
        if (argc < 2)
        {
            std::cerr << "Usage: " << argv[0] << " <dir> [phrase1 phrase2 ...]\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        fs::path dir = argv[1];
        if (!fs::exists(dir) || !fs::is_directory(dir))
        {
            std::cerr << "Not a directory: " << argv[1] << "\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        if (argc >= 3)
        {
            for (int i = 2; i < argc; ++i)
                phrases.emplace_back(argv[i]);
        }
        else
        {
            phrases = DEFAULT_PHRASES;
        }

        // Collect all files
        for (const auto &entry : fs::directory_iterator(dir))
        {
            if (fs::is_regular_file(entry.path()))
                files.push_back(entry.path());
        }

        std::cout << "Processing " << files.size() << " files with "
                  << (world_size - 1) << " workers...\n";
    }

    // Broadcast phrases to all processes
    int phrase_count = phrases.size();
    MPI_Bcast(&phrase_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (world_rank != 0)
    {
        phrases.resize(phrase_count);
    }

    for (int i = 0; i < phrase_count; ++i)
    {
        int phrase_len = (world_rank == 0) ? phrases[i].size() : 0;
        MPI_Bcast(&phrase_len, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (world_rank != 0)
        {
            phrases[i].resize(phrase_len);
        }

        MPI_Bcast(&phrases[i][0], phrase_len, MPI_CHAR, 0, MPI_COMM_WORLD);
    }

    // Master-worker pattern
    if (world_rank == 0)
    {
        master_process(files, phrases, world_size);
    }
    else
    {
        worker_process(phrases);
    }

    MPI_Finalize();
    return 0;
}