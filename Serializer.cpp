#include "Serializer.h"
#include <cstring>
#include <sstream>
#include <iostream>
#include <algorithm>

std::vector<char> Serializer::serialize_string_list(const std::vector<std::string>& list) {
    std::ostringstream oss;
    oss << list.size() << '\n';
    for (const auto& s : list) {
        oss << s.size() << '\n' << s;
    }
    std::string str = oss.str();
    return std::vector<char>(str.begin(), str.end());
}

std::vector<std::string> Serializer::deserialize_string_list(const std::vector<char>& buffer) {
    std::vector<std::string> result;
    if (buffer.empty()) return result;

    std::string str(buffer.begin(), buffer.end());
    std::istringstream iss(str);
    size_t count;
    if (!(iss >> count)) return {};
    iss.ignore();

    for (size_t i = 0; i < count; ++i) {
        size_t len;
        iss >> len;
        iss.ignore();
        std::string s(len, '\0');
        iss.read(&s[0], len);
        result.push_back(s);
    }
    return result;
}

std::vector<char> Serializer::serialize_counts(const std::vector<int>& counts) {
    std::vector<char> buffer(counts.size() * sizeof(int));
    std::memcpy(buffer.data(), counts.data(), buffer.size());
    return buffer;
}

std::vector<int> Serializer::deserialize_counts(const std::vector<char>& buffer, size_t count) {
    std::vector<int> counts(count);
    std::memcpy(counts.data(), buffer.data(), count * sizeof(int));
    return counts;
}

std::vector<char> Serializer::serialize_per_hour(const std::map<std::string, std::map<std::string, int>>& data) {
    std::ostringstream oss;
    oss << data.size() << '\n';
    for (const auto& [phrase, hour_map] : data) {
        oss << phrase << '\n' << hour_map.size() << '\n';
        for (const auto& [hour, count] : hour_map) {
            oss << hour << '\n' << count << '\n';
        }
    }
    std::string str = oss.str();
    return std::vector<char>(str.begin(), str.end());
}

std::map<std::string, std::map<std::string, int>> Serializer::deserialize_per_hour(const std::vector<char>& buffer) {
    std::map<std::string, std::map<std::string, int>> result;
    if (buffer.empty()) return result;

    std::string str(buffer.begin(), buffer.end());
    std::istringstream iss(str);
    size_t phrase_count;
    if (!(iss >> phrase_count)) return {};
    iss.ignore();

    for (size_t i = 0; i < phrase_count; ++i) {
        std::string phrase;
        std::getline(iss, phrase);
        size_t hour_count;
        iss >> hour_count;
        iss.ignore();
        for (size_t j = 0; j < hour_count; ++j) {
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

std::vector<char> Serializer::serialize_matches(const std::map<std::string, std::vector<std::string>>& data) {
    std::ostringstream oss;
    oss << data.size() << '\n';
    for (const auto& [phrase, lines] : data) {
        oss << phrase << '\n' << lines.size() << '\n';
        for (const auto& line : lines) {
            oss << line.size() << '\n' << line;
        }
    }
    std::string str = oss.str();
    return std::vector<char>(str.begin(), str.end());
}

std::map<std::string, std::vector<std::string>> Serializer::deserialize_matches(const std::vector<char>& buffer) {
    std::map<std::string, std::vector<std::string>> result;
    if (buffer.empty()) return result;

    std::string str(buffer.begin(), buffer.end());
    std::istringstream iss(str);
    size_t phrase_count;
    if (!(iss >> phrase_count)) return {};
    iss.ignore();

    for (size_t i = 0; i < phrase_count; ++i) {
        std::string phrase;
        std::getline(iss, phrase);
        size_t line_count;
        iss >> line_count;
        iss.ignore();
        for (size_t j = 0; j < line_count; ++j) {
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