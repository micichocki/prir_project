#ifndef SERIALIZER_H
#define SERIALIZER_H

#include <vector>
#include <string>
#include <map>

class Serializer {
public:
    static std::vector<char> serialize_string_list(const std::vector<std::string>& list);
    static std::vector<std::string> deserialize_string_list(const std::vector<char>& buffer);

    // from MPI version
    static std::vector<char> serialize_counts(const std::vector<int>& counts);
    static std::vector<int> deserialize_counts(const std::vector<char>& buffer, size_t count);

    static std::vector<char> serialize_per_hour(const std::map<std::string, std::map<std::string, int>>& data);
    static std::map<std::string, std::map<std::string, int>> deserialize_per_hour(const std::vector<char>& buffer);

    static std::vector<char> serialize_matches(const std::map<std::string, std::vector<std::string>>& data);
    static std::map<std::string, std::vector<std::string>> deserialize_matches(const std::vector<char>& buffer);
};

#endif