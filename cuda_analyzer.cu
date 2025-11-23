#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <string>

// Kernel CUDA: Każdy wątek sprawdza jedno miejsce w pliku
__global__ void count_keywords_kernel(const char* data, int data_size,
                                      const char* flat_keywords, const int* keyword_offsets, const int* keyword_lengths,
                                      int num_keywords, int* d_counts)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= data_size) return;

    // Każdy wątek sprawdza, czy od jego pozycji (idx) zaczyna się któreś ze słów kluczowych
    for (int k = 0; k < num_keywords; ++k) {
        int len = keyword_lengths[k];

        // Zabezpieczenie przed wyjściem poza plik
        if (idx + len > data_size) continue;

        // Ręczne porównanie stringów (implementacja strncmp na GPU)
        bool match = true;
        int offset = keyword_offsets[k];
        for (int i = 0; i < len; ++i) {
            if (data[idx + i] != flat_keywords[offset + i]) {
                match = false;
                break;
            }
        }

        if (match) {
            // atomicAdd jest konieczny, bo wiele wątków może pisać do tego samego licznika jednocześnie
            atomicAdd(&d_counts[k], 1);
        }
    }
}

// Funkcja Helper (Wrapper), którą wywołasz z kodu C++
extern "C" bool run_cuda_analysis(const char* file_content, size_t file_size,
                                  const std::vector<std::string>& phrases,
                                  std::vector<int>& results)
{
    // 1. Przygotowanie słów kluczowych do formatu przyjaznego dla C (flat array)
    std::string flat_phrases;
    std::vector<int> offsets;
    std::vector<int> lengths;

    for (const auto& p : phrases) {
        offsets.push_back(flat_phrases.size());
        lengths.push_back(p.size());
        flat_phrases += p;
    }

    // 2. Alokacja pamięci na GPU
    char* d_data;
    char* d_keywords;
    int* d_offsets;
    int* d_lengths;
    int* d_counts;

    cudaError_t err;

    err = cudaMalloc(&d_data, file_size);
    if(err != cudaSuccess) return false;

    cudaMalloc(&d_keywords, flat_phrases.size());
    cudaMalloc(&d_offsets, offsets.size() * sizeof(int));
    cudaMalloc(&d_lengths, lengths.size() * sizeof(int));
    cudaMalloc(&d_counts, phrases.size() * sizeof(int));

    // Zerowanie wyników na GPU
    cudaMemset(d_counts, 0, phrases.size() * sizeof(int));

    // 3. Kopiowanie danych Host -> Device
    cudaMemcpy(d_data, file_content, file_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_keywords, flat_phrases.data(), flat_phrases.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, offsets.data(), offsets.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lengths, lengths.data(), lengths.size() * sizeof(int), cudaMemcpyHostToDevice);

    // 4. Konfiguracja siatki wątków
    int blockSize = 256;
    int numBlocks = (file_size + blockSize - 1) / blockSize;

    // 5. Uruchomienie Kernela
    count_keywords_kernel<<<numBlocks, blockSize>>>(d_data, file_size, d_keywords, d_offsets, d_lengths, phrases.size(), d_counts);

    // Czekaj na zakończenie
    cudaDeviceSynchronize();

    // 6. Kopiowanie wyników Device -> Host
    std::vector<int> gpu_counts(phrases.size());
    cudaMemcpy(gpu_counts.data(), d_counts, phrases.size() * sizeof(int), cudaMemcpyDeviceToHost);

    // Przypisanie do wyniku wyjściowego
    results = gpu_counts;

    // 7. Sprzątanie
    cudaFree(d_data);
    cudaFree(d_keywords);
    cudaFree(d_offsets);
    cudaFree(d_lengths);
    cudaFree(d_counts);

    return true;
}