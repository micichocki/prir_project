#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

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

// Funkcja Helper (Wrapper), wywoływana z C++
// Przyjmuje surowe wskaźniki zamiast std::vector/std::string
extern "C" bool run_cuda_analysis_raw(const char* file_content, size_t file_size,
                                      const char* flat_phrases, int flat_phrases_size,
                                      const int* offsets,
                                      const int* lengths,
                                      int num_keywords,
                                      int* out_results,
                                      int blockSize)
{
    // 2. Alokacja pamięci na GPU
    char* d_data = nullptr;
    char* d_keywords = nullptr;
    int* d_offsets = nullptr;
    int* d_lengths = nullptr;
    int* d_counts = nullptr;

    cudaError_t err;

    err = cudaMalloc(&d_data, file_size);
    if(err != cudaSuccess) return false;

    if (cudaMalloc(&d_keywords, flat_phrases_size) != cudaSuccess) { cudaFree(d_data); return false; }
    if (cudaMalloc(&d_offsets, num_keywords * sizeof(int)) != cudaSuccess) { cudaFree(d_data); cudaFree(d_keywords); return false; }
    if (cudaMalloc(&d_lengths, num_keywords * sizeof(int)) != cudaSuccess) { cudaFree(d_data); cudaFree(d_keywords); cudaFree(d_offsets); return false; }
    if (cudaMalloc(&d_counts, num_keywords * sizeof(int)) != cudaSuccess) { cudaFree(d_data); cudaFree(d_keywords); cudaFree(d_offsets); cudaFree(d_lengths); return false; }

    // Zerowanie wyników na GPU
    cudaMemset(d_counts, 0, num_keywords * sizeof(int));

    // 3. Kopiowanie danych Host -> Device
    cudaMemcpy(d_data, file_content, file_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_keywords, flat_phrases, flat_phrases_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, offsets, num_keywords * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lengths, lengths, num_keywords * sizeof(int), cudaMemcpyHostToDevice);

    // 4. Konfiguracja siatki wątków
    int numBlocks = (int)((file_size + blockSize - 1) / blockSize);

    // 5. Uruchomienie Kernela
    count_keywords_kernel<<<numBlocks, blockSize>>>(d_data, (int)file_size, d_keywords, d_offsets, d_lengths, num_keywords, d_counts);

    // Czekaj na zakończenie
    cudaDeviceSynchronize();

    // 6. Kopiowanie wyników Device -> Host
    cudaMemcpy(out_results, d_counts, num_keywords * sizeof(int), cudaMemcpyDeviceToHost);

    // 7. Sprzątanie
    cudaFree(d_data);
    cudaFree(d_keywords);
    cudaFree(d_offsets);
    cudaFree(d_lengths);
    cudaFree(d_counts);

    return true;
}