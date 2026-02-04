#include "params.h"
#include "simulator.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <windows.h>
#include <ctime>
#include <iomanip>
#include <cstdint>

/* ================= CUDA Error Checking Macro ================= */
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                  << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while (0)

/* =====================================================
   GPU Kernel: SPAD Single-Photon First-Photon Detection Simulation
   Parallel Dimensions: Pixel ¡Á Pulse
   ===================================================== */
__global__ void spad_sim_kernel(
    const Point* d_points,   // Point cloud (per pulse)
    const float* d_Nspp,     // Signal photon expectation
    const float* d_Nbpp,     // Background photon expectation
    const float* d_tof,      // Flight time
    int16_t* d_record,           // Output: First photon bin
    int N_row,
    int N_col,
    int N_bin,
    float DCR,
    float r_bg,
    float delta_t,
    float tau,
    int N_pulse,
    int start_idx,
    int total_points,
    float sigma_pix,         // Spatial diffusion (pixels)
    unsigned long long seed_offset)
{
    /* ----------- Global Thread Index ----------- */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = N_row * N_col * N_pulse;
    if (idx >= total_threads) return;

    /* ----------- Pixel & Pulse Index ----------- */
    int pix_idx   = idx / N_pulse;
    int pulse_idx = idx % N_pulse;
    int global_pulse_idx = start_idx + pulse_idx;
    if (global_pulse_idx >= total_points) return;

    int v = pix_idx / N_col;  // ÐÐ
    int u = pix_idx % N_col;  // ÁÐ

    /* ----------- The target corresponding to the current pulse ----------- */
    Point pt = d_points[global_pulse_idx];
    int u0 = pt.x;
    int v0 = pt.y;

    /* ----------- Initialise the random number state ----------- */
    curandState state;
    curand_init(1234ULL + idx + seed_offset, 0, 0, &state);

    int16_t first_bin = -1;

    /* ========== Spatial diffusion (two-dimensional Gaussian PSF) ========== */
    float dx = float(u - u0);
    float dy = float(v - v0);
    float spatial_weight =
        expf(-(dx * dx) / (2.0f * sigma_pix) - (dy * dy) / (2.0f * sigma_pix));

    float lambda_signal = d_Nspp[global_pulse_idx] * spatial_weight;
    float lambda_noise  = d_Nbpp[global_pulse_idx] * spatial_weight;

    /* ========== Time jitter ========== */
    float jitter = curand_normal(&state) * tau;
    int signal_bin = int((d_tof[global_pulse_idx] + jitter) / delta_t);

    /* ========== Time bin traversal (first photon) ========== */
    for (int b = 0; b < N_bin; ++b)
    {
        // dark counting
        float lambda = DCR * delta_t + r_bg * delta_t * (1 - spatial_weight);

        // signal photon
        if (b == signal_bin && signal_bin >= 0 && signal_bin < N_bin)
        {
            lambda += lambda_signal + lambda_noise / N_bin;
        }

        // Background noise (uniformly distributed)
        lambda += lambda_noise / N_bin;

        // Poisson sampling
        int count = curand_poisson(&state, lambda);
        if (count > 0)
        {
            first_bin = b;
            break;  // ¡ï First-photon triggering
        }
    }

    d_record[pix_idx * N_pulse + pulse_idx] = first_bin;
}

void Simulator::SpadSimulator(
    const std::vector<Point>& pixels,
    const std::string& save_filename)
{
    // ================= GPU device information =================
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    std::cout << "Using GPU devices: " << prop.name << std::endl;
    std::cout << "Computing power: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Graphics memory size: "
              << prop.totalGlobalMem / (1024.0 * 1024.0)
              << " MB" << std::endl;

    // ================= Start time recorded =================
    clock_t start_time = clock();
    std::cout << "Commencing SPAD photon counting simulation..." << std::endl;

    std::vector<Point> points(
        pixels.begin() + 1,
        pixels.end()
    );

    std::cout << "Number of points read: " << points.size() << std::endl;

    /* ========== System Parameters ========== */
    int N_row = Params::N_row;
    int N_col = Params::N_col;
    int N_pix = N_row * N_col;
    int N_bin = Params::N_bin;
    int N_pulse = Params::N_pulse;

    float PDE = Params::PDE;
    float DCR = Params::DCR;
    float delta_t = Params::delta_t;
    float tau = Params::tau;
    float sigma_pix = Params::sigma_pix;

    float N_s = Params::N_s;
    float N_b = Params::N_b;
    float r_bg = Params::r_bg;
    float R = Params::R_ref;
    float c = Params::c;

    /* ========== Calculate Ns / Nb / ToF ========== */
    std::vector<float> Nspp(points.size());
    std::vector<float> Nbpp(points.size());
    std::vector<float> tof(points.size());

    for (size_t i = 0; i < points.size(); ++i)
    {
        float att = (R * R) / (points[i].z * points[i].z);
        Nspp[i] = N_s * att * PDE;
        Nbpp[i] = N_b * att * PDE;
        tof[i]  = 2.0f * points[i].z / c;
    }

    /* ========== GPU memory allocation ========== */
    Point* d_points;
    float *d_Nspp, *d_Nbpp, *d_tof;
    CUDA_CHECK(cudaMalloc(&d_points, points.size() * sizeof(Point)));
    CUDA_CHECK(cudaMalloc(&d_Nspp,  points.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Nbpp,  points.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tof,   points.size() * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_points, points.data(), points.size() * sizeof(Point), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Nspp, Nspp.data(), points.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Nbpp, Nbpp.data(), points.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tof, tof.data(), points.size() * sizeof(float), cudaMemcpyHostToDevice));

    
    int num_batches = (points.size() + N_pulse - 1) / N_pulse;
    int blockSize = 256;

    long long total_events = 0;

    /* ========== Batch processing simulation ========== */
    for (int b = 0; b < num_batches; ++b)
    {
        int start_idx = b * N_pulse;
        int cur_pulse = std::min(N_pulse, (int)points.size() - start_idx);
        int total_threads = N_pix * cur_pulse;
        int gridSize = (total_threads + blockSize - 1) / blockSize;

        int16_t* d_record;
        CUDA_CHECK(cudaMalloc(&d_record, total_threads * sizeof(int16_t)));
        CUDA_CHECK(cudaMemset(d_record, 0xFF, total_threads * sizeof(int16_t)));

        unsigned long long seed_offset = (unsigned long long)b * total_threads;

        spad_sim_kernel<<<gridSize, blockSize>>>(
            d_points, d_Nspp, d_Nbpp, d_tof,
            d_record,
            N_row, N_col, N_bin,
            DCR, r_bg, delta_t, tau,
            cur_pulse, start_idx,
            points.size(),
            sigma_pix, 
            seed_offset);

        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<int16_t> record(total_threads);
        CUDA_CHECK(cudaMemcpy(record.data(), d_record,
            total_threads * sizeof(int16_t), cudaMemcpyDeviceToHost));

        for (int x : record)
            if (x >= 0) total_events++;

        std::ofstream ofs(
            save_filename + "/batch_" + std::to_string(b) + ".bin",
            std::ios::binary
        );

        ofs.write(reinterpret_cast<char*>(record.data()), record.size() * sizeof(int16_t));
        ofs.close();

        CUDA_CHECK(cudaFree(d_record));

        std::cout << "\rProgress: "
                  << std::fixed << std::setprecision(1)
                  << 100.0 * (b + 1) / num_batches << "%";
    }

    CUDA_CHECK(cudaFree(d_points));
    CUDA_CHECK(cudaFree(d_Nspp));
    CUDA_CHECK(cudaFree(d_Nbpp));
    CUDA_CHECK(cudaFree(d_tof));

    // ================= End of statistics =================
    clock_t end_time = clock();
    double total_time =
        double(end_time - start_time) / CLOCKS_PER_SEC;

    std::cout << "\n==========================================" << std::endl;
    std::cout << "GPU photon counting simulation completed!" << std::endl;
    std::cout << "Total photon events: " << total_events << std::endl;
    std::cout << "Total time taken: " << total_time << " seconde" << std::endl;
    std::cout << "Temps moyen par point : "
              << total_time * 1e6 / points.size()
              << " microseconde" << std::endl;
    std::cout << "Les fichiers de sortie sont enregistr¨¦s dans : " << save_filename << "/" << std::endl;
    std::cout << "==========================================" << std::endl;
}
