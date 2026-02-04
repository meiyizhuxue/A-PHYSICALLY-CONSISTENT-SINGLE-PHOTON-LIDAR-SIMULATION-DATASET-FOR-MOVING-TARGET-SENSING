#ifndef PARAMS_H
#define PARAMS_H

#include <cmath>
#include <vector>
#include <array>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <algorithm>

namespace Params {
    // base
    constexpr double dt = 0.0005;          // Sampling step
    constexpr int num_points = 20000;     // Number of sampling points

    // Initial position
    constexpr double x0 = 0.0;             // Initial x
    constexpr double y0 = 0.0;             // Initial y
    constexpr double z0 = 100.0;           // Initial z

    // Expected trajectory function
    inline double f_x(double t) {
        return x0 + 30.0 * std::sin(0.15 * t) + 5.0 * std::sin(0.5 * t);
        // return x0 + 10 * t;
    }

    inline double f_y(double t) {
        return y0 + std::sin(0.3 * t);
        // return y0;
    }

    inline double f_z(double t) {
        return z0 + 20.0 * std::sin(0.1 * t) + 10.0 * std::sin(0.4 * t) + 5.0 * std::sin(0.5 * t);
        // return z0;
    }

    // =============================
    // Camera / Lidar Pose (World Coordinate System)
    // =============================

    // Camera position T = [-120; 0; 100]
    constexpr std::array<double, 3> cam_pos = { -120.0, 0.0, 100.0 };

    // forward direction
    constexpr std::array<double, 3> cam_forward = { 1.0, 0.0, 0.0 };

    // Up direction in the world coordinate system
    constexpr std::array<double, 3> world_up = { 0.0, 0.0, 1.0 };

    // =============================
    // Imaging parameters
    // =============================

    constexpr int N_row = 128;
    constexpr int N_col = 128;

    constexpr double fx = 200.0;
    constexpr double fy = 200.0;

    // =============================
    // SPAD / SPL Detection Parameters
    // =============================

    constexpr float PDE = 0.2f;        // Photon Detection Efficiency
    constexpr float DCR = 2e4f;        // Dark Count Rate (Hz)

    constexpr float delta_t = 1e-9f;   // Time bin width (seconds)
    constexpr int N_bin = 4000;        // nombre de bacs

    constexpr float tau = 2e-9f;       // Écart type de la gigue temporelle (s)

    constexpr int N_pulse = 1000;      // Nombre d'impulsions

    constexpr float sigma_pix = 1.5f;  // Coefficient de diffusion spatiale

    constexpr float c = 3e8f;          // vitesse de la lumière

    // =============================
    // Paramètres statistiques des photons
    // =============================

    constexpr float N_s = 5.0f;        // Attente de photons de signal
    constexpr float N_b = 0.0f;        // Attente de photons de fond
    constexpr float r_bg = 4e5f;       // Taux de bruit de fond pour les autres cibles (Hz)

    // Distance de référence (pour l'atténuation quadratique, etc.)
    constexpr float R_ref = 100.0f;
}

#endif // TRAJECTORY_PARAMS_H