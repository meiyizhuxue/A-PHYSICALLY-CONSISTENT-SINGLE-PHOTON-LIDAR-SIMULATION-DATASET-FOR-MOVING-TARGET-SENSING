#ifndef SIMULATOR_H
#define SIMULATOR_H

#include <cmath>
#include <vector>
#include <array>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <algorithm>

// =============================
// Basic data structures
// =============================

struct Point
{
    double t;
    double x, y, z;
};

// =============================
// Simulation
// =============================

class Simulator
{
public:
    // Generation of flight paths for drones
    std::vector<Point> TrajectorySimulator();

    // Pixel coordinate mapping generation
    std::vector<Point> CameraProjector(
        const std::vector<Point>& traj);

    // Save the text file
    void saveToTxt(
        const std::string& filename,
        const std::vector<Point>& point);

    // Photon Event Simulation
    void SpadSimulator(
        const std::vector<Point>& pixels,
        const std::string& save_filename
    );

    // Save parameters
    void SaveParamsToFile(
        const std::string& filename
    );
};

#endif