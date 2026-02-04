#include "simulator.h"
#include <filesystem>
#include <string>
#include <iostream>

namespace fs = std::filesystem;

// Ensure the directory exists; if it does not exist, create it.
void EnsureDirectory(const std::string& path)
{
    fs::path dir(path);
    if (!fs::exists(dir))
    {
        if (fs::create_directories(dir))
            std::cout << "Directory creation successful: " << path << std::endl;
        else
            std::cerr << "Failed to create directory: " << path << std::endl;
    }
}

// Usage example
int main()
{
    std::string save_dir = "D:\\cu_file\\20260101\\rbg_400k_data";
    EnsureDirectory(save_dir);

    Simulator sim;

    sim.SaveParamsToFile(save_dir + "\\params.txt");
    
    auto traj = sim.TrajectorySimulator();
    sim.saveToTxt(save_dir + "\\traj.txt", traj);

    auto pixels = sim.CameraProjector(traj);
    sim.saveToTxt(save_dir + "\\pixels.txt", pixels);

    sim.SpadSimulator(pixels, save_dir);  // Direct directory transfer

    return 0;
}


