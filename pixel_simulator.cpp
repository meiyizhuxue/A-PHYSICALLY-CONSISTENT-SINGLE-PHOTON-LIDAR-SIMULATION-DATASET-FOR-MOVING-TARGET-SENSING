#include "params.h"
#include "simulator.h"

#include <fstream>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <vector>
#include <Eigen/Dense>

// #include <Eigen/Dense>

std::vector<Point> Simulator::TrajectorySimulator()
{	
	std::vector<Point> traj; // World coordinates of sampling points
	traj.reserve(Params::num_points); // Pre-allocated size

    for (int i = 0; i <= Params::num_points; ++i)
    {
        double t = i * Params::dt; // Sampling timestamp
        // Obtain the expected trajectory
        double xd = Params::f_x(t);
        double yd = Params::f_y(t);
        double zd = Params::f_z(t);

        // Obtain simulated trajectory
        traj.push_back({ t, xd, yd, zd });
    }
    return traj;
}

std::vector<Point> Simulator::CameraProjector(const std::vector<Point>& traj)
{
    // =============================
    // 1. Camera Pose (Eigen)
    // =============================
    Eigen::Vector3d T(Params::cam_pos[0], Params::cam_pos[1], Params::cam_pos[2]);
    Eigen::Vector3d forward(Params::cam_forward[0], Params::cam_forward[1], Params::cam_forward[2]);
    Eigen::Vector3d up(Params::world_up[0], Params::world_up[1], Params::world_up[2]);

    // =============================
    // 2. Constructing a rotation matrix
    // =============================
    Eigen::Vector3d z = forward.normalized();

    Eigen::Vector3d x = up.cross(z);
    if (x.norm() < 1e-6)
    {
        x = Eigen::Vector3d(1, 0, 0).cross(z);
        if (x.norm() < 1e-6)
            x = Eigen::Vector3d(0, 0, 1).cross(z);
    }
    x.normalize();

    Eigen::Vector3d y = z.cross(x).normalized();

    Eigen::Matrix3d R;
    R.col(0) = x;
    R.col(1) = y;
    R.col(2) = z;

    // Orthogonality correction
    if ((x.cross(y) - z).norm() > 1e-6)
    {
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(
            R, Eigen::ComputeFullU | Eigen::ComputeFullV);
        R = svd.matrixU() * svd.matrixV().transpose();
    }

    Eigen::Matrix3d Rt = R.transpose(); 

    // =============================
    // 3. Internal Reference
    // =============================
    const double cx = (Params::N_col + 1) / 2.0;
    const double cy = (Params::N_row + 1) / 2.0;

    std::vector<Point> pixels;
    pixels.reserve(traj.size());

    // =============================
    // 4. World ¡ú Camera ¡ú Pixel
    // =============================
    for (const auto& p : traj)
    {
        Eigen::Vector3d Pw(p.x, p.y, p.z);
        Eigen::Vector3d Pc = Rt * (Pw - T);

        double zc = Pc.z();
        if (zc <= 1e-6)
            continue;

        double u_f = Params::fx * Pc.x() / zc + cx;
        double v_f = -Params::fy * Pc.y() / zc + cy;

        double u = static_cast<int>(std::round(u_f));
        double v = static_cast<int>(std::round(v_f));

        if (u < 1 || u > Params::N_col || v < 1 || v > Params::N_row)
            continue;

        pixels.push_back({ p.t, u, v, zc });
    }
    return pixels;
}

void Simulator::saveToTxt(const std::string& filename, const std::vector<Point>& point)
{
    std::ofstream ofs(filename);
    ofs << "# t x(u) y(v) z(d)\n";
    for (const auto& p : point)
    {
        ofs << p.t << " "
            << p.x << " "
            << p.y << " "
            << p.z << "\n";
    }
    ofs.close();
    std::cout << "The text file has been saved to " << filename << std::endl;
}

void Simulator::SaveParamsToFile(const std::string& filename)
{
    std::ofstream ofs(filename);
    if (!ofs) {
        std::cerr << "Unable to open the file: " << filename << std::endl;
        return;
    }

    ofs << std::fixed << std::setprecision(6);

    // ================= Constant parameter =================
    ofs << "=== Basic Parameters ===\n";
    ofs << "dt = " << Params::dt << "\n";
    ofs << "num_points = " << Params::num_points << "\n\n";

    ofs << "=== Initial position ===\n";
    ofs << "x0 = " << Params::x0 << "\n";
    ofs << "y0 = " << Params::y0 << "\n";
    ofs << "z0 = " << Params::z0 << "\n\n";

    ofs << "=== Optical axis orientation ===\n";
    ofs << "x = " << Params::cam_forward[0] << "\n";
    ofs << "y = " << Params::cam_forward[1] << "\n";
    ofs << "z = " << Params::cam_forward[2] << "\n\n";

    ofs << "=== SPAD/SPL parameters ===\n";
    ofs << "PDE = " << Params::PDE << "\n";
    ofs << "DCR = " << Params::DCR << "\n";
    ofs << "delta_t = " << Params::delta_t << "\n";
    ofs << "N_bin = " << Params::N_bin << "\n";
    ofs << "tau = " << Params::tau << "\n";
    ofs << "N_pulse = " << Params::N_pulse << "\n";
    ofs << "sigma_pix = " << Params::sigma_pix << "\n";
    ofs << "c = " << Params::c << "\n\n";

    ofs << "=== Statistical parameters of photons ===\n";
    ofs << "N_s = " << Params::N_s << "\n";
    ofs << "N_b = " << Params::N_b << "\n";
    ofs << "r_bg = " << Params::r_bg << "\n";
    ofs << "R_ref = " << Params::R_ref << "\n\n";

    ofs.close();
    std::cout << "Parameters have been saved to " << filename << std::endl;
}
