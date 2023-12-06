#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>

#include <opencv2/opencv.hpp>

#include "simulator.hpp"


/**
 * Constructor for the NSsimulator class.
 * @param U: The radial velocity at edge of the drum.
 * @param R: The radius of the drum.
 * @param nu: The viscosity of the fluid.
 * @param r_min: The minimum radius to use in calculation (close to 0).
 * @param r_max: The maximum radius to use in calculation (inf).
 * @param t_max: The maximum time to use in calculation.
 * @param dr: The step size in radius.
 * @param dt: The step size in time.
 * @return NSsimulator object.
*/
NSsimulator::NSsimulator(float U, 
                         float R, 
                         float nu,
                         float r_min, 
                         float r_max, 
                         float t_max, 
                         float dr, 
                         float dt) {

    U_ = U;
    R_ = R;
    nu_ = nu;
    r_min_ = r_min;
    r_max_ = r_max;
    t_max_ = t_max;
    dr_ = dr;
    dt_ = dt;

    inv_2dr_ = 1.0f / (2 * dr_);
    inv_dr2_ = 1.0f / (dr_ * dr_);

    r_space_ = linearSpace(r_min_, r_max_, dr_);
    t_space_ = linearSpace(0, t_max_, dt_);

    steps_to_R_ = static_cast<int>((R - r_min_) / dr_);

    u_theta_ = Eigen::MatrixXf(r_space_.rows(), t_space_.rows());
    u_theta_.setZero();

    // init value u(r, 0) = 0
    for (int r_step = 0; r_step < r_space_.rows(); r_step++) {
        u_theta_(r_step, 0) = 0;
    }

    // boundary condition u(R, t) = omega(t) * R
    for (int t_step = 0; t_step < t_space_.rows(); t_step++) {
        u_theta_(steps_to_R_, t_step) = omega(t_space_(t_step)) * R;
    }

    // boundary condition u(inf, t) = 0
    for (int t_step = 0; t_step < t_space_.rows(); t_step++) {
        u_theta_(r_space_.rows() - 1, t_step) = 0;
    }
}

/**
 * Solves the Navier-Stokes equation for the rotating drum.
*/
void NSsimulator::solveNavierStokesEuler() {
    for (int t_step = 0; t_step < t_space_.rows() - 1; t_step++) {
        for (int r_step = 1; r_step < r_space_.rows() - 1; r_step++) {
            const float u = u_theta_(r_step, t_step);
            const float u_prev_r = u_theta_(r_step - 1, t_step);
            const float u_next_r = u_theta_(r_step + 1, t_step);

            const float dudr = (u_next_r - u_prev_r) * inv_2dr_;
            const float d2udr2 = (u_next_r - 2 * u + u_prev_r) * inv_dr2_;

            u_theta_(r_step, t_step + 1) = u + dt_ * dudt(r_space_(r_step), u, dudr, d2udr2);      
        }

        // boundary conditions
        u_theta_(r_space_.rows() - 1, t_step + 1) = 0;
        u_theta_(steps_to_R_, t_step + 1) = omega(t_space_(t_step + 1)) * R_;
    }

    // check if the solution diverges
    const float max_u = u_theta_.maxCoeff();
    const float min_u = u_theta_.minCoeff();
    if (std::isnan(max_u) || std::isnan(min_u)) {
        std::cout << "Solution diverged, use a smaller dt." << std::endl;
        std::exit(0);
    }
}


/**
 * Solves the Navier-Stokes equation for the rotating drum.
*/
void NSsimulator::solveNavierStokesRK4() {
    for (int t_step = 0; t_step < t_space_.rows() - 1; t_step++) {
        for (int r_step = 1; r_step < r_space_.rows() - 1; r_step++) {
            const float u = u_theta_(r_step, t_step);
            const float u_prev_r = u_theta_(r_step - 1, t_step);
            const float u_next_r = u_theta_(r_step + 1, t_step);

            const float dudr = (u_next_r - u_prev_r) * inv_2dr_;
            const float d2udr2 = (u_next_r - 2 * u + u_prev_r) * inv_dr2_;

            const float k1 = dt_ * dudt(r_space_(r_step), u, dudr, d2udr2);
            const float k2 = dt_ * dudt(r_space_(r_step), u + k1 / 2, dudr, d2udr2);
            const float k3 = dt_ * dudt(r_space_(r_step), u + k2 / 2, dudr, d2udr2);
            const float k4 = dt_ * dudt(r_space_(r_step), u + k3, dudr, d2udr2);

            u_theta_(r_step, t_step + 1) = u + (k1 + 2 * k2 + 2 * k3 + k4) / 6;      
        }

        // boundary conditions
        u_theta_(r_space_.rows() - 1, t_step + 1) = 0;
        u_theta_(steps_to_R_, t_step + 1) = omega(t_space_(t_step + 1)) * R_;
    }

    // check if the solution diverges
    const float max_u = u_theta_.maxCoeff();
    const float min_u = u_theta_.minCoeff();
    if (std::isnan(max_u) || std::isnan(min_u)) {
        std::cout << "Solution diverged, use a smaller dt." << std::endl;
        std::exit(0);
    }
}


/**
 * Simulates particles in the rotating drum.
 * @param particles_over_time: The particles simulated over time.
 * @param num_particles: The number of particles to simulate.
 * @param fps: The number of frames per second to simulate.
*/
void NSsimulator::simulateParticles(std::vector<Eigen::MatrixXf>& particles_over_time, 
                                    const int num_particles, 
                                    const int fps) {

    // setup the particles
    const int time_step_skip = static_cast<int>(1.0 / dt_ / fps);

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);

    std::uniform_real_distribution<float> radial_distribution(R_, r_max_*1.5);
    std::uniform_real_distribution<float> theta_distribution(0, 2 * M_PI);

    Eigen::MatrixXf particles(num_particles, 2);
    for (int i = 0; i < num_particles; i++) {
        particles(i, 0) = radial_distribution(generator);
        particles(i, 1) = theta_distribution(generator);
    }

    // simulate the particles
    for (int t_step = 0; t_step < u_theta_.cols(); t_step+=time_step_skip) {
        for (int i = 0; i < particles.rows(); i++) {
            const float r = particles(i, 0);
            const float theta = particles(i, 1);

            // update the position
            int r_step_int = static_cast<int>((r - r_min_) / dr_);
            const float r_step = (r - r_min_) / dr_;
            const float r_step_frac = r_step - r_step_int;

            if (r_step_int + 1 >= u_theta_.rows()) {
                r_step_int = u_theta_.rows() - 2;
            }

            const float u_theta_r = u_theta_(r_step_int, t_step) * (1 - r_step_frac) + u_theta_(r_step_int + 1, t_step) * r_step_frac;

            float new_r = r + radialVelocity(r) * dt_ * time_step_skip;
            const float new_theta = theta + u_theta_r * dt_ * time_step_skip;

            if (new_r < r_min_) {
                new_r = radial_distribution(generator);
            }

            particles(i, 0) = new_r;
            particles(i, 1) = new_theta;
        }
        particles_over_time.push_back(particles);
    }
}


/**
 * Visualizes the particles in the rotating drum.
 * @param particles_over_time: The particles simulated over time.
 * @param fps: The number of frames per second to simulate.
*/
void NSsimulator::visualizeParticles(const std::vector<Eigen::MatrixXf>& particles_over_time, 
                                     const int fps, 
                                     const int particle_size) {
    // visualize the particles
    std::vector<cv::Mat> images;

    float current_angle = 0;

    // get 30 images per second
    const float time_step_skip = 1.0 / dt_ / static_cast<float>(fps);
    const int image_size = static_cast<int>(sqrt(2) * (r_max_ - r_min_) / dr_);

    for (int t_step = 0; t_step < particles_over_time.size(); t_step++) {
        cv::Mat image(image_size, image_size, CV_8UC3, cv::Scalar(0, 0, 0));
        const Eigen::MatrixXf& particles_this_time = particles_over_time[t_step];

        // add the particles
        drawParticles(image, particles_this_time, particle_size);

        // add the drum filter and integrate the angle 
        current_angle += omega(t_step * dt_ * time_step_skip) * dt_ * time_step_skip;
        drawDrumFilter(image, current_angle);

        // resize the image
        cv::resize(image, image, cv::Size(1000, 1000), 0, 0, cv::INTER_LINEAR);
        images.push_back(image);
    }

    // show the images as a video
    for (int i = 0; i < images.size(); i++) {
        cv::imshow("Particles", images[i]);
        cv::waitKey(static_cast<int>(1000 / fps));
    }
}


/**
 * Creates a linear space of floats.
 * @param start: The start of the linear space.
 * @param end: The end of the linear space.
 * @param step: The step size of the linear space.
 * @return A linear space of floats.
*/
Eigen::MatrixXf NSsimulator::linearSpace(const float start, 
                                         const float end, 
                                         const float step) {

    const int size = (end - start) / step + 1;
    Eigen::MatrixXf result(size, 1);
    for (int i = 0; i < size; i++) {
        result(i, 0) = start + i * step;
    }
    return result;
}

/**
 * Arbitrary function for aziumthal velocity at R.
 * @param t: The time to calculate the angular velocity at.
 * @return The angular velocity of the drum at time t.
*/
inline float NSsimulator::omega(const float t) {
    return 2*sin(t);
}

/**
 * Calculates the time derivative of the azimuthal velocity.
 * @param r: The radius to calculate the time derivative at.
 * @param u: Current azimuthal velocity.
 * @param dudr: The radial derivative of the azimuthal velocity.
 * @param d2udr2: The second radial derivative of the azimuthal velocity.
 * @return The time derivative of the azimuthal velocity.
*/
inline float NSsimulator::dudt(const float r, const float u, const float dudr, const float d2udr2) {
    const float inv_r = 1.0f / r;
    const float inv_r2 = inv_r * inv_r;

    return nu_ * (d2udr2 + inv_r * dudr - u * inv_r2) + U_ * R_ * inv_r * dudr + U_ * R_ * inv_r2 * u;
}


/**
 * Calculates the radial velocity of the fluid.
 * @param r: The radius to calculate the radial velocity at.
 * @return The radial velocity of the fluid.
*/
inline float NSsimulator::radialVelocity(const float r) {
    return - U_ * R_ / r;
}


/**
 * Draws the drum filter on the image.
 * @param image: The image to draw the drum filter on.
 * @param current_angle: The current angle of the drum.
*/
void NSsimulator::drawDrumFilter(cv::Mat& image, const float current_angle) {
    const int num_drums = 20;
    const float delta_theta = 2 * M_PI / num_drums;
    const cv::Point center(image.rows / 2, image.cols / 2);

    for (int i = 0; i < num_drums; i++) {
        const float x_img = image.rows / 2 + steps_to_R_ * sin(current_angle + delta_theta * i);
        const float y_img = image.cols / 2 + steps_to_R_ * cos(current_angle + delta_theta * i);
        const cv::Point end(x_img, y_img);
        cv::line(image, center, end, cv::Scalar(75, 75, 75), 2);
    }

    // draw the drum circle
    cv::circle(image, center, steps_to_R_, cv::Scalar(75, 75, 75), 2);
}


/**
 * Draws the particles on the image.
 * @param image: The image to draw the particles on.
 * @param particles: The particles to draw on the image.
*/
void NSsimulator::drawParticles(cv::Mat& image, const Eigen::MatrixXf& particles, const int particle_size) {
    const int image_size = image.rows;

    for (int i = 0; i < particles.rows(); i++) {
        const float r = particles(i, 0);
        const float theta = particles(i, 1);

        // convert to cartesian image coordinates
        const float x_img = (r - r_min_) / dr_ * sin(theta) + image_size / 2.0;
        const float y_img = (r - r_min_) / dr_ * cos(theta) + image_size / 2.0;

        if (x_img < 0 || x_img >= image_size || y_img < 0 || y_img >= image_size) { continue; }

        if (r < R_) {
            // The particle is inside the drum filter
            // draw with darker color
            const int darkness = static_cast<int>(0.5*255.0 * pow(r / R_, particle_size + 1));
            cv::circle(image, cv::Point(x_img, y_img), particle_size, cv::Scalar(darkness, darkness, darkness), -1);
        }
        else {
            cv::circle(image, cv::Point(x_img, y_img), particle_size, cv::Scalar(255, 255, 255), -1);
        }
    }
}


int main(int argc, char **argv) {
    float U = 0.75; 
    float R = 1.5;
    float nu = 0.8;
    float r_min = 0.1;
    float r_max = 5;
    float t_max = 10;
    float dr = 0.01;
    float dt = 0.00001;
    int fps = 30;
    int num_particles = 100000;
    int particle_size = 0;

    NSsimulator simulator(U, R, nu, r_min, r_max, t_max, dr, dt);

    std::cout << "Solving Navier Stokes..." << std::endl;

    auto start = std::chrono::system_clock::now();
    simulator.solveNavierStokesEuler();
    auto end = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed_seconds = end-start;
    std::cout << "Solved Navier Stokes in " << elapsed_seconds.count() << " seconds" << std::endl;
    std::cout << "Starting particle simulation" << std::endl;

    std::vector<Eigen::MatrixXf> particles_over_time;
    auto start2 = std::chrono::system_clock::now();
    simulator.simulateParticles(particles_over_time, num_particles, fps);
    auto end2 = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed_seconds2 = end2-start2;
    std::cout << "Simulated " << num_particles << " particles in " 
              << elapsed_seconds2.count() << " seconds" << std::endl;

    std::cout << "Starting visualization..." << std::endl;
    simulator.visualizeParticles(particles_over_time, fps, particle_size);
    return 0;
}