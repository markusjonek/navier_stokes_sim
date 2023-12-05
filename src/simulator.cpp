#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#include "simulator.hpp"

NSsimulator::NSsimulator(float U, float R, float nu,float r_min, float r_max, float t_max, float dr, float dt) {
    U_ = U;
    R_ = R;
    nu_ = nu;
    r_min_ = r_min;
    r_max_ = r_max;
    t_max_ = t_max;
    dr_ = dr;
    dt_ = dt;

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

Eigen::MatrixXf NSsimulator::linearSpace(float start, float end, float step) {
    int size = (end - start) / step + 1;
    Eigen::MatrixXf result(size, 1);
    for (int i = 0; i < size; i++) {
        result(i, 0) = start + i * step;
    }
    return result;
}

float NSsimulator::omega(float t) {
    return 2*sin(M_PI * t);
}

float NSsimulator::dudt(float r, float u, float dudr, float d2udr2) {
    float nu_side = nu_ * (d2udr2 + (1/r) * dudr - u / pow(r, 2));
    float other_side = (U_ * R_ / r) * dudr + (U_ * R_ / pow(r, 2)) * u;

    return nu_side + other_side;
}

float NSsimulator::radialVelocity(float r) {
    return - U_ * R_ / r;
}

void NSsimulator::solveNavierStokes() {
    for (int t_step = 0; t_step < t_space_.rows() - 1; t_step++) {
        for (int r_step = 1; r_step < r_space_.rows() - 1; r_step++) {

            float u = u_theta_(r_step, t_step);
            float dudr = (u_theta_(r_step + 1, t_step) - u_theta_(r_step - 1, t_step)) / (2*dr_);
            float d2udr2 = (u_theta_(r_step + 1, t_step) - 2 * u_theta_(r_step, t_step) + u_theta_(r_step - 1, t_step)) / pow(dr_, 2);
            u_theta_(r_step, t_step + 1) = u_theta_(r_step, t_step) + dt_ * dudt(r_space_(r_step), u, dudr, d2udr2);

            // boundary condition u(inf, t) = 0
            u_theta_(r_space_.rows() - 1, t_step + 1) = 0;

            // boundary condition u(R, t) = omega(t) * R
            u_theta_(steps_to_R_, t_step + 1) = omega(t_space_(t_step + 1)) * R_;

        }
    }
}

void NSsimulator::simulateParticles(std::vector<Eigen::MatrixXf> &particles_over_time, int num_particles, int fps) {
    // setup the particles
    int time_step_skip = (int)(1.0 / dt_ / fps);

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
            float r = particles(i, 0);
            float theta = particles(i, 1);

            // update the position
            int r_step_int = (int)((r - r_min_) / dr_);
            float r_step = (r - r_min_) / dr_;
            float r_step_frac = r_step - r_step_int;

            if (r_step_int + 1 >= u_theta_.rows()) {
                r_step_int = u_theta_.rows() - 2;
            }

            float u_theta_r = u_theta_(r_step_int, t_step) * (1 - r_step_frac) + u_theta_(r_step_int + 1, t_step) * r_step_frac;

            float new_r = r + radialVelocity(r) * dt_ * time_step_skip;
            float new_theta = theta + u_theta_r * dt_ * time_step_skip;

            if (new_r < r_min_) {
                new_r = radial_distribution(generator);
            }

            particles(i, 0) = new_r;
            particles(i, 1) = new_theta;
        }
        particles_over_time.push_back(particles);
    }
}


void NSsimulator::drawDrumFilter(cv::Mat &image, float current_angle) {
    int num_drum = 20;
    float delta_theta = 2 * M_PI / num_drum;
    int steps_to_R = (int) (R_ / dr_);
    cv::Point center(image.rows / 2, image.cols / 2);

    for (int i = 0; i < num_drum; i++) {
        float x_img = image.rows / 2 + steps_to_R * sin(current_angle + delta_theta * i);
        float y_img = image.cols / 2 + steps_to_R * cos(current_angle + delta_theta * i);
        cv::Point end(x_img, y_img);
        cv::line(image, center, end, cv::Scalar(75, 75, 75), 2);
    }

    // draw the drum circle
    cv::circle(image, cv::Point(image.rows / 2, image.cols / 2), (int)(R_ / dr_), cv::Scalar(75, 75, 75), 2);
}


void NSsimulator::visualizeParticles(std::vector<Eigen::MatrixXf> &particles_over_time, int fps) {
    // visualize the particles
    std::vector<cv::Mat> images;

    float current_angle = 0;

    // get 30 images per second
    float time_step_skip = 1.0 / dt_ / (float)fps;
    int image_size = int((r_max_ - r_min_) / dr_);


    for (int t_step = 0; t_step < particles_over_time.size(); t_step++) {
        cv::Mat image(image_size, image_size, CV_8UC3, cv::Scalar(0, 0, 0));
        Eigen::MatrixXf particles_this_time = particles_over_time[t_step];

        for (int i = 0; i < particles_this_time.rows(); i++) {
            float r = particles_this_time(i, 0);
            float theta = particles_this_time(i, 1);

            float x_img = (r - r_min_) / dr_ * sin(theta) + image_size / 2.0;
            float y_img = (r - r_min_) / dr_ * cos(theta) + image_size / 2.0;

            if (x_img < 0 || x_img >= image_size || y_img < 0 || y_img >= image_size) { continue; }

            if (r < R_) {
                int darkness = (int) (0.5*255.0 * pow(r / R_, 2));
                cv::circle(image, cv::Point(x_img, y_img), 2, cv::Scalar(darkness, darkness, darkness), -1);
            }
            else {
                cv::circle(image, cv::Point(x_img, y_img), 2, cv::Scalar(255, 255, 255), -1);
            }
        }

        // add the drum filter
        current_angle += omega(t_step * dt_ * time_step_skip) * dt_ * time_step_skip;
        drawDrumFilter(image, current_angle);

        // resize the image
        cv::resize(image, image, cv::Size(1000, 1000), 0, 0, cv::INTER_LINEAR);
        images.push_back(image);
    }
    // show the images as a video
    for (int i = 0; i < images.size(); i++) {
        cv::imshow("Particles", images[i]);
        cv::waitKey((int)(1000 / fps));
    }
}



int main(int argc, char **argv) {
    float U = 1.0;
    float R = 1.5;
    float nu = 2.0;
    float r_min = 0.001;
    float r_max = 10;
    float t_max = 10;
    float dr = 0.01;
    float dt = 0.00001;
    int fps = 30;
    int num_particles = 10000;

    std::cout << "Solving Navier Stokes..." << std::endl;
    auto start = std::chrono::system_clock::now();
    NSsimulator simulator(U, R, nu, r_min, r_max, t_max, dr, dt);
    simulator.solveNavierStokes();
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::cout << "Solved Navier Stokes in " << elapsed_seconds.count() << " seconds" << std::endl;

    std::cout << "Starting particle simulation" << std::endl;
    auto start2 = std::chrono::system_clock::now();
    std::vector<Eigen::MatrixXf> particles_over_time;
    simulator.simulateParticles(particles_over_time, num_particles, fps);
    auto end2 = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds2 = end2-start2;
    std::cout << "Simulated particles in " << elapsed_seconds2.count() << " seconds" << std::endl;

    std::cout << "Starting visualization..." << std::endl;

    simulator.visualizeParticles(particles_over_time, fps);
    return 0;
}