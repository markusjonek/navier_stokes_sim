#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <chrono>
#include <ctime>

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>


class NSsimulator {
    public:
        NSsimulator(float U, float R, float nu, float r_min, float r_max, float t_max, float dr, float dt);
        void solveNavierStokes();
        void simulateParticles(std::vector<Eigen::MatrixXf> &particles_over_time, int num_particles, int fps);
        void visualizeParticles(std::vector<Eigen::MatrixXf> &particles_over_time, int fps);
    private:
        float U_;
        float R_;
        float nu_;
        float r_min_;
        float r_max_;
        float t_max_;
        float dr_;
        float dt_;
        int steps_to_R_;
        Eigen::MatrixXf r_space_;
        Eigen::MatrixXf t_space_;
        Eigen::MatrixXf u_theta_;
        Eigen::MatrixXf linearSpace(float start, float end, float step);
        float omega(float t);
        float dudt(float r, float u, float dudr, float d2udr2);
        float radialVelocity(float r);
        void drawDrumFilter(cv::Mat &image, float current_angle);
};