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

/**
  * Class for solving and simulating the Navier-Stokes equation
  * for a rotating drum.
*/
class NSsimulator {
public:
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
    NSsimulator(float U, 
                float R, 
                float nu, 
                float r_min, 
                float r_max, 
                float t_max, 
                float dr, 
                float dt);

    /**
     * Solves the Navier-Stokes equation for the rotating drum.
    */
    void solveNavierStokesEuler();

    /**
     * Solves the Navier-Stokes equation for the rotating drum.
    */
    void solveNavierStokesRK4();

    /**
     * Simulates particles in the rotating drum.
     * @param particles_over_time: The particles simulated over time.
     * @param num_particles: The number of particles to simulate.
     * @param fps: The number of frames per second to simulate.
    */
    void simulateParticles(std::vector<Eigen::MatrixXf>& particles_over_time, 
                           const int num_particles, 
                           const int fps);

    /**
     * Visualizes the particles in the rotating drum.
     * @param particles_over_time: The particles simulated over time.
     * @param fps: The number of frames per second to simulate.
    */
    void visualizeParticles(const std::vector<Eigen::MatrixXf>& particles_over_time, 
                            const int fps,
                            const int particle_size=0);

private:
    float U_;
    float R_;
    float nu_;
    float r_min_;
    float r_max_;
    float t_max_;
    float dr_;
    float dt_;

    float inv_2dr_;
    float inv_dr2_;

    int steps_to_R_;

    Eigen::MatrixXf r_space_;
    Eigen::MatrixXf t_space_;
    Eigen::MatrixXf u_theta_;

    /**
     * Creates a linear space of floats.
     * @param start: The start of the linear space.
     * @param end: The end of the linear space.
     * @param step: The step size of the linear space.
     * @return A linear space of floats.
    */
    Eigen::MatrixXf linearSpace(const float start, 
                                const float end, 
                                const float step);
    
    /**
     * Arbitrary function for aziumthal velocity at R.
     * @param t: The time to calculate the angular velocity at.
     * @return The angular velocity of the drum at time t.
    */
    inline float omega(const float t);

    /**
     * Calculates the time derivative of the azimuthal velocity.
     * @param r: The radius to calculate the time derivative at.
     * @param u: Current azimuthal velocity.
     * @param dudr: The radial derivative of the azimuthal velocity.
     * @param d2udr2: The second radial derivative of the azimuthal velocity.
     * @return The time derivative of the azimuthal velocity.
    */
    inline float dudt(const float r, 
                      const float u, 
                      const float dudr, 
                      const float d2udr2);

    /**
     * Calculates the radial velocity of the fluid.
     * @param r: The radius to calculate the radial velocity at.
     * @return The radial velocity of the fluid.
    */
    inline float radialVelocity(const float r);

    /**
     * Draws the drum filter on the image.
     * @param image: The image to draw the drum filter on.
     * @param current_angle: The current angle of the drum.
    */
    void drawDrumFilter(cv::Mat& image, const float current_angle);

    /**
     * Draws the particles on the image.
     * @param image: The image to draw the particles on.
     * @param particles: The particles to draw on the image.
    */
    void drawParticles(cv::Mat& image, const Eigen::MatrixXf& particles, const int particle_size);
};