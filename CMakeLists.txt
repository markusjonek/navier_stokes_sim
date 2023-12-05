# Set the minimum required version of CMake
cmake_minimum_required(VERSION 3.0)

# Set the project name
project(NSsimulator)

# Find the Eigen library
find_package(Eigen3 REQUIRED)

# Find the OpenCV library
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-O3")

# Add the include directories for Eigen and OpenCV
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(include)

# Add the executable target
add_executable(particle_simulation src/simulator.cpp)
target_link_libraries(particle_simulation Eigen3::Eigen ${OpenCV_LIBS})
