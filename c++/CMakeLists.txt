cmake_minimum_required(VERSION 3.0)
project(c__)

set(CMAKE_CXX_STANDARD 11)
# set(CMAKE_BUILD_TYPE "Release" CACHE STRING
#       "Choose the type of build, options are: Debug Release
# RelWithDebInfo MinSizeRel."
#       FORCE)
find_package(MPI REQUIRED)

add_executable(dummy main.cpp src/discrete_environment.cpp src/random_utils.cpp src/envs/frozen_lake.cpp src/envs/frozen_lake_ext.cpp  src/envs/copy.cpp)
target_include_directories(dummy PUBLIC include)
target_include_directories(dummy PUBLIC ${MPI_INCLUDE_PATH})
target_link_libraries(dummy ${MPI_LIBRARIES})
