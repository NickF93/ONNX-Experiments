cmake_minimum_required(VERSION 3.10)

# Set the project name
project(onnx_inference_project)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add the executable
add_executable(onnx_inference onnx.cpp)

# Include directories
include_directories(/usr/local/include/onnxruntime)

# Link directories
link_directories(/usr/local/lib)

# Find and link OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Set enhanced compiler flags specifically for onnx.cpp to optimize for speed and size
set_source_files_properties(onnx.cpp PROPERTIES COMPILE_FLAGS "-O3 -m64 -march=native -flto -fno-ident -ffunction-sections -fdata-sections -Wl,--gc-sections -fomit-frame-pointer -fmerge-all-constants -fno-stack-protector")

# Enable verbose makefile output
set(CMAKE_VERBOSE_MAKEFILE ON)

# Link libraries using the "keyword" mode consistently
target_link_libraries(onnx_inference PRIVATE onnxruntime ${OpenCV_LIBS})
