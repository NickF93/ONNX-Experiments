cmake_minimum_required(VERSION 3.10)

# Set the project name
project(onnx_inference_project)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

# Create options for building the project in Debug or Release mode
set(CMAKE_C_FLAGS "-fPIC")
set(CMAKE_C_FLAGS_RELEASE "-O3 -m64 -march=native -flto -s")
set(CMAKE_C_FLAGS_DEBUG "-g3 -Og -m64 -march=native -flto")

set(CMAKE_CXX_FLAGS "-Wno-builtin-macro-redefined -fPIC")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -m64 -march=native -flto -fno-ident -ffunction-sections -fdata-sections -Wl,--gc-sections -fomit-frame-pointer -fmerge-all-constants -fno-stack-protector -s")
set(CMAKE_CXX_FLAGS_DEBUG "-g3 -Og -m64 -march=native -flto")

# Define the executable
add_executable(onnx_inference onnx.cpp)

# Include directories
include_directories(/usr/local/include/onnxruntime)

# Link directories
link_directories(/usr/local/lib)

# Find and link OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Apply different compiler flags depending on the build type
if(CMAKE_BUILD_TYPE MATCHES Debug)
    set_source_files_properties(onnx.cpp PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS_DEBUG}")
elseif(CMAKE_BUILD_TYPE MATCHES Release)
    set_source_files_properties(onnx.cpp PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS_RELEASE}")
endif()

# Enable verbose makefile output
set(CMAKE_VERBOSE_MAKEFILE ON)

# Print compiler commands
set(CMAKE_RULE_MESSAGES ON)

# Link libraries using the "keyword" mode consistently
target_link_libraries(onnx_inference PRIVATE onnxruntime ${OpenCV_LIBS})
