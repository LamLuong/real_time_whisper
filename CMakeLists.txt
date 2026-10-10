cmake_minimum_required(VERSION 3.10)
project(realtime_whisper VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

find_package(PkgConfig REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)

pkg_check_modules(LIBAV REQUIRED IMPORTED_TARGET
    libavcodec
    libavformat
    libavutil
    libswresample
)

link_directories(
   $ENV{TENSORRT_DIR}/lib
)

add_executable(${PROJECT_NAME}
    main.cpp
    whisper.cpp
    trt_infer.cpp
)

target_include_directories(${PROJECT_NAME}
                            PRIVATE
                            ${CUDAToolkit_INCLUDE_DIRS}
                            $ENV{TENSORRT_DIR}/include)

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${OpenCV_LIBS}
    PkgConfig::LIBAV
    CUDA::cudart
    nvinfer
)
target_compile_options(${PROJECT_NAME} PRIVATE -Wno-deprecated-declarations)