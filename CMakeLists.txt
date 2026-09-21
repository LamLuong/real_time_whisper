cmake_minimum_required(VERSION 3.10)
project(realtime_whisper VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PkgConfig REQUIRED)

pkg_check_modules(LIBAV REQUIRED IMPORTED_TARGET
    libavcodec
    libavformat
    libavutil
    libswresample
)

add_executable(${PROJECT_NAME}
    main.cpp
    whisper.cpp
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    PkgConfig::LIBAV
)