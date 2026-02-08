cmake_minimum_required(VERSION 3.20)

project(hvlib
    VERSION 0.1.0
    LANGUAGES CXX
)

# ---- Global options ----
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING
        "Choose the type of build." FORCE)
endif()

option(NORM_COSINE "Use normalized cosine similarity" OFF)

# ---- Subdirectories ----
add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(targets)