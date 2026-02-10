cmake_minimum_required(VERSION 3.20)
project(grasslm VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Dependencies ---

# Eigen (header-only, installed via: brew install eigen)
find_package(Eigen3 REQUIRED NO_MODULE)

# Apple Accelerate framework (macOS BLAS/LAPACK)
if(APPLE)
    find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
    # Use updated CBLAS/LAPACK headers (required since macOS 13.3)
    add_compile_definitions(ACCELERATE_NEW_LAPACK)
endif()

# --- Static library: libgrasslm.a ---

set(LIB_SOURCES
    src/tensor.cpp
    src/layers.cpp
    src/model.cpp
    src/tokenizer.cpp
    src/generator.cpp
    src/weight_loader.cpp
)

add_library(grasslm_lib STATIC ${LIB_SOURCES})
set_target_properties(grasslm_lib PROPERTIES OUTPUT_NAME grasslm)

target_include_directories(grasslm_lib
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(grasslm_lib PUBLIC Eigen3::Eigen)

if(APPLE)
    target_link_libraries(grasslm_lib PUBLIC ${ACCELERATE_FRAMEWORK})
endif()

# --- CLI executable ---

add_executable(grasslm_cli grasslm_cli.cpp)
target_link_libraries(grasslm_cli PRIVATE grasslm_lib)

# --- Tests ---

option(GRASSLM_BUILD_TESTS "Build tests" ON)

if(GRASSLM_BUILD_TESTS)
    enable_testing()

    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    # Prevent GoogleTest from overriding compiler/linker settings on Windows
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    include(GoogleTest)

    # test_tensor
    add_executable(test_tensor tests/test_tensor.cpp)
    target_link_libraries(test_tensor PRIVATE grasslm_lib GTest::gtest_main)
    gtest_discover_tests(test_tensor)

    # test_layers
    add_executable(test_layers tests/test_layers.cpp)
    target_link_libraries(test_layers PRIVATE grasslm_lib GTest::gtest_main)
    gtest_discover_tests(test_layers)

    # test_numerical (Python <-> C++ parity)
    add_executable(test_numerical tests/test_numerical.cpp)
    target_link_libraries(test_numerical PRIVATE grasslm_lib GTest::gtest_main)
    gtest_discover_tests(test_numerical)
endif()
