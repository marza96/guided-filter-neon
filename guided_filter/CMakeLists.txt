cmake_minimum_required(VERSION 4.0.0)

project(GuidedFilter)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "Build for Apple Silicon")

enable_testing()

if(APPLE)
    if(EXISTS "/opt/homebrew/opt/libomp")
        set(OpenMP_ROOT "/opt/homebrew/opt/libomp")
    elseif(EXISTS "/usr/local/opt/libomp")
        set(OpenMP_ROOT "/usr/local/opt/libomp")
    endif()
endif()

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

add_library(guided_filter_naive src/guided_filter_naive.cpp)
target_include_directories(guided_filter_naive PUBLIC include)
target_link_libraries(guided_filter_naive ${OpenCV_LIBS} OpenMP::OpenMP_CXX)

add_library(guided_filter_neon src/guided_filter_neon.cpp)
target_include_directories(guided_filter_neon PUBLIC include)
target_link_libraries(guided_filter_neon ${OpenCV_LIBS} OpenMP::OpenMP_CXX)


if (RUN_TEST STREQUAL "True")
    find_package(GTest REQUIRED)

    include(GoogleTest)

    add_executable(test_guided_filter_naive test/test_guided_filter_naive.cpp)
    target_include_directories(test_guided_filter_naive PUBLIC include)

    target_link_libraries(test_guided_filter_naive ${OpenCV_LIBS})
    target_link_libraries(test_guided_filter_naive GTest::gtest_main)
    target_link_libraries(test_guided_filter_naive guided_filter_naive)
    target_link_libraries(test_guided_filter_naive OpenMP::OpenMP_CXX)


    add_executable(test_guided_filter_neon test/test_guided_filter_neon.cpp)
    target_include_directories(test_guided_filter_neon PUBLIC include)

    target_link_libraries(test_guided_filter_neon ${OpenCV_LIBS})
    target_link_libraries(test_guided_filter_neon GTest::gtest_main)
    target_link_libraries(test_guided_filter_neon guided_filter_naive)
    target_link_libraries(test_guided_filter_neon guided_filter_neon)
    target_link_libraries(test_guided_filter_neon OpenMP::OpenMP_CXX)

    gtest_discover_tests(
        test_guided_filter_neon 
        DISCOVERY_MODE PRE_TEST
        DISCOVERY_TIMEOUT 30
    )

    gtest_discover_tests(
        test_guided_filter_naive 
        DISCOVERY_MODE PRE_TEST
        DISCOVERY_TIMEOUT 30
    )
endif()