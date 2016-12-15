cmake_minimum_required (VERSION 3.6)
enable_language (C)
find_package (Threads)
find_library (OPENCL_LIBRARIES OpenCL)
project (clcurl)

file(GLOB SRC src/*.c)
set (SOURCES ${SRC})

set (LIBRARY_NAME clcurl)

set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pthread -Wall")

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin) 
SET (CTEST_BUILD_TARGET "DEVELOPMENT")

enable_testing()
add_subdirectory (test)
add_subdirectory (src)

