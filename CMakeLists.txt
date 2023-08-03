cmake_minimum_required (VERSION 3.2)
project (stpphawkes)

# Enable CTest
enable_testing()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
find_package(R REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Celero)

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()


set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mtune=native -march=native -pedantic -Wall -Wextra -Wcast-align   -Wdisabled-optimization  -Wlogical-op  -Wnoexcept   -Wsign-promo -Wstrict-null-sentinel  -Wswitch-default -Wundef -Wno-unused")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mtune=native -march=native -Wall -pedantic -Wall -Wextra -Wcast-align  -Wdisabled-optimization -Wlogical-op  -Wnoexcept  -Wsign-promo -Wstrict-null-sentinel -Wswitch-default -Wundef -Wno-unused")

#find_package(Armadillo REQUIRED)

#find_package(GDAL REQUIRED)
#find_package(Boost COMPONENTS program_options unit_test_framework)

# Set default build type to release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
	endif()


# Use C++11
set(CMAKE_CXX_STANDARD 11)

#SET(CXX_FLAGS "${CXX_FLAGS} -mtune=native -fopenmp -O3 -Wall -fpic  -O2 -g -pipe -Wall -Wp,-D_FORTIFY_SOURCE=2
#-fexceptions -fstack-protector-strong --param=ssp-buffer-size=4 -grecord-gcc-switches") 

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(src)

