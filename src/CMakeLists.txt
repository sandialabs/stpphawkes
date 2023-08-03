# Build Common Library first

SET(COMMON_SOURCES helper_functions.cpp)

add_library(stpphawkes_common ${COMMON_SOURCES})

target_link_libraries(stpphawkes_common ${R_LIBRARIES})

include_directories(libstpphawkes ${R_INCLUDE_DIR} ${RCPP_INCLUDE})

# Build Temporal Code

SET(STPP_SOURCES condInt_mcmc_temporal_branching_md.cpp
                 simulate_temporal_hawkes.cpp
                 temporal_common.cpp
                 utilities.cpp
                 )
                 
add_library(libstpphawkes ${STPP_SOURCES})

target_link_libraries(libstpphawkes ${R_LIBRARIES} stpphawkes_common)

include_directories(libstpphawkes ${R_INCLUDE_DIR} ${RCPP_INCLUDE})


SET(CATMARK_SOURCES temporal_catmark.cpp
                    temporal_catmark_common.cpp
                    temporal_catmark_md.cpp
                 )
find_package(GSL REQUIRED)                   
add_library(catmark ${CATMARK_SOURCES})

target_link_libraries(catmark ${R_LIBRARIES} libstpphawkes stpphawkes_common GSL::gsl)

include_directories(catmark ${R_INCLUDE_DIR} ${RCPP_INCLUDE})

set(SPATIO_TEMPORAL_SOURCES condInt_mcmc_stpp_branching.cpp condInt_mcmc_stpp_branching_md.cpp spatio_temporal_common.cpp)
add_library(spatiotemporal ${SPATIO_TEMPORAL_SOURCES})
target_link_libraries(spatiotemporal libstpphawkes)
include_directories(spatiotemporal ${R_INCLUDE_DIR} ${RCPP_INCLUDE})

