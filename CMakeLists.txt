cmake_minimum_required(VERSION 3.4)

################################################################################
## General settings
################################################################################

project("Pypp-NeuroEvolution")
message("\n####################################################################"
    "############\n## CMakeFile for ${PROJECT_NAME}")

set(CMAKE_CXX_FLAGS "-Wall -Wextra -pedantic")

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")

################################################################################
## Dependencies
################################################################################

if (${NO_GVC})
    message("Not searching for gvc")
else()
    find_package(PkgConfig)
    pkg_search_module(gvc libgvc libcgraph libcdt)
    if (${gvc_FOUND})
        add_definitions(-DWITH_GVC)
        include_directories(${gvc_INCLUDE_DIRS})
        list(APPEND CORE_LIBS ${gvc_LIBRARIES})
        message("-- gvc found at ${gvc_INCLUDEDIR}")
    else()
        message("Failed to find gvc")
    endif()
endif()

#set(PYBIND11_INCLUDE ${PROJECT_SOURCE_DIR}/extern/pybind11/include)

################################################################################
## Source files
################################################################################

FUNCTION(PREPEND output prefix)
   SET(listVar "")
   FOREACH(f ${${output}})
      LIST(APPEND listVar "${prefix}/${f}")
   ENDFOREACH(f)
   SET(${output} "${listVar}" PARENT_SCOPE)
ENDFUNCTION(PREPEND)

set(CPP_SRC
    "config.cpp"

    "misc/constants.h"
    "misc/point.hpp"

    "genotype.h"

    "phenotype/cppn.cpp"
#    "phenotype/ann.cpp"
#    "phenotype/eshn.cpp"
)
PREPEND(CPP_SRC "cpp")

set(PYBIND_SRC
    "module.cpp"
    "config.cpp"

#    "misc/constants.cpp"
    "genotype.cpp"

    "phenotype/cppn.cpp"
#    "phenotype/ann.cpp"
#    "phenotype/eshn.cpp"
)
PREPEND(PYBIND_SRC "pybind")

set(LIB pyne_cpp)
add_subdirectory(extern/pybind11)
pybind11_add_module(${LIB} ${CPP_SRC} ${PYBIND_SRC})

################################################################################
## Target (cppn test)
################################################################################
#if (WITH_TESTS)
#    add_executable(
#      cppn_test
#      "src/tests/cppn.cpp")
#  target_link_libraries(cppn_test ${CORE_LIBS} eshn-core)
#endif()

#################################################################################
### Additional flags
#################################################################################


option(WITH_DEBUG_INFO "Sets whether to maximize debug data collection" OFF)
message("> With debug info " ${WITH_DEBUG_INFO})
if(WITH_DEBUG_INFO)
    add_definitions(-DWITH_DEBUG_INFO)

    # Enable small memory error detector (fuse=gold fixes linker errors)
#    set(ASAN "-fsanitize=thread")
    set(ASAN "-fsanitize=address -fsanitize=undefined")
    string(APPEND CMAKE_CXX_FLAGS " -g ${ASAN} -fno-omit-frame-pointer")
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        string(APPEND CMAKE_CXX_FLAGS " -fuse-ld=gold")
    endif()
endif()

option(NO_GVC "Whether to prevent looking for the gvc library" OFF)
message("> Disable gvc: " ${NO_GVC})

option(WITH_TESTS "Whether or not to build all test targets" OFF)
message("> Building tests: " ${WITH_TESTS})

option(ESHN_WITH_DISTANCE "Whether CPPN inputs include connection length" OFF)
message("> CPPN distance: ${ESHN_WITH_DISTANCE}")
if (ESHN_WITH_DISTANCE)
    add_definitions(-DESHN_WITH_DISTANCE)
    list(APPEND KGD_DEFINITIONS -DESHN_WITH_DISTANCE)
endif()

set(ESHN_SUBSTRATE_DIMENSION "2" CACHE STRING
    "Dimension of the ANN substrate (2 or 3)")
set_property(CACHE ESHN_SUBSTRATE_DIMENSION PROPERTY STRINGS 2 3)
message("> Substrate dimension: ${ESHN_SUBSTRATE_DIMENSION}")
add_definitions(-DESHN_SUBSTRATE_DIMENSION=${ESHN_SUBSTRATE_DIMENSION})
list(APPEND KGD_DEFINITIONS -DESHN_SUBSTRATE_DIMENSION=${ESHN_SUBSTRATE_DIMENSION})


#################################################################################
### Generate configuration files
#################################################################################

set(CPPN_CONSTANTS ${PROJECT_SOURCE_DIR}/cpp/misc/constants)

## CPPN Inputs (depends on arguments)
set(CPPN_INPUTS_LIST "")
set(CPPN_INPUT_NAMES "")
foreach(i RANGE 0 1)
    string(APPEND CPPN_INPUTS_LIST "X${i} Y${i} ")
    string(APPEND CPPN_INPUT_NAMES "x_${i} y_${i} ")
    if (${ESHN_SUBSTRATE_DIMENSION} EQUAL 3)
        string(APPEND CPPN_INPUTS_LIST "Z${i} ")
        string(APPEND CPPN_INPUT_NAMES "z_${i} ")
    endif()
endforeach()
if(ESHN_WITH_DISTANCE)
    string(APPEND CPPN_INPUTS_LIST "Length ")
    string(APPEND CPPN_INPUT_NAMES "l ")
endif()
string(APPEND CPPN_INPUTS_LIST "Bias")
string(APPEND CPPN_INPUT_NAMES "b")
string(REGEX REPLACE "([^ ]+)" "\"\\1\"," CPPN_INPUT_NAMES ${CPPN_INPUT_NAMES})
string(REGEX MATCHALL "[^ ]+" tokens ${CPPN_INPUTS_LIST})
list(LENGTH tokens CPPN_INPUTS_COUNT)
message("> CPPN inputs: ${CPPN_INPUTS_LIST}")

## CPPN Outputs (constant for now)
set(CPPN_OUTPUTS_LIST "Weight LEO Bias")
set(CPPN_OUTPUT_NAMES "w l b")
string(REGEX MATCHALL "[^ ]+" tokens ${CPPN_OUTPUTS_LIST})
string(REGEX REPLACE "([^ ]+)" "\"\\1\"," CPPN_OUTPUT_NAMES ${CPPN_OUTPUT_NAMES})
list(LENGTH tokens CPPN_OUTPUTS_COUNT)

string(REPLACE " " ", " CPPN_INPUTS_LIST ${CPPN_INPUTS_LIST})
string(REPLACE " " ", " CPPN_OUTPUTS_LIST ${CPPN_OUTPUTS_LIST})
configure_file(${CPPN_CONSTANTS}_template.h ${CPPN_CONSTANTS}.h)
#message("Generated cppn constants config '${CPPN_CONSTANTS}.h'")

#################################################################################
### Install cpp library
#################################################################################

execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "if True:
        from distutils.sysconfig import get_python_lib
        print(get_python_lib())"
    OUTPUT_VARIABLE PYTHON_LIB_PATH OUTPUT_STRIP_TRAILING_WHITESPACE)
message("-- Installation folder is: '${PYTHON_LIB_PATH}' for ${PYTHON_EXECUTABLE}")

INSTALL(TARGETS ${LIB} DESTINATION ${PYTHON_LIB_PATH})
