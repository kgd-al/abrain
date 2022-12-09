cmake_minimum_required(VERSION 3.4)

################################################################################
## General settings
################################################################################

project("Pypp-NeuroEvolution")
set(PROJECT_SHORT "pyne")
message("\n####################################################################"
    "############\n## CMakeFile for ${PROJECT_NAME}")

set(CMAKE_CXX_FLAGS "-Wall -Wextra -pedantic")

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")

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
    "phenotype/ann.cpp"
    "phenotype/eshn.cpp"
)
PREPEND(CPP_SRC "src/pyne/_cpp")

set(PYBIND_SRC
    "module.cpp"
    "config.cpp"

    "genotype.cpp"

    "phenotype/cppn.cpp"
    "phenotype/ann.cpp"
)
PREPEND(PYBIND_SRC "src/pyne/_bindings")

set(LIB_CPP _cpp)
add_subdirectory(extern/pybind11)
#find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(${LIB_CPP} ${CPP_SRC} ${PYBIND_SRC})
target_compile_definitions(${LIB_CPP}
                           PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})

################################################################################
## C++ Coverage
################################################################################
if(WITH_COVERAGE)
    message("#### Searching for packages")
    find_program(LCOV lcov REQUIRED)
    find_program(GENHTML genhtml REQUIRED)
    if(NOT LCOV OR NOT GENHTML)
        message(SEND_ERROR "Could not find lcov/genhtml program(s). "
            "Either install missing components or run without coverage tests ("
            "${LCOV} & ${GENHTML})")
    endif()

    add_custom_target(coverage
        COMMAND ${LCOV} --directory . --capture --output-file coverage.info
        COMMAND ${GENHTML} --demangle-cpp -o coverage/cpp coverage.info
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
endif()

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

message("> Build type: ${CMAKE_BUILD_TYPE}")

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

option(WITH_COVERAGE "Whether to generate coverage information" OFF)
message("> With coverage ${WITH_COVERAGE}")
if(WITH_COVERAGE)
    message("  > lcov at: " ${LCOV})
    message("  > genhtml at: " ${GENHTML})
    string(APPEND CMAKE_CXX_FLAGS " -O0 -coverage")
endif()

if(NOT CMAKE_BUILD_TYPE MATCHES Debug)
    if(WITH_DEBUG_INFO)
        message(SEND_ERROR
            "Generating debug information is only available in debug mode "
            "(currently using ${CMAKE_BUILD_TYPE})")
    endif()
    if(WITH_COVERAGE)
        message(SEND_ERROR
            "Generating coverage information is only available in debug mode "
            "(currently using ${CMAKE_BUILD_TYPE})")
    endif()
endif()

option(ESHN_WITH_DISTANCE "Whether CPPN inputs include connection length" ON)
message("> CPPN distance: ${ESHN_WITH_DISTANCE}")
if (ESHN_WITH_DISTANCE)
    add_definitions(-DESHN_WITH_DISTANCE)
endif()

set(ESHN_SUBSTRATE_DIMENSION "3" CACHE STRING
    "Dimension of the ANN substrate (2 or 3)")
set_property(CACHE ESHN_SUBSTRATE_DIMENSION PROPERTY STRINGS 2 3)
message("> Substrate dimension: ${ESHN_SUBSTRATE_DIMENSION}")
add_definitions(-DESHN_SUBSTRATE_DIMENSION=${ESHN_SUBSTRATE_DIMENSION})


#################################################################################
### Generate configuration files
#################################################################################

set(CPPN_CONSTANTS ${PROJECT_SOURCE_DIR}/src/pyne/_cpp/misc/constants)

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
string(REGEX REPLACE "([^ ,]+)" "\"\\1\""
    CPPN_INPUT_ENUM_NAMES ${CPPN_INPUTS_LIST})
string(REPLACE " " ", " CPPN_OUTPUTS_LIST ${CPPN_OUTPUTS_LIST})
string(REGEX REPLACE "([^ ,]+)" "\"\\1\""
    CPPN_OUTPUT_ENUM_NAMES ${CPPN_OUTPUTS_LIST})
string(REGEX REPLACE "([^ ]+)" "CPPN_OUTPUT::\\1"    
    CPPN_QUALIFIED_OUTPUTS ${CPPN_OUTPUTS_LIST})

configure_file(${CPPN_CONSTANTS}_template.h ${CPPN_CONSTANTS}.h)
message("-+ Generated cppn constants config '${CPPN_CONSTANTS}.h'")

#################################################################################
### Install cpp library
#################################################################################

add_custom_command(
    TARGET ${LIB_CPP} POST_BUILD
    COMMAND echo "binary dir is: ${PROJECT_BINARY_DIR}"
    COMMAND echo "lib has been built as: $<TARGET_FILE:${LIB_CPP}>"
    COMMAND PYTHONPATH=. pybind11-stubgen -o ${PROJECT_BINARY_DIR}/
        --skip-signature-downgrade --no-setup-py --log-level DEBUG
        ${LIB_CPP}
    WORKING_DIRECTORY $<TARGET_FILE_DIR:${LIB_CPP}>
    COMMENT "-+ Generating python stubs from c++ bindings"
)

#install(TARGETS ${LIB_CPP} DESTINATION ${LIB_CPP})
#install(
#    DIRECTORY ${PROJECT_BINARY_DIR}/_cpp-stubs/
#    DESTINATION ${LIB_CPP}/
#    PATTERN "*.pyi"
#        PERMISSIONS OWNER_READ GROUP_READ WORLD_READ)

#if (DEV_BUILD)
#    cmake_policy(SET CMP0087 NEW)   # Allow evaluation of generator expressions
#    message("-+ Dev mod: Not installing, instead sym-linking to python")
#    INSTALL(CODE
#        "execute_process( \
#            COMMAND ln -svf \
#            $<TARGET_FILE:${LIB_CPP}> ${PROJECT_SOURCE_DIR}/python \
#        )"
#    )
#    INSTALL(CODE
#        "execute_process( \
#            COMMAND ln -svf ${PROJECT_BINARY_DIR}/_cpp-stubs/ \
#                            ${PROJECT_SOURCE_DIR}/python/${LIB_CPP} \
#        )"
#    )

#else()
#    execute_process(
#        COMMAND ${PYTHON_EXECUTABLE} -c "if True:
#            from distutils.sysconfig import get_python_lib
#            print(get_python_lib())"
#        OUTPUT_VARIABLE PYTHON_LIB_PATH OUTPUT_STRIP_TRAILING_WHITESPACE)
#    string(APPEND PYTHON_LIB_PATH "/${PROJECT_SHORT}/")
#    message("-+ Installation folder is: '${PYTHON_LIB_PATH}' for
#        ${PYTHON_EXECUTABLE}")

#    INSTALL(TARGETS ${LIB_CPP} DESTINATION ${PYTHON_LIB_PATH})
#    INSTALL(
#        DIRECTORY ${PROJECT_BINARY_DIR}/_cpp-stubs/
#        DESTINATION ${PYTHON_LIB_PATH}/${LIB_CPP}/
#        PATTERN "*.pyi"
#            PERMISSIONS OWNER_READ GROUP_READ WORLD_READ)
#endif()


#################################################################################
### Ugly hack to get additional files
#################################################################################

file(GLOB FUNCS
    ${CMAKE_SOURCE_DIR}/ps/*.ps
    ${CMAKE_SOURCE_DIR}/ps/*.png)
add_custom_command(
    OUTPUT ${CMAKE_SOURCE_DIR}/ps/id.png
    COMMAND ${CMAKE_SOURCE_DIR}/ps/plotter.sh
    COMMENT "-+ Generating ps/png file for cppn rendering"
    BYPRODUCTS ${FUNCS})
