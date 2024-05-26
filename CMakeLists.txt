cmake_minimum_required(VERSION 3.5)

################################################################################
## General settings
################################################################################

project("abrain")
message("\n####################################################################"
    "############\n## CMakeFile for ${PROJECT_NAME} ${VERSION_INFO}")

# set(CMAKE_CXX_FLAGS "")
if(LINUX)
  string(APPEND CMAKE_CXX_FLAGS " -Wall -Wextra -pedantic")
endif()
if(MSVC)
  string(APPEND CMAKE_CXX_FLAGS " /EHsc")
endif()

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-fno-elide-constructors -fno-default-inline")

################################################################################
## Dependencies
################################################################################

if (Python3_EXECUTABLE)
    get_filename_component(PYTHON_BIN ${Python3_EXECUTABLE} DIRECTORY)
    set(ENV{PATH} "$ENV{PATH}:${PYTHON_BIN}")
    message("Provided python: ${Python3_EXECUTABLE} (added to path)")
endif()
#
#find_package (Python3 COMPONENTS Interpreter Development)
#message("Using python* ${Python_EXECUTABLE} ${PYTHON_EXECUTABLE}")
#message("Using python3 ${Python3_EXECUTABLE}")

find_package(pybind11 REQUIRED)

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

#    "misc/constants.h"
    "misc/point.hpp"

    "genotype.h"
    "innovations.cpp"
    "innovations.h"

    "phenotype/cppn.cpp"
    "phenotype/cppn.h"
    "phenotype/ann.cpp"
    "phenotype/eshn.cpp"
)
PREPEND(CPP_SRC "src/abrain/_cpp")

set(PYBIND_SRC
    "module.cpp"
    "config.cpp"

    "genotype.cpp"
    "innovations.cpp"

    "phenotype/cppn.cpp"
    "phenotype/ann.cpp"

    "utils.hpp"
)
PREPEND(PYBIND_SRC "src/abrain/_bindings")

set(LIB_CPP _cpp)
pybind11_add_module(${LIB_CPP} ${CPP_SRC} ${PYBIND_SRC})
target_compile_definitions(${LIB_CPP}
                           PRIVATE VERSION_INFO=${VERSION_INFO})

################################################################################
## C++ Coverage
################################################################################
message("> With coverage ${WITH_COVERAGE}")
if(WITH_COVERAGE)
    message("#### Searching for packages")
    find_program(LCOV lcov)
    find_program(GENHTML genhtml)
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

#################################################################################
### Additional flags
#################################################################################

if (DEV_BUILD)
    set(DEV_SUFFIX " (dev)")
endif()

message("> Build type: ${CMAKE_BUILD_TYPE}${DEV_SUFFIX}")
message(">    Version: ${VERSION_INFO}")

message("> On n cores: ${CMAKE_BUILD_PARALLEL_LEVEL}")

option(MAKE_STUBS "Sets whether to generate pyi files from module" ON)
message("> Generate stubs " ${MAKE_STUBS})

option(AUTO_INSTALL "Sets whether to copy compiled library to source folder during build" OFF)
message("> Auto install " ${AUTO_INSTALL})

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
#     if(WITH_DEBUG_INFO)
#         message(SEND_ERROR
#             "Generating debug information is only available in debug mode "
#             "(currently using ${CMAKE_BUILD_TYPE})")
#     endif()
    if(WITH_COVERAGE)
        message(SEND_ERROR
            "Generating coverage information is only available in debug mode "
            "(currently using ${CMAKE_BUILD_TYPE})")
    endif()
endif()

message("> Compile with: ${CMAKE_CXX_FLAGS}")

#################################################################################
### Install cpp library
#################################################################################

if (MAKE_STUBS)
    set(STUBS_DIR ${PROJECT_SOURCE_DIR}/src/${PROJECT_NAME}/${LIB_CPP})
    set(PATH_SEP ":")
    if (WIN32)
      set(PATH_SEP ";")
    endif()
    find_program(STUBGEN pybind11-stubgen REQUIRED)
    message("Generating stubs with ${STUBGEN}")
    add_custom_command(
        TARGET ${LIB_CPP} POST_BUILD
        COMMAND echo "binary dir is: ${PROJECT_BINARY_DIR}"
        COMMAND echo "lib has been built as: $<TARGET_FILE:${LIB_CPP}>"
        COMMAND echo "CMAKE_PREFIX_PATH: '${CMAKE_PREFIX_PATH}'"
        COMMAND echo "             PATH: '$ENV{PATH}'"
        COMMAND echo "  BUILD_TIME_PATH: '${BUILD_TIME_PATH}'"
        COMMAND echo "      PYTHON_PATH: '$ENV{PYTHONPATH}'"
        COMMAND pwd
        COMMAND ls -d $<TARGET_FILE_DIR:${LIB_CPP}>
        COMMAND ls $<TARGET_FILE_DIR:${LIB_CPP}>
        COMMAND which ${STUBGEN}
        COMMAND ${CMAKE_COMMAND} -E
            env "PATH=${BUILD_TIME_PATH}"
            env "PYTHONPATH=.${PATH_SEP}$ENV{PYTHONPATH}"
            ${STUBGEN} -o ${PROJECT_BINARY_DIR}/
                #--skip-signature-downgrade --no-setup-py --log-level DEBUG
                --exit-code
                ${LIB_CPP}
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${PROJECT_BINARY_DIR}/${LIB_CPP}/ ${STUBS_DIR}
        WORKING_DIRECTORY $<TARGET_FILE_DIR:${LIB_CPP}>
        COMMENT "-+ Generating python stubs from c++ bindings"
    )
endif()

if (AUTO_INSTALL)
    set(LIB_DEST "${PROJECT_SOURCE_DIR}/src/abrain/")
    add_custom_command(
        TARGET ${LIB_CPP} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${LIB_CPP}> ${LIB_DEST}
            COMMAND ${CMAKE_COMMAND} -E echo "Copied $<TARGET_FILE:${LIB_CPP}> to ${LIB_DEST}"
    )
    message(">> Library ${LIB_CPP} will be automatically copied to ${LIB_DEST}")
endif()

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
