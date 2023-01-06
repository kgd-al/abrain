#ifndef CPPN_CONSTANTS_TEMPLATE_H
#define CPPN_CONSTANTS_TEMPLATE_H

typedef unsigned int uint;

namespace kgd::abrain {

static constexpr const char* BUILD_TYPE = "${CMAKE_BUILD_TYPE}";

static constexpr const char* CXX_FLAGS = "${CMAKE_CXX_FLAGS}";

static constexpr const char* CXX_STANDARD = "${CMAKE_CXX_STANDARD}";

} // end of namespace kgd::abrain

namespace kgd::eshn::cppn {

enum class CPPN_INPUT {
    ${CPPN_INPUTS_LIST}
};

static constexpr const char* CPPN_INPUT_ENUM_NAMES [] {
    ${CPPN_INPUT_ENUM_NAMES}
};

enum class CPPN_OUTPUT {
    ${CPPN_OUTPUTS_LIST}
};

static constexpr const char* CPPN_OUTPUT_ENUM_NAMES [] {
    ${CPPN_OUTPUT_ENUM_NAMES}
};

static constexpr CPPN_OUTPUT CPPN_OUTPUT_LIST [] {
    ${CPPN_QUALIFIED_OUTPUTS}
};

static constexpr int CPPN_INPUTS =  ${CPPN_INPUTS_COUNT};
static constexpr int CPPN_OUTPUTS = ${CPPN_OUTPUTS_COUNT};

static constexpr const char* CPPN_INPUT_NAMES [] {
    ${CPPN_INPUT_NAMES}
};

static constexpr const char* CPPN_OUTPUT_NAMES [] {
    ${CPPN_OUTPUT_NAMES}
};

} // end of namespace kgd::eshn::cppn

#endif // CPPN_CONSTANTS_TEMPLATE_H
