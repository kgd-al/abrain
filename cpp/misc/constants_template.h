#ifndef CPPN_CONSTANTS_TEMPLATE_H
#define CPPN_CONSTANTS_TEMPLATE_H

namespace kgd::eshn::cppn {

enum class CPPN_INPUT {
    ${CPPN_INPUTS_LIST}
};

enum class CPPN_OUTPUT {
    ${CPPN_OUTPUTS_LIST}
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
