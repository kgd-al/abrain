#ifndef CPPN_CONSTANTS_TEMPLATE_H
#define CPPN_CONSTANTS_TEMPLATE_H

namespace kgd::pyne {

static constexpr const char* BUILD_TYPE = "Debug";

static constexpr const char* CXX_FLAGS = "-Wall -Wextra -pedantic -O0 -coverage";

static constexpr const char* CXX_STANDARD = "17";

} // end of namespace kgd::pyne

namespace kgd::eshn::cppn {

enum class CPPN_INPUT {
    X0, Y0, Z0, X1, Y1, Z1, Length, Bias
};

static constexpr const char* CPPN_INPUT_ENUM_NAMES [] {
    "X0", "Y0", "Z0", "X1", "Y1", "Z1", "Length", "Bias"
};

enum class CPPN_OUTPUT {
    Weight, LEO, Bias
};

static constexpr const char* CPPN_OUTPUT_ENUM_NAMES [] {
    "Weight", "LEO", "Bias"
};

static constexpr CPPN_OUTPUT CPPN_OUTPUT_LIST [] {
    CPPN_OUTPUT::Weight, CPPN_OUTPUT::LEO, CPPN_OUTPUT::Bias
};

static constexpr int CPPN_INPUTS =  8;
static constexpr int CPPN_OUTPUTS = 3;

static constexpr const char* CPPN_INPUT_NAMES [] {
    "x_0", "y_0", "z_0", "x_1", "y_1", "z_1", "l", "b",
};

static constexpr const char* CPPN_OUTPUT_NAMES [] {
    "w", "l", "b",
};

} // end of namespace kgd::eshn::cppn

#endif // CPPN_CONSTANTS_TEMPLATE_H
