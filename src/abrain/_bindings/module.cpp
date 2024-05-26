#include "../_cpp/phenotype/ann.h"
#include "pybind11/pybind11.h"
namespace py = pybind11;

namespace kgd::eshn::pybind {

void init_genotype (py::module_ &m);
void init_innovations (py::module_ &m);

void init_generic_cppn_phenotype(py::module_ &m);

template <unsigned int DI> void init_point_type (py::module_ &m, const char *name);
template <typename CPPN> void init_eshn_cppn_phenotype (
        py::module_ &m, const char *name, const char *pname);

template <typename ANN> void init_ann_phenotype (py::module_ &m, const char *name);

void init_config (py::module_ &m);

PYBIND11_MODULE(_cpp, main) {
  main.doc() = "Docstring main module";

  auto config = main.def_submodule(
          "config",
          "Docstring for config submodule");
  init_config(config);

  auto genotype = main.def_submodule(
        "genotype",
        "Docstring for genotype submodule");
  init_genotype(genotype);
  init_innovations(genotype);

  auto phenotype = main.def_submodule(
        "phenotype",
        "Docstring for phenotype submodule");
  init_point_type<2>(phenotype, "Point2D");
  init_point_type<3>(phenotype, "Point3D");
  init_generic_cppn_phenotype(phenotype);
  init_eshn_cppn_phenotype<phenotype::CPPN2D>(phenotype, "CPPN2D", "Point2D");
  init_eshn_cppn_phenotype<phenotype::CPPN3D>(phenotype, "CPPN3D", "Point3D");
  init_ann_phenotype<phenotype::ANN2D>(phenotype, "ANN2D");
  init_ann_phenotype<phenotype::ANN3D>(phenotype, "ANN3D");

  auto misc = main.def_submodule(
        "misc",
        "Docstring for phenotype submodule");

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)
#ifdef VERSION_INFO
    main.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    main.attr("__version__") = "dev";
#endif
}

} // kgd::eshn::pybind
