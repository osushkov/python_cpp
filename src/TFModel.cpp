#include "TFModel.hpp"
#include "PythonUtil.hpp"

#include <boost/python.hpp>
#include <boost/python/numeric.hpp>
#include <boost/python/numpy.hpp>
#include <boost/python/stl_iterator.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <iostream>
#include <mutex>
#include <vector>

namespace np = boost::python::numpy;
namespace bp = boost::python;

typedef vector<np::ndarray> MyList;

class ModelInstance {
public:
  ModelInstance() = default;
  virtual ~ModelInstance() = default;

  virtual void BuildGraph(void) = 0;
  virtual np::ndarray Inference(const np::ndarray &input) = 0;
  virtual void SetModelParams(MyList params) = 0;
};

class PyModelInstance final : public ModelInstance,
                              public bp::wrapper<ModelInstance> {
public:
  using ModelInstance::ModelInstance;

  void BuildGraph(void) override { get_override("BuildGraph")(); }

  np::ndarray Inference(const np::ndarray &input) {
    return get_override("Inference")(input);
  }

  void SetModelParams(MyList params) { get_override("SetModelParams")(params); }
};

BOOST_PYTHON_MODULE(ModelFramework) {
  np::initialize();
  bp::class_<MyList>("MyList").def(bp::vector_indexing_suite<MyList, true>());
  bp::class_<PyModelInstance, boost::noncopyable>("ModelInstance");
}

struct TFModel::TFModelImpl {
  bp::object model;
  std::mutex m;

  TFModelImpl() {
    std::lock_guard<std::mutex> l(m);
    try {
      PythonUtil::Initialise();
      PyImport_AppendInittab("ModelFramework", &initModelFramework);

      bp::object main = bp::import("__main__");
      bp::object globals = main.attr("__dict__");
      bp::object module =
          PythonUtil::Import("model", "src/python/model.py", globals);
      bp::object Model = module.attr("Model");
      model = Model();
    } catch (const bp::error_already_set &e) {
      std::cerr << std::endl << PythonUtil::ParseException() << std::endl;
      throw e;
    }
  }

  np::ndarray Inference(const np::ndarray &input) {
    std::lock_guard<std::mutex> l(m);
    try {
      return bp::extract<np::ndarray>(model.attr("Inference")(input));
    } catch (const bp::error_already_set &e) {
      std::cerr << std::endl << PythonUtil::ParseException() << std::endl;
      throw e;
    }
  }

  void SetModelParams(MyList params) {
    std::lock_guard<std::mutex> l(m);
    try {
      // boost::python::call<double>
      model.attr("SetModelParams")(params);
    } catch (const bp::error_already_set &e) {
      std::cerr << std::endl << PythonUtil::ParseException() << std::endl;
      throw e;
    }
  }
};

TFModel::TFModel() : impl(new TFModelImpl()) {}
TFModel::~TFModel() = default;

np::ndarray TFModel::Inference(const np::ndarray &input) {
  return impl->Inference(input);
}

void TFModel::SetModelParams(const vector<np::ndarray> &params) {
  impl->SetModelParams(params);
}
