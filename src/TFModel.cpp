#include "TFModel.hpp"
#include "PythonUtil.hpp"

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <cassert>
#include <iostream>

namespace np = boost::python::numpy;
namespace bp = boost::python;

class ModelInstance {
public:
  virtual ~ModelInstance() = default;

  virtual np::ndarray Inference(const np::ndarray &input) = 0;
  virtual void SetModelParams(bp::object params) = 0;
};

class PyModelInstance final : public ModelInstance,
                              public bp::wrapper<ModelInstance> {
public:
  using ModelInstance::ModelInstance;

  np::ndarray Inference(const np::ndarray &input) {
    return get_override("Inference")(input);
  }

  void SetModelParams(bp::object params) {
    get_override("SetModelParams")(params);
  }
};

BOOST_PYTHON_MODULE(ModelFramework) {
  np::initialize();
  bp::class_<PyModelInstance, boost::noncopyable>("ModelInstance");
}

struct TFModel::TFModelImpl {
  bp::object model;

  TFModelImpl(unsigned batchSize) {
    assert(batchSize >= 1);

    try {
      PyImport_AppendInittab("ModelFramework", &initModelFramework);

      bp::object main = bp::import("__main__");
      bp::object globals = main.attr("__dict__");
      bp::object modelModule = PythonUtil::Import("model", "src/python/model.py", globals);

      bp::object Model = modelModule.attr("Model");
      model = Model(batchSize);
    } catch (const bp::error_already_set &e) {
      std::cerr << std::endl << PythonUtil::ParseException() << std::endl;
      throw e;
    }
  }

  np::ndarray Inference(const np::ndarray &input) {
    try {
      return bp::extract<np::ndarray>(model.attr("Inference")(input));
    } catch (const bp::error_already_set &e) {
      std::cerr << std::endl << PythonUtil::ParseException() << std::endl;
      throw e;
    }
  }

  void SetModelParams(const bp::object &params) {
    try {
      model.attr("SetModelParams")(params);
    } catch (const bp::error_already_set &e) {
      std::cerr << std::endl << PythonUtil::ParseException() << std::endl;
      throw e;
    }
  }
};

TFModel::TFModel(unsigned batchSize) : impl(new TFModelImpl(batchSize)) {}
TFModel::~TFModel() = default;

np::ndarray TFModel::Inference(const np::ndarray &input) {
  return impl->Inference(input);
}

void TFModel::SetModelParams(const bp::object &params) {
  impl->SetModelParams(params);
}
