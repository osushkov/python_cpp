#include "TFLearner.hpp"
#include "PythonUtil.hpp"

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <boost/python/stl_iterator.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <iostream>
#include <mutex>
#include <vector>

namespace np = boost::python::numpy;
namespace bp = boost::python;

class LearnerInstance {
public:
  LearnerInstance() = default;
  virtual ~LearnerInstance() = default;

  virtual void LearnIterations(unsigned iters) = 0;
  virtual std::vector<np::ndarray> GetModelParams(void) = 0;
};

class PyLearnerInstance final : public LearnerInstance,
                                public bp::wrapper<LearnerInstance> {
public:
  using LearnerInstance::LearnerInstance;

  void LearnIterations(unsigned iters) override {
    get_override("LearnIterations")(iters);
  }

  std::vector<np::ndarray> GetModelParams(void) override {
    return get_override("GetModelParams")();
  }
};

BOOST_PYTHON_MODULE(LearnerFramework) {
  np::initialize();

  bp::class_<PyLearnerInstance, boost::noncopyable>("LearnerInstance");
}

struct TFLearner::TFLearnerImpl {
  bp::object learner;
  std::mutex m;

  TFLearnerImpl() {
    std::lock_guard<std::mutex> l(m);
    try {
      PythonUtil::Initialise();
      PyImport_AppendInittab("LearnerFramework", &initLearnerFramework);

      bp::object main = bp::import("__main__");
      bp::object globals = main.attr("__dict__");
      bp::object module =
          PythonUtil::Import("learner", "src/python/learner.py", globals);
      bp::object Learner = module.attr("Learner");
      learner = Learner();
    } catch (const bp::error_already_set &e) {
      std::cerr << std::endl << PythonUtil::ParseException() << std::endl;
      throw e;
    }
  }

  void LearnIterations(unsigned iters) {
    std::lock_guard<std::mutex> l(m);

    try {
      learner.attr("LearnIterations")(iters);
    } catch (const bp::error_already_set &e) {
      std::cerr << std::endl << PythonUtil::ParseException() << std::endl;
      throw e;
    }
  }

  std::vector<np::ndarray> GetModelParams(void) {
    std::lock_guard<std::mutex> l(m);

    try {
      return PythonUtil::ToStdVector<np::ndarray>(
          learner.attr("GetModelParams")());
    } catch (const bp::error_already_set &e) {
      std::cerr << std::endl << PythonUtil::ParseException() << std::endl;
      throw e;
    }
  }
};

TFLearner::TFLearner() : impl(new TFLearnerImpl()) {}
TFLearner::~TFLearner() = default;

void TFLearner::LearnIterations(unsigned iters) {
  impl->LearnIterations(iters);
}

std::vector<np::ndarray> TFLearner::GetModelParams(void) {
  return impl->GetModelParams();
}
