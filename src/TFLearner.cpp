#include "TFLearner.hpp"
#include "PythonUtil.hpp"

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <iostream>

namespace np = boost::python::numpy;
namespace bp = boost::python;

class LearnerInstance {
public:
  LearnerInstance() = default;
  virtual ~LearnerInstance() = default;

  virtual void LearnIterations(unsigned iters) = 0;
  virtual bp::object GetModelParams(void) = 0;
};

class PyLearnerInstance final : public LearnerInstance,
                                public bp::wrapper<LearnerInstance> {
public:
  using LearnerInstance::LearnerInstance;

  void LearnIterations(unsigned iters) override {
    get_override("LearnIterations")(iters);
  }

  bp::object GetModelParams(void) override {
    return get_override("GetModelParams")();
  }
};

BOOST_PYTHON_MODULE(LearnerFramework) {
  np::initialize();
  bp::class_<PyLearnerInstance, boost::noncopyable>("LearnerInstance");
}

struct TFLearner::TFLearnerImpl {
  bp::object learner;

  TFLearnerImpl() {
    try {
      PyImport_AppendInittab("LearnerFramework", &initLearnerFramework);

      bp::object main = bp::import("__main__");
      bp::object globals = main.attr("__dict__");
      bp::object learnerModule =
          PythonUtil::Import("learner", "src/python/learner.py", globals);

      bp::object Learner = learnerModule.attr("Learner");
      learner = Learner();
    } catch (const bp::error_already_set &e) {
      std::cerr << std::endl << PythonUtil::ParseException() << std::endl;
      throw e;
    }
  }

  void LearnIterations(unsigned iters) {
    try {
      learner.attr("LearnIterations")(iters);
    } catch (const bp::error_already_set &e) {
      std::cerr << std::endl << PythonUtil::ParseException() << std::endl;
      throw e;
    }
  }

  bp::object GetModelParams(void) {
    try {
      return learner.attr("GetModelParams")();
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

bp::object TFLearner::GetModelParams(void) {
  return impl->GetModelParams();
}
