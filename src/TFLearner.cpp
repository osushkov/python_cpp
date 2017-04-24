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

  virtual void BuildGraph(void) = 0;
  virtual void LearnIterations(unsigned iters) = 0;
  // virtual np::ndarray doubled(void) = 0;
};

class PyLearnerInstance final : public LearnerInstance,
                                public bp::wrapper<LearnerInstance> {
public:
  using LearnerInstance::LearnerInstance;

  void BuildGraph(void) override { get_override("BuildGraph")(); }

  void LearnIterations(unsigned iters) override {
    get_override("LearnIterations")(iters);
  }

  // np::ndarray doubled(void) override { return get_override("doubled")(); }
};

BOOST_PYTHON_MODULE(LearnerFramework) {
  np::initialize();
  bp::class_<PyLearnerInstance, boost::noncopyable>("LearnerInstance");
}

struct TFLearner::TFLearnerImpl {
  bp::object learner;

  TFLearnerImpl() {
    try {
      PythonUtil::Initialise();
      PyImport_AppendInittab("LearnerFramework", &initLearnerFramework);

      bp::object main = bp::import("__main__");
      bp::object globals = main.attr("__dict__");
      bp::object module =
          PythonUtil::Import("strategy", "src/python/learner.py", globals);
      bp::object Learner = module.attr("Learner");
      learner = Learner();
    } catch (const bp::error_already_set &e) {
      std::cerr << std::endl << PythonUtil::ParseException() << std::endl;
      throw e;
    }
  }

  void BuildGraph(void) {
    try {
      learner.attr("BuildGraph")();
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

  // np::ndarray doubled(void) {
  //   return bp::extract<np::ndarray>(strategy.attr("doubled")());
  // }
};

TFLearner::TFLearner() : impl(new TFLearnerImpl()) {}
TFLearner::~TFLearner() = default;

void TFLearner::BuildGraph(void) { impl->BuildGraph(); }

void TFLearner::LearnIterations(unsigned iters) {
  impl->LearnIterations(iters);
}
