#include <iostream>
#include <vector>
#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

using namespace std;

typedef vector<float> PyVec;

class StrategyInstance {
public:
  StrategyInstance() = default;
  virtual ~StrategyInstance() = default;

  virtual void eval(const PyVec &vals) = 0;
};

namespace bp = boost::python;

class PyStrategyInstance final : public StrategyInstance,
                                 public bp::wrapper<StrategyInstance> {
  using StrategyInstance::StrategyInstance;

  void eval(const PyVec &vals) override { get_override("eval")(vals); }
};

BOOST_PYTHON_MODULE(StrategyFramework) {
  bp::class_<PyVec>("PyVec")
        .def(bp::vector_indexing_suite<PyVec>() );

  bp::class_<PyStrategyInstance, boost::noncopyable>("StrategyInstance");
}

bp::object import(const std::string &module, const std::string &path,
                  bp::object &globals) {
  bp::dict locals;
  locals["module_name"] = module;
  locals["path"] = path;

  bp::exec("import imp\n"
           "new_module = imp.load_module(module_name, open(path), path, ('py', "
           "'U', imp.PY_SOURCE))\n",
           globals, locals);
  return locals["new_module"];
}

int main(int argc, char **argv) {
  std::cout << "hello world!!" << std::endl;

  Py_Initialize();
  PyImport_AppendInittab("StrategyFramework", &initStrategyFramework);

  bp::object main     = bp::import("__main__");
  bp::object globals  = main.attr("__dict__");
  bp::object module   = import("strategy", "src/strategy.py", globals);
  bp::object Strategy = module.attr("Strategy");
  bp::object strategy = Strategy();

  vector<float> vals{1.0f, 2.0f, 3.0f};

  strategy.attr("eval")(vals);
  strategy.attr("eval")(vals);

  std::cout << "bye world" << std::endl;
  return 0;
}
