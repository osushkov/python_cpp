
#include "TFLearner.hpp"

#include <boost/python/numpy.hpp>
#include <iostream>
#include <vector>

using namespace std;

std::ostream &operator<<(std::ostream &stream, const np::ndarray &array) {
  stream << boost::python::extract<char const *>(boost::python::str(array));
  return stream;
}

int main(int argc, char **argv) {
  std::cout << "hello world!!" << std::endl;

  TFLearner learner;
  learner.BuildGraph();
  learner.LearnIterations(5);
  // std::cout << learner.generate() << std::endl;
  // np::initialize();

  // PyImport_AppendInittab("StrategyFramework", &initStrategyFramework);
  //
  // bp::object main = bp::import("__main__");
  // bp::object globals = main.attr("__dict__");
  // bp::object module = import("strategy", "src/strategy.py", globals);
  // bp::object Strategy = module.attr("Strategy");
  // bp::object strategy = Strategy();
  //
  // vector<float> vals{1.0f, 2.0f, 3.0f};
  //
  // strategy.attr("eval")(vals);
  // strategy.attr("eval")(vals);
  //
  // bp::object r = strategy.attr("generate")();
  // int size = bp::extract<int>(r.attr("size"));
  // std::cout << "b: " << size << std::endl;
  //
  // np::ndarray ndr = bp::extract<np::ndarray>(r);
  // std::cout << "wooo: " << bp::extract<char const *>(bp::str(ndr)) <<
  // std::endl;
  //
  // strategy.attr("store")(ndr);
  // ndr = bp::extract<np::ndarray>(strategy.attr("doubled")());
  //
  // std::cout << "wooo: " << bp::extract<char const *>(bp::str(ndr)) <<
  // std::endl;
  //
  // PyStrategyInstance pyStrat;
  // ndr = pyStrat.generate();
  // std::cout << "wooo2: " << bp::extract<char const *>(bp::str(ndr))
  //           << std::endl;
  //
  // // for (auto v : b) {
  // //   std::cout << "v: " << v << std::endl;
  // // }

  std::cout << "bye world" << std::endl;
  return 0;
}
