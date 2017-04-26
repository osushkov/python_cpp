
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <string>
#include <ostream>

namespace np = boost::python::numpy;
namespace bp = boost::python;

namespace PythonUtil {

void Initialise(void);

bp::object Import(const std::string &module, const std::string &path,
                  bp::object &globals);

std::string ParseException(void);

template <typename T>
inline std::vector<T> ToStdVector(const bp::object &iterable) {
  return std::vector<T>(bp::stl_input_iterator<T>(iterable),
                        bp::stl_input_iterator<T>());
}

np::ndarray ArrayFromVector(const std::vector<float>& data);

class GIL {
public:
  GIL() { state_ = PyGILState_Ensure(); }
  ~GIL() { PyGILState_Release(state_); }

  GIL(const GIL &) = delete;
  GIL &operator=(const GIL &) = delete;

private:
  PyGILState_STATE state_;
};
}

std::ostream &operator<<(std::ostream &stream, const np::ndarray &array);
