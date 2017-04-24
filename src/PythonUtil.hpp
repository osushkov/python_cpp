
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <string>

namespace np = boost::python::numpy;
namespace bp = boost::python;

namespace PythonUtil {

void Initialise(void);

bp::object Import(const std::string &module, const std::string &path,
                  bp::object &globals);

std::string ParseException(void);
}
