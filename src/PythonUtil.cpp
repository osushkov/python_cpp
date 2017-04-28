
#include "PythonUtil.hpp"
#include "util/Common.hpp"
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <boost/python/stl_iterator.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <mutex>
#include <vector>

using namespace std;

static std::once_flag initialiseFlag;
static uptr<PythonMainContext> globalContext;

PythonMainContext& PythonUtil::GlobalContext(void) {
  return *globalContext;
}

void PythonUtil::Initialise(void) {
  std::call_once(initialiseFlag, []() {
    Py_Initialize();
    PyEval_InitThreads();
    np::initialize();

    globalContext = make_unique<PythonMainContext>();
  });
}

bp::object PythonUtil::Import(const std::string &module,
                              const std::string &path, bp::object &globals) {
  try {
    bp::dict locals;
    locals["module_name"] = module;
    locals["path"] = path;

    bp::exec(
        "import imp\n"
        "import sys\n"
        "sys.argv = [module_name]\n"
        "new_module = imp.load_module(module_name, open(path), path, ('py', "
        "'U', imp.PY_SOURCE))\n",
        globals, locals);
    return locals["new_module"];
  } catch (const bp::error_already_set &e) {
    std::cerr << std::endl << PythonUtil::ParseException() << std::endl;
    throw e;
  }
}

std::string PythonUtil::ParseException(void) {
  PyObject *type = nullptr, *value = nullptr, *traceback = nullptr;
  PyErr_Fetch(&type, &value, &traceback);
  std::string ret("Unfetchable Python error");

  if (type != nullptr) {
    bp::handle<> hType(type);
    bp::str typeStr(hType);
    bp::extract<std::string> eTypeStr(typeStr);
    ret = eTypeStr.check() ? eTypeStr() : "Unknown exception type";
  }

  if (value != nullptr) {
    bp::handle<> hVal(value);
    bp::str a(hVal);
    bp::extract<std::string> returned(a);
    ret += returned.check() ? (": " + returned())
                            : std::string(": Unparseable Python error: ");
  }

  if (traceback != nullptr) {
    bp::handle<> hTb(traceback);
    bp::object tb(bp::import("traceback"));
    bp::object fmtTb(tb.attr("format_tb"));
    bp::object tbList(fmtTb(hTb));
    bp::object tbStr(bp::str("\n").join(tbList));
    bp::extract<std::string> returned(tbStr);
    ret += returned.check() ? (": " + returned())
                            : std::string(": Unparseable Python traceback");
  }

  return ret;
}

np::ndarray PythonUtil::ArrayFromVector(const std::vector<float> &data) {
  bp::tuple shape = bp::make_tuple(data.size());
  bp::tuple stride = bp::make_tuple(sizeof(float));

  return np::from_data(data.data(), np::dtype::get_builtin<float>(), shape,
                       stride, bp::object());
}

std::ostream &operator<<(std::ostream &stream, const np::ndarray &array) {
  stream << bp::extract<char const *>(bp::str(array));
  return stream;
}
