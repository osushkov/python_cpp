
#include "PythonUtil.hpp"
#include <mutex>

std::once_flag initialiseFlag;

void PythonUtil::Initialise(void) {
  std::call_once(initialiseFlag, []() { Py_Initialize(); });
}

bp::object PythonUtil::Import(const std::string &module,
                              const std::string &path, bp::object &globals) {
  bp::dict locals;
  locals["module_name"] = module;
  locals["path"] = path;

  bp::exec("import imp\n"
           "new_module = imp.load_module(module_name, open(path), path, ('py', "
           "'U', imp.PY_SOURCE))\n",
           globals, locals);
  return locals["new_module"];
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
    ret += returned.check() ? (": " + returned()) : std::string(": Unparseable Python error: ");
  }

  if (traceback != nullptr) {
    bp::handle<> hTb(traceback);
    bp::object tb(bp::import("traceback"));
    bp::object fmtTb(tb.attr("format_tb"));
    bp::object tbList(fmtTb(hTb));
    bp::object tbStr(bp::str("\n").join(tbList));
    bp::extract<std::string> returned(tbStr);
    ret += returned.check() ? (": " + returned()) : std::string(": Unparseable Python traceback");
  }

  return ret;
}

std::ostream &operator<<(std::ostream &stream, const np::ndarray &array) {
  stream << bp::extract<char const *>(bp::str(array));
  return stream;
}
