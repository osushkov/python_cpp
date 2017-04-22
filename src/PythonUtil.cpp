
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
