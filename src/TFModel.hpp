#pragma once

#include "util/Common.hpp"
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

namespace bp = boost::python;
namespace np = boost::python::numpy;

class TFModel {
public:
  TFModel(unsigned batchSize);
  virtual ~TFModel();

  np::ndarray Inference(const np::ndarray &input);
  void SetModelParams(const bp::object &params);

private:
  struct TFModelImpl;
  uptr<TFModelImpl> impl;
};
