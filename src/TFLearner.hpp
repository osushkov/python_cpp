#pragma once

#include "util/Common.hpp"
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <vector>

namespace bp = boost::python;
namespace np = boost::python::numpy;

class TFLearner {
public:
  TFLearner();
  virtual ~TFLearner();

  // noncopyable
  TFLearner(const TFLearner &other) = delete;
  TFLearner &operator=(TFLearner &other) = delete;

  void LearnIterations(unsigned iters);
  bp::object GetModelParams(void);

private:
  struct TFLearnerImpl;
  uptr<TFLearnerImpl> impl;
};
