#pragma once

#include "util/Common.hpp"
#include <boost/python/numpy.hpp>

namespace np = boost::python::numpy;

class TFLearner {
public:
  TFLearner();
  virtual ~TFLearner();

  // noncopyable
  TFLearner(const TFLearner &other) = delete;
  TFLearner &operator=(TFLearner &other) = delete;

  void BuildGraph(void);
  void LearnIterations(unsigned iters);
  // np::ndarray doubled(void);

private:
  struct TFLearnerImpl;
  uptr<TFLearnerImpl> impl;
};
