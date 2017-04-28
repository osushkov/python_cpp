#pragma once

#include "util/Common.hpp"
#include <boost/python.hpp>

namespace bp = boost::python;

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
