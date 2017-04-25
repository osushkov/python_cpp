#include "TFModel.hpp"
#include "PythonUtil.hpp"

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <boost/python/stl_iterator.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <iostream>
#include <mutex>
#include <vector>

struct TFModel::TFModelImpl {
  TFModelImpl() {}

  np::ndarray Inference(const np::ndarray &input) { return input; }

  void SetModelParams(const vector<np::ndarray> &params) {}
};

TFModel::TFModel() : impl(new TFModelImpl()) {}
TFModel::~TFModel() = default;

np::ndarray TFModel::Inference(const np::ndarray &input) {
  return impl->Inference(input);
}

void TFModel::SetModelParams(const vector<np::ndarray> &params) {
  impl->SetModelParams(params);
}
