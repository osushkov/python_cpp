
#include "PythonContext.hpp"
#include "PythonUtil.hpp"
#include "TFLearner.hpp"
#include "TFModel.hpp"

#include <atomic>
#include <boost/python/numpy.hpp>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <vector>

using namespace std;

int main(int argc, char **argv) {
  PythonUtil::Initialise();

  PythonContext ctx;

  std::cout << "hello world!!" << std::endl;

  vector<float> woo{1.0f, 5.0f, 6.0f, 3.2f, 10.0f};
  std::cout << PythonUtil::ArrayFromVector(woo) << std::endl;

  TFLearner learner;
  TFModel model(5);

  atomic<int> learnIters(0);
  thread learnThread([&learner, &learnIters, &ctx] {
    for (unsigned i = 0; i < 1000; i++) {
      // std::cout << "li: " << i << std::endl;

      PythonContextLock pl(ctx);
      learner.LearnIterations(10);
      learnIters++;
    }
  });

  int iters = 0;
  while ((iters = learnIters.load()) < 1000) {
    PythonContextLock pl(ctx);
    // std::cout << learner.GetModelParams()[0] << std::endl;

    // model.SetModelParams(learner.GetModelParams());
    std::cout << iters << ": "
              << model.Inference(PythonUtil::ArrayFromVector(woo)) << std::endl;
    std::cout << iters << std::endl;
  }

  learnThread.join();

  // for (unsigned i = 0; i < 1000; i++) {
  // learner.LearnIterations(10);
  // model.SetModelParams(learner.GetModelParams());
  // std::cout << model.Inference(PythonUtil::ArrayFromVector(woo)) <<
  // std::endl; getchar();
  // }

  std::cout << "bye world" << std::endl;
  return 0;
}
