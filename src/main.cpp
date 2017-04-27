
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
#include <chrono>

using namespace std;

static constexpr unsigned TOTAL_ITERS = 1000;

int main(int argc, char **argv) {
  PythonUtil::Initialise();
  PySys_SetArgv(argc, argv);
  
  PythonMainContext mainCtx;
  PythonThreadContext thread0Ctx(mainCtx);
  PythonThreadContext thread1Ctx(mainCtx);

  TFLearner learner;
  TFModel model(5);

  atomic<int> learnIters(0);
  thread learnThread([&learner, &learnIters, &thread0Ctx] {
    for (unsigned i = 0; i < TOTAL_ITERS; i++) {
      PythonContextLock pl(thread0Ctx);
      learner.LearnIterations(10);
      learnIters++;
    }
  });

  thread evalThread([&learner, &model, &learnIters, &thread1Ctx] {
    vector<float> woo{1.0f, 5.0f, 6.0f, 3.2f, 10.0f};

    int iters = 0;
    while ((iters = learnIters.load()) < TOTAL_ITERS) {
      PythonContextLock pl(thread1Ctx);

      auto p = learner.GetModelParams();
      model.SetModelParams(p);
      std::cout << iters << ": "
                << model.Inference(PythonUtil::ArrayFromVector(woo))
                << std::endl;
    }
  });



  evalThread.join();
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
