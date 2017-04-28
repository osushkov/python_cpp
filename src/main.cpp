
#include "PythonContext.hpp"
#include "PythonUtil.hpp"
#include "TFLearner.hpp"
#include "TFModel.hpp"

#include <atomic>
#include <iostream>
#include <thread>

using namespace std;

static constexpr unsigned TOTAL_ITERS = 1000;

int main(int argc, char **argv) {
  PythonUtil::Initialise();

  TFLearner learner;
  TFModel model(5);

  atomic<unsigned> learnIters(0);
  thread learnThread([&learner, &learnIters] {
    PythonThreadContext threadCtx(PythonUtil::GlobalContext());
    for (unsigned i = 0; i < TOTAL_ITERS; i++) {
      PythonContextLock pl(threadCtx);
      learner.LearnIterations(10);
      learnIters++;
    }
  });

  thread evalThread([&learner, &model, &learnIters] {
    PythonThreadContext threadCtx(PythonUtil::GlobalContext());
    vector<float> woo{1.0f, 5.0f, 6.0f, 3.2f, 10.0f};

    unsigned iters = 0;
    while ((iters = learnIters.load()) < TOTAL_ITERS) {
      PythonContextLock pl(threadCtx);

      model.SetModelParams(learner.GetModelParams());
      std::cout << iters << ": "
                << model.Inference(PythonUtil::ArrayFromVector(woo))
                << std::endl;
    }
  });



  evalThread.join();
  learnThread.join();

  std::cout << "bye world" << std::endl;
  return 0;
}
