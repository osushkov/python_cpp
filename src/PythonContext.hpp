#pragma once

#include <boost/python.hpp>
#include <mutex>
#include <iostream>
#include <chrono>
#include <thread>

struct PythonMainContext {
  PythonMainContext()
      : threadState(PyThreadState_Get()),
        interpreterState(threadState->interp) {
    PyEval_ReleaseLock();
  }

  ~PythonMainContext() {
    std::cout << "PythonMainContext DESTRUCTOR!!" << std::endl;
    PyEval_RestoreThread(threadState);
    Py_Finalize();
    std::cout << "PythonMainContext FINISHED DESTRUCTOR!!" << std::endl;
  }

  PyThreadState *threadState;
  PyInterpreterState *interpreterState;

  std::mutex m;
};

struct PythonThreadContext {
  PythonThreadContext(PythonMainContext &mainCtx) : mainCtx(mainCtx) {
    PyEval_AcquireLock();
    threadState = PyThreadState_New(mainCtx.interpreterState);
    PyEval_ReleaseLock();
  }

  ~PythonThreadContext() {
    std::cout << "PythonThreadContext DESTRUCTOR!!" << std::endl;
    PyEval_AcquireLock();
    PyThreadState_Swap(mainCtx.threadState);
    PyThreadState_Clear(threadState);
    PyThreadState_Delete(threadState);
    PyEval_ReleaseLock();
    std::cout << "PythonThreadContext FINISHED DESTRUCTOR!!" << std::endl;
  }

  PythonMainContext &mainCtx;
  PyThreadState *threadState;
};

struct PythonContextLock {
  PythonContextLock(PythonThreadContext &ctx) {
    std::this_thread::sleep_for(std::chrono::nanoseconds(1));
    lock = std::unique_lock<std::mutex>(ctx.mainCtx.m);

    PyEval_AcquireLock();
    prevThreadState = PyThreadState_Swap(ctx.threadState);
  }

  ~PythonContextLock() {
    PyThreadState_Swap(prevThreadState);
    PyEval_ReleaseLock();
  }

  PyThreadState *prevThreadState;
  std::unique_lock<std::mutex> lock;
};
