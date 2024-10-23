/*******************************************************************************
 * This file is part of MEGA++.
 * Copyright (c) 2023 Will Roper (w.roper@sussex.ac.uk)
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * This header file contains the defintion of the threadpool class used to
 * distribute local work over local threads. This implementation uses pthreads.
 ******************************************************************************/
#ifndef THREADPOOL_H
#define THREADPOOL_H

#include <atomic>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <mutex>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

// Keys for thread specific data.
extern pthread_key_t threadpool_tid;

/**
 * @brief The ThreadPool class provides a flexible and efficient mechanism
 * for parallelizing tasks across multiple threads using pthreads.
 *
 * The class is designed to distribute local workloads over local threads,
 * offering a simplified interface for parallelizing the execution of a
 * specified function over a given array of data. The implementation ensures
 * synchronization and coordination among the worker threads.
 *
 * Functions mapped over must have the following signature:
 *   void mapFunction(void *mapData, int size, void *extraData)
 *
 * Key Features:
 * - Automatic and uniform chunking options for workload distribution.
 * - Support for additional data to be passed to the map function.
 * - Dynamic control over the number of worker threads in the pool.
 * - Thread safety and efficient handling of synchronization using
 *   condition variables and atomic operations.
 *
 * Usage:
 * - Create an instance of the ThreadPool with the desired number of threads.
 * - Use the `map` function to apply a given function to an array of data
 *   in parallel, providing options for chunking and additional data.
 * - The class ensures proper initialization, cleanup, and coordination of
 *   worker threads through its constructor and destructor.
 */
class ThreadPool {
public:
  // Constants
  static const int threadpool_auto_chunk_size = 0;
  static const int threadpool_uniform_chunk_size = -1;
  static const int threadpool_default_chunk_ratio = 7;

private:
  // Number of threads
  int numThreads;

  // Define the index of the current threadpool task.
  std::atomic<size_t> taskInd;

  // Member variables
  std::vector<std::thread> threads;
  std::mutex waitMutex, runMutex;
  std::condition_variable waitCondition, runCondition;

  // Map function and data
  std::function<void(void *, int, void *)> mapFunction;
  void *mapData;
  size_t mapDataSize;
  size_t mapDataCount;
  size_t mapDataStride;
  size_t mapDataChunk;
  void *mapExtraData;

  // Number of threads running and finished
  std::atomic<int> numThreadsRunning;
  std::atomic<int> numThreadsFinished;

  // Flag for when we are done mapping
  bool done;

  std::mutex coutMutex;

public:
  // Constructor
  ThreadPool(int numThreads);

  // Destructor
  ~ThreadPool();

  // Map function to apply a given function to an array of data in parallel
  void map(std::function<void(void *, int, void *)> mapFunction, void *mapData,
           size_t dataSize, int chunk, void *extraData = nullptr);

private:
  // Struct to store log entry information
  struct LogEntry {
    int tid;
    size_t chunkSize;
    std::function<void(void *, int, void *)> mapFunction;
    std::chrono::steady_clock::time_point tic, toc;
  };

  // Struct to store log entries for each thread
  struct MapperLog {
    std::vector<LogEntry> log;
    int count;
  };

  // Helper function to initialize threads
  void initializeThreads();

  // Worker thread function
  void workerThread(int tid);
};

#endif // THREADPOOL_H
