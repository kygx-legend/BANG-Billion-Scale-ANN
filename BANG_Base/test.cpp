/* Copyright 2024 Indian Institute Of Technology Hyderbad, India. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
	http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
// Authors: Karthik V., Saim Khan, Somesh Singh
//
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <cstring>
#include <omp.h>
#include <sys/stat.h>
#include <cmath>
#include <vector>
#include <set>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <assert.h>
#include "bang.h"
#include <chrono>
#include "concurrent_queue.h"
#include <thread>
#include <random>
#include <algorithm>
#include <atomic>

using namespace std;

// Type aliases
using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;

// Query context holding shared state
struct QueryContext {
    std::vector<TimePoint> start_times;
    std::vector<TimePoint> finish_times;
    moodycamel::ConcurrentQueue<int> query_queue;
    std::atomic<bool> done{false};

    QueryContext(int num_queries) {
        start_times.resize(num_queries);
        finish_times.resize(num_queries);
    }
};

void client(QueryContext& ctx, int qps, int offset, int num_queries) {
    std::mt19937 rng(0);
    std::exponential_distribution<double> arrival_distribution(qps);

    for (int i = offset; i < offset + num_queries; ++i) {
        ctx.start_times[i] = Clock::now();
        ctx.query_queue.enqueue(i);

        std::this_thread::sleep_for(std::chrono::duration<double>(arrival_distribution(rng)));
    }
}

void worker(QueryContext& ctx, int max_batch_size, std::chrono::milliseconds max_wait_time) {
    while (!ctx.done.load()) { std::vector<int> batch;
        auto batch_start = Clock::now();

        while (batch.size() < max_batch_size) {
            int query_id;
            if (ctx.query_queue.try_dequeue(query_id)) {
                batch.push_back(query_id);
            } else {
                auto now = Clock::now();
                if (now - batch_start >= max_wait_time) {
                    break;
                }
            }
        }

        if (!batch.empty()) {
            cout << "batch: " << batch.size() << std::endl;
            for (auto i : batch) {
                cout << i << " ";
            }
            cout << endl;
            //bang.bang_query(batch);
            auto finish_time = Clock::now();
            for (int query_id : batch) {
                ctx.finish_times[query_id] = finish_time;
            }
        }
    }
}

int main(int argc, char **argv) {
  int numQueries = atoi(argv[1]);

  QueryContext qc(numQueries);

  std::vector<std::thread> threads;
  // worker
  std::thread workerThread = std::thread([&qc]() {
      worker(qc, 1000, std::chrono::milliseconds(1000));
  });
  // warm up
  std::thread warmUpThread = std::thread([&qc]() {
      client(qc, 1000, 0, 1000);
  });
  // client 
  std::thread clientThread = std::thread([&qc]() {
      std::this_thread::sleep_for(std::chrono::seconds(5));
      client(qc, 1000, 1000, 9000);
  });

  workerThread.join();
  warmUpThread.join();
  clientThread.join();

  return 0;
}
