/* Copyright 2024 Indian Institute Of Technology Hyderbad, India. All Rights
Reserved. Licensed under the Apache License, Version 2.0 (the "License"); you
may not use this file except in compliance with the License. You may obtain a
copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless
required by applicable law or agreed to in writing, software distributed under
the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS
OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.
==============================================================================*/
// Authors: Karthik V., Saim Khan, Somesh Singh
//
#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "bang.h"
#include "concurrent_queue.h"

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

double calculate_percentile(std::vector<double> data, double percentile) {
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty.");
    }
    if (percentile < 0.0 || percentile > 100.0) {
        throw std::invalid_argument("Percentile must be between 0 and 100.");
    }

    std::sort(data.begin(), data.end());
    double index = (percentile / 100.0) * (data.size() - 1);

    if (index == static_cast<int>(index)) {
        return data[static_cast<int>(index)];
    } else {
        int lower_index = static_cast<int>(index);
        int upper_index = lower_index + 1;
        return data[lower_index] + (index - lower_index) * (data[upper_index] - data[lower_index]);
    }
}

inline unsigned long long checkpoint_time_millisec() {
  const std::chrono::time_point<std::chrono::system_clock> now =
      std::chrono::system_clock::now();
  auto duration = now.time_since_epoch();
  auto millis =
      std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
  return millis;
}

double calculate_recall(int offset, int batchSize, unsigned* gold_std, float* gs_dist, size_t dim_gs, result_ann_t* our_results, unsigned dim_or, unsigned recall_at) {
  double total_recall = 0;
  std::set<unsigned> gt, res;

  for (int i = 0; i < batchSize; i++) {
    // cout << "Query : " << i + offset << endl;
    gt.clear();
    res.clear();
    unsigned* gt_vec = gold_std + dim_gs * (offset + i);
    result_ann_t* res_vec = our_results + dim_or * i;
    size_t tie_breaker = recall_at;

    if (gs_dist != nullptr) {
      tie_breaker = recall_at - 1;
      float* gt_dist_vec = gs_dist + dim_gs * (offset + i);
      while (tie_breaker < dim_gs && gt_dist_vec[tie_breaker] == gt_dist_vec[recall_at - 1])
        tie_breaker++;
    }

    gt.insert(gt_vec, gt_vec + tie_breaker);
    res.insert(res_vec, res_vec + recall_at);
    /*cout << "Results: ";
    for (int nIterInner = 0; nIterInner < recall_at; nIterInner++)
    {
            cout << our_results[(i*recall_at) + nIterInner] << "\t" ;
    }
    cout << endl;
    cout << "GT: ";*/
    unsigned cur_recall = 0;
    for (auto& v : gt) {
      // cout << v << "\t" ;
      if (res.find(v) != res.end()) {
        cur_recall++;
      }
    }
    // cout << endl;
    total_recall += cur_recall;
  }

  // std::cout << "total_recall = " << total_recall << " " << "num_queries = "
  // << num_queries << " recall_at " << recall_at << endl;
  return total_recall / (batchSize) * (100.0 / recall_at);
}

inline bool file_exists(const std::string& name) {
  struct stat buffer;
  auto val = stat(name.c_str(), &buffer);
  // std::cout << " Stat(" << name.c_str() << ") returned: " << val <<
  // std::endl;
  return (val == 0);
}

class cached_ifstream {
 public:
  cached_ifstream() {}
  cached_ifstream(const std::string& filename, uint64_t cacheSize)
      : cache_size(cacheSize), cur_off(0) {
    this->open(filename, cache_size);
  }
  ~cached_ifstream() {
    delete[] cache_buf;
    reader.close();
  }

  void open(const std::string& filename, uint64_t cacheSize);
  size_t get_file_size();

  void read(char* read_buf, uint64_t n_bytes);

 private:
  // underlying ifstream
  std::ifstream reader;
  // # bytes to cache in one shot read
  uint64_t cache_size = 0;
  // underlying buf for cache
  char* cache_buf = nullptr;
  // offset into cache_buf for cur_pos
  uint64_t cur_off = 0;
  // file size
  uint64_t fsize = 0;
};

/*Helper function*/
void cached_ifstream ::open(const std::string& filename, uint64_t cacheSize) {
  this->cur_off = 0;
  reader.open(filename, std::ios::binary | std::ios::ate);
  fsize = reader.tellg();
  reader.seekg(0, std::ios::beg);
  assert(reader.is_open());
  assert(cacheSize > 0);
  cacheSize = (std::min)(cacheSize, fsize);
  this->cache_size = cacheSize;
  cache_buf = new char[cacheSize];
  reader.read(cache_buf, cacheSize);
  // cout << "Opened: " << filename.c_str() << ", size: " << fsize  << ",
  // cache_size: " << cacheSize << std::endl;
}

size_t cached_ifstream ::get_file_size() {
  return fsize;
}

void cached_ifstream ::read(char* read_buf, uint64_t n_bytes) {
  assert(cache_buf != nullptr);
  assert(read_buf != nullptr);
  if (n_bytes <= (cache_size - cur_off)) {
    // case 1: cache contains all data
    memcpy(read_buf, cache_buf + cur_off, n_bytes);
    cur_off += n_bytes;
  } else {
    // case 2: cache contains some data
    uint64_t cached_bytes = cache_size - cur_off;
    if (n_bytes - cached_bytes > fsize - reader.tellg()) {
      std::stringstream stream;
      stream << "Reading beyond end of file" << std::endl;
      stream << "n_bytes: " << n_bytes << " cached_bytes: " << cached_bytes
             << " fsize: " << fsize << " current pos:" << reader.tellg()
             << std::endl;
      cout << stream.str() << std::endl;
      exit(1);
    }
    memcpy(read_buf, cache_buf + cur_off, cached_bytes);

    reader.read(read_buf + cached_bytes, n_bytes - cached_bytes);
    cur_off = cache_size;

    uint64_t size_left = fsize - reader.tellg();

    if (size_left >= cache_size) {
      reader.read(cache_buf, cache_size);
      cur_off = 0;
    }
  }
}

// compute ground truth
inline void load_truthset(const std::string& bin_file,
                          uint32_t*& ids,
                          float*& dists,
                          size_t& npts,
                          size_t& dim) {
  uint64_t read_blk_size = 64 * 1024 * 1024;
  cached_ifstream reader(bin_file, read_blk_size);
  // std::cout << "Reading truthset file " << bin_file.c_str() << " ..." <<
  // std::endl;
  size_t actual_file_size = reader.get_file_size();

  int npts_i32, dim_i32;
  reader.read((char*)&npts_i32, sizeof(int));
  reader.read((char*)&dim_i32, sizeof(int));
  npts = (unsigned)npts_i32;
  dim = (unsigned)dim_i32;

  // std::cout << "Metadata: #pts = " << npts << ", #dims = " << dim << "..." <<
  // std::endl;

  size_t expected_actual_file_size =
      2 * npts * dim * sizeof(uint32_t) + 2 * sizeof(uint32_t);
  if (actual_file_size != expected_actual_file_size) {
    std::stringstream stream;
    stream << "Error. File size mismatch. Actual size is " << actual_file_size
           << " while expected size is  " << expected_actual_file_size
           << " npts = " << npts << " dim = " << dim << std::endl;
    std::cout << stream.str();
    //      throw ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
    //                                  __LINE__);
    exit(1);
  }

  ids = new uint32_t[npts * dim];
  reader.read((char*)ids, npts * dim * sizeof(uint32_t));
  dists = new float[npts * dim];
  reader.read((char*)dists, npts * dim * sizeof(float));
}

const string DT_UINT8("uint8");
const string DT_INT8("int8");
const string DT_FLOAT("float");
const string DISTFUNC_MIPS("mips");

void client(QueryContext& ctx, int qps, int offset, int num_queries) {
  std::mt19937 rng(0);
  std::exponential_distribution<double> arrival_distribution(qps);

  for (int i = offset; i < offset + num_queries; ++i) {
    ctx.start_times[i] = Clock::now();
    ctx.query_queue.enqueue(i);

    std::this_thread::sleep_for(std::chrono::duration<double>(arrival_distribution(rng)));
  }
}

template <typename T>
void worker(QueryContext& ctx, BANGSearch<T>& bang, int numQueries, T* queriesFP, int dim, int max_batch_size, int recall_param, bool calc_recall_flag, unsigned* gt_ids, float* gt_dists, size_t gt_dim, std::chrono::milliseconds max_wait_time) {
  std::atomic<int> numFinishQueries{0};

  while (!ctx.done.load()) {
    std::vector<int> batch;
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
      // conduct query execution
      cout << "batch: " << batch.size() << std::endl;
      /*
      cout << "batch start: " << batch.front() << std::endl;
      cout << "batch start: " << batch.back() << std::endl;
      for (auto i : batch) {
        cout << i << " ";
      }
      cout << endl;
      */
      result_ann_t* nearestNeighbours = (result_ann_t*) malloc(sizeof(result_ann_t) * recall_param * batch.size());
      float* nearestNeighbours_dist = (float*) malloc(sizeof(float) * recall_param * batch.size());

      auto batch_prepare_start = Clock::now();
      bang.bang_alloc(batch.size());
      bang.bang_init(batch.size());
      auto batch_prepare_end = Clock::now();

      bang.bang_query(queriesFP + dim * batch.front(), batch.size(), nearestNeighbours, nearestNeighbours_dist);

      auto finish_time = Clock::now();
      for (int query_id : batch) {
        //ctx.finish_times[query_id] = finish_time;
        ctx.finish_times[query_id] = finish_time - (batch_prepare_end - batch_prepare_start);
      }

      numFinishQueries += batch.size();
      if (numFinishQueries == numQueries) {
        // compute the recall for the last batch
        if (calc_recall_flag) {
          double recall = calculate_recall(batch.front(), batch.size(), gt_ids, gt_dists, gt_dim, nearestNeighbours, recall_param, recall_param);
          std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
          std::cout.precision(2);
          std::string recall_string = "Recall@" + std::to_string(recall_param) + ": ";
          std::cout << recall_string << recall << std::endl;
        }
        // exit the loop
        break;
      }

      free(nearestNeighbours);
      nearestNeighbours = NULL;
      free(nearestNeighbours_dist);
      nearestNeighbours_dist = NULL;
      bang.bang_free();
    }
  }
}

template <typename T>
int run_anns(int argc, char** argv) {
  // load index files
  BANGSearch<T> objBANG;
  if (false == objBANG.bang_load(argv[1])) {
    cout << "Error: Bang_load failed" << endl;
    return -1;
  }

  //  load the queries
  T* queriesFP = NULL;
  int numQueries = atoi(argv[4]);
  int maxBatchSize = atoi(argv[5]);

  string queryPointsFP_file = string(argv[2]);
  ifstream in4(queryPointsFP_file, std::ios::binary);
  if (!in4.is_open()) {
    printf("Error.. Could not open the Query File: %s\n", queryPointsFP_file.c_str());
    return -1;
  }
  in4.seekg(4);
  int dim = 0;
  in4.read((char*)&dim, sizeof(int));
  // full floating point coordinates of queries
  queriesFP = (T*)malloc(numQueries * dim * sizeof(T));
  if (NULL == queriesFP) {
    printf("Error.. Malloc failed for queriesFP");
    return -1;
  }
  in4.read((char*)queriesFP, sizeof(T) * dim * numQueries);
  in4.close();

  int recall_param = atoi(argv[6]);
  int nWLLen = atoi(argv[9]);
  // load ground truth
  unsigned* gt_ids = nullptr;
  float* gt_dists = nullptr;
  size_t gt_num, gt_dim;
  bool calc_recall_flag = false;

  if (file_exists(argv[3])) {
    load_truthset(argv[3], gt_ids, gt_dists, gt_num, gt_dim);
    calc_recall_flag = true;
    std::cout << "Groundtruth file " << argv[3] << " loaded" << std::endl;
  } else {
    std::cout << "Groundtruth file could not be loaded:" << argv[3] << std::endl;
    exit(1);
  }

  sleep(1);

  DistFunc uDistFunc = ENUM_DIST_L2;
  if (DISTFUNC_MIPS == argv[8]) {
    uDistFunc = ENUM_DIST_MIPS;
  }
  objBANG.bang_set_searchparams(recall_param, nWLLen, uDistFunc);

  int qps = atoi(argv[10]);
  QueryContext qc(numQueries);
  std::vector<std::thread> threads;
  // worker
  std::thread workerThread = std::thread([&qc, &objBANG, numQueries, queriesFP, dim, maxBatchSize, recall_param, calc_recall_flag, gt_ids, gt_dists, gt_dim]() {
      worker(qc, objBANG, numQueries, queriesFP, dim, maxBatchSize, recall_param, calc_recall_flag, gt_ids, gt_dists, gt_dim, std::chrono::milliseconds(1000));

      // calculate query time 
      std::vector<double> queryTime(numQueries);
      for (int i = 1000; i < numQueries; i++) {
        queryTime[i] = std::chrono::duration_cast<std::chrono::milliseconds>(qc.finish_times[i] - qc.start_times[i]).count();
      }

      // calculate p50 and p99
      double p50 = calculate_percentile(queryTime, 50.0);
      double p99 = calculate_percentile(queryTime, 99.0);

      std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
      std::cout.precision(2);
      std::cout << "P50: " << p50 << std::endl;
      std::cout << "P99: " << p99 << std::endl;
  });
  // warm up
  std::thread warmUpThread = std::thread([&qc, qps]() {
      client(qc, qps, 0, 1000);
  });
  // client
  std::thread clientThread = std::thread([&qc, qps]() {
      std::this_thread::sleep_for(std::chrono::seconds(10));
      client(qc, qps, 1000, 9000);
  });

  workerThread.join();
  warmUpThread.join();
  clientThread.join();

  objBANG.bang_unload();
  free(queriesFP);
  delete[] gt_ids;
  delete[] gt_dists;

  return 0;
}

int main(int argc, char** argv) {
    if (argc < 11) {
        cerr << "Too few parameters! " << argv[0] << " "
            << "<path with file prefix to the director with index files > \
            <query file> <GroundTruth File> <NumQueries> <maxBatchSize> <recall parameter k> <data type : uint8, int8 or float> <dist funct: l2 or mips>"
            << endl;
        exit(1);
    }

    if (DT_UINT8 == argv[7]) {
        return run_anns<uint8_t>(argc, argv);
    } else if (DT_INT8 == argv[7]) {
        return run_anns<int8_t>(argc, argv);
    } else if (DT_FLOAT == argv[7]) {
        return run_anns<float>(argc, argv);
    } else {
        cerr << "Invalid data type specified" << endl;
        exit(1);
    }
}
