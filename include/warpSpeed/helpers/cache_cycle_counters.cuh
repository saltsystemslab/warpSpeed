#ifndef CACHE_CYCLE_COUNTS_HT
#define CACHE_CYCLE_COUNTS_HT


#define TIMER_RATIO 10

#if COUNT_CACHE_CYCLES

__device__ __managed__ uint64_t total_throughput = 0;
__device__ __managed__ uint64_t total_calls = 0;


__device__ __managed__ uint64_t fast_throughput = 0;
__device__ __managed__ uint64_t fast_calls = 0;

__device__ __managed__ uint64_t host_throughput = 0;
__device__ __managed__ uint64_t host_calls = 0;

__device__ __managed__ uint64_t queue_throughput = 0;
__device__ __managed__ uint64_t queue_calls = 0;

#define START_TOTAL_THROUGHPUT \
  uint64_t total_start_time = clock64();

#define END_TOTAL_THROUGHPUT \
  if (my_tile.thread_rank() == 0){ \
    uint64_t end_time = clock64(); \
    atomicAdd(&total_throughput, (unsigned long long int) (end_time-total_start_time)/TIMER_RATIO); \
    atomicAdd(&total_calls, 1); }

#define END_FAST_THROUGHPUT \
  if (my_tile.thread_rank() == 0){ \
    uint64_t end_time = clock64(); \
    atomicAdd(&fast_throughput, (unsigned long long int) (end_time-total_start_time)/TIMER_RATIO); \
    atomicAdd(&fast_calls, 1); }

#define START_QUEUE_READ_THROUGHPUT \
  uint64_t queue_start_time = clock64();

#define END_QUEUE_READ_THROUGHPUT \
  if (my_tile.thread_rank() == 0){ \
    uint64_t queue_end_time = clock64(); \
    atomicAdd(&queue_throughput, (unsigned long long int) (queue_end_time-queue_start_time)/TIMER_RATIO); \
    atomicAdd(&queue_calls, 1); }

#define START_HOST_READ_THROUGHPUT \
  uint64_t host_start_time = clock64();

#define END_HOST_READ_THROUGHPUT \
  if (my_tile.thread_rank() == 0){ \
    uint64_t host_end_time = clock64(); \
    atomicAdd(&host_throughput, (unsigned long long int) (host_end_time-host_start_time)/TIMER_RATIO); \
    atomicAdd(&host_calls, 1); }


namespace helpers {


inline uint64_t print_total_cycle_data() {
  cudaDeviceSynchronize();

  uint64_t throughput = (total_throughput/total_calls)*TIMER_RATIO;

  total_throughput = 0;
  total_calls = 0;
  cudaDeviceSynchronize();
  printf("Total cycles: %lu\n", throughput);
  return throughput;
}

inline uint64_t print_fast_cycle_data() {
  cudaDeviceSynchronize();

  uint64_t throughput = (fast_throughput/fast_calls)*TIMER_RATIO;

  fast_throughput = 0;
  fast_calls = 0;
  cudaDeviceSynchronize();
  printf("Fast cycles: %lu\n", throughput);
  return throughput;
}

inline uint64_t print_host_cycle_data() {
  cudaDeviceSynchronize();

  uint64_t throughput = (host_throughput/host_calls)*TIMER_RATIO;

  host_throughput = 0;
  host_calls = 0;
  cudaDeviceSynchronize();
  printf("Host cycles: %lu\n", throughput);
  return throughput;
}

inline uint64_t print_queue_cycle_data() {
  cudaDeviceSynchronize();

  uint64_t throughput = (queue_throughput/queue_calls)*TIMER_RATIO;

  queue_throughput = 0;
  queue_calls = 0;
  cudaDeviceSynchronize();
  printf("Queue cycles: %lu\n", throughput);
  return throughput;
}


}  // namespace bght
#else

#define START_TOTAL_THROUGHPUT
#define END_TOTAL_THROUGHPUT 
#define END_FAST_THROUGHPUT
#define START_QUEUE_READ_THROUGHPUT
#define END_QUEUE_READ_THROUGHPUT
#define START_HOST_READ_THROUGHPUT
#define END_HOST_READ_THROUGHPUT

namespace helpers {



 
} 

#endif

#endif