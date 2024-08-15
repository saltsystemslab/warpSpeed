#ifndef CACHE_COUNTS_HT
#define CACHE_COUNTS_HT




#if COUNT_CACHE_OPS

__device__ __managed__ uint64_t negative_query_count_cache = 0;
__device__ __managed__ uint64_t query_count_cache = 0;
__device__ __managed__ uint64_t insert_count_cache = 0;
__device__ __managed__ uint64_t insert_empty_count_cache = 0;
__device__ __managed__ uint64_t remove_count_cache = 0;


#define ADD_QUERY \
  if (my_tile.thread_rank() == 0)   \
    atomicAdd(&query_count_cache, 1);

#define ADD_QUERY_NEGATIVE \
  if (my_tile.thread_rank() == 0)   \
    atomicAdd(&negative_query_count_cache, 1);

#define ADD_INSERT \
  if (my_tile.thread_rank() == 0)   \
    atomicAdd(&insert_count_cache, 1);

#define ADD_INSERT_EMPTY \
  if (my_tile.thread_rank() == 0)   \
    atomicAdd(&insert_empty_count_cache, 1);


#define ADD_REMOVE \
  if (my_tile.thread_rank() == 0)   \
    atomicAdd(&remove_count_cache, 1);


namespace helpers {

inline uint64_t get_num_cache_queries() {
  cudaDeviceSynchronize();
  auto count = query_count_cache;
  query_count_cache = 0;
  cudaDeviceSynchronize();
  return count;
}

inline uint64_t get_num_cache_inserts() {
  cudaDeviceSynchronize();
  auto count = insert_count_cache;
  insert_count_cache = 0;
  cudaDeviceSynchronize();
  return count;
}

inline uint64_t get_num_cache_inserts_empty() {
  cudaDeviceSynchronize();
  auto count = insert_empty_count_cache;
  insert_empty_count_cache = 0;
  cudaDeviceSynchronize();
  return count;
}

inline uint64_t get_num_cache_removes() {
  cudaDeviceSynchronize();
  auto count = remove_count_cache;
  remove_count_cache = 0;
  cudaDeviceSynchronize();
  return count;
}


inline uint64_t get_num_cache_queries_negative() {
  cudaDeviceSynchronize();
  auto count = negative_query_count_cache;
  negative_query_count_cache = 0;
  cudaDeviceSynchronize();
  return count;
}



}  // namespace bght
#else

#define ADD_QUERY
#define ADD_QUERY_NEGATIVE
#define ADD_INSERT
#define ADD_INSERT_EMPTY
#define ADD_REMOVE


namespace helpers {
inline uint64_t get_num_cache_queries() {
  return 0;
}

inline uint64_t get_num_cache_queries_negative() {
  return 0;
}

inline uint64_t get_num_cache_inserts() {
  return 0;
}

inline uint64_t get_num_cache_inserts_empty() {
  return 0;
}

inline uint64_t get_num_cache_removes() {
  return 0;
}
} 

#endif

#endif