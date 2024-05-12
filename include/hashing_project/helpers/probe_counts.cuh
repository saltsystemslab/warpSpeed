#ifndef PROBE_COUNTS_HT
#define PROBE_COUNTS_HT


#if COUNT_PROBES
__device__ __managed__ uint32_t helper_global_probes_count = 0;
#define ADD_PROBE_TILE \
  if (my_tile.thread_rank() == 0)   \
    atomicAdd(&helper_global_probes_count, 1);


#define ADD_PROBE_ADJUSTED \
  if (my_tile.thread_rank() == 0) \
    atomicAdd(&helper_global_probes_count, ((sizeof(Key)+sizeof(Val))*my_tile.size()-1)/128+1);

#define ADD_PROBE_BUCKET \
  if (my_tile.thread_rank() == 0)   \
    atomicAdd(&helper_global_probes_count, (uint32_t) bucket_size*(sizeof(Key)+sizeof(Val))/128);

#define ADD_PROBE atomicAdd(&helper_global_probes_count, 1);
namespace helpers {
inline uint32_t get_num_probes() {
  cudaDeviceSynchronize();
  auto count = helper_global_probes_count;
  helper_global_probes_count = 0;
  cudaDeviceSynchronize();
  return count;
}
}  // namespace bght
#else
#define ADD_PROBE_TILE
#define ADD_PROBE
#define ADD_PROBE_BUCKET
#define ADD_PROBE_ADJUSTED
namespace helpers {
inline uint32_t get_num_probes() {
  return 0;
}
} 

#endif

#endif