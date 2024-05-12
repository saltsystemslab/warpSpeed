#ifndef LD_HELPER
#define LD_HELPER


#include <gallatin/allocators/alloc_utils.cuh>



#if LOAD_CHEAP

template <typename T>
__device__ T hash_table_load(const T * address){

  return address[0];

}

#else

template <typename T>
__device__ T hash_table_load(const T * address){

  return gallatin::utils::ld_acq(address);

}


#endif

#endif