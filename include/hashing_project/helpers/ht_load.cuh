#ifndef LD_HELPER
#define LD_HELPER


#include <gallatin/allocators/alloc_utils.cuh>
#include <hashing_project/helpers/ht_pairs.cuh>

#define STABLE_HT_LOCKLESS_QUERY 0

#if LOAD_CHEAP

template <typename T>
__device__ T hash_table_load(const T * address){

  return address[0];

}

template<typename T>
__device__ inline void ht_store(T * p, T store_val){

  p[0] = store_val;

}


//large vector load.
template <template<typename, typename> typename pair, typename Key, typename Val>
__device__ inline pair<Key, Val> ht_load_packed_pair (pair<Key, Val> * address) {

  return address[0];

}

template <typename pair>
__device__ inline pair ht_load_metadata (const uint16_t * address) {

  return ((pair *) address)[0];

}


template <template<typename, typename> typename pair, typename Key, typename Val>
__device__ inline void ht_store_packed_pair(pair<Key, Val> * address, pair<Key, Val> data) {

  address[0] = data;

}

#else

template <typename T>
__device__ inline T hash_table_load(const T * address){

  //#pragma message "You shouldn't see this compile - Hash table loading value that is too large"
  //static_assert(1 == 0);
  asm volatile ("trap;");


}


template <>
__device__ inline uint64_t hash_table_load(const uint64_t *p) {
  uint64_t res;
  asm volatile("ld.gpu.acquire.u64 %0, [%1];" : "=l"(res) : "l"(p));
  return res;

  // return atomicOr((unsigned long long int *)p, 0ULL);
}

template <>
__device__ inline uint32_t hash_table_load(const uint32_t *p) {
  uint32_t res;
  asm volatile("ld.gpu.acquire.u32 %0, [%1];" : "=r"(res) : "l"(p));
  return res;

  // return atomicOr((unsigned long long int *)p, 0ULL);
}

template <>
__device__ inline uint16_t hash_table_load(const uint16_t *p) {
  uint16_t res;
  asm volatile("ld.gpu.acquire.u16 %0, [%1];" : "=h"(res) : "l"(p));
  return res;
}



template<typename T>
__device__ inline void ht_store(const T * p, T store_val){

  asm volatile("trap;");

}

template<>
__device__ inline void ht_store(const uint64_t *p, uint64_t store_val) {
  
  asm volatile("st.gpu.release.u64 [%0], %1;" :: "l"(p), "l"(store_val) : "memory");

  // return atomicOr((unsigned long long int *)p, 0ULL);
}

template<>
__device__ inline void ht_store(const uint32_t *p, uint32_t store_val) {
  
  asm volatile("st.gpu.release.u32 [%0], %1;" :: "l"(p), "r"(store_val) : "memory");

  // return atomicOr((unsigned long long int *)p, 0ULL);
}


template<>
__device__ inline void ht_store(const uint16_t *p, uint16_t store_val) {
  
  asm volatile("st.gpu.release.u16 [%0], %1;" :: "l"(p), "h"(store_val) : "memory");

  // return atomicOr((unsigned long long int *)p, 0ULL);
}




//large vector load.
template <template<typename, typename> typename pair, typename Key, typename Val>
__device__ inline pair<Key, Val> ht_load_packed_pair (pair<Key, Val> * address) {


  if constexpr  (sizeof(Key) + sizeof(Val) == 16){

    pair<Key, Val> loaded_pair;

    asm volatile("ld.gpu.acquire.v2.u64 {%0,%1}, [%2];" : "=l"(loaded_pair.key), "=l"(loaded_pair.val) : "l"(address));

    return loaded_pair;

  }

  else if constexpr (sizeof(Key) + sizeof(Val) == 8){

    uint64_t return_val;

    asm volatile("ld.gpu.acquire.u64 %0, [%1];" : "=l"(return_val) : "l"(address));


    using pair_type = pair<Key,Val>;

    return ((pair_type *) &return_val)[0];

  }

}

//large vector store
//large vector load.
template <template<typename, typename> typename pair, typename Key, typename Val>
__device__ inline void ht_store_packed_pair(pair<Key, Val> * address, pair<Key, Val> data) {

  if constexpr  (sizeof(Key) + sizeof(Val) == 16){


    asm volatile("st.gpu.release.v2.u64 [%0], {%1,%2};" :: "l"(address), "l"(data.key), "l"(data.val) : "memory");

    return;

  }

  else if constexpr (sizeof(Key) + sizeof(Val) == 8){


    asm volatile("st.gpu.release.v2.u32 [%0], {%1,%2};" :: "l"(address), "r"(data.key), "r"(data.val) : "memory");

    //asm volatile("st.gpu.release.u64 [%0], %1;" :: "l"((uint64_t *) address), "l"((uint64_t) data) : "memory");

    return;

  }

}


template <typename pair>
__device__ inline pair ht_load_metadata (const uint16_t * address) {


  pair loaded_pair;

  asm volatile("ld.gpu.acquire.v2.u64 {%0,%1}, [%2];" : "=l"(loaded_pair.first), "=l"(loaded_pair.second) : "l"(address));

  return loaded_pair;

}


#endif

#endif