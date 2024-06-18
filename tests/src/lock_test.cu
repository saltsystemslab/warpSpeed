/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */



#define COUNT_PROBES 1

#include <gallatin/allocators/global_allocator.cuh>

#include <gallatin/allocators/timer.cuh>

#include <warpcore/single_value_hash_table.cuh>
#include <gallatin/allocators/timer.cuh>

#include <bght/p2bht.hpp>

#include <hashing_project/cache.cuh>

#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <chrono>
#include <openssl/rand.h>


#include <hashing_project/table_wrappers/p2_wrapper.cuh>
#include <hashing_project/table_wrappers/dummy_ht.cuh>
#include <hashing_project/table_wrappers/iht_wrapper.cuh>
#include <hashing_project/table_wrappers/warpcore_wrapper.cuh>

#include <iostream>
#include <locale>


//thrust stuff.
#include <thrust/shuffle.h>
#include <thrust/random.h>

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

using namespace gallatin::allocators;



#define MEASURE_FAILS 1

#define MEASURE_INDEPENDENT 1


#define DATA_TYPE uint64_t





struct lock_bucket
{
   
   uint64_t lock;

   uint64_t data[15];

   __device__ void init(){
      lock = 0;

      for (int i =0; i < 15; i++){
         data[i] = 0;
      }
   }

};


__global__ void init_buckets(lock_bucket * buckets, uint64_t n_buckets){

   uint64_t tid = gallatin::utils::get_tid();

   if (tid >= n_buckets) return;

   buckets[tid].init();

}


template <uint32_t tile_size>
__global__ void test_locks_internal(uint64_t * accesses, uint64_t n_ops, lock_bucket * buckets, uint64_t n_buckets){

   
   auto thread_block = cg::this_thread_block();

   cg::thread_block_tile<tile_size> my_tile = cg::tiled_partition<tile_size>(thread_block);

   uint64_t tid = gallatin::utils::get_tile_tid(my_tile);


   if (tid >= n_ops) return;


   uint64_t my_bucket_addr = accesses[tid] % n_buckets;


   lock_bucket * my_bucket = &buckets[my_bucket_addr];
   //acquire lock.

   if (my_tile.thread_rank() == 0){
      while (atomicOr((unsigned long long int *)&my_bucket->lock, (unsigned long long int) SET_BIT_MASK(0)) != 0);
   }

   my_tile.sync();


   if (my_tile.thread_rank() < 15){


      uint64_t data = gallatin::utils::ld_acq(&my_bucket->data[my_tile.thread_rank()]);
   

      if (my_tile.thread_rank() == 0){

         gallatin::utils::typed_atomic_write(&my_bucket->data[0], data, tid);
      }


   }



   my_tile.sync();

   atomicAnd((unsigned long long int *)&my_bucket->lock, (unsigned long long int) ~SET_BIT_MASK(0));


}


template <uint32_t tile_size>
__global__ void test_locks_external(uint64_t * accesses, uint64_t n_ops, lock_bucket * buckets, uint64_t n_buckets, uint64_t * locks){

   
   auto thread_block = cg::this_thread_block();

   cg::thread_block_tile<tile_size> my_tile = cg::tiled_partition<tile_size>(thread_block);

   uint64_t tid = gallatin::utils::get_tile_tid(my_tile);


   if (tid >= n_ops) return;


   uint64_t my_bucket_addr = accesses[tid] % n_buckets;


   lock_bucket * my_bucket = &buckets[my_bucket_addr];

   //acquire lock.

   if (my_tile.thread_rank() == 0){

      uint64_t high = my_bucket_addr/64;
      uint64_t low = my_bucket_addr % 64;

      while (atomicOr((unsigned long long int *)&locks[high], (unsigned long long int) SET_BIT_MASK(low)) & SET_BIT_MASK(low));

   }

   my_tile.sync();


   if (my_tile.thread_rank() < 15){


      uint64_t data = gallatin::utils::ld_acq(&my_bucket->data[my_tile.thread_rank()]);
   

      if (my_tile.thread_rank() == 0){

         gallatin::utils::typed_atomic_write(&my_bucket->data[0], data, tid);
      }


   }



   my_tile.sync();

   uint64_t high = my_bucket_addr/64;
   uint64_t low = my_bucket_addr % 64;

   //if old is 0, SET_BIT_MASK & 0 is 0 - loop exit.
   atomicAnd((unsigned long long int *)&locks[high], (unsigned long long int) ~SET_BIT_MASK(low));

}

template <typename T>
__host__ T * generate_data(uint64_t nitems){


   //malloc space

   T * vals;

   cudaMallocHost((void **)&vals, sizeof(T)*nitems);


   //          100,000,000
   uint64_t cap = 100000000ULL;

   for (uint64_t to_fill = 0; to_fill < nitems; to_fill+=0){

      uint64_t togen = (nitems - to_fill > cap) ? cap : nitems - to_fill;


      RAND_bytes((unsigned char *) (vals + to_fill), togen * sizeof(T));



      to_fill += togen;

      //printf("Generated %llu/%llu\n", to_fill, nitems);

   }

   printf("Generation done\n");
   return vals;
}


//generate data within the range from 0, cutoff
//modulus any keys which exceed the range.
//this generates a random list of keys to operate on.
template <typename T>
__host__ T * generate_clipped_data(uint64_t nitems, uint64_t cutoff){


   T * host_data = generate_data<T>(nitems);

   for (uint64_t i =0; i < nitems; i++){
      host_data[i] = host_data[i] % cutoff;
   }


   return host_data;


}





//stolen shamelessy from https://stackoverflow.com/questions/43482488/how-to-format-a-number-with-thousands-separator-in-c-c
// struct separate_thousands : std::numpunct<char> {
//     char_type do_thousands_sep() const override { return ','; }  // separate with commas
//     string_type do_grouping() const override { return "\3"; } // groups of 3 digit
// };



template <uint tile_size>
__host__ void internal_test(uint64_t n_buckets, DATA_TYPE * access_pattern, uint64_t n_ops){


   lock_bucket * buckets = gallatin::utils::get_device_version<lock_bucket>(n_buckets);

   init_buckets<<<(n_buckets-1)/256+1,256>>>(buckets, n_buckets);

   cudaDeviceSynchronize();

   gallatin::utils::timer lock_timer;

   test_locks_internal<tile_size><<<(n_ops*tile_size-1)/256+1,256>>>(access_pattern, n_ops, buckets, n_buckets);

   lock_timer.sync_end();

   lock_timer.print_throughput("Internal Operated", n_ops);

   cudaFree(buckets);


}


template <uint tile_size>
__host__ void external_test(uint64_t n_buckets, DATA_TYPE * access_pattern, uint64_t n_ops){


   uint64_t n_locks = (n_buckets-1)/64+1;

   uint64_t * locks = gallatin::utils::get_device_version<uint64_t>(n_locks);

   cudaMemset(locks, 0, sizeof(uint64_t)*n_locks);

   lock_bucket * buckets = gallatin::utils::get_device_version<lock_bucket>(n_buckets);

   init_buckets<<<(n_buckets-1)/256+1,256>>>(buckets, n_buckets);

   cudaDeviceSynchronize();

   gallatin::utils::timer lock_timer;

   test_locks_external<tile_size><<<(n_ops*tile_size-1)/256+1,256>>>(access_pattern, n_ops, buckets, n_buckets, locks);

   lock_timer.sync_end();

   lock_timer.print_throughput("External Operated", n_ops);

   cudaFree(buckets);
   cudaFree(locks);

}



template <uint tile_size>
__host__ void test_both(uint64_t n_buckets, DATA_TYPE * access_pattern, uint64_t n_ops){

   printf("Tile size: %u\n", tile_size);
   internal_test<tile_size>(n_buckets, access_pattern, n_ops);
   external_test<tile_size>(n_buckets, access_pattern, n_ops);

}

int main(int argc, char** argv) {

   uint64_t n_buckets;

   uint64_t n_rounds;


   if (argc < 2){
      n_buckets = 1000000;
   } else {
      n_buckets = std::stoull(argv[1]);
   }

   if (argc < 3){
      n_rounds = 1000;
   } else {
      n_rounds = std::stoull(argv[2]);
   }

   auto access_pattern = generate_data<DATA_TYPE>(n_rounds);
   
   //auto access_pattern = generate_clipped_data<uint64_t>(table_capacity+replacement_items, table_capacity);
   
   test_both<1>(n_buckets, access_pattern, n_rounds);
   test_both<2>(n_buckets, access_pattern, n_rounds);
   test_both<4>(n_buckets, access_pattern, n_rounds);
   test_both<8>(n_buckets, access_pattern, n_rounds);
   test_both<16>(n_buckets, access_pattern, n_rounds);
   test_both<32>(n_buckets, access_pattern, n_rounds);
   //p2


   cudaFreeHost(access_pattern);

   //cache_test<hashing_project::wrappers::p2_wrapper, 32>(host_items, cache_items, n_ops);



   cudaDeviceReset();
   return 0;

}
