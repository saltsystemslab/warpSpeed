#ifndef WARPSPEED_CUCKOO
#define WARPSPEED_CUCKOO

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <gallatin/allocators/alloc_utils.cuh>
#include <warpSpeed/helpers/const_cuckoo_vector.cuh>
#include <warpSpeed/helpers/ht_load.cuh>

#include "assert.h"
#include "stdio.h"

namespace cg = cooperative_groups;

// helper_macro
// define macros
#define MAX_VALUE(nbits) ((1ULL << (nbits)) - 1)
#define BITMASK(nbits) ((nbits) == 64 ? 0xffffffffffffffff : MAX_VALUE(nbits))

#define SET_BIT_MASK(index) ((1ULL << index))

// a pointer list managing a set section of device memory

#define N_CUCKOO_HASHES 3
#define CUCKOO_MAX_PROBES 20

#define MAX_CUCKOO_ATTEMPTS 500

#define MEASURE_LOCKS 1

#define DEBUG_PRINTS 0

//cache protocol
//query cache
//on success add to pin?
//need delete from potential buckets implementation - need to download warpcore...
//buidld with primary p2bht first.



namespace warpSpeed {

namespace tables {


   //investigate this.
   // __device__ inline void st_rel(const uint64_t *p, uint64_t store_val) {
  
   //   asm volatile("st.gpu.release.u64 [%0], %1;" :: "l"(p), "l"(store_val) : "memory");

   //   // return atomicOr((unsigned long long int *)p, 0ULL);

   //   // atom{.sem}{.scope}{.space}.cas.b128 d, [a], b, c {, cache-policy};
   // }

template <typename HT, uint tile_size>
   __global__ void cuckoo_get_fill_kernel(HT * metadata_table, uint64_t n_buckets, uint64_t * item_count){


      auto thread_block = cg::this_thread_block();

      cg::thread_block_tile<tile_size> my_tile = cg::tiled_partition<tile_size>(thread_block);


      uint64_t tid = gallatin::utils::get_tile_tid(my_tile);

      if (tid >= n_buckets) return;

      uint64_t n_items_in_bucket = metadata_table->get_bucket_fill(my_tile, tid);

      if (my_tile.thread_rank() == 0){
         atomicAdd((unsigned long long int *)item_count, n_items_in_bucket);
      }


   }

   template <typename Key, Key defaultKey, Key tombstoneKey, typename Val, Val defaultVal, Val tombstoneVal, uint partition_size, uint bucket_size>
   struct cuckoo_bucket {

      //uint64_t lock_and_size;

      using pair_type = ht_pair<Key, Val>;

      pair_type slots[bucket_size];

      static const uint64_t n_traversals = ((bucket_size-1)/partition_size+1)*partition_size;


      __device__ void init(){

         pair_type sentinel_pair{defaultKey, defaultVal};

         //lock_and_size = 0;

         for (uint i=0; i < bucket_size; i++){

            slots[i] = sentinel_pair;

         }

         __threadfence();
      }



      __device__ int insert(cg::thread_block_tile<partition_size> my_tile, Key ext_key, Val ext_val){


         //first read size
         // internal_read_size = gallatin::utils::ldcv(&size);

         // //failure means resize has started...
         // if (internal_read_size != expected_size) return false;

         for (int i = my_tile.thread_rank(); i < n_traversals; i+=my_tile.size()){

            bool key_match = (i < bucket_size);

            Key loaded_key;

            if (key_match) loaded_key = hash_table_load(&slots[i].key);

            //early drop if loaded_key is gone
            bool ballot = key_match && (loaded_key == defaultKey);

            auto ballot_result = my_tile.ballot(ballot);

            while (ballot_result){

                  ballot = false;

                  const auto leader = __ffs(ballot_result)-1;

                  if (leader == my_tile.thread_rank()){


                     ballot = typed_atomic_write(&slots[i].key, defaultKey, ext_key);

                     ADD_PROBE
                     if (ballot){

                        ht_store(&slots[i].val, ext_val);
                        //gallatin::utils::st_rel(&slots[i].val, ext_val);
                        //typed_atomic_exchange(&slots[i].val, ext_val);
                     }
                  } 

     

                  //if leader succeeds return
                  if (my_tile.ballot(ballot)){
                     return __ffs(my_tile.ballot(ballot))-1;
                  }
                  

                  //if we made it here no successes, decrement leader
                  ballot_result  ^= 1UL << leader;

                  //printf("Stalling in insert_into_bucket keys\n");

               }

               ballot = key_match && (loaded_key == tombstoneKey);

               ballot_result = my_tile.ballot(ballot);

               while (ballot_result){

                  ballot = false;

                  const auto leader = __ffs(ballot_result)-1;

                  if (leader == my_tile.thread_rank()){

                     ballot = typed_atomic_write(&slots[i].key, tombstoneKey, ext_key);

                     ADD_PROBE

                     if (ballot){

                        //loop and wait on tombstone val to be done.

                        // Val loaded_val = hash_table_load(&slots[i].val);

                        // while(loaded_val != tombstoneVal){
                        //    loaded_val = hash_table_load(&slots[i].val);
                        //    __threadfence();

                        //    //printf("Looping in tombstone\n");
                        // }

                        // __threadfence();
                        ht_store(&slots[i].val, ext_val);
                        //gallatin::utils::st_rel(&slots[i].val, ext_val);
                        //typed_atomic_write(&slots[i].val, ext_val);
                     }
                  } 

     

                  //if leader succeeds return
                  if (my_tile.ballot(ballot)){
                     return __ffs(my_tile.ballot(ballot))-1;
                  }
                  

                  //if we made it here no successes, decrement leader
                  ballot_result  ^= 1UL << leader;

                  //printf("Stalling in insert_into_bucket\n");
                  //printf("Stalling in insert_into_bucket tombstone\n");

               }


         }


         return -1;

      }

      //insert based on match_ballots
      //makes 1 attempt - on first failure trigger reload - this is needed for load balancing.
      __device__ bool insert_ballots(cg::thread_block_tile<partition_size> my_tile, Key ext_key, Val ext_val, uint empty_match, uint tombstone_match){


         //first read size
         // internal_read_size = gallatin::utils::ldcv(&size);

         // //failure means resize has started...
         // if (internal_read_size != expected_size) return false;

         //attempt inserts on tombstones

         for (uint i = my_tile.thread_rank(); i < n_traversals; i+=my_tile.size()){

            uint offset = i - my_tile.thread_rank();

            bool empty_ballot = false;
            bool tombstone_ballot = false;
            bool any_ballot = false;

            bool key_match = (i < bucket_size);

            Key loaded_key;

            if (key_match){

               empty_ballot = empty_match & SET_BIT_MASK(i);
               tombstone_ballot = tombstone_match & SET_BIT_MASK(i);

               any_ballot = empty_ballot || tombstone_ballot;
            }

            //early drop if loaded_key is gone

            auto ballot_result = my_tile.ballot(any_ballot);

            while (ballot_result){

               bool ballot = false;

               const auto leader = __ffs(ballot_result)-1;


               if (leader == my_tile.thread_rank() && empty_ballot){

                  ADD_PROBE

                  ballot = typed_atomic_write(&slots[i].key, defaultKey, ext_key);
                  if (ballot){

                     ht_store(&slots[i].val, ext_val);
                     //gallatin::utils::st_rel(&slots[i].val, ext_val);
                     //typed_atomic_exchange(&slots[i].val, ext_val);

                  }
               } 

  

               //if leader succeeds return
               if (my_tile.ballot(ballot)){
                  return true;
               }
               

               //check tombstone

               if (leader == my_tile.thread_rank() && tombstone_ballot){

                  ballot = typed_atomic_write(&slots[i].key, tombstoneKey, ext_key);

                  ADD_PROBE

                  if (ballot){

                     //loop and wait on tombstone val to be done.

                     // Val loaded_val = hash_table_load(&slots[i].val);

                     // while(loaded_val != tombstoneVal){

                     //    //this may be an issue if a stored value is legitimately a tombstone - need special logic in delete?
                     //    loaded_val = hash_table_load(&slots[i].val);
                     //    __threadfence();
                     //    #if DEBUG_PRINTS
                     //    printf("Looping in tombstone 2\n");
                     //    #endif
                     // }

                     
                     ht_store(&slots[i].val, ext_val);
                     //gallatin::utils::st_rel(&slots[i].val, ext_val);

                     __threadfence();


                  }

               }

               //if leader succeeds return
               if (my_tile.ballot(ballot)){
                  return true;
               }
                  

               //if we made it here no successes, decrement leader
               ballot_result  ^= 1UL << leader;

               //printf("Stalling in insert_into_bucket keys\n");


            }

         }


         return false;

      }



      //attempt to insert into the table based on an existing mapping.
      __device__ int upsert_existing(cg::thread_block_tile<partition_size> my_tile, Key ext_key, Val ext_val, uint upsert_mapping){


         //first read size
         // internal_read_size = gallatin::utils::ldcv(&size);

         // //failure means resize has started...
         // if (internal_read_size != expected_size) return false;

         for (int i = my_tile.thread_rank(); i < n_traversals; i+=my_tile.size()){

            //key needs to exist && upsert mapping shows a key exists.
            bool key_match = (i < bucket_size) && (SET_BIT_MASK(i) & upsert_mapping);


            //early drop if loaded_key is gone
            // bool ballot = key_match 


            auto ballot_result = my_tile.ballot(key_match);

            while (ballot_result){

               bool ballot = false;

               const auto leader = __ffs(ballot_result)-1;

               if (leader == my_tile.thread_rank()){

                  ADD_PROBE
                  //ballot = typed_atomic_write(&slots[i].key, ext_key, ext_key);
                  ballot = true;
                  if (ballot){

                     ht_store(&slots[i].val, ext_val);
                     //gallatin::utils::st_rel(&slots[i].val, ext_val);
                     __threadfence();
                     //typed_atomic_exchange(&slots[i].val, ext_val);
                  }
               }

     

                  //if leader succeeds return
                  if (my_tile.ballot(ballot)){
                     return __ffs(my_tile.ballot(ballot))-1;
                  }
                  

                  //if we made it here no successes, decrement leader
                  ballot_result  ^= 1UL << leader;

                  //printf("Stalling in insert_into_bucket keys\n");

               }

         }


         return -1;

      }

      __device__ int upsert_existing_func(cg::thread_block_tile<partition_size> my_tile, Key ext_key, Val ext_val, uint upsert_mapping, void (*replace_func)(pair_type *, Key, Val)){


         //first read size
         // internal_read_size = gallatin::utils::ldcv(&size);

         // //failure means resize has started...
         // if (internal_read_size != expected_size) return false;

         for (int i = my_tile.thread_rank(); i < n_traversals; i+=my_tile.size()){

            //key needs to exist && upsert mapping shows a key exists.
            bool key_match = (i < bucket_size) && (SET_BIT_MASK(i) & upsert_mapping);


            //early drop if loaded_key is gone
            // bool ballot = key_match 


            auto ballot_result = my_tile.ballot(key_match);

            while (ballot_result){

               bool ballot = false;

               const auto leader = __ffs(ballot_result)-1;

               if (leader == my_tile.thread_rank()){

                  ADD_PROBE
                  //ballot = typed_atomic_write(&slots[i].key, ext_key, ext_key);
                  ballot = true;
                  if (ballot){

                     replace_func(&slots[i], ext_key, ext_val);
                     //gallatin::utils::st_rel(&slots[i].val, ext_val);
                     __threadfence();
                     //typed_atomic_exchange(&slots[i].val, ext_val);
                  }
               }

     

                  //if leader succeeds return
                  if (my_tile.ballot(ballot)){
                     return __ffs(my_tile.ballot(ballot))-1;
                  }
                  

                  //if we made it here no successes, decrement leader
                  ballot_result  ^= 1UL << leader;

                  //printf("Stalling in insert_into_bucket keys\n");

               }

         }


         return -1;

      }



      //calculate the available slots int the bucket
      //deposit into a trio of uints for return
      //these each are the result of the a block-wide ballot on empty, tombstone, or key_match
      //allowing one ld_acq of the bucket to service all requests.
      __device__ void load_fill_ballots(const cg::thread_block_tile<partition_size> & my_tile, const Key & upsert_key, __restrict__ uint & empty_match, __restrict__ uint & tombstone_match, __restrict__ uint & key_match){

         //wipe previous
         empty_match = 0U;
         tombstone_match = 0U;
         key_match = 0U;


         for (uint i = my_tile.thread_rank(); i < n_traversals; i+=my_tile.size()){

            //step in clean intervals of my_tile. 
            uint offset = i - my_tile.thread_rank();

            bool valid = i < bucket_size;

            ADD_PROBE_ADJUSTED
            

            bool found_empty = false;
            bool found_tombstone = false;
            bool found_exact = false;

            if (valid){

               Key loaded_key = hash_table_load(&slots[i].key);

               found_empty = (loaded_key == defaultKey);
               found_tombstone = (loaded_key == tombstoneKey);
               found_exact = (loaded_key == upsert_key);

            }

            empty_match |= (my_tile.ballot(found_empty) << offset);
            tombstone_match |= (my_tile.ballot(found_tombstone) << offset);
            key_match |= (my_tile.ballot(found_exact) << offset);

         }

         return;

      }


      __device__ void load_fill_ballots_match(const cg::thread_block_tile<partition_size> & my_tile, const Key & upsert_key, __restrict__ uint & key_match){


         
         //wipe previous
         key_match = 0U;

         
         for (uint i = my_tile.thread_rank(); i < n_traversals; i+=my_tile.size()){

            ADD_PROBE_ADJUSTED

            //step in clean intervals of my_tile. 
            uint offset = i - my_tile.thread_rank();

            bool valid = i < bucket_size;


            bool found_exact = false;

            bool found_empty = false;

            if (valid){

               Key loaded_key = hash_table_load(&slots[i].key);


               found_exact = (loaded_key == upsert_key);

               found_empty = (loaded_key == defaultKey);

            }

            key_match |= (my_tile.ballot(found_exact) << offset);

            if (key_match) return;

            if (my_tile.ballot(found_empty)) return;

         }

         return;

      }



      __device__ bool load_fill_ballots_upserts(const cg::thread_block_tile<partition_size> & my_tile, const Key & upsert_key, const Val & upsert_val, __restrict__ uint & tombstone_match){


         //wipe previous
         tombstone_match = 0U;



         for (uint i = my_tile.thread_rank(); i < n_traversals; i+=my_tile.size()){


            ADD_PROBE_ADJUSTED

            //step in clean intervals of my_tile. 
            uint offset = i - my_tile.thread_rank();

            bool valid = i < bucket_size;

            

            bool found_empty = false;
            bool found_tombstone = false;
            bool found_exact = false;

            if (valid){

               Key loaded_key = hash_table_load(&slots[i].key);

               found_empty = (loaded_key == defaultKey);
               found_tombstone = (loaded_key == tombstoneKey);
               found_exact = (loaded_key == upsert_key);

            }

           
            tombstone_match |= (my_tile.ballot(found_tombstone) << offset);





            //if (empty_match || key_match) return;

            int leader = __ffs(my_tile.ballot(found_exact))-1;

            bool ballot = false;

            if (leader == my_tile.thread_rank()){

               ADD_PROBE

               if (gallatin::utils::typed_atomic_write(&slots[i].key, upsert_key, upsert_key)){

                  ht_store(&slots[i].val, upsert_val);
                  //gallatin::utils::st_rel(&slots[i].val, upsert_val);
                  ballot = true;


               }

            }

            //upserted.
            if (my_tile.ballot(ballot)) return true;

            leader = __ffs(my_tile.ballot(found_empty))-1;

            if (leader == my_tile.thread_rank()){

               ADD_PROBE

               if (gallatin::utils::typed_atomic_write(&slots[i].key, defaultKey, upsert_key)){

                  ht_store(&slots[i].val, upsert_val);
                  //gallatin::utils::st_rel(&slots[i].val, upsert_val);
                  ballot = true;
               }

            }

            if (my_tile.ballot(ballot)) return true;

         }

         return false;

      }


      __device__ bool query(const cg::thread_block_tile<partition_size> & my_tile, Key ext_key, Val & return_val){

         

         for (uint i = my_tile.thread_rank(); i < n_traversals; i+=my_tile.size()){

            ADD_PROBE_ADJUSTED

            uint offset = i - my_tile.thread_rank();

            bool valid = i < bucket_size;

            bool found_ballot = false;

            Val loaded_val;

            if (valid){
               Key loaded_key = hash_table_load(&slots[i].key);

               found_ballot = (loaded_key == ext_key);

               if (found_ballot){
                  loaded_val = hash_table_load(&slots[i].val);
               }
            }


            int found = __ffs(my_tile.ballot(found_ballot))-1;

            if (found == -1) continue;

            return_val = my_tile.shfl(loaded_val, found);

            return true;



         }


         return false;

      }

      __device__ Val * query_ptr_ballot(const cg::thread_block_tile<partition_size> & my_tile, __restrict__ uint & match_ballot){

         //Val * return_val = nullptr;


            int found = __ffs(match_ballot)-1;

            if (found == -1) return nullptr;

            //return_val = my_tile.shfl(loaded_val, found);

            return &slots[found].val;


      }

      __device__ pair_type * query_pair_ballot(const cg::thread_block_tile<partition_size> & my_tile, __restrict__ uint & match_ballot){

         //Val * return_val = nullptr;


            int found = __ffs(match_ballot)-1;

            if (found == -1) return nullptr;

            //return_val = my_tile.shfl(loaded_val, found);

            return &slots[found];


      }

      __device__ bool erase(cg::thread_block_tile<partition_size> my_tile, Key ext_key){


         //acquire lock.


         for (uint i = my_tile.thread_rank(); i < n_traversals; i+=my_tile.size()){

            uint offset = i - my_tile.thread_rank();

            bool valid = i < bucket_size;

            bool found_ballot = false;

            Val loaded_val;

            if (valid){
               Key loaded_key = hash_table_load(&slots[i].key);

               found_ballot = (loaded_key == ext_key);

            }

            uint ballot_result = my_tile.ballot(found_ballot);

            while (ballot_result){

               bool ballot = false;

               const auto leader = __ffs(ballot_result)-1;

               if (leader == my_tile.thread_rank()){

                  ADD_PROBE
                  ballot = typed_atomic_write(&slots[i].key, ext_key, tombstoneKey);
                  if (ballot){

                     //force store
                     //gallatin::utils::st_rel(&slots[i].val, tombstoneVal);
                     //typed_atomic_exchange(&slots[i].val, ext_val);
                  }
               }

     

               //if leader succeeds return
               if (my_tile.ballot(ballot)){
                  return true;
               }
                  

                  //if we made it here no successes, decrement leader
                  ballot_result  ^= 1UL << leader;

                  //printf("Stalling in insert_into_bucket keys\n");

            }

         }



         return false;
      }


   };


   template <typename table>
   __global__ void init_cuckoo_table_kernel(table * hash_table){

      uint64_t tid = gallatin::utils::get_tid();

      if (tid >= hash_table->n_buckets_primary) return;

      hash_table->init_bucket_and_locks(tid);
      

   }





   template <typename Key, Key defaultKey, Key tombstoneKey, typename Val, Val defaultVal, Val tombstoneVal, uint partition_size, uint bucket_size>
   struct cuckoo_table {


      using my_type = cuckoo_table<Key, defaultKey, tombstoneKey, Val, defaultVal, tombstoneVal, partition_size, bucket_size>;


      using tile_type = cg::thread_block_tile<partition_size>;

      using bucket_type = cuckoo_bucket<Key, defaultKey, tombstoneKey, Val, defaultVal, tombstoneVal, partition_size, bucket_size>;

      using pair_type = ht_pair<Key, Val>;

      using vector_type = warpSpeed::data_structs::const_cuckoo_vector<warpSpeed::data_structs::const_vector_pair<Key>, CUCKOO_MAX_PROBES+1>;

      bucket_type * primary_buckets;
      uint64_t * primary_locks;

      uint64_t n_buckets_primary;
      uint64_t n_buckets_alt;
      uint64_t seed;

      //dummy handle
      static __host__ my_type * generate_on_device(uint64_t cache_capacity, uint64_t ext_seed){

         my_type * host_version = gallatin::utils::get_host_version<my_type>();

         uint64_t ext_n_buckets = (cache_capacity-1)/bucket_size+1;

         host_version->n_buckets_primary = ext_n_buckets;

         //host_version->n_buckets_alt = 10;

         //printf("Iceberg table has %lu total: %lu primary and %lu alt\n", ext_n_buckets, host_version->n_buckets_primary, host_version->n_buckets_alt);

         host_version->primary_buckets = gallatin::utils::get_device_version<bucket_type>(host_version->n_buckets_primary);
         host_version->primary_locks = gallatin::utils::get_device_version<uint64_t>( (host_version->n_buckets_primary-1)/64+1);

         host_version->seed = ext_seed;


         my_type * device_version = gallatin::utils::move_to_device<my_type>(host_version);


         //this is the issue
         init_cuckoo_table_kernel<my_type><<<(ext_n_buckets-1)/256+1,256>>>(device_version);

         cudaDeviceSynchronize();

         return device_version;

      }

      __device__ void init_bucket_and_locks(uint64_t tid){

         if (tid < n_buckets_primary){
            primary_buckets[tid].init();
            unlock_bucket_one_thread_primary(tid);

            //if (tid == n_buckets_primary-1) printf("On last item\n");
         }

      }


      __device__ void stall_lock(tile_type my_tile, uint64_t bucket){

         if (my_tile.thread_rank() == 0){
            stall_lock_one_thread_primary(bucket);
         }

         my_tile.sync();

      }

      __device__ void stall_lock_one_thread_primary(uint64_t bucket){

         #if LOAD_CHEAP
         return;
         #endif

         uint64_t high = bucket/64;
         uint64_t low = bucket % 64;

         // uint64_t current_lock = hash_table_load(&primary_locks[bucket]);


         // while (true){

         //    while (current_lock % 2 != 0){
         //       __threadfence();
         //       current_lock = hash_table_load(&primary_locks[bucket]);
         //    }

         //    uint64_t next_lock = atomicCAS((unsigned long long int *)&primary_locks[bucket], current_lock, current_lock+1);

         //    //acquired! Else its what is currently stored
         //    if (next_lock == current_lock) return;

         //    current_lock = next_lock;
         //    __threadfence();

         // }


         //if old is 0, SET_BIT_MASK & 0 is 0 - loop exit.

         do {
            ADD_PROBE
            #if DEBUG_PRINTS
            printf("Looping on lock %lu\n", bucket);
            #endif
         }
         while (atomicOr((unsigned long long int *)&primary_locks[high], (unsigned long long int) SET_BIT_MASK(low)) & SET_BIT_MASK(low));

         #if DEBUG_PRINTS
         printf("Lock %lu acquired\n", bucket);
         #endif


      }



      __device__ void unlock(tile_type my_tile, uint64_t bucket){

         if (my_tile.thread_rank() == 0){
            unlock_bucket_one_thread_primary(bucket);
         }

         my_tile.sync();

      }


      __device__ void unlock_bucket_one_thread_primary(uint64_t bucket){

         #if LOAD_CHEAP
         return;
         #endif

         uint64_t high = bucket/64;
         uint64_t low = bucket % 64;

         ADD_PROBE

         //if old is 0, SET_BIT_MASK & 0 is 0 - loop exit.
         atomicAnd((unsigned long long int *)&primary_locks[high], (unsigned long long int) ~SET_BIT_MASK(low));
      }



      __device__ void lock_key_buckets(uint64_t bucket_0, uint64_t bucket_1, uint64_t bucket_2){


         if (bucket_0 > bucket_1){
            uint64_t temp = bucket_1;
            bucket_1 = bucket_0;
            bucket_0 = temp;
         }

         //after this, bucket_0 < bucket 1

         if (bucket_1 > bucket_2){
            uint64_t temp = bucket_2;
            bucket_2 = bucket_1;
            bucket_1 = bucket_0;

            //after this point, bucket 2 is the largest key
            //but maybe bucket 1 < bucket_0

            if (bucket_0 > bucket_1){
               temp = bucket_1;
               bucket_1 = bucket_0;
               bucket_0 = temp;
            }


         }



         //from here, locks are ordered
         stall_lock_one_thread_primary(bucket_0);

         if (bucket_1 != bucket_0){
            stall_lock_one_thread_primary(bucket_1);
         }

         if (bucket_2 != bucket_0 && bucket_2 != bucket_1){
            stall_lock_one_thread_primary(bucket_2);
         }



      }


      __device__ void unlock_key_buckets(uint64_t bucket_0, uint64_t bucket_1, uint64_t bucket_2){


         if (bucket_0 > bucket_1){
            uint64_t temp = bucket_1;
            bucket_1 = bucket_0;
            bucket_0 = temp;
         }

         //after this, bucket_0 < bucket 1

         if (bucket_1 > bucket_2){
            uint64_t temp = bucket_2;
            bucket_2 = bucket_1;
            bucket_1 = bucket_0;

            //after this point, bucket 2 is the largest key
            //but maybe bucket 1 < bucket_0

            if (bucket_0 > bucket_1){
               temp = bucket_1;
               bucket_1 = bucket_0;
               bucket_0 = temp;
            }


         }

         if (bucket_0 > bucket_1 || bucket_1 > bucket_2) printf("Bad sort\n");


         //from here, locks are ordered
         unlock_bucket_one_thread_primary(bucket_0);

         if (bucket_1 != bucket_0){
            unlock_bucket_one_thread_primary(bucket_1);
         }

         if (bucket_2 != bucket_0 && bucket_2 != bucket_1){
            unlock_bucket_one_thread_primary(bucket_2);
         }


      }


      //device-side murmurhash64a
      __device__ uint64_t hash ( const void * key, int len, uint64_t seed )
      {
         const uint64_t m = 0xc6a4a7935bd1e995;
         const int r = 47;

         uint64_t h = seed ^ (len * m);

         const uint64_t * data = (const uint64_t *)key;
         const uint64_t * end = data + (len/8);

         while(data != end)
         {
            uint64_t k = *data++;

            k *= m; 
            k ^= k >> r; 
            k *= m; 

            h ^= k;
            h *= m; 
         }

         const unsigned char * data2 = (const unsigned char*)data;

         switch(len & 7)
         {
            case 7: h ^= (uint64_t)data2[6] << 48;
            case 6: h ^= (uint64_t)data2[5] << 40;
            case 5: h ^= (uint64_t)data2[4] << 32;
            case 4: h ^= (uint64_t)data2[3] << 24;
            case 3: h ^= (uint64_t)data2[2] << 16;
            case 2: h ^= (uint64_t)data2[1] << 8;
            case 1: h ^= (uint64_t)data2[0];
                        h *= m;
         };

         h ^= h >> r;
         h *= m;
         h ^= h >> r;

         return h;
      }



      static __host__ void free_on_device(my_type * device_version){

         my_type * host_version = gallatin::utils::move_to_host<my_type>(device_version);

         cudaFree(host_version->primary_buckets);
         cudaFree(host_version->primary_locks);



         cudaFreeHost(host_version);
         
         return;

      }


      __device__ bucket_type * get_bucket_ptr_primary(uint64_t bucket_addr){

         return &primary_buckets[bucket_addr];

      }

      __device__ bucket_type * get_bucket_ptr_alt(uint64_t bucket_addr){

         return &primary_buckets[bucket_addr];

      }


      //boot keys one at a time based on our optimistic route.
      //This takes a single key and it's current bucket and uses that to boot.
      //returns false if the route changed in any detectable way - implies failure in routing and retry needed.
      //As a potential optimization this could take in the address of the next empty location
      // but currently it locks the region and performs a series of atomics
      // this temporarily results in a double copy of a key, but the key can only be manipulated while holding the primary lock
      // so this should be safe.
      __device__ bool boot_key_next(const tile_type & my_tile, const Key & key, int current_bucket_index){

         uint64_t primary_bucket = get_current_bucket(key, 0);

         //lock bucket.
         stall_lock(my_tile, primary_bucket);


         uint64_t current_bucket = get_current_bucket(key, current_bucket_index);
         bucket_type * current_bucket_ptr = get_bucket_ptr_primary(current_bucket);

         uint64_t next_bucket = get_current_bucket(key, (current_bucket_index + 1) % N_CUCKOO_HASHES);

         bucket_type * next_bucket_ptr = get_bucket_ptr_primary(next_bucket);

         Val queried_val;


         uint bucket_match;

         current_bucket_ptr->load_fill_ballots_match(my_tile, key, bucket_match);

         //failed find.
         if (__popc(bucket_match) == 0){

            unlock(my_tile, primary_bucket);
            return false;
         }

         pair_type * val_loc = current_bucket_ptr->query_pair_ballot(my_tile, bucket_match);

         if (my_tile.thread_rank() == 0){
            queried_val = hash_table_load(&val_loc->val);
         }

         queried_val = my_tile.shfl(queried_val, 0);


         uint bucket_empty;
         uint bucket_tombstone;

         next_bucket_ptr->load_fill_ballots(my_tile, key, bucket_empty, bucket_tombstone, bucket_match);


         if (!next_bucket_ptr->insert_ballots(my_tile, key, queried_val, bucket_empty, bucket_tombstone)){

            //not insertable - someone preempted us.
            unlock(my_tile, primary_bucket);
            return false;
         }

         my_tile.sync();

         Val alt_queried_val;

         //this doesn't trigger?
         // if (!next_bucket_ptr->query(my_tile, key, alt_queried_val)){
         //    printf("Failed to cuckoo and query key %lu\n", key); 
         // }

         //at this point the key is moved and the chain can proceed.
         //remove. this cannot fail - so we just write empty!

         if (my_tile.thread_rank() == 0){

            //force st.rel and threadfence to enforce morally strong.
            //should be visible to future threads
            // overwrite of val is not necessary - future inserts will overwrite in a morally strong fashion.
            // gated by lock acquistion of the key in future efforts.

            // if (!gallatin::utils::typed_atomic_write(&val_loc->key, key, tombstoneKey)){
            //    printf("Failed to erase\n");
            // }
            ht_store(&val_loc->key, tombstoneKey);
            __threadfence();

         }

         my_tile.sync();

         unlock(my_tile, primary_bucket);

         return true;

      }


      __device__ bool upsert_function(const tile_type & my_tile, const Key & key, const Val & val, void (*replace_func)(pair_type *, Key, Val)){


         __shared__ vector_type data_vectors[64];

         // uint64_t bucket_0 = hash(&key, sizeof(Key), seed) % n_buckets_primary;
         
         // stall_lock(my_tile, bucket_0);


         // Val * existing_loc = query_reference_nolock(my_tile, key);

         // if (existing_loc != nullptr){


         //    if (my_tile.thread_rank() == 0){

         //       gallatin::utils::st_rel(existing_loc, val);
         //      __threadfence();

         //    }

         //    //this syncs.
         //    unlock(my_tile, bucket_0);
         //    //my_tile.sync();


         //    return true;
         // }


         //unlock(my_tile, bucket_0);

         //if (replace_reference(my_tile, key, val)) return true;



         uint64_t bucket_0 = get_current_bucket(key, 0);

         uint64_t bucket_1 = get_current_bucket(key, 1);
         uint64_t bucket_2 = get_current_bucket(key, 2);

         // if (my_tile.thread_rank() == 0){
         //    lock_key_buckets(bucket_0, bucket_1, bucket_2);
         // }
         // my_tile.sync();


         stall_lock(my_tile, bucket_0);



         //bool upserted = false;
         bool upserted = upsert_primary_buckets_function(my_tile, key, val, bucket_0, bucket_1, bucket_2, replace_func);


         unlock(my_tile, bucket_0);


         // if (my_tile.thread_rank() == 0){
         //    unlock_key_buckets(bucket_0, bucket_1, bucket_2);
         // }
         // my_tile.sync();

         if (upserted) return true;


         //return false;


         int n_iters = 0;

         vector_type * my_vector = &data_vectors[my_tile.meta_group_rank()];

         if (my_tile.thread_rank() == 0){

            my_vector->reset({0,0,0});
         }


         my_tile.sync();


         int n_tries = 0;


         while (n_tries < MAX_CUCKOO_ATTEMPTS){


            //else, unlock and generate

            //bool success = false;

            if (!generate_path(my_tile, key, my_vector)){
               //too full!
               n_tries++;

               if (my_tile.thread_rank() == 0){

                  my_vector->reset({0,0,0});

               }
               __threadfence();

               my_tile.sync();
               continue;
            }


            //verify_path(my_tile, my_vector);

            my_tile.sync();

            bool return_val = upsert_function_internal(my_tile, key, val, my_vector, replace_func);

            my_tile.sync();


            // if (my_tile.thread_rank() == 0){

            //    for (uint64_t i=(sorted_hashes->size-1); i > 0; i--){
            //       unlock_bucket_one_thread_primary((*sorted_hashes)[i].hash);
            //    }

            //    unlock_bucket_one_thread_primary((*sorted_hashes)[0].hash);

            // }


            my_tile.sync();


            if (my_tile.thread_rank() == 0){

               my_vector->reset({0,0,0});

            }



            my_tile.sync();

            if (return_val){

               #if DEBUG_PRINTS
               printf("Exiting success\n");
               #endif

               return true;
            } 

            //printf("Looping\n");
            #if DEBUG_PRINTS
            printf("Exiting failure\n");
            #endif


            n_tries++;

            if (n_tries >= MAX_CUCKOO_ATTEMPTS){

               //if (my_tile.thread_rank() == 0) printf("Exceeded max cuckoo depth\n");
               return false;
            }

            __threadfence();

         }


       }


       __device__ bool upsert_replace(const tile_type & my_tile, const Key & key, const Val & val){


         __shared__ vector_type data_vectors[64];


         uint64_t bucket_0 = get_current_bucket(key, 0);

         uint64_t bucket_1 = get_current_bucket(key, 1);
         uint64_t bucket_2 = get_current_bucket(key, 2);


         stall_lock(my_tile, bucket_0);



         //bool upserted = false;
         bool upserted = upsert_primary_buckets(my_tile, key, val, bucket_0, bucket_1, bucket_2);


         unlock(my_tile, bucket_0);


         if (upserted) return true;


         //int n_iters = 0;

         vector_type * my_vector = &data_vectors[my_tile.meta_group_rank()];

         if (my_tile.thread_rank() == 0){

            my_vector->reset({0,0,0});
         }


         my_tile.sync();


         int n_tries = 0;


         while (n_tries < MAX_CUCKOO_ATTEMPTS){


            if (!generate_path(my_tile, key, my_vector)){
               //too full!
               n_tries++;

               if (my_tile.thread_rank() == 0){

                  my_vector->reset({0,0,0});

               }
               __threadfence();

               my_tile.sync();
               continue;
            }



            my_tile.sync();

            bool return_val = upsert_replace_internal(my_tile, key, val, my_vector);

            my_tile.sync();


            my_tile.sync();


            if (my_tile.thread_rank() == 0){

               my_vector->reset({0,0,0});

            }



            my_tile.sync();

            if (return_val){

               #if DEBUG_PRINTS
               printf("Exiting success\n");
               #endif

               return true;
            } 

            //printf("Looping\n");
            #if DEBUG_PRINTS
            printf("Exiting failure\n");
            #endif


            n_tries++;

            if (n_tries >= MAX_CUCKOO_ATTEMPTS){

               //if (my_tile.thread_rank() == 0) printf("Exceeded max cuckoo depth\n");
               return false;
            }

            __threadfence();

         }


      }


      //primary buckets are already locked, try buckets in order
       //if any space exists done.
      __device__ bool upsert_primary_buckets(const tile_type & my_tile, const Key & key, const Val & val, uint64_t bucket_0, uint64_t bucket_1, uint64_t bucket_2){




         bucket_type * bucket_0_ptr = get_bucket_ptr_primary(bucket_0);


         uint bucket_0_empty;
         uint bucket_0_tombstone;
         uint bucket_0_match;

         bucket_0_ptr->load_fill_ballots(my_tile, key, bucket_0_empty, bucket_0_tombstone, bucket_0_match);

         if (bucket_0_ptr->upsert_existing(my_tile, key, val, bucket_0_match) != -1) return true;

         if (__popc(bucket_0_empty) != 0){

            if (bucket_0_ptr->insert_ballots(my_tile, key, val, bucket_0_empty, bucket_0_tombstone)) return true;

         }

         //if popc(bucket_empty) try pure insert on empty - implies not inserted yet.
         //this should go in load fill ballots.

         uint bucket_1_empty;
         uint bucket_1_tombstone;
         uint bucket_1_match;

         bucket_type * bucket_1_ptr = get_bucket_ptr_primary(bucket_1);

         bucket_1_ptr->load_fill_ballots(my_tile, key, bucket_1_empty, bucket_1_tombstone, bucket_1_match);

         if (bucket_1_ptr->upsert_existing(my_tile, key, val, bucket_1_match) != -1) return true;

         if (__popc(bucket_1_empty) != 0){

            if (bucket_1_ptr->insert_ballots(my_tile, key, val, bucket_1_empty, bucket_1_tombstone)) return true;

         }


         uint bucket_2_empty;
         uint bucket_2_tombstone;
         uint bucket_2_match;

         bucket_type * bucket_2_ptr = get_bucket_ptr_primary(bucket_2);

         bucket_2_ptr->load_fill_ballots(my_tile, key, bucket_2_empty, bucket_2_tombstone, bucket_2_match);

         if (bucket_2_ptr->upsert_existing(my_tile, key, val, bucket_2_match) != -1) return true;

         if (__popc(bucket_2_empty) != 0){

            if (bucket_2_ptr->insert_ballots(my_tile, key, val, bucket_2_empty, bucket_2_tombstone)) return true;

         }


         if (bucket_0_ptr->insert_ballots(my_tile, key, val, bucket_0_empty, bucket_0_tombstone)) return true;

         if (bucket_1_ptr->insert_ballots(my_tile, key, val, bucket_1_empty, bucket_1_tombstone)) return true;

         if (bucket_2_ptr->insert_ballots(my_tile, key, val, bucket_2_empty, bucket_2_tombstone)) return true;

         return false;

      }


      __device__ bool upsert_primary_buckets_function(const tile_type & my_tile, const Key & key, const Val & val, uint64_t bucket_0, uint64_t bucket_1, uint64_t bucket_2, void (*replace_func)(pair_type *, Key, Val)){




         bucket_type * bucket_0_ptr = get_bucket_ptr_primary(bucket_0);


         uint bucket_0_empty;
         uint bucket_0_tombstone;
         uint bucket_0_match;

         bucket_0_ptr->load_fill_ballots(my_tile, key, bucket_0_empty, bucket_0_tombstone, bucket_0_match);

         if (bucket_0_ptr->upsert_existing_func(my_tile, key, val, bucket_0_match, replace_func) != -1) return true;

         if (__popc(bucket_0_empty) != 0){

            if (bucket_0_ptr->insert_ballots(my_tile, key, val, bucket_0_empty, bucket_0_tombstone)) return true;

         }

         //if popc(bucket_empty) try pure insert on empty - implies not inserted yet.
         //this should go in load fill ballots.

         uint bucket_1_empty;
         uint bucket_1_tombstone;
         uint bucket_1_match;

         bucket_type * bucket_1_ptr = get_bucket_ptr_primary(bucket_1);

         bucket_1_ptr->load_fill_ballots(my_tile, key, bucket_1_empty, bucket_1_tombstone, bucket_1_match);

         if (bucket_1_ptr->upsert_existing_func(my_tile, key, val, bucket_1_match, replace_func) != -1) return true;

         if (__popc(bucket_1_empty) != 0){

            if (bucket_1_ptr->insert_ballots(my_tile, key, val, bucket_1_empty, bucket_1_tombstone)) return true;

         }


         uint bucket_2_empty;
         uint bucket_2_tombstone;
         uint bucket_2_match;

         bucket_type * bucket_2_ptr = get_bucket_ptr_primary(bucket_2);

         bucket_2_ptr->load_fill_ballots(my_tile, key, bucket_2_empty, bucket_2_tombstone, bucket_2_match);

         if (bucket_2_ptr->upsert_existing_func(my_tile, key, val, bucket_2_match, replace_func) != -1) return true;

         if (__popc(bucket_2_empty) != 0){

            if (bucket_2_ptr->insert_ballots(my_tile, key, val, bucket_2_empty, bucket_2_tombstone)) return true;

         }


         if (bucket_0_ptr->insert_ballots(my_tile, key, val, bucket_0_empty, bucket_0_tombstone)) return true;

         if (bucket_1_ptr->insert_ballots(my_tile, key, val, bucket_1_empty, bucket_1_tombstone)) return true;

         if (bucket_2_ptr->insert_ballots(my_tile, key, val, bucket_2_empty, bucket_2_tombstone)) return true;

         return false;

      }

       // __device__ void release_vector(const tile_type & my_tile, vector_type & sort_vector){

       //   if (my_tile.thread_rank() == 0){
       //      sort_vector.free_vector();
       //   }

       //   my_tile.sync();
       // }

      __device__ void acquire_locks(const tile_type & my_tile, vector_type & sorted_hashes){

         if (my_tile.thread_rank() == 0){


               for (uint64_t i = 0; i < sorted_hashes.size; i++){
                  stall_lock_one_thread_primary(sorted_hashes[i].hash);
               }


               printf("All locks acquired\n");

         }


         my_tile.sync();


       }


      __device__ void verify_sort(const tile_type & my_tile, vector_type * sorted_hashes){

         if (my_tile.thread_rank() == 0){


               for (uint64_t i = 1; i < sorted_hashes->size; i++){

                  if ((*sorted_hashes)[i-1].hash >= (*sorted_hashes)[i].hash) printf("Bad sort\n");
               }

         }

         my_tile.sync();


       }

      __device__ void release_locks(const tile_type & my_tile, vector_type & sorted_hashes){

         if (my_tile.thread_rank() == 0){


               for (uint64_t i = 0; i < sorted_hashes.size; i++){
                  unlock_bucket_one_thread_primary(sorted_hashes[i].hash);
               }

         }

         my_tile.sync();


       }

       __device__ void sort_locks(const tile_type & my_tile, vector_type * lock_vector, vector_type * sort_vector){


         // if (lock_vector.size >= 2){
         //    printf("In function that's interesting\n");
         // }

         if (my_tile.thread_rank() == 0){

            //sort locks for acquisition.

            //sort_vector.init(CUCKOO_MAX_PROBES);

            uint64_t hash_min = ~0ULL;
            Key min_key;

            for (int i = 0; i < lock_vector->size; i++){

               //calculate min.
               if ((*lock_vector)[i].hash < hash_min){
                  hash_min = (*lock_vector)[i].hash;
                  min_key = (*lock_vector)[i].key;
               } 


            }


            sort_vector->push_back({hash_min, min_key});

            //uint64_t smallest_current_hash = 0;

            while (true){

               uint64_t smallest_current_hash = ~0ULL;

               for (int i = 0; i < lock_vector->size; i++){

                  if ((*lock_vector)[i].hash > hash_min && (*lock_vector)[i].hash < smallest_current_hash){
                     smallest_current_hash = (*lock_vector)[i].hash;
                     min_key = (*lock_vector)[i].key;

                  }


               }


               if (smallest_current_hash == ~0ULL) break;

               if (hash_min >= smallest_current_hash){
                  printf("Bad hash sort\n");
               }

               hash_min = smallest_current_hash;

               sort_vector->push_back({hash_min, min_key});

               //printf("Looping in lock sort, smallest is %lx\n", hash_min);
               


            }

         }


         //acquire locks.  
         //return sort_vector;

       }


       // __device__ int next_bucket(const Key & key, uint64_t current_bucket_hash){

       //   for (int i = 0; i < N_CUCKOO_HASHES; i++){

       //      uint64_t ith_hash = hash(&key, sizeof(Key), seed+i) % n_buckets_primary;

       //      if (ith_hash == current_bucket_hash) return (i + 1) % N_CUCKOO_HASHES;

       //   }

       //   asm volatile("trap;");

       //   return -1;

       // }


      __device__ uint64_t get_next_bucket(const Key & key, int bucket_id){

         Key working_key = key;

         bucket_id = (bucket_id+1) % N_CUCKOO_HASHES;

         return hash(&working_key, sizeof(Key), seed + bucket_id) % n_buckets_primary;
      }

      __device__ int get_current_bucket(const Key & key, int bucket_id){

         Key working_key = key;

         bucket_id = bucket_id % N_CUCKOO_HASHES;

         return hash(&working_key, sizeof(Key), seed + bucket_id) % n_buckets_primary;
       }


      __device__ int get_bucket_id_from_hash(const Key & key, uint64_t current_bucket_hash){

         for (int i = 0; i < N_CUCKOO_HASHES; i++){

            Key working_key = key;

            uint64_t ith_hash = hash(&working_key, sizeof(Key), seed+i) % n_buckets_primary;

            if (ith_hash == current_bucket_hash) return i;

         }

         //asm volatile("trap;");

         //printf("Failed to generate get_bucket_id_from_hash\n");

         return -1;

       }



       // __device__ uint64_t hash_with_cuckoo_num(const Key & key, int bucket_id){

       //   Key working_key = key;

       //   uint64_t hash = hash(&working_key, sizeof(Key), seed+bucket_id) % n_buckets_primary;

       //   return hash;

       // }


      __device__ void verify_path(const tile_type & my_tile, vector_type & item_vector){


         if (my_tile.thread_rank() == 0){

            for (int i = 1; i < item_vector.size; i++){

               Key previous_key = item_vector[i-1].key;

               int current_bucket_id = item_vector[i-1].bucket_id;

               uint64_t next_hash = item_vector[i].hash;

               if (get_next_bucket(previous_key, current_bucket_id) != next_hash){
                  printf("Hashes don't line up\n");
               }

            }

         }


      }

       //does not contain the last item (tombstone)
       __device__ bool generate_path(const tile_type & my_tile, const Key & key, vector_type * item_vector){

         Key working_key = key;
         //setting it to this value wraps get_next to 0.
         int current_bucket_id = 2;

         // uint64_t next_bucket = get_current_bucket(working_key, current_bucket_id);

         // uint64_t final_query_bucket = get_current_bucket(working_key, 2);

         // if (my_tile.thread_rank() == 0){

         //    item_vector->push_back({final_query_bucket, working_key, 2});


         //    item_vector->push_back({next_bucket, working_key, current_bucket_id});

         // }

         //acts as exiting while loop
         for (int i =0; i < CUCKOO_MAX_PROBES; i++){


            //uint64_t current_bucket = get_current_bucket(working_key, current_bucket_id);

            uint64_t next_bucket = get_next_bucket(working_key, current_bucket_id);

            //bucket_type * bucket_primary_ptr = get_bucket_ptr_primary(current_bucket);

            bucket_type * next_bucket_ptr = get_bucket_ptr_primary(next_bucket);

            uint primary_empty;
            uint primary_tombstone;
            uint primary_match;

            //ADD_PROBE_ADJUSTED

            //load
            next_bucket_ptr->load_fill_ballots(my_tile, key, primary_empty, primary_tombstone, primary_match);

            if (__popc(primary_tombstone) != 0){
               // if (my_tile.thread_rank() == 0){
               //    item_vector.push_back(bucket_primary, tombstoneKey, -1);
               // }

               return true;
            }

            if (__popc(primary_empty) != 0){

               // if (my_tile.thread_rank() == 0){
               //    item_vector.push_back(bucket_primary, defaultKey, -1);
               // }
               return true;
          
            }

            //else get key

            if (my_tile.thread_rank() == 0){

                  uint64_t random_address = clock64() % bucket_size;

                  working_key = hash_table_load(&next_bucket_ptr->slots[random_address].key);


            }

            working_key = my_tile.shfl(working_key, 0);

            current_bucket_id = get_bucket_id_from_hash(working_key, next_bucket);

            if (working_key == tombstoneKey || working_key == defaultKey) return true;

            //hash = get_current_bucket(working_key, 0);

            // if (current_bucket_id == -1 && my_tile.thread_rank() == 0){
            //    printf("Failed movement for key %lu\n", working_key);

            //    return false;
            // }

            if (my_tile.thread_rank() == 0){
                item_vector->push_back({next_bucket, working_key, current_bucket_id});
            }



         }

         //printf("Exceeded max depth!\n");

         return false;

       }

      //start by booting all keys, then circle back and see if there is a replacement we can do once our lock is acquired.
      __device__ bool upsert_replace_internal(const tile_type & my_tile, const Key & key, const Val & val, vector_type * order_vector){



         uint64_t size = my_tile.shfl(order_vector->size,0);

         //ignore current.
         for (int i = size-1; i>=0; i--){
            //shuffle data

            Key current_key = (*order_vector)[i].key;
            int current_bucket_id = (*order_vector)[i].bucket_id;

            // if (my_tile.thread_rank() == 0){
            //    current_key = (*order_vector)[i].key;
            //    current_bucket_id = (*order_vector)[i].bucket_id;
            // } 

            // current_key = my_tile.shfl(current_key, 0);
            // current_bucket_id = my_tile.shfl(current_bucket_id, 0);

            if (!boot_key_next(my_tile, current_key, current_bucket_id)){
               return false;
            }


         }

         //insert final key.
         uint64_t bucket_0 = get_current_bucket(key, 0);
         uint64_t bucket_1 = get_current_bucket(key, 1);
         uint64_t bucket_2 = get_current_bucket(key, 2);

         //bucket_type * current_bucket_ptr = get_bucket_ptr_primary(current_bucket);


         stall_lock(my_tile, bucket_0);

         bool result = upsert_primary_buckets(my_tile, key, val, bucket_0, bucket_1, bucket_2);

         unlock(my_tile, bucket_0);
         return result;


      }

      __device__ bool upsert_function_internal(const tile_type & my_tile, const Key & key, const Val & val, vector_type * order_vector, void (*replace_func)(pair_type *, Key, Val)){



         uint64_t size = my_tile.shfl(order_vector->size,0);

         //ignore current.
         for (int i = size-1; i>=0; i--){
            //shuffle data

            Key current_key = (*order_vector)[i].key;
            int current_bucket_id = (*order_vector)[i].bucket_id;

            // if (my_tile.thread_rank() == 0){
            //    current_key = (*order_vector)[i].key;
            //    current_bucket_id = (*order_vector)[i].bucket_id;
            // } 

            // current_key = my_tile.shfl(current_key, 0);
            // current_bucket_id = my_tile.shfl(current_bucket_id, 0);

            if (!boot_key_next(my_tile, current_key, current_bucket_id)){
               return false;
            }


         }

         //insert final key.
         uint64_t bucket_0 = get_current_bucket(key, 0);
         uint64_t bucket_1 = get_current_bucket(key, 1);
         uint64_t bucket_2 = get_current_bucket(key, 2);

         //bucket_type * current_bucket_ptr = get_bucket_ptr_primary(current_bucket);


         stall_lock(my_tile, bucket_0);

         bool result = upsert_primary_buckets_function(my_tile, key, val, bucket_0, bucket_1, bucket_2, replace_func);

         unlock(my_tile, bucket_0);
         return result;


      }


      //find the reference of the value if it exists
      __device__ Val * query_reference(tile_type my_tile, Key key){


         return nullptr;

         //loop through hashes




         //return nullptr;


         // uint64_t bucket_primary = hash(&key, sizeof(Key), seed) % n_buckets_primary;


         // bucket_type * bucket_primary_ptr = get_bucket_ptr_primary(bucket_primary);

         // uint primary_empty;
         // uint primary_tombstone;
         // uint primary_match;

         // bucket_primary_ptr->load_fill_ballots(my_tile, key, primary_empty, primary_tombstone, primary_match);

         // Val * val_loc = bucket_primary_ptr->query_ptr_ballot(my_tile, primary_match);

         // if (val_loc != nullptr) return val_loc;

         // //shortcutting
         // if (__popc(primary_empty) > 0) return nullptr;


         // uint64_t primary_step = hash(&key, sizeof(Key), seed+1);
         // uint64_t alternate_step = hash(&key, sizeof(Key), seed+2);

         // //setup for alterna


         // for (int i = 0; i < CUCKOO_MAX_PROBES; i++){


         //    uint64_t bucket_index = (primary_step + alternate_step*i) % n_buckets_alt;
         //    bucket_type * bucket_ptr = get_bucket_ptr_alt(bucket_index);
      
         //    uint bucket_empty;
         //    uint bucket_tombstone;
         //    uint bucket_match;


         //    bucket_ptr->load_fill_ballots(my_tile, key, bucket_empty, bucket_tombstone, bucket_match);

         //    val_loc = bucket_primary_ptr->query_ptr_ballot(my_tile, primary_match);

         //    if (val_loc != nullptr) return val_loc;

         //    //shortcutting
         //    if (__popc(primary_empty) > 0) return nullptr;


         // }


         // return nullptr;

      }

      __device__ pair_type * query_packed_reference(tile_type my_tile, Key key){

         for (int i = 0; i < N_CUCKOO_HASHES; i++){
            uint64_t bucket = get_current_bucket(key, i);

            bucket_type * bucket_ptr = get_bucket_ptr_primary(bucket);
      
            //uint bucket_empty;
            //uint bucket_tombstone;
            uint bucket_match;




            bucket_ptr->load_fill_ballots_match(my_tile, key, bucket_match);


            pair_type * val_loc = bucket_ptr->query_pair_ballot(my_tile, bucket_match);

            if (val_loc != nullptr) return val_loc;


         }

         return nullptr;


      }

      __device__ pair_type * find_pair_no_lock(tile_type my_tile, Key key){

         uint64_t lock_bucket = get_current_bucket(key, 0);

         stall_lock(my_tile, lock_bucket);

         for (int i = 0; i < N_CUCKOO_HASHES; i++){
            uint64_t bucket = get_current_bucket(key, i);


            bucket_type * bucket_ptr = get_bucket_ptr_primary(bucket);

      
            //uint bucket_empty;
            //uint bucket_tombstone;
            uint bucket_match;




            bucket_ptr->load_fill_ballots_match(my_tile, key, bucket_match);


            pair_type * val_loc = bucket_ptr->query_pair_ballot(my_tile, bucket_match);

            if (val_loc != nullptr){
               unlock(my_tile, lock_bucket);
               return val_loc;
            }


         }

         unlock(my_tile, lock_bucket);
         return nullptr;


      }

      __device__ uint64_t get_lock_bucket(tile_type my_tile, Key key){

         uint64_t my_slot = get_current_bucket(key, 0);

         return my_slot;

      }

      __device__ pair_type * find_pair_no_lock_2(tile_type my_tile, Key key){

         for (int i = 0; i < N_CUCKOO_HASHES; i++){
            uint64_t bucket = get_current_bucket(key, i);

            bucket_type * bucket_ptr = get_bucket_ptr_primary(bucket);
      
            //uint bucket_empty;
            //uint bucket_tombstone;
            uint bucket_match;




            bucket_ptr->load_fill_ballots_match(my_tile, key, bucket_match);


            pair_type * val_loc = bucket_ptr->query_pair_ballot(my_tile, bucket_match);

            if (val_loc != nullptr) return val_loc;


         }

         return nullptr;


      }


      __device__ bool replace_reference(tile_type my_tile, Key key, Val val){


         // for (int i = 0; i < N_CUCKOO_HASHES; i++){

         //    uint64_t current_bucket = get_current_bucket(key, i);

         //    stall_lock(current_bucket);


         // }


         pair_type * found_pair = query_packed_reference(my_tile, key);


         if (found_pair == nullptr){

            return false;


         }

         ht_store(&found_pair->val, val);
         //gallatin::utils::st_rel(&found_pair->val, val);

         return true;


      }


      __device__ bool query_internal(tile_type my_tile, Key key, Val & val, uint64_t bucket_0, uint64_t bucket_1, uint64_t bucket_2){


         bucket_type * bucket_0_ptr = get_bucket_ptr_primary(bucket_0);
         bucket_type * bucket_1_ptr = get_bucket_ptr_primary(bucket_1);
         bucket_type * bucket_2_ptr = get_bucket_ptr_primary(bucket_2);

         if (bucket_0_ptr->query(my_tile, key, val)) return true;
         if (bucket_1_ptr->query(my_tile, key, val)) return true;
         if (bucket_2_ptr->query(my_tile, key, val)) return true;


         return false;



      }


      // //nope! no storage
      [[nodiscard]] __device__ bool find_with_reference(tile_type my_tile, Key key, Val & val){


         // for (int i = 0; i < N_CUCKOO_HASHES; i++){

         //    uint64_t current_bucket = get_current_bucket(key, i);

         //    stall_lock(current_bucket);


         // }

         uint64_t bucket_0 = get_current_bucket(key, 0);
         uint64_t bucket_1 = get_current_bucket(key, 1);
         uint64_t bucket_2 = get_current_bucket(key, 2);

         // if (my_tile.thread_rank() == 0){
         //    lock_key_buckets(bucket_0, bucket_1, bucket_2);
         // }
         // my_tile.sync();

         stall_lock(my_tile, bucket_0);

         bool found = query_internal(my_tile, key, val, bucket_0, bucket_1, bucket_2);

         // if (my_tile.thread_rank() == 0){
         //    unlock_key_buckets(bucket_0, bucket_1, bucket_2);
         // }
         // my_tile.sync();

         unlock(my_tile, bucket_0);

         return found;





      }


      [[nodiscard]] __device__ bool find_with_reference_no_lock(tile_type my_tile, Key key, Val & val){


         // for (int i = 0; i < N_CUCKOO_HASHES; i++){

         //    uint64_t current_bucket = get_current_bucket(key, i);

         //    stall_lock(current_bucket);


         // }

         for (int i = 0; i < N_CUCKOO_HASHES; i++){

            uint64_t current_bucket = get_current_bucket(key, i);

            bucket_type * current_bucket_ptr = get_bucket_ptr_primary(current_bucket);


            if (current_bucket_ptr->query(my_tile, key, val)){

            
               return true;
            }


         }

         return false;


      }

      __device__ bool remove(tile_type my_tile, Key key){


         uint64_t bucket_0 = get_current_bucket(key, 0);
         uint64_t bucket_1 = get_current_bucket(key, 1);
         uint64_t bucket_2 = get_current_bucket(key, 2);

         // if (my_tile.thread_rank() == 0){
         //    lock_key_buckets(bucket_0, bucket_1, bucket_2);
         // }
         // my_tile.sync();

         stall_lock(my_tile, bucket_0);



         pair_type * found_pair = query_packed_reference(my_tile, key);

         if (found_pair == nullptr){

            unlock(my_tile, bucket_0);
            return false;
         }

         bool ballot = false;

         //erase
         if (my_tile.thread_rank() == 0){

            ADD_PROBE
            ballot = typed_atomic_write(&found_pair->key, key, tombstoneKey);
            //if (ballot){

               //force store
               //gallatin::utils::st_rel(&found_pair->val, tombstoneVal);
               //typed_atomic_exchange(&slots[i].val, ext_val);
            //}

         }

         __threadfence();

        
         bool result = my_tile.ballot(ballot);
         if (result){
            
            unlock(my_tile, bucket_0);

            return true;
         }


         //printf("Looping in remove\n");

         unlock(my_tile, bucket_0);

         return false;





      }

      // __device__ bool upsert(tile_type my_tile, Key old_key, Val old_val, Key new_key, Val new_val){
      //    return internal_table->upsert_exact(my_tile, old_key, old_val, new_key, new_val);
      // }

      // __device__ bool upsert(tile_type my_tile, pair_type old_pair, pair_type new_pair){
      //    return upsert(my_tile, old_pair.first, old_pair.second, new_pair.first, new_pair.second);
      // }

      // __device__ bool insert_if_not_exists(tile_type my_tile, Key key, Val val){
      //    return internal_table->insert_exact(my_tile, key, val);
      // }
      
      // __device__ pair_type find_replaceable_pair(tile_type my_tile, Key key){
      //    return internal_table->find_smaller_hash(my_tile, key);
      // }

      static __device__ pair_type pack_together(Key key, Val val){
         return pair_type{key, val};
      }

      __host__ float load(){

         return 0;

      }

      static std::string get_name(){
         return "cuckoo_hashing";
      }


      __host__ void print_space_usage(){

         my_type * host_version = gallatin::utils::copy_to_host<my_type>(this);
            
         uint64_t capacity = host_version->n_buckets_primary*sizeof(bucket_type) + (host_version->n_buckets_primary-1)/8+1; 

         cudaFreeHost(host_version);

         printf("Cuckoo using %lu bytes\n", capacity);

      }

      __host__ void print_fill(){
      
         uint64_t * n_items;

         cudaMallocManaged((void **)&n_items, sizeof(uint64_t));

         n_items[0] = 0;

         my_type * host_version = gallatin::utils::copy_to_host<my_type>(this);

         uint64_t n_buckets = host_version->n_buckets_primary;


         cuckoo_get_fill_kernel<my_type, partition_size><<<(n_buckets*partition_size-1)/256+1,256>>>(this, n_buckets, n_items);

         cudaDeviceSynchronize();
         printf("fill: %lu/%lu = %f%%\n", n_items[0], n_buckets*bucket_size, 100.0*n_items[0]/(n_buckets*bucket_size));

         cudaFree(n_items);
         cudaFreeHost(host_version);


      }

      __device__ uint64_t get_bucket_fill(tile_type my_tile, uint64_t bucket){


         bucket_type * bucket_ptr = get_bucket_ptr_primary(bucket);


         uint32_t bucket_empty;
         uint32_t bucket_tombstone;
         uint32_t bucket_match;

         bucket_ptr->load_fill_ballots(my_tile, defaultKey, bucket_empty, bucket_tombstone, bucket_match);

         return bucket_size-__popc(bucket_empty) - __popc(bucket_tombstone);

      }

      __host__ uint64_t get_num_locks(){

         my_type * host_version = gallatin::utils::copy_to_host<my_type>(this);

         uint64_t nblocks = host_version->n_buckets_primary;

         cudaFreeHost(host_version);

         return nblocks;
      }


   };

template <typename T>
constexpr T generate_cuckoo_tombstone(uint64_t offset) {
  return (~((T) 0)) - offset;
};

template <typename T>
constexpr T generate_cuckoo_sentinel() {
  return ((T) 0);
};


// template <typename Key, Key sentinel, Key tombstone, typename Val, Val defaultVal, Val tombstoneVal, uint partition_size, uint bucket_size>
  
template <typename Key, typename Val, uint tile_size, uint bucket_size>
using cuckoo_generic = typename warpSpeed::tables::cuckoo_table<Key,
                                    generate_cuckoo_sentinel<Key>(),
                                    generate_cuckoo_tombstone<Key>(0),
                                    Val,
                                    generate_cuckoo_sentinel<Val>(),
                                    generate_cuckoo_tombstone<Val>(0),
                                    tile_size,
                                    bucket_size>;




} //namespace wrappers

}  // namespace ht_project

#endif  // GPU_BLOCK_