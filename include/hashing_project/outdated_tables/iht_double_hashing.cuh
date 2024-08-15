#ifndef OUT_IHT_DOUBLE
#define OUT_IHT_DOUBLE

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <gallatin/allocators/alloc_utils.cuh>
#include <hashing_project/helpers/ht_load.cuh>


#include "assert.h"
#include "stdio.h"

namespace cg = cooperative_groups;

// helper_macro
// define macros
#define MAX_VALUE(nbits) ((1ULL << (nbits)) - 1)
#define BITMASK(nbits) ((nbits) == 64 ? 0xffffffffffffffff : MAX_VALUE(nbits))

#define SET_BIT_MASK(index) ((1ULL << index))

// a pointer list managing a set section of device memory

#define FRONT_TOTAL_RATIO .9
//#define BACK_PROBES 20


//cache protocol
//query cache
//on success add to pin?
//need delete from potential buckets implementation - need to download warpcore...
//buidld with primary p2bht first.



namespace hashing_project {

namespace tables {


   //investigate this.
   // __device__ inline void st_rel(const uint64_t *p, uint64_t store_val) {
  
   //   asm volatile("st.gpu.release.u64 [%0], %1;" :: "l"(p), "l"(store_val) : "memory");

   //   // return atomicOr((unsigned long long int *)p, 0ULL);

   //   // atom{.sem}{.scope}{.space}.cas.b128 d, [a], b, c {, cache-policy};
   // }


  


   template <typename Key, typename Val>
   struct iht_pair{
      Key key;
      Val val;
   };


   template <typename Key, Key defaultKey, Key tombstoneKey, typename Val, Val defaultVal, Val tombstoneVal, uint partition_size, uint bucket_size>
   struct iht_bucket {

      //uint64_t lock_and_size;

      using pair_type = iht_pair<Key, Val>;

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



      // __device__ int insert(Key ext_key, Val ext_val, cg::thread_block_tile<partition_size> my_tile){


      //    //first read size
      //    // internal_read_size = gallatin::utils::ldcv(&size);

      //    // //failure means resize has started...
      //    // if (internal_read_size != expected_size) return false;

      //    for (int i = my_tile.thread_rank(); i < n_traversals; i+=my_tile.size()){

      //       bool key_match = (i < num_pairs);

      //       Key loaded_key;

      //       if (key_match) loaded_key = hash_table_load(&slots[i].key);

      //       //early drop if loaded_key is gone
      //       bool ballot = key_match && (loaded_key == defaultKey);

      //       auto ballot_result = my_tile.ballot(ballot);

      //          while (ballot_result){

      //             ballot = false;

      //             const auto leader = __ffs(ballot_result)-1;

      //             if (leader == my_tile.thread_rank()){


      //                ballot = typed_atomic_write(&slots[i].key, defaultKey, ext_key);
      //                if (ballot){

      //                   gallatin::utils::st_rel(&slots[i].val, ext_val);
      //                   //typed_atomic_exchange(&slots[i].val, ext_val);
      //                }
      //             } 

     

      //             //if leader succeeds return
      //             if (my_tile.ballot(ballot)){
      //                return __ffs(my_tile.ballot(ballot))-1;
      //             }
                  

      //             //if we made it here no successes, decrement leader
      //             ballot_result  ^= 1UL << leader;

      //             //printf("Stalling in insert_into_bucket keys\n");

      //          }

      //          ballot = key_match && (loaded_key == tombstoneKey);

      //          ballot_result = my_tile.ballot(ballot);

      //          while (ballot_result){

      //             ballot = false;

      //             const auto leader = __ffs(ballot_result)-1;

      //             if (leader == my_tile.thread_rank()){
      //                ballot = typed_atomic_write(&slots[i].key, tombstoneKey, ext_key);
      //                if (ballot){

      //                   //loop and wait on tombstone val to be done.

      //                   Val loaded_val = hash_table_load(&slots[i].val);

      //                   while(loaded_val != tombstoneVal){
      //                      loaded_val = hash_table_load(&slots[i].val);
      //                      __threadfence();
      //                   }

      //                   __threadfence();

      //                   gallatin::utils::st_rel(&slots[i].val, ext_val);
      //                   //typed_atomic_write(&slots[i].val, ext_val);
      //                }
      //             } 

     

      //             //if leader succeeds return
      //             if (my_tile.ballot(ballot)){
      //                return __ffs(my_tile.ballot(ballot))-1;
      //             }
                  

      //             //if we made it here no successes, decrement leader
      //             ballot_result  ^= 1UL << leader;

      //             //printf("Stalling in insert_into_bucket\n");
      //             //printf("Stalling in insert_into_bucket tombstone\n");

      //          }


      //    }


      //    return -1;

      // }

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
               bool ballot_exists = false;

               const auto leader = __ffs(ballot_result)-1;


               if (leader == my_tile.thread_rank() && empty_ballot){

                  ballot_exists = true;

                  ballot = typed_atomic_write(&slots[i].key, defaultKey, ext_key);
                  if (ballot){

                     gallatin::utils::st_rel(&slots[i].val, ext_val);
                     //typed_atomic_exchange(&slots[i].val, ext_val);
                  }
               } 

  

               //if leader succeeds return
               if (my_tile.ballot(ballot_exists)){
                  return my_tile.ballot(ballot);
               }
               

               //check tombstone

               if (leader == my_tile.thread_rank() && tombstone_ballot){

                  ballot_exists = true;

                  ballot = typed_atomic_write(&slots[i].key, tombstoneKey, ext_key);

                  if (ballot){

                     //loop and wait on tombstone val to be done.

                     Val loaded_val = hash_table_load(&slots[i].val);

                     while(loaded_val != tombstoneVal){

                        //this may be an issue if a stored value is legitimately a tombstone - need special logic in delete?
                        loaded_val = hash_table_load(&slots[i].val);
                        __threadfence();
                     }

                     __threadfence();

                     gallatin::utils::st_rel(&slots[i].val, ext_val);


                  }

               }

               //if leader succeeds return
               if (my_tile.ballot(ballot_exists)){
                  return my_tile.ballot(ballot);
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


                  ballot = typed_atomic_write(&slots[i].key, ext_key, ext_key);
                  if (ballot){

                     gallatin::utils::st_rel(&slots[i].val, ext_val);
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
      //allowing one hash_table_load of the bucket to service all requests.
      __device__ void load_fill_ballots(const cg::thread_block_tile<partition_size> & my_tile, const Key & upsert_key, __restrict__ uint & empty_match, __restrict__ uint & tombstone_match, __restrict__ uint & key_match){


         //wipe previous
         empty_match = 0U;
         tombstone_match = 0U;
         key_match = 0U;

         int my_count = 0;

         for (uint i = my_tile.thread_rank(); i < n_traversals; i+=my_tile.size()){

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

            empty_match |= (my_tile.ballot(found_empty) << offset);
            tombstone_match |= (my_tile.ballot(found_tombstone) << offset);
            key_match |= (my_tile.ballot(found_exact) << offset);

         }

         return;

      }



      __device__ bool query(const cg::thread_block_tile<bucket_size> & my_tile, Key ext_key, Val & return_val){


         for (uint i = my_tile.thread_rank(); i < n_traversals; i+=my_tile.size()){

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

      __device__ Val * query_ptr_ballot(const cg::thread_block_tile<bucket_size> & my_tile, __restrict__ uint & match_ballot){

         //Val * return_val = nullptr;


            int found = __ffs(match_ballot)-1;

            if (found == -1) return nullptr;

            //return_val = my_tile.shfl(loaded_val, found);

            return &slots[found].val;


      }

      __device__ pair_type * query_pair_ballot(const cg::thread_block_tile<bucket_size> & my_tile, __restrict__ uint & match_ballot){

         //Val * return_val = nullptr;


            int found = __ffs(match_ballot)-1;

            if (found == -1) return nullptr;

            //return_val = my_tile.shfl(loaded_val, found);

            return &slots[found];


      }

      __device__ bool erase(cg::thread_block_tile<bucket_size> my_tile, Key ext_key){


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


                  ballot = typed_atomic_write(&slots[i].key, ext_key, tombstoneKey);
                  if (ballot){

                     //force store
                     gallatin::utils::st_rel(&slots[i].val, tombstoneVal);
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
   __global__ void init_iht_table_kernel(table * hash_table){

      uint64_t tid = gallatin::utils::get_tid();

      hash_table->init_bucket_and_locks(tid);
      

   }



   template <typename Key, Key defaultKey, Key tombstoneKey, typename Val, Val defaultVal, Val tombstoneVal, uint partition_size, uint bucket_size>
   struct iht_double_table {


      using my_type = iht_double_table<Key, defaultKey, tombstoneKey, Val, defaultVal, tombstoneVal, partition_size, bucket_size>;


      using tile_type = cg::thread_block_tile<bucket_size>;

      using bucket_type = iht_bucket<Key, defaultKey, tombstoneKey, Val, defaultVal, tombstoneVal, partition_size, bucket_size>;

      using packed_pair_type = iht_pair<Key, Val>;

      bucket_type * primary_buckets;
      bucket_type * alt_buckets;
      uint64_t * primary_locks;
      uint64_t * alt_locks;

      uint64_t n_buckets_primary;
      uint64_t n_buckets_alt;
      uint64_t seed;

      //dummy handle
      static __host__ my_type * generate_on_device(uint64_t cache_capacity, uint64_t ext_seed){

         my_type * host_version = gallatin::utils::get_host_version<my_type>();

         uint64_t ext_n_buckets = (cache_capacity-1)/bucket_size+1;

         host_version->n_buckets_primary = ext_n_buckets*FRONT_TOTAL_RATIO;
         host_version->n_buckets_alt = ext_n_buckets*(1.0-FRONT_TOTAL_RATIO)+1;

         //host_version->n_buckets_alt = 10;

         printf("Iceberg table has %lu total: %lu primary and %lu alt\n", ext_n_buckets, host_version->n_buckets_primary, host_version->n_buckets_alt);

         host_version->primary_buckets = gallatin::utils::get_device_version<bucket_type>(host_version->n_buckets_primary);
         host_version->alt_buckets = gallatin::utils::get_device_version<bucket_type>(host_version->n_buckets_alt);

         host_version->primary_locks = gallatin::utils::get_device_version<uint64_t>( (host_version->n_buckets_primary-1)/64+1);
         host_version->alt_locks = gallatin::utils::get_device_version<uint64_t>( (host_version->n_buckets_alt-1)/64+1);

         host_version->seed = ext_seed;


         uint64_t n_buckets = host_version->n_buckets_primary+host_version->n_buckets_alt;

         my_type * device_version = gallatin::utils::move_to_device<my_type>(host_version);


         //this is the issue
         init_iht_table_kernel<my_type><<<(n_buckets-1)/256+1,256>>>(device_version);

         cudaDeviceSynchronize();

         return device_version;

      }

      __device__ void init_bucket_and_locks(uint64_t tid){

         if (tid < n_buckets_primary){
            primary_buckets[tid].init();
            unlock_bucket_one_thread_primary(tid);

            //if (tid == n_buckets_primary-1) printf("On last item\n");
         }

         if (tid < n_buckets_alt){

            alt_buckets[tid].init();
            unlock_bucket_one_thread_alt(tid);
         }

      }


      __device__ void stall_lock_primary(tile_type my_tile, uint64_t bucket){

         if (my_tile.thread_rank() == 0){
            stall_lock_one_thread_primary(bucket);
         }

         my_tile.sync();

      }

      __device__ void stall_lock_one_thread_primary(uint64_t bucket){

         uint64_t high = bucket/64;
         uint64_t low = bucket % 64;

         //if old is 0, SET_BIT_MASK & 0 is 0 - loop exit.
         while (atomicOr((unsigned long long int *)&primary_locks[high], (unsigned long long int) SET_BIT_MASK(low)) & SET_BIT_MASK(low)){
           //printf("TID %llu Stalling for Bucket %llu/%llu\n", threadIdx.x+blockIdx.x*blockDim.x, bucket, num_buckets_);
         }


      }


      __device__ void stall_lock_alt(tile_type my_tile, uint64_t bucket){

         if (my_tile.thread_rank() == 0){
            stall_lock_one_thread_primary(bucket);
         }

         my_tile.sync();

      }

      __device__ void stall_lock_one_thread_alt(uint64_t bucket){

         uint64_t high = bucket/64;
         uint64_t low = bucket % 64;

         //if old is 0, SET_BIT_MASK & 0 is 0 - loop exit.
         while (atomicOr((unsigned long long int *)&alt_locks[high], (unsigned long long int) SET_BIT_MASK(low)) & SET_BIT_MASK(low)){
           //printf("TID %llu Stalling for Bucket %llu/%llu\n", threadIdx.x+blockIdx.x*blockDim.x, bucket, num_buckets_);
         }


      }

      __device__ void unlock_primary(tile_type my_tile, uint64_t bucket){

         if (my_tile.thread_rank() == 0){
            unlock_bucket_one_thread_primary(bucket);
         }

         my_tile.sync();

      }


      __device__ void unlock_bucket_one_thread_primary(uint64_t bucket){

         uint64_t high = bucket/64;
         uint64_t low = bucket % 64;

         //if old is 0, SET_BIT_MASK & 0 is 0 - loop exit.
         atomicAnd((unsigned long long int *)&primary_locks[high], (unsigned long long int) ~SET_BIT_MASK(low));

      }

      __device__ void unlock_alt(tile_type my_tile, uint64_t bucket){

         if (my_tile.thread_rank() == 0){
            unlock_bucket_one_thread_alt(bucket);
         }

         my_tile.sync();

      }


      __device__ void unlock_bucket_one_thread_alt(uint64_t bucket){

         uint64_t high = bucket/64;
         uint64_t low = bucket % 64;

         //if old is 0, SET_BIT_MASK & 0 is 0 - loop exit.
         atomicAnd((unsigned long long int *)&alt_locks[high], (unsigned long long int) ~SET_BIT_MASK(low));

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

         cudaFree(host_version->alt_buckets);
         cudaFree(host_version->alt_locks);

         cudaFreeHost(host_version);
         
         return;

      }


      __device__ bucket_type * get_bucket_ptr_primary(uint64_t bucket_addr){

         return &primary_buckets[bucket_addr];

      }

      __device__ bucket_type * get_bucket_ptr_alt(uint64_t bucket_addr){

         return &primary_buckets[bucket_addr];

      }


       __device__ bool upsert_generic(const tile_type & my_tile, const Key & key, const Val & val){

         uint64_t bucket_0 = hash(&key, sizeof(Key), seed) % n_buckets_primary;
         
         stall_lock_primary(my_tile, bucket_0);


         Val * existing_loc = query_reference(my_tile, key, bucket_0);

         if (existing_loc != nullptr){


            if (my_tile.thread_rank() == 0){

               gallatin::utils::st_rel(existing_loc, val);
              __threadfence();

            }

            //this syncs.
            unlock_primary(my_tile, bucket_0);
            //my_tile.sync();


            return true;
         }     

         bool return_val = upsert_generic_nolock(my_tile, key, val);

         unlock_primary(my_tile, bucket_0);

         return return_val;

       }


      __device__ bool upsert_generic_nolock(const tile_type & my_tile, const Key & key, const Val & val){



         uint64_t bucket_primary = hash(&key, sizeof(Key), seed) % n_buckets_primary;


         bucket_type * bucket_primary_ptr = get_bucket_ptr_primary(bucket_primary);
         //bucket_type * bucket_1_ptr = get_bucket_ptr(bucket_1);


         //first pass is the attempt to upsert/shortcut on primary
         //if this fails enter generic load loop

         //arguments for primary bucket are defined herer - needed for the primary upsert.
         //secondary come before the main non-shortcut loop - shorcut saves registers if possible.
         uint primary_empty;
         uint primary_tombstone;
         uint primary_match;


         //global load occurs here - if counting loads this is the spot for bucket 0.
         bucket_primary_ptr->load_fill_ballots(my_tile, key, primary_empty, primary_tombstone, primary_match);

         //size is bucket_size - empty slots (empty + tombstone)
         //this saves an op over (bucket_size - __popc(bucket_0_empty)) - __popc(bucket_0_tombstone);
         uint primary_size = bucket_size - __popc(primary_empty | primary_tombstone);

         //always attempt primary insertion
         while (primary_size < bucket_size){


            // if (__popc(primary_match) != 0){


            //    printf("Replacement triggered\n");


            // if (bucket_primary_ptr->upsert_existing(my_tile, key, val, primary_match) != -1){
            //    return true;
            // }

            // //match was observed but has changed - move on to tombstone.
            // //because of lock other threads cannot interfere in this upsert other than deletion
            // //due to stability + lock the key must not exist anymore so we can proceed with insertion.
            // //__threadfence();
            // // continue;

            // }

            if (bucket_primary_ptr->insert_ballots(my_tile, key, val, primary_empty, primary_tombstone)) return true;

            //reload values
            bucket_primary_ptr->load_fill_ballots(my_tile, key, primary_empty, primary_tombstone, primary_match);

            primary_size = bucket_size - __popc(primary_empty | primary_tombstone);
            
         }


         //double hashing.

         uint64_t primary_step = hash(&key, sizeof(Key), seed+1);
         uint64_t alternate_step = hash(&key, sizeof(Key), seed+2);

         //setup for alterna


         for (int i = 0; i < BACK_PROBES; i++){


            uint64_t bucket_index = (primary_step + alternate_step*i) % n_buckets_alt;
            bucket_type * bucket_ptr = get_bucket_ptr_alt(bucket_index);
      
            uint bucket_empty;
            uint bucket_tombstone;
            uint bucket_match;




            bucket_ptr->load_fill_ballots(my_tile, key, bucket_empty, bucket_tombstone, bucket_match);


            while (__popc(bucket_empty | bucket_tombstone) != 0){


               // if (__popc(bucket_match != 0)){

               //    printf("Replacement triggered\n");

               //    if (bucket_ptr->upsert_existing(my_tile, key, val, bucket_match) != -1){
               //       return true;
               //    }

               // }

               if (bucket_ptr->insert_ballots(my_tile, key, val, bucket_empty, bucket_tombstone)) return true;

               bucket_ptr->load_fill_ballots(my_tile, key, bucket_empty, bucket_tombstone, bucket_match);


            }


            //if (my_tile.thread_rank() == 0) printf("%d done\n", i);


         }

         //if (my_tile.thread_rank() == 0) printf("%lu exceeded\n", BACK_PROBES);

         return false;

      }


      //find the reference of the value if it exists
      __device__ Val * query_reference(tile_type my_tile, Key key, uint64_t bucket_primary){


         //uint64_t bucket_primary = hash(&key, sizeof(Key), seed) % n_buckets_primary;


         bucket_type * bucket_primary_ptr = get_bucket_ptr_primary(bucket_primary);

         uint primary_empty;
         uint primary_tombstone;
         uint primary_match;

         bucket_primary_ptr->load_fill_ballots(my_tile, key, primary_empty, primary_tombstone, primary_match);

         Val * val_loc = bucket_primary_ptr->query_ptr_ballot(my_tile, primary_match);

         if (val_loc != nullptr) return val_loc;

         //shortcutting
         if (__popc(primary_empty) > 0) return nullptr;


         uint64_t primary_step = hash(&key, sizeof(Key), seed+1);
         uint64_t alternate_step = hash(&key, sizeof(Key), seed+2);

         //setup for alterna


         for (int i = 0; i < BACK_PROBES; i++){


            uint64_t bucket_index = (primary_step + alternate_step*i) % n_buckets_alt;
            bucket_type * bucket_ptr = get_bucket_ptr_alt(bucket_index);
      
            uint bucket_empty;
            uint bucket_tombstone;
            uint bucket_match;


            bucket_ptr->load_fill_ballots(my_tile, key, bucket_empty, bucket_tombstone, bucket_match);

            val_loc = bucket_primary_ptr->query_ptr_ballot(my_tile, primary_match);

            if (val_loc != nullptr) return val_loc;

            //shortcutting
            if (__popc(primary_empty) > 0) return nullptr;


         }


         return nullptr;

      }

      __device__ packed_pair_type * query_packed_reference(tile_type my_tile, Key key, uint bucket_primary){



         bucket_type * bucket_primary_ptr = get_bucket_ptr_primary(bucket_primary);

         uint primary_empty;
         uint primary_tombstone;
         uint primary_match;

         bucket_primary_ptr->load_fill_ballots(my_tile, key, primary_empty, primary_tombstone, primary_match);

         packed_pair_type * val_loc = bucket_primary_ptr->query_pair_ballot(my_tile, primary_match);

         if (val_loc != nullptr) return val_loc;

         //shortcutting
         if (__popc(primary_empty) > 0) return nullptr;


         uint64_t primary_step = hash(&key, sizeof(Key), seed+1);
         uint64_t alternate_step = hash(&key, sizeof(Key), seed+2);

         //setup for alterna


         for (int i = 0; i < BACK_PROBES; i++){


            uint64_t bucket_index = (primary_step + alternate_step*i) % n_buckets_alt;
            bucket_type * bucket_ptr = get_bucket_ptr_alt(bucket_index);
      
            uint bucket_empty;
            uint bucket_tombstone;
            uint bucket_match;


            bucket_ptr->load_fill_ballots(my_tile, key, bucket_empty, bucket_tombstone, bucket_match);

            val_loc = bucket_primary_ptr->query_pair_ballot(my_tile, primary_match);

            if (val_loc != nullptr) return val_loc;

            //shortcutting
            if (__popc(primary_empty) > 0) return nullptr;


         }


         return nullptr;





      }

      // //nope! no storage
      __device__ bool find_with_reference(tile_type my_tile, Key key, Val & val){


         //return false;

         uint64_t bucket_primary = hash(&key, sizeof(Key), seed) % n_buckets_primary;

         stall_lock(my_tile, bucket_primary);


         Val * val_location = query_reference(my_tile, key, bucket_primary);

         if (val_location == nullptr){
            unlock(my_tile, bucket_primary);
            return false;
         } 

         val = hash_table_load(val_location);
         __threadfence();

         unlock(my_tile, bucket_primary);

         return true;

      }

      __device__ bool remove(tile_type my_tile, Key key){



         uint64_t bucket_primary = hash(&key, sizeof(Key), seed) % n_buckets_primary;

         stall_lock(my_tile, bucket_primary);

         packed_pair_type * found_pair = query_packed_reference(my_tile, key, bucket_primary);

         if (found_pair == nullptr){

            unlock(my_tile, bucket_primary);
            return false;
         }

         bool ballot = false;
         //erase
         if (my_tile.thread_rank() == 0){


            ballot = typed_atomic_write(&found_pair->key, key, tombstoneKey);
            if (ballot){

               //force store
               gallatin::utils::st_rel(&found_pair->val, tombstoneVal);
               //typed_atomic_exchange(&slots[i].val, ext_val);
            }

         }

        
         bool result = my_tile.ballot(ballot);
         if (result){
            unlock(my_tile, bucket_primary);
            return true;
         } 

         __threadfence();

         unlock(my_tile, bucket_primary);
         return false;






      }

      // __device__ bool upsert(tile_type my_tile, Key old_key, Val old_val, Key new_key, Val new_val){
      //    return internal_table->upsert_exact(my_tile, old_key, old_val, new_key, new_val);
      // }

      // __device__ bool upsert(tile_type my_tile, packed_pair_type old_pair, packed_pair_type new_pair){
      //    return upsert(my_tile, old_pair.first, old_pair.second, new_pair.first, new_pair.second);
      // }

      // __device__ bool insert_if_not_exists(tile_type my_tile, Key key, Val val){
      //    return internal_table->insert_exact(my_tile, key, val);
      // }
      
      // __device__ packed_pair_type find_replaceable_pair(tile_type my_tile, Key key){
      //    return internal_table->find_smaller_hash(my_tile, key);
      // }

      static __device__ packed_pair_type pack_together(Key key, Val val){
         return packed_pair_type{key, val};
      }

      __host__ float load(){

         return 0;

      }

      static char * get_name(){
         return "iht_double_hashing";
      }


   };

template <typename T>
constexpr T generate_iht_tombstone(uint64_t offset) {
  return (~((T) 0)) - offset;
};

template <typename T>
constexpr T generate_iht_sentinel() {
  return ((T) 0);
};


// template <typename Key, Key sentinel, Key tombstone, typename Val, Val defaultVal, Val tombstoneVal, uint partition_size, uint bucket_size>
  
template <typename Key, typename Val, uint bucket_size>
using iht_generic = typename hashing_project::tables::iht_double_table<Key,
                                    generate_iht_sentinel<Key>(),
                                    generate_iht_tombstone<Key>(0),
                                    Val,
                                    generate_iht_sentinel<Val>(),
                                    generate_iht_tombstone<Val>(0),
                                    bucket_size,
                                    bucket_size>;




} //namespace wrappers

}  // namespace ht_project

#endif  // GPU_BLOCK_