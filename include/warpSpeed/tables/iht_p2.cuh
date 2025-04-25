#ifndef OUR_IHT_P2
#define OUR_IHT_P2

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <gallatin/allocators/alloc_utils.cuh>
#include <warpSpeed/helpers/probe_counts.cuh>
#include <warpSpeed/helpers/ht_load.cuh>

#include <warpSpeed/helpers/ht_bucket.cuh>


#include "assert.h"
#include "stdio.h"

namespace cg = cooperative_groups;

// helper_macro
// define macros
#define MAX_VALUE(nbits) ((1ULL << (nbits)) - 1)
#define BITMASK(nbits) ((nbits) == 64 ? 0xffffffffffffffff : MAX_VALUE(nbits))

#define SET_BIT_MASK(index) ((1ULL << index))

// a pointer list managing a set section of device memory

#define FRONT_TOTAL_RATIO .83


//cache protocol
//query cache
//on success add to pin?
//need delete from potential buckets implementation - need to download warpcore...
//build with primary p2bht first.



namespace warpSpeed {

namespace tables {


   //investigate this.
   // __device__ inline void st_rel(const uint64_t *p, uint64_t store_val) {
  
   //   asm volatile("st.gpu.release.u64 [%0], %1;" :: "l"(p), "l"(store_val) : "memory");

   //   // return atomicOr((unsigned long long int *)p, 0ULL);

   //   // atom{.sem}{.scope}{.space}.cas.b128 d, [a], b, c {, cache-policy};
   // }


   template <typename HT, uint tile_size>
   __global__ void iht_get_fill_kernel(HT * metadata_table, uint64_t n_buckets_primary, uint64_t n_buckets_alt, uint64_t * item_count){


      auto thread_block = cg::this_thread_block();

      cg::thread_block_tile<tile_size> my_tile = cg::tiled_partition<tile_size>(thread_block);


      uint64_t tid = gallatin::utils::get_tile_tid(my_tile);

      if (tid >= n_buckets_primary) return;

      uint64_t n_items_in_bucket = metadata_table->get_bucket_fill_primary(my_tile, tid);

      if (my_tile.thread_rank() == 0){
         atomicAdd((unsigned long long int *)item_count, n_items_in_bucket);
      }

      if (tid >= n_buckets_alt) return;

      n_items_in_bucket = metadata_table->get_bucket_fill_alt(my_tile, tid);

      if (my_tile.thread_rank() == 0){
         atomicAdd((unsigned long long int *)item_count, n_items_in_bucket);
      }


   }


   template <typename table>
   __global__ void init_iht_p2_table_kernel(table * hash_table){

      uint64_t tid = gallatin::utils::get_tid();

      hash_table->init_bucket_and_locks(tid);
      

   }



   template <typename Key, Key defaultKey, Key tombstoneKey, typename Val, Val defaultVal, Val tombstoneVal, uint partition_size, uint bucket_size>
   struct iht_p2_table {


      using my_type = iht_p2_table<Key, defaultKey, tombstoneKey, Val, defaultVal, tombstoneVal, partition_size, bucket_size>;


      using tile_type = cg::thread_block_tile<partition_size>;

      using bucket_type = hash_table_bucket<Key, defaultKey, tombstoneKey, tombstoneKey-1, Val, defaultVal, tombstoneVal, partition_size, bucket_size>;

      using packed_pair_type = ht_pair<Key, Val>;

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

         //printf("Iceberg table has %lu total: %lu primary and %lu alt\n", ext_n_buckets, host_version->n_buckets_primary, host_version->n_buckets_alt);

         host_version->primary_buckets = gallatin::utils::get_device_version<bucket_type>(host_version->n_buckets_primary);
         host_version->alt_buckets = gallatin::utils::get_device_version<bucket_type>(host_version->n_buckets_alt);

         host_version->primary_locks = gallatin::utils::get_device_version<uint64_t>( (host_version->n_buckets_primary-1)/64+1);
         host_version->alt_locks = gallatin::utils::get_device_version<uint64_t>( (host_version->n_buckets_alt-1)/64+1);

         host_version->seed = ext_seed;


         uint64_t n_buckets = host_version->n_buckets_primary+host_version->n_buckets_alt;

         my_type * device_version = gallatin::utils::move_to_device<my_type>(host_version);


         //this is the issue
         init_iht_p2_table_kernel<my_type><<<(n_buckets-1)/256+1,256>>>(device_version);

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

      __device__ void stall_lock(tile_type my_tile, uint64_t bucket){

         stall_lock_primary(my_tile, bucket);

      }

      __device__ void stall_lock_one_thread_primary(uint64_t bucket){

         #if LOAD_CHEAP
         return;
         #endif

         uint64_t high = bucket/64;
         uint64_t low = bucket % 64;

         //if old is 0, SET_BIT_MASK & 0 is 0 - loop exit.
         do {
            ADD_PROBE
         }
         while (atomicOr((unsigned long long int *)&primary_locks[high], (unsigned long long int) SET_BIT_MASK(low)) & SET_BIT_MASK(low));

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
         do {
            ADD_PROBE
         }
         while (atomicOr((unsigned long long int *)&alt_locks[high], (unsigned long long int) SET_BIT_MASK(low)) & SET_BIT_MASK(low));


      }

      __device__ void unlock_primary(tile_type my_tile, uint64_t bucket){

         if (my_tile.thread_rank() == 0){
            unlock_bucket_one_thread_primary(bucket);
         }

         my_tile.sync();

      }

      __device__ void unlock(tile_type my_tile, uint64_t bucket){

         unlock_primary(my_tile, bucket);
      
      }



      __device__ void unlock_bucket_one_thread_primary(uint64_t bucket){

         #if LOAD_CHEAP
         return;
         #endif

         uint64_t high = bucket/64;
         uint64_t low = bucket % 64;

         //if old is 0, SET_BIT_MASK & 0 is 0 - loop exit.
         atomicAnd((unsigned long long int *)&primary_locks[high], (unsigned long long int) ~SET_BIT_MASK(low));
         ADD_PROBE

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
         ADD_PROBE

      }

      __device__ void lock_key(tile_type my_tile, Key key){

         uint64_t key_hash = hash(&key, sizeof(Key), seed);

         uint64_t bucket_0 = get_first_bucket(key_hash);

         stall_lock(my_tile, bucket_0);

      }

      __device__ void unlock_key(tile_type my_tile, Key key){

         uint64_t key_hash = hash(&key, sizeof(Key), seed);

         uint64_t bucket_0 = get_first_bucket(key_hash);

         unlock(my_tile, bucket_0);

      }

      __device__ uint64_t get_lock_bucket(tile_type my_tile, Key key){

         uint64_t key_hash = hash(&key, sizeof(Key), seed);

         uint64_t bucket_0 = get_first_bucket(key_hash);

         return bucket_0;

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


      __device__ uint64_t get_first_bucket(uint64_t hash){
         return (hash & BITMASK(32)) % n_buckets_primary;
      }



      __host__ uint64_t get_num_locks(){


         my_type * host_version = gallatin::utils::copy_to_host<my_type>(this);

         uint64_t nblocks = host_version->n_buckets_primary;

         cudaFreeHost(host_version);

         return nblocks;

      }


      __device__ uint64_t get_second_bucket(uint64_t hash){
         return ((hash) >> 32) % n_buckets_alt;
      }


      //combination of first two hashes - should be 2-way independent?
      // __device__ uint64_t get_third_bucket(uint64_t hash){

      //    uint64_t first = get_first_bucket(hash);

      //    return (first & (hash >> 32)) % n_buckets_alt;

      // }



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

         return &alt_buckets[bucket_addr];

      }


      __device__ bool upsert_replace(const tile_type & my_tile, const Key & key, const Val & val){


         uint64_t key_hash = hash(&key, sizeof(Key), seed);
         uint64_t bucket_0 = get_first_bucket(key_hash);
         
         stall_lock_primary(my_tile, bucket_0);

         bool return_val = upsert_replace_internal(my_tile, key, val, bucket_0, key_hash);

         unlock_primary(my_tile, bucket_0);

         return return_val;

       }

      __device__ bool upsert_function(const tile_type & my_tile, const Key & key, const Val & val, void (*replace_func)(packed_pair_type *, Key, Val)){


         uint64_t key_hash = hash(&key, sizeof(Key), seed);
         uint64_t bucket_0 = get_first_bucket(key_hash);
         
         stall_lock_primary(my_tile, bucket_0);

         bool return_val = upsert_function_internal(my_tile, key, val, bucket_0, key_hash, replace_func);

         unlock_primary(my_tile, bucket_0);

         return return_val;

       }

      __device__ bool upsert_function_no_lock(const tile_type & my_tile, const Key & key, const Val & val, void (*replace_func)(packed_pair_type *, Key, Val)){


         uint64_t key_hash = hash(&key, sizeof(Key), seed);
         uint64_t bucket_0 = get_first_bucket(key_hash);
         
         //stall_lock_primary(my_tile, bucket_0);

         bool return_val = upsert_function_internal(my_tile, key, val, bucket_0, key_hash, replace_func);

         //unlock_primary(my_tile, bucket_0);

         return return_val;

       }

      __device__ bool upsert_no_lock(const tile_type & my_tile, const Key & key, const Val & val){


         uint64_t key_hash = hash(&key, sizeof(Key), seed);
         uint64_t bucket_0 = get_first_bucket(key_hash);
         
         //stall_lock_primary(my_tile, bucket_0);

         bool return_val = upsert_replace_internal(my_tile, key, val, bucket_0, key_hash);

         //unlock_primary(my_tile, bucket_0);

         return return_val;

      }

      __device__ bool upsert_function_internal(const tile_type & my_tile, const Key & key, const Val & val, uint64_t bucket_primary, uint64_t key_hash, void (*replace_func)(packed_pair_type *, Key, Val)){



         //uint64_t bucket_primary = hash(&key, sizeof(Key), seed) % n_buckets_primary;


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
         if (bucket_primary_ptr->load_fill_ballots_upserts_func(my_tile, key, val, primary_empty, primary_tombstone, primary_match, replace_func)){
            return true;
         }

         //size is bucket_size - empty slots (empty + tombstone)
         //this saves an op over (bucket_size - __popc(bucket_0_empty)) - __popc(bucket_0_tombstone);
         uint primary_size = bucket_size - __popc(primary_empty);

         //always attempt primary insertion (shortcutting)
         //this holds IFF empty.
         while (primary_size < bucket_size){


            if (__popc(primary_match) != 0){



               if (bucket_primary_ptr->upsert_existing_func(my_tile, key, val, primary_match, replace_func) != -1){
                  return true;

            }

            //match was observed but has changed - move on to tombstone.
            //because of lock other threads cannot interfere in this upsert other than deletion
            //due to stability + lock the key must not exist anymore so we can proceed with insertion.
            //__threadfence();
            // continue;

            }

            if (bucket_primary_ptr->insert_ballots(my_tile, key, val, primary_empty, primary_tombstone)) return true;

            //reload values
            ADD_PROBE_BUCKET
            bucket_primary_ptr->load_fill_ballots(my_tile, key, primary_empty, primary_tombstone, primary_match);

            primary_size = bucket_size - __popc(primary_empty);
            
         }


         //p2_hash

         //uint64_t bucket_0 = hash(&key, sizeof(Key), seed+1) % n_buckets_alt;
         uint64_t bucket_0 = get_second_bucket(key_hash);

         uint64_t bucket_1 = hash(&key, sizeof(Key), seed+2) % n_buckets_alt;
         //setup for alterna

         uint bucket_0_empty;
         uint bucket_0_tombstone;
         uint bucket_0_match;

         uint bucket_1_empty;
         uint bucket_1_tombstone;
         uint bucket_1_match;


         bucket_type * bucket_0_ptr = get_bucket_ptr_alt(bucket_0);
         bucket_type * bucket_1_ptr = get_bucket_ptr_alt(bucket_1);


         //global load occurs here - if counting loads this is the spot for bucket 0.
         ADD_PROBE_BUCKET
         bucket_0_ptr->load_fill_ballots(my_tile, key, bucket_0_empty, bucket_0_tombstone, bucket_0_match);
         ADD_PROBE_BUCKET
         bucket_1_ptr->load_fill_ballots(my_tile, key, bucket_1_empty, bucket_1_tombstone, bucket_1_match);


         //check upserts
         if (__popc(bucket_0_match) != 0){

            if (bucket_0_ptr->upsert_existing_func(my_tile, key, val, bucket_0_match, replace_func) != -1){
               return true;
            }

         }

         if (__popc(bucket_1_match) != 0){

            if (bucket_1_ptr->upsert_existing_func(my_tile, key, val, bucket_1_match, replace_func) != -1){
               return true;
            }

         }


         primary_size = bucket_size - __popc(primary_empty | primary_tombstone);

         //always attempt primary insertion (shortcutting)
         //this holds IFF empty.
         while (primary_size < bucket_size){

            //match was observed but has changed - move on to tombstone.
            //because of lock other threads cannot interfere in this upsert other than deletion
            //due to stability + lock the key must not exist anymore so we can proceed with insertion.
            //__threadfence();
            // continue;


            if (bucket_primary_ptr->insert_ballots(my_tile, key, val, primary_empty, primary_tombstone)) return true;

            //reload values
            ADD_PROBE_BUCKET
            bucket_primary_ptr->load_fill_ballots(my_tile, key, primary_empty, primary_tombstone, primary_match);

            primary_size = bucket_size - __popc(primary_empty);
            
         }

         uint bucket_0_size = bucket_size - __popc(bucket_0_empty | bucket_0_tombstone);

         uint bucket_1_size = bucket_size - __popc(bucket_1_empty | bucket_1_tombstone);


        


         //loop on P2
         while (bucket_0_size != bucket_size || bucket_1_size != bucket_size){



            if (bucket_0_size <= bucket_1_size){

               if (bucket_0_ptr->insert_ballots(my_tile, key, val, bucket_0_empty, bucket_0_tombstone)) return true;

            } else {

               if (bucket_1_ptr->insert_ballots(my_tile, key, val, bucket_1_empty, bucket_1_tombstone)) return true;

            }


            //reload
            bucket_0_ptr->load_fill_ballots(my_tile, key, bucket_0_empty, bucket_0_tombstone, bucket_0_match);
            ADD_PROBE_BUCKET
            bucket_1_ptr->load_fill_ballots(my_tile, key, bucket_1_empty, bucket_1_tombstone, bucket_1_match);
            ADD_PROBE_BUCKET

            bucket_1_size = bucket_size - __popc(bucket_1_empty | bucket_1_tombstone);
            bucket_0_size = bucket_size - __popc(bucket_0_empty | bucket_0_tombstone);




         }

         return false;

      }

      __device__ bool upsert_replace_internal(const tile_type & my_tile, const Key & key, const Val & val, uint64_t bucket_primary, uint64_t key_hash){



         //uint64_t bucket_primary = hash(&key, sizeof(Key), seed) % n_buckets_primary;


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
         if (bucket_primary_ptr->load_fill_ballots_upserts(my_tile, key, val, primary_empty, primary_tombstone, primary_match)){
            return true;
         }

         //size is bucket_size - empty slots (empty + tombstone)
         //this saves an op over (bucket_size - __popc(bucket_0_empty)) - __popc(bucket_0_tombstone);
         uint primary_size = bucket_size - __popc(primary_empty);

         //always attempt primary insertion (shortcutting)
         //this holds IFF empty.
         while (primary_size < bucket_size){


            if (__popc(primary_match) != 0){



               if (bucket_primary_ptr->upsert_existing(my_tile, key, val, primary_match) != -1){
                  return true;

            }

            //match was observed but has changed - move on to tombstone.
            //because of lock other threads cannot interfere in this upsert other than deletion
            //due to stability + lock the key must not exist anymore so we can proceed with insertion.
            //__threadfence();
            // continue;

            }

            if (bucket_primary_ptr->insert_ballots(my_tile, key, val, primary_empty, primary_tombstone)) return true;

            //reload values
            ADD_PROBE_BUCKET
            bucket_primary_ptr->load_fill_ballots(my_tile, key, primary_empty, primary_tombstone, primary_match);

            primary_size = bucket_size - __popc(primary_empty);
            
         }


         //p2_hash

         //uint64_t bucket_0 = hash(&key, sizeof(Key), seed+1) % n_buckets_alt;
         uint64_t bucket_0 = get_second_bucket(key_hash);

         uint64_t bucket_1 = hash(&key, sizeof(Key), seed+2) % n_buckets_alt;
         //setup for alterna

         uint bucket_0_empty;
         uint bucket_0_tombstone;
         uint bucket_0_match;

         uint bucket_1_empty;
         uint bucket_1_tombstone;
         uint bucket_1_match;


         bucket_type * bucket_0_ptr = get_bucket_ptr_alt(bucket_0);
         bucket_type * bucket_1_ptr = get_bucket_ptr_alt(bucket_1);


         //global load occurs here - if counting loads this is the spot for bucket 0.
         ADD_PROBE_BUCKET
         bucket_0_ptr->load_fill_ballots(my_tile, key, bucket_0_empty, bucket_0_tombstone, bucket_0_match);
         ADD_PROBE_BUCKET
         bucket_1_ptr->load_fill_ballots(my_tile, key, bucket_1_empty, bucket_1_tombstone, bucket_1_match);


         //check upserts
         if (__popc(bucket_0_match) != 0){

            if (bucket_0_ptr->upsert_existing(my_tile, key, val, bucket_0_match) != -1){
               return true;
            }

         }

         if (__popc(bucket_1_match) != 0){

            if (bucket_1_ptr->upsert_existing(my_tile, key, val, bucket_1_match) != -1){
               return true;
            }

         }


         primary_size = bucket_size - __popc(primary_empty | primary_tombstone);

         //always attempt primary insertion (shortcutting)
         //this holds IFF empty.
         while (primary_size < bucket_size){

            //match was observed but has changed - move on to tombstone.
            //because of lock other threads cannot interfere in this upsert other than deletion
            //due to stability + lock the key must not exist anymore so we can proceed with insertion.
            //__threadfence();
            // continue;


            if (bucket_primary_ptr->insert_ballots(my_tile, key, val, primary_empty, primary_tombstone)) return true;

            //reload values
            ADD_PROBE_BUCKET
            bucket_primary_ptr->load_fill_ballots(my_tile, key, primary_empty, primary_tombstone, primary_match);

            primary_size = bucket_size - __popc(primary_empty);
            
         }

         uint bucket_0_size = bucket_size - __popc(bucket_0_empty | bucket_0_tombstone);

         uint bucket_1_size = bucket_size - __popc(bucket_1_empty | bucket_1_tombstone);


        


         //loop on P2
         while (bucket_0_size != bucket_size || bucket_1_size != bucket_size){



            if (bucket_0_size <= bucket_1_size){

               if (bucket_0_ptr->insert_ballots(my_tile, key, val, bucket_0_empty, bucket_0_tombstone)) return true;

            } else {

               if (bucket_1_ptr->insert_ballots(my_tile, key, val, bucket_1_empty, bucket_1_tombstone)) return true;

            }


            //reload
            bucket_0_ptr->load_fill_ballots(my_tile, key, bucket_0_empty, bucket_0_tombstone, bucket_0_match);
            ADD_PROBE_BUCKET
            bucket_1_ptr->load_fill_ballots(my_tile, key, bucket_1_empty, bucket_1_tombstone, bucket_1_match);
            ADD_PROBE_BUCKET

            bucket_1_size = bucket_size - __popc(bucket_1_empty | bucket_1_tombstone);
            bucket_0_size = bucket_size - __popc(bucket_0_empty | bucket_0_tombstone);




         }

         return false;

      }



      // //nope! no storage
      [[nodiscard]] __device__ bool find_with_reference(tile_type my_tile, Key key, Val & val){



         uint64_t key_hash = hash(&key, sizeof(Key), seed);
         uint64_t bucket_primary = get_first_bucket(key_hash);

         bucket_type * bucket_primary_ptr = get_bucket_ptr_primary(bucket_primary);


         if (bucket_primary_ptr->query(my_tile, key, val)){
            return true;
         }

         //uint64_t bucket_0 = hash(&key, sizeof(Key), seed+1) % n_buckets_alt;
         uint64_t bucket_0 = get_second_bucket(key_hash);


         bucket_type * bucket_0_ptr = get_bucket_ptr_alt(bucket_0);


         if (bucket_0_ptr->query(my_tile, key, val)){
            return true;
         }


         uint64_t bucket_1 = hash(&key, sizeof(Key), seed+2) % n_buckets_alt;

         bucket_type * bucket_1_ptr = get_bucket_ptr_alt(bucket_1);


         if (bucket_1_ptr->query(my_tile, key, val)){
            return true;
         }

         return false;

      }

      __device__ bool remove(tile_type my_tile, Key key){


         uint64_t key_hash = hash(&key, sizeof(Key), seed);
         uint64_t bucket_primary = get_first_bucket(key_hash);
         

         //uint64_t bucket_primary = hash(&key, sizeof(Key), seed) % n_buckets_primary;

         stall_lock(my_tile, bucket_primary);


         bucket_type * bucket_primary_ptr = get_bucket_ptr_primary(bucket_primary);


         if (bucket_primary_ptr->erase(my_tile, key)){
            unlock(my_tile, bucket_primary);
            return true;
         }

         //uint64_t bucket_0 = hash(&key, sizeof(Key), seed+1) % n_buckets_alt;
         uint64_t bucket_0 = get_second_bucket(key_hash);


         bucket_type * bucket_0_ptr = get_bucket_ptr_alt(bucket_0);


         if (bucket_0_ptr->erase(my_tile, key)){
            unlock(my_tile, bucket_primary);
            return true;
         }


         uint64_t bucket_1 = hash(&key, sizeof(Key), seed+2) % n_buckets_alt;

         bucket_type * bucket_1_ptr = get_bucket_ptr_alt(bucket_1);


         if (bucket_1_ptr->erase(my_tile, key)){
            unlock(my_tile, bucket_primary);
            return true;
         }

         unlock(my_tile, bucket_primary);

         return false;





      }


      [[nodiscard]] __device__ bool find_with_reference_no_lock(tile_type my_tile, Key key, Val & val){



         uint64_t key_hash = hash(&key, sizeof(Key), seed);
         uint64_t bucket_primary = get_first_bucket(key_hash);
         

         //stall_lock(my_tile, bucket_primary);


         bucket_type * bucket_primary_ptr = get_bucket_ptr_primary(bucket_primary);

         if (bucket_primary_ptr->query(my_tile, key, val)){
            //unlock(my_tile, bucket_primary);
            return true;
         }

         //uint64_t bucket_0 = hash(&key, sizeof(Key), seed+1) % n_buckets_alt;
         uint64_t bucket_0 = get_second_bucket(key_hash);


         bucket_type * bucket_0_ptr = get_bucket_ptr_alt(bucket_0);


         if (bucket_0_ptr->query(my_tile, key, val)){
            //unlock(my_tile, bucket_primary);
            return true;
         }


         uint64_t bucket_1 = hash(&key, sizeof(Key), seed+2) % n_buckets_alt;

         bucket_type * bucket_1_ptr = get_bucket_ptr_alt(bucket_1);


         if (bucket_1_ptr->query(my_tile, key, val)){
            //unlock(my_tile, bucket_primary);
            return true;
         }

         //unlock(my_tile, bucket_primary);

         return false;

      }

      __device__ bool remove_no_lock(tile_type my_tile, Key key){


         uint64_t key_hash = hash(&key, sizeof(Key), seed);
         uint64_t bucket_primary = get_first_bucket(key_hash);
         

         //uint64_t bucket_primary = hash(&key, sizeof(Key), seed) % n_buckets_primary;

         //stall_lock(my_tile, bucket_primary);


         bucket_type * bucket_primary_ptr = get_bucket_ptr_primary(bucket_primary);


         if (bucket_primary_ptr->erase(my_tile, key)){
            //unlock(my_tile, bucket_primary);
            return true;
         }

         //uint64_t bucket_0 = hash(&key, sizeof(Key), seed+1) % n_buckets_alt;
         uint64_t bucket_0 = get_second_bucket(key_hash);


         bucket_type * bucket_0_ptr = get_bucket_ptr_alt(bucket_0);

 
         if (bucket_0_ptr->erase(my_tile, key)){
            //unlock(my_tile, bucket_primary);
            return true;
         }


         uint64_t bucket_1 = hash(&key, sizeof(Key), seed+2) % n_buckets_alt;

         bucket_type * bucket_1_ptr = get_bucket_ptr_alt(bucket_1);

         if (bucket_1_ptr->erase(my_tile, key)){
            //unlock(my_tile, bucket_primary);
            return true;
         }

         //unlock(my_tile, bucket_primary);

         return false;





      }


      [[nodiscard]] __device__ packed_pair_type * find_pair(tile_type my_tile, Key key){


         //return false;


         packed_pair_type * return_pair = nullptr;

         uint64_t key_hash = hash(&key, sizeof(Key), seed);
         uint64_t bucket_primary = get_first_bucket(key_hash);


         bucket_type * bucket_primary_ptr = get_bucket_ptr_primary(bucket_primary);


         bool found_empty;

         return_pair = bucket_primary_ptr->query_pair(my_tile, key, found_empty);


         if (return_pair != nullptr || found_empty){
            return return_pair;
         }

         //uint64_t bucket_0 = hash(&key, sizeof(Key), seed+1) % n_buckets_alt;
         uint64_t bucket_0 = get_second_bucket(key_hash);


         bucket_type * bucket_0_ptr = get_bucket_ptr_alt(bucket_0);


         return_pair = bucket_0_ptr->query_pair(my_tile, key, found_empty);


         if (return_pair != nullptr){

            return return_pair;
         }


         uint64_t bucket_1 = hash(&key, sizeof(Key), seed+2) % n_buckets_alt;

         bucket_type * bucket_1_ptr = get_bucket_ptr_alt(bucket_1);


         return_pair = bucket_1_ptr->query_pair(my_tile, key, found_empty);


         return return_pair;


      }

      [[nodiscard]] __device__ packed_pair_type * find_pair_no_lock(tile_type my_tile, Key key){


         //return false;


         packed_pair_type * return_pair = nullptr;

         uint64_t key_hash = hash(&key, sizeof(Key), seed);
         uint64_t bucket_primary = get_first_bucket(key_hash);
         


         bucket_type * bucket_primary_ptr = get_bucket_ptr_primary(bucket_primary);

         bool found_empty;

         return_pair = bucket_primary_ptr->query_pair(my_tile, key, found_empty);


         if (return_pair != nullptr || found_empty){
            
            return return_pair;
         }

         //uint64_t bucket_0 = hash(&key, sizeof(Key), seed+1) % n_buckets_alt;
         uint64_t bucket_0 = get_second_bucket(key_hash);


         bucket_type * bucket_0_ptr = get_bucket_ptr_alt(bucket_0);


         return_pair = bucket_0_ptr->query_pair(my_tile, key, found_empty);
         if (return_pair != nullptr || found_empty){
           
            return return_pair;
         }


         uint64_t bucket_1 = hash(&key, sizeof(Key), seed+2) % n_buckets_alt;

         bucket_type * bucket_1_ptr = get_bucket_ptr_alt(bucket_1);


         return_pair = bucket_1_ptr->query_pair(my_tile, key, found_empty);



         return return_pair;


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

      static std::string get_name(){
         return "iht_p2_hashing";
      }

      __host__ void print_space_usage(){

         my_type * host_version = gallatin::utils::copy_to_host<my_type>(this);
            
         uint64_t capacity = host_version->n_buckets_primary*sizeof(bucket_type) + (host_version->n_buckets_primary-1)/8+1; 

         capacity += host_version->n_buckets_alt*sizeof(bucket_type);


         cudaFreeHost(host_version);

         printf("iht_p2_hashing using %lu bytes\n", capacity);

      }

      __host__ void print_fill(){

         uint64_t n_items = get_fill();

         my_type * host_version = gallatin::utils::copy_to_host<my_type>(this);

         uint64_t n_buckets_primary = host_version->n_buckets_primary;

         uint64_t n_buckets_alt = host_version->n_buckets_alt;

         uint64_t n_buckets = n_buckets_primary+n_buckets_alt;

         printf("fill: %lu/%lu = %f%%\n", n_items, n_buckets*bucket_size, 100.0*n_items/(n_buckets*bucket_size));

         cudaFreeHost(host_version);



      }

      __host__ uint64_t get_fill(){


         uint64_t * n_items;

         cudaMallocManaged((void **)&n_items, sizeof(uint64_t));

         n_items[0] = 0;

         my_type * host_version = gallatin::utils::copy_to_host<my_type>(this);

         uint64_t n_buckets_primary = host_version->n_buckets_primary;

         uint64_t n_buckets_alt = host_version->n_buckets_alt;

         iht_get_fill_kernel<my_type, partition_size><<<(n_buckets_primary*partition_size-1)/256+1,256>>>(this, n_buckets_primary, n_buckets_alt, n_items);

         cudaDeviceSynchronize();

         uint64_t return_items = n_items[0];

         cudaFree(n_items);
         cudaFreeHost(host_version);

         return return_items;


      }

      __device__ uint64_t get_bucket_fill_primary(tile_type my_tile, uint64_t bucket){


         bucket_type * bucket_ptr = get_bucket_ptr_primary(bucket);


         uint32_t bucket_empty;
         uint32_t bucket_tombstone;
         uint32_t bucket_match;

         bucket_ptr->load_fill_ballots(my_tile, defaultKey, bucket_empty, bucket_tombstone, bucket_match);

         return bucket_size-__popc(bucket_empty) - __popc(bucket_tombstone);

      }

      __device__ uint64_t get_bucket_fill_alt(tile_type my_tile, uint64_t bucket){


         bucket_type * bucket_ptr = get_bucket_ptr_alt(bucket);


         uint32_t bucket_empty;
         uint32_t bucket_tombstone;
         uint32_t bucket_match;

         bucket_ptr->load_fill_ballots(my_tile, defaultKey, bucket_empty, bucket_tombstone, bucket_match);

         return bucket_size-__popc(bucket_empty) - __popc(bucket_tombstone);

      }


   };

template <typename T>
constexpr T generate_iht_p2_tombstone(uint64_t offset) {
  return (~((T) 0)) - offset;
};

template <typename T>
constexpr T generate_iht_p2_sentinel() {
  return ((T) 0);
};


// template <typename Key, Key sentinel, Key tombstone, typename Val, Val defaultVal, Val tombstoneVal, uint partition_size, uint bucket_size>
  
template <typename Key, typename Val, uint tile_size, uint bucket_size>
using iht_p2_generic = typename warpSpeed::tables::iht_p2_table<Key,
                                    generate_iht_p2_sentinel<Key>(),
                                    generate_iht_p2_tombstone<Key>(0),
                                    Val,
                                    generate_iht_p2_sentinel<Val>(),
                                    generate_iht_p2_tombstone<Val>(0),
                                    tile_size,
                                    bucket_size>;




} //namespace wrappers

}  // namespace ht_project

#endif  // GPU_BLOCK_