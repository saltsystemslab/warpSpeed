#ifndef WARPSPEED_P2_HASH_META
#define WARPSPEED_P2_HASH_META

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <gallatin/allocators/alloc_utils.cuh>
#include <gallatin/data_structs/ds_utils.cuh>

#include <warpSpeed/helpers/ht_pairs.cuh>
#include <warpSpeed/helpers/probe_counts.cuh>
#include <warpSpeed/helpers/ht_load.cuh>

#include <warpSpeed/helpers/ht_metadata_bucket.cuh>


#include "assert.h"
#include "stdio.h"

namespace cg = cooperative_groups;

// helper_macro
// define macros
#define MAX_VALUE(nbits) ((1ULL << (nbits)) - 1)
#define BITMASK(nbits) ((nbits) == 64 ? 0xffffffffffffffff : MAX_VALUE(nbits))

#define SET_BIT_MASK(index) ((1ULL << index))

// a pointer list managing a set section of device memory

#define COUNT_INSERT_PROBES 1

#define LARGE_MD_LOAD 1

#define P2_MD_CUTOFF .75


#define ATOMIC_VERIFY 0

//modifications required to use buckets > 32
//experimental and full support is untested.
#define LARGE_BUCKET_MODS 0



//cache protocol
//query cache
//on success add to pin?
//need delete from potential buckets implementation - need to download warpcore...
//buidld with primary p2bht first.



namespace warpSpeed {

namespace tables {


   //load 8 tags and pack them
   __device__ packed_tags load_multi_tags(const uint16_t * start_of_bucket){


      // packed_tags load_type;

      // asm volatile("ld.gpu.acquire.v2.u64 {%0,%1}, [%2];" : "=l"(load_type.first), "=l"(load_type.second) : "l"(start_of_bucket));
      // return load_type;

      return ht_load_metadata<packed_tags>(start_of_bucket);

   }



   //investigate this.
   // __device__ inline void st_rel(const uint64_t *p, uint64_t store_val) {
  
   //   asm volatile("st.gpu.release.u64 [%0], %1;" :: "l"(p), "l"(store_val) : "memory");

   //   // return atomicOr((unsigned long long int *)p, 0ULL);

   //   // atom{.sem}{.scope}{.space}.cas.b128 d, [a], b, c {, cache-policy};
   // }


   template <typename HT, uint tile_size>
   __global__ void p2_md_get_fill_kernel(HT * metadata_table, uint64_t n_buckets, uint64_t * item_count){


      auto thread_block = cg::this_thread_block();

      cg::thread_block_tile<tile_size> my_tile = cg::tiled_partition<tile_size>(thread_block);


      uint64_t tid = gallatin::utils::get_tile_tid(my_tile);

      if (tid >= n_buckets) return;

      uint64_t n_items_in_bucket = metadata_table->get_bucket_fill(my_tile, tid);

      if (my_tile.thread_rank() == 0){
         atomicAdd((unsigned long long int *)item_count, n_items_in_bucket);
      }


   }


   template <typename table>
   __global__ void init_p2_md_table_kernel(table * hash_table){

      uint64_t tid = gallatin::utils::get_tid();

      hash_table->init_bucket_and_locks(tid);
      

   }



   template <typename Key, Key defaultKey, Key tombstoneKey, typename Val, Val defaultVal, Val tombstoneVal, uint partition_size, uint bucket_size>
   struct p2_metadata_table {


      using my_type = p2_metadata_table<Key, defaultKey, tombstoneKey, Val, defaultVal, tombstoneVal, partition_size, bucket_size>;


      using tile_type = cg::thread_block_tile<partition_size>;

      using md_bucket_type = hash_table_metadata<Key, defaultKey, tombstoneKey, Val, tombstoneVal, partition_size, bucket_size>;

      using bucket_type = hash_table_metadata_bucket<Key, defaultKey, tombstoneKey, tombstoneKey-1, Val, defaultVal, tombstoneVal, partition_size, bucket_size>;

      using packed_pair_type = ht_pair<Key, Val>;



      md_bucket_type * metadata;
      bucket_type * buckets;
      uint64_t * locks;

      uint64_t n_buckets;
      uint64_t seed;

      //dummy handle
      static __host__ my_type * generate_on_device(uint64_t cache_capacity, uint64_t ext_seed){

         my_type * host_version = gallatin::utils::get_host_version<my_type>();

         uint64_t ext_n_buckets = (cache_capacity-1)/bucket_size+1;

         host_version->n_buckets = ext_n_buckets;

         host_version->buckets = gallatin::utils::get_device_version<bucket_type>(ext_n_buckets);

         host_version->metadata = gallatin::utils::get_device_version<md_bucket_type>(ext_n_buckets);

         host_version->locks = gallatin::utils::get_device_version<uint64_t>( (ext_n_buckets-1)/64+1);

         host_version->seed = ext_seed;

         my_type * device_version = gallatin::utils::move_to_device<my_type>(host_version);

         init_p2_md_table_kernel<my_type><<<(ext_n_buckets-1)/256+1,256>>>(device_version);

         cudaDeviceSynchronize();

         return device_version;

      }

      __device__ void init_bucket_and_locks(uint64_t tid){

         if (tid < n_buckets){
            metadata[tid].init();
            buckets[tid].init();
            unlock_bucket_one_thread(tid);
         }

      }


      __device__ void stall_lock(tile_type my_tile, uint64_t bucket){

         if (my_tile.thread_rank() == 0){
            stall_lock_one_thread(bucket);
         }

         my_tile.sync();

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

      __device__ void stall_lock_one_thread(uint64_t bucket){

         #if LOAD_CHEAP
         return;
         #endif

         uint64_t high = bucket/64;
         uint64_t low = bucket % 64;

         //if old is 0, SET_BIT_MASK & 0 is 0 - loop exit.

         do {
            ADD_PROBE
            //printf("Looping in lock\n");
         }
         while (atomicOr((unsigned long long int *)&locks[high], (unsigned long long int) SET_BIT_MASK(low)) & SET_BIT_MASK(low));

         //printf("Exiting lock\n");

      }

      __device__ void unlock(tile_type my_tile, uint64_t bucket){

         if (my_tile.thread_rank() == 0){
            unlock_bucket_one_thread(bucket);
         }

         my_tile.sync();

      }


      __device__ void unlock_bucket_one_thread(uint64_t bucket){

         #if LOAD_CHEAP
         return;
         #endif

         uint64_t high = bucket/64;
         uint64_t low = bucket % 64;

         //if old is 0, SET_BIT_MASK & 0 is 0 - loop exit.
         atomicAnd((unsigned long long int *)&locks[high], (unsigned long long int) ~SET_BIT_MASK(low));
         ADD_PROBE

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
         return (hash & BITMASK(32)) % n_buckets;
      }

      __host__ uint64_t get_num_locks(){

         my_type * host_version = gallatin::utils::copy_to_host<my_type>(this);

         uint64_t nblocks = host_version->n_buckets;

         cudaFreeHost(host_version);

         return nblocks;

      }


      __device__ uint64_t get_second_bucket(uint64_t hash){
         return (hash >> 32) % n_buckets;
      }


      static __host__ void free_on_device(my_type * device_version){

         my_type * host_version = gallatin::utils::move_to_host<my_type>(device_version);

         cudaFree(host_version->metadata);
         cudaFree(host_version->buckets);
         cudaFree(host_version->locks);

         cudaFreeHost(host_version);
         
         return;

      }


      __device__ bucket_type * get_bucket_ptr(uint64_t bucket_addr){

         return &buckets[bucket_addr];

      }

      __device__ md_bucket_type * get_metadata(uint64_t bucket_addr){
         return &metadata[bucket_addr];
      }


       __device__ bool upsert_replace(const tile_type & my_tile, const Key & key, const Val & val){
 

         uint64_t key_hash = hash(&key, sizeof(Key), seed);

         uint64_t bucket_0 = get_first_bucket(key_hash);
         //uint64_t bucket_0 = hash(&key, sizeof(Key), seed) % n_buckets;
         
         stall_lock(my_tile, bucket_0);

         bool return_val = upsert_replace_internal(my_tile, key, val, key_hash, bucket_0);

         unlock(my_tile, bucket_0);

         return return_val;

       }

      __device__ bool upsert_function(const tile_type & my_tile, const Key & key, const Val & val, void (*replace_func)(packed_pair_type *, Key, Val)){
 

         uint64_t key_hash = hash(&key, sizeof(Key), seed);

         uint64_t bucket_0 = get_first_bucket(key_hash);
         //uint64_t bucket_0 = hash(&key, sizeof(Key), seed) % n_buckets;
         
         stall_lock(my_tile, bucket_0);

         bool return_val = upsert_function_internal(my_tile, key, val, key_hash, bucket_0, replace_func);

         unlock(my_tile, bucket_0);

         return return_val;

      }

      __device__ bool upsert_function_no_lock(const tile_type & my_tile, const Key & key, const Val & val, void (*replace_func)(packed_pair_type *, Key, Val)){
 

         uint64_t key_hash = hash(&key, sizeof(Key), seed);

         uint64_t bucket_0 = get_first_bucket(key_hash);
         //uint64_t bucket_0 = hash(&key, sizeof(Key), seed) % n_buckets;

         bool return_val = upsert_function_internal(my_tile, key, val, key_hash, bucket_0, replace_func);

         return return_val;

      }


      __device__ bool upsert_function_internal(const tile_type & my_tile, const Key & key, const Val & val, uint64_t key_hash, uint64_t bucket_0, void (*replace_func)(packed_pair_type *, Key, Val)){

         //uint64_t bucket_0 = hash(&key, sizeof(Key), seed) % n_buckets;
        

         md_bucket_type * md_bucket_0 = get_metadata(bucket_0);
         bucket_type * bucket_0_ptr = get_bucket_ptr(bucket_0);
        

         //first pass is the attempt to upsert/shortcut on primary
         //if this fails enter generic load loop

         //arguments for primary bucket are defined herer - needed for the primary upsert.
         //secondary come before the main non-shortcut loop - shorcut saves registers if possible.


         #if LARGE_BUCKET_MODS
         uint64_t bucket_0_empty;
         uint64_t bucket_0_tombstone;
         uint64_t bucket_0_match;
         #else

         uint32_t bucket_0_empty;
         uint32_t bucket_0_tombstone;
         uint32_t bucket_0_match;

         #endif


         //global load occurs here - if counting loads this is the spot for bucket 0.
         // #if COUNT_INSERT_PROBES
         // ADD_PROBE_BUCKET
         // #endif

         #if LARGE_MD_LOAD
         md_bucket_0->load_fill_ballots_huge(my_tile, key, bucket_0_empty, bucket_0_tombstone, bucket_0_match);

         #else 

         md_bucket_0->load_fill_ballots(my_tile, key, bucket_0_empty, bucket_0_tombstone, bucket_0_match);

         #endif


         // uint bucket_0_empty_small;
         // uint bucket_0_tombstone_small;
         // uint bucket_0_match_small;

         // md_bucket_0->load_fill_ballots_big(my_tile, key, bucket_0_empty_small, bucket_0_tombstone_small, bucket_0_match_small);

         // if (bucket_0_empty != bucket_0_empty_small){
         //    printf("Mismatch %u != %u\n", bucket_0_empty, bucket_0_empty_small);
         // }

         
         //bucket_0_ptr->load_fill_ballots_big(my_tile, key, bucket_0_empty, bucket_0_tombstone, bucket_0_match);

         //size is bucket_size - empty slots (empty + tombstone)
         //this saves an op over (bucket_size - __popcll(bucket_0_empty)) - __popcll(bucket_0_tombstone);

         #if LARGE_BUCKET_MODS
         uint bucket_0_size = bucket_size - __popcll(bucket_0_empty ); //| bucket_0_tombstone
         #else
         uint bucket_0_size = bucket_size - __popc(bucket_0_empty ); //| bucket_0_tombstone
         #endif

         //.75 shortcut for the moment.
         while (bucket_0_size < bucket_size*P2_MD_CUTOFF){


            //if (my_tile.thread_rank() == 0) printf("In shortcut loop %lu, %x ballot_map\n", bucket_0_size, bucket_0_empty);

            #if LARGE_BUCKET_MODS
            if (__popcll(bucket_0_match) != 0)
            #else
            if (__popc(bucket_0_match) != 0)
            #endif
            {

               if (bucket_0_ptr->upsert_existing_func(my_tile, key, val, bucket_0_match, replace_func) != -1){
                  return true;
               }

               //match was observed but has changed - move on to tombstone.
               //because of lock other threads cannot interfere in this upsert other than deletion
               //due to stability + lock the key must not exist anymore so we can proceed with insertion.
               //__threadfence();
               // continue;

            }

            int tombstone_pos = md_bucket_0->match_tombstone(my_tile, key, bucket_0_tombstone);

            if (tombstone_pos != -1){

               if (my_tile.thread_rank()  == (tombstone_pos % partition_size)){

                  //temporarily make this atomic?

                  // if (!gallatin::utils::typed_atomic_write(&bucket_0_ptr->slots[tombstone_pos].key, tombstoneKey, key)){

                  //    uint16_t my_tag = md_bucket_0->get_tag(key);
                  //    printf("Failed to replace tombstone %u\n\n", my_tag);
                  // }
                  ADD_PROBE

                  ht_store_packed_pair(&bucket_0_ptr->slots[tombstone_pos], {key,val});

                  // #if ATOMIC_VERIFY

                  // if (!gallatin::utils::typed_atomic_write(&bucket_0_ptr->slots[tombstone_pos].key, tombstoneKey, key)){

                  //    uint16_t my_tag = md_bucket_0->get_tag(key);
                  //    printf("Failed to replace tombstone %u\n\n", my_tag);
                  // }

                  // #else
                  // ht_store(&bucket_0_ptr->slots[tombstone_pos].key, key);
                  // #endif
                  // ht_store(&bucket_0_ptr->slots[tombstone_pos].val, val);

                  __threadfence();

               }

               my_tile.sync();
               return true;

            }

            int empty_pos = md_bucket_0->match_empty(my_tile, key, bucket_0_empty);

            if (empty_pos != -1){

               if (my_tile.thread_rank()  == (empty_pos % partition_size)){


                  // if (!gallatin::utils::typed_atomic_write(&bucket_0_ptr->slots[empty_pos].key, defaultKey, key)){
                  //    printf("Failed to replace empty\n");
                  // }

                  ht_store_packed_pair(&bucket_0_ptr->slots[empty_pos], {key,val});
                  ADD_PROBE

                  // #if ATOMIC_VERIFY
                  // if (!gallatin::utils::typed_atomic_write(&bucket_0_ptr->slots[empty_pos].key, defaultKey, key)){
                  //    printf("Failed to replace empty\n");
                  // }
                  // #else
                  // ht_store(&bucket_0_ptr->slots[empty_pos].key, key);

                  // #endif
                  // ht_store(&bucket_0_ptr->slots[empty_pos].val, val);

                  __threadfence();

               }

               my_tile.sync();
               return true;


            }

            //reload
            #if LARGE_MD_LOAD
               md_bucket_0->load_fill_ballots_huge(my_tile, key, bucket_0_empty, bucket_0_tombstone, bucket_0_match);

            #else 

               md_bucket_0->load_fill_ballots(my_tile, key, bucket_0_empty, bucket_0_tombstone, bucket_0_match);

            #endif


            #if LARGE_BUCKET_MODS
            bucket_0_size = bucket_size - __popcll(bucket_0_empty); // | bucket_0_tombstone);
            #else
            bucket_0_size = bucket_size - __popc(bucket_0_empty);
            #endif


            
         }

         //setup for alternate
         bucket_0_size = bucket_size - __popcll(bucket_0_empty | bucket_0_tombstone); //| bucket_0_tombstone

         uint64_t bucket_1 = get_second_bucket(key_hash);
         md_bucket_type * md_bucket_1 = get_metadata(bucket_1);
         bucket_type * bucket_1_ptr = get_bucket_ptr(bucket_1);

         #if LARGE_BUCKET_MODS
         uint64_t bucket_1_empty;
         uint64_t bucket_1_tombstone;
         uint64_t bucket_1_match;
         #else
         uint32_t bucket_1_empty;
         uint32_t bucket_1_tombstone;
         uint32_t bucket_1_match;
         #endif
         // #if COUNT_INSERT_PROBES
         // ADD_PROBE_BUCKET
         // #endif

         #if LARGE_MD_LOAD
         md_bucket_1->load_fill_ballots_huge(my_tile, key, bucket_1_empty, bucket_1_tombstone, bucket_1_match);

         #else 

         md_bucket_1->load_fill_ballots(my_tile, key, bucket_1_empty, bucket_1_tombstone, bucket_1_match);

         #endif

         #if LARGE_BUCKET_MODS
         uint bucket_1_size = bucket_size - __popcll(bucket_1_empty | bucket_1_tombstone);
         #else
         uint bucket_1_size = bucket_size - __popc(bucket_1_empty | bucket_1_tombstone);
         #endif


         //return false;
         //printf("Reached alternate\n");

         //generic tile loop to perform operations.
         while (bucket_0_size != bucket_size || bucket_1_size != bucket_size){

            //check upserts
            #if LARGE_BUCKET_MODS
            if (__popcll(bucket_0_match) != 0){

               if (bucket_0_ptr->upsert_existing_func(my_tile, key, val, bucket_0_match, replace_func) != -1){
                  return true;
               }

            }

            if (__popcll(bucket_1_match) != 0){

               if (bucket_1_ptr->upsert_existing_func(my_tile, key, val, bucket_1_match, replace_func) != -1){
                  return true;
               }

            }

            #else

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

            #endif

            //check main insert

            if (bucket_0_size <= bucket_1_size){


               int tombstone_pos_0 = md_bucket_0->match_tombstone(my_tile, key, bucket_0_tombstone);

               if (tombstone_pos_0 != -1){

                  if (my_tile.thread_rank()  == (tombstone_pos_0 % partition_size)){

                     ADD_PROBE

                     ht_store_packed_pair(&bucket_0_ptr->slots[tombstone_pos_0], {key,val});
                     // #if ATOMIC_VERIFY

                     // if (!gallatin::utils::typed_atomic_write(&bucket_0_ptr->slots[tombstone_pos_0].key, tombstoneKey, key)){

                     //    uint16_t my_tag = md_bucket_0->get_tag(key);
                     //    printf("Failed to replace tombstone %u\n\n", my_tag);
                     // }

                     // #else
                     // ht_store(&bucket_0_ptr->slots[tombstone_pos_0].key, key);
                     // #endif
                     // ht_store(&bucket_0_ptr->slots[tombstone_pos_0].val, val);

                     __threadfence();

                  }

                  my_tile.sync();
                  return true;

               }

               int empty_pos_0 = md_bucket_0->match_empty(my_tile, key, bucket_0_empty);

               if (empty_pos_0 != -1){

                  if (my_tile.thread_rank()  == (empty_pos_0 % partition_size)){

                     ADD_PROBE

                     ht_store_packed_pair(&bucket_0_ptr->slots[empty_pos_0], {key,val});
                     
                     // #if ATOMIC_VERIFY
                     // if (!gallatin::utils::typed_atomic_write(&bucket_0_ptr->slots[empty_pos_0].key, defaultKey, key)){
                     //    printf("Failed to replace empty\n");
                     // }
                     // #else
                     // ht_store(&bucket_0_ptr->slots[empty_pos_0].key, key);

                     // #endif

                     // ht_store(&bucket_0_ptr->slots[empty_pos_0].val, val);

                     __threadfence();

                  }

                  my_tile.sync();
                  return true;


               }


            } else {

               int tombstone_pos_1 = md_bucket_1->match_tombstone(my_tile, key, bucket_1_tombstone);

               if (tombstone_pos_1 != -1){

                  if (my_tile.thread_rank()  == (tombstone_pos_1 % partition_size)){

                     ADD_PROBE

                     ht_store_packed_pair(&bucket_1_ptr->slots[tombstone_pos_1], {key,val});
                     // #if ATOMIC_VERIFY

                     // if (!gallatin::utils::typed_atomic_write(&bucket_1_ptr->slots[tombstone_pos_1].key, tombstoneKey, key)){

                     //    uint16_t my_tag = md_bucket_0->get_tag(key);
                     //    printf("Failed to replace tombstone %u\n\n", my_tag);
                     // }

                     // #else
                     // ht_store(&bucket_1_ptr->slots[tombstone_pos_1].key, key);
                     // #endif
         
                     // ht_store(&bucket_1_ptr->slots[tombstone_pos_1].val, val);

                     __threadfence();

                  }

                  my_tile.sync();
                  return true;

               }

               int empty_pos_1 = md_bucket_1->match_empty(my_tile, key, bucket_1_empty);

               if (empty_pos_1 != -1){

                  if (my_tile.thread_rank()  == (empty_pos_1 % partition_size)){

                     ADD_PROBE
                     
                     ht_store_packed_pair(&bucket_1_ptr->slots[empty_pos_1], {key,val});
                     // #if ATOMIC_VERIFY
                     // if (!gallatin::utils::typed_atomic_write(&bucket_1_ptr->slots[empty_pos_1].key, defaultKey, key)){
                     //    printf("Failed to replace empty\n");
                     // }
                     // #else
                     // ht_store(&bucket_1_ptr->slots[empty_pos_1].key, key);

                     // #endif

                     // ht_store(&bucket_1_ptr->slots[empty_pos_1].val, val);

                     __threadfence();

                  }

                  my_tile.sync();
                  return true;


               }
            
            }

            #if LARGE_MD_LOAD
               md_bucket_0->load_fill_ballots_huge(my_tile, key, bucket_0_empty, bucket_0_tombstone, bucket_0_match);

            #else 

               md_bucket_0->load_fill_ballots(my_tile, key, bucket_0_empty, bucket_0_tombstone, bucket_0_match);

            #endif

            #if LARGE_MD_LOAD
               md_bucket_1->load_fill_ballots_huge(my_tile, key, bucket_1_empty, bucket_1_tombstone, bucket_1_match);

            #else 

               md_bucket_1->load_fill_ballots(my_tile, key, bucket_1_empty, bucket_1_tombstone, bucket_1_match);

            #endif


            #if LARGE_BUCKET_MODS
            bucket_1_size = bucket_size - __popcll(bucket_1_empty | bucket_1_tombstone);
            bucket_0_size = bucket_size - __popcll(bucket_0_empty | bucket_0_tombstone);
            #else

            bucket_1_size = bucket_size - __popc(bucket_1_empty | bucket_1_tombstone);
            bucket_0_size = bucket_size - __popc(bucket_0_empty | bucket_0_tombstone);

            #endif


            
         }


         return false;


      }


      __device__ bool upsert_no_lock(const tile_type & my_tile, const Key & key, const Val & val){

         //uint64_t bucket_0 = hash(&key, sizeof(Key), seed) % n_buckets;
        

         uint64_t key_hash = hash(&key, sizeof(Key), seed);

         uint64_t bucket_0 = get_first_bucket(key_hash);

         md_bucket_type * md_bucket_0 = get_metadata(bucket_0);
         bucket_type * bucket_0_ptr = get_bucket_ptr(bucket_0);
        

         //first pass is the attempt to upsert/shortcut on primary
         //if this fails enter generic load loop

         //arguments for primary bucket are defined herer - needed for the primary upsert.
         //secondary come before the main non-shortcut loop - shorcut saves registers if possible.


         #if LARGE_BUCKET_MODS
         uint64_t bucket_0_empty;
         uint64_t bucket_0_tombstone;
         uint64_t bucket_0_match;
         #else

         uint32_t bucket_0_empty;
         uint32_t bucket_0_tombstone;
         uint32_t bucket_0_match;

         #endif


         //global load occurs here - if counting loads this is the spot for bucket 0.
         // #if COUNT_INSERT_PROBES
         // ADD_PROBE_BUCKET
         // #endif

         #if LARGE_MD_LOAD
         md_bucket_0->load_fill_ballots_huge(my_tile, key, bucket_0_empty, bucket_0_tombstone, bucket_0_match);

         #else 

         md_bucket_0->load_fill_ballots(my_tile, key, bucket_0_empty, bucket_0_tombstone, bucket_0_match);

         #endif


         // uint bucket_0_empty_small;
         // uint bucket_0_tombstone_small;
         // uint bucket_0_match_small;

         // md_bucket_0->load_fill_ballots_big(my_tile, key, bucket_0_empty_small, bucket_0_tombstone_small, bucket_0_match_small);

         // if (bucket_0_empty != bucket_0_empty_small){
         //    printf("Mismatch %u != %u\n", bucket_0_empty, bucket_0_empty_small);
         // }

         
         //bucket_0_ptr->load_fill_ballots_big(my_tile, key, bucket_0_empty, bucket_0_tombstone, bucket_0_match);

         //size is bucket_size - empty slots (empty + tombstone)
         //this saves an op over (bucket_size - __popcll(bucket_0_empty)) - __popcll(bucket_0_tombstone);

         #if LARGE_BUCKET_MODS
         uint bucket_0_size = bucket_size - __popcll(bucket_0_empty ); //| bucket_0_tombstone
         #else
         uint bucket_0_size = bucket_size - __popc(bucket_0_empty ); //| bucket_0_tombstone
         #endif

         //.75 shortcut for the moment.
         while (bucket_0_size < bucket_size*P2_MD_CUTOFF){


            //if (my_tile.thread_rank() == 0) printf("In shortcut loop %lu, %x ballot_map\n", bucket_0_size, bucket_0_empty);

            #if LARGE_BUCKET_MODS
            if (__popcll(bucket_0_match) != 0)
            #else
            if (__popc(bucket_0_match) != 0)
            #endif
            {

               if (bucket_0_ptr->upsert_existing(my_tile, key, val, bucket_0_match) != -1){
                  return true;
               }

               //match was observed but has changed - move on to tombstone.
               //because of lock other threads cannot interfere in this upsert other than deletion
               //due to stability + lock the key must not exist anymore so we can proceed with insertion.
               //__threadfence();
               // continue;

            }

            int tombstone_pos = md_bucket_0->match_tombstone(my_tile, key, bucket_0_tombstone);

            if (tombstone_pos != -1){

               if (my_tile.thread_rank()  == (tombstone_pos % partition_size)){

                  //temporarily make this atomic?

                  // if (!gallatin::utils::typed_atomic_write(&bucket_0_ptr->slots[tombstone_pos].key, tombstoneKey, key)){

                  //    uint16_t my_tag = md_bucket_0->get_tag(key);
                  //    printf("Failed to replace tombstone %u\n\n", my_tag);
                  // }
                  ADD_PROBE

                  ht_store_packed_pair(&bucket_0_ptr->slots[tombstone_pos], {key,val});

                  // #if ATOMIC_VERIFY

                  // if (!gallatin::utils::typed_atomic_write(&bucket_0_ptr->slots[tombstone_pos].key, tombstoneKey, key)){

                  //    uint16_t my_tag = md_bucket_0->get_tag(key);
                  //    printf("Failed to replace tombstone %u\n\n", my_tag);
                  // }

                  // #else
                  // ht_store(&bucket_0_ptr->slots[tombstone_pos].key, key);
                  // #endif
                  // ht_store(&bucket_0_ptr->slots[tombstone_pos].val, val);

                  __threadfence();

               }

               my_tile.sync();
               return true;

            }

            int empty_pos = md_bucket_0->match_empty(my_tile, key, bucket_0_empty);

            if (empty_pos != -1){

               if (my_tile.thread_rank()  == (empty_pos % partition_size)){


                  // if (!gallatin::utils::typed_atomic_write(&bucket_0_ptr->slots[empty_pos].key, defaultKey, key)){
                  //    printf("Failed to replace empty\n");
                  // }
                  ADD_PROBE

                  ht_store_packed_pair(&bucket_0_ptr->slots[empty_pos], {key,val});

                  // #if ATOMIC_VERIFY
                  // if (!gallatin::utils::typed_atomic_write(&bucket_0_ptr->slots[empty_pos].key, defaultKey, key)){
                  //    printf("Failed to replace empty\n");
                  // }
                  // #else
                  // ht_store(&bucket_0_ptr->slots[empty_pos].key, key);

                  // #endif
                  // ht_store(&bucket_0_ptr->slots[empty_pos].val, val);

                  __threadfence();

               }

               my_tile.sync();
               return true;


            }

            //reload
            #if LARGE_MD_LOAD
               md_bucket_0->load_fill_ballots_huge(my_tile, key, bucket_0_empty, bucket_0_tombstone, bucket_0_match);

            #else 

               md_bucket_0->load_fill_ballots(my_tile, key, bucket_0_empty, bucket_0_tombstone, bucket_0_match);

            #endif


            #if LARGE_BUCKET_MODS
            bucket_0_size = bucket_size - __popcll(bucket_0_empty); // | bucket_0_tombstone);
            #else
            bucket_0_size = bucket_size - __popc(bucket_0_empty);
            #endif


            
         }

         //setup for alternate
         bucket_0_size = bucket_size - __popcll(bucket_0_empty | bucket_0_tombstone); //| bucket_0_tombstone

         uint64_t bucket_1 = get_second_bucket(key_hash);
         md_bucket_type * md_bucket_1 = get_metadata(bucket_1);
         bucket_type * bucket_1_ptr = get_bucket_ptr(bucket_1);

         #if LARGE_BUCKET_MODS
         uint64_t bucket_1_empty;
         uint64_t bucket_1_tombstone;
         uint64_t bucket_1_match;
         #else
         uint32_t bucket_1_empty;
         uint32_t bucket_1_tombstone;
         uint32_t bucket_1_match;
         #endif
         // #if COUNT_INSERT_PROBES
         // ADD_PROBE_BUCKET
         // #endif

         #if LARGE_MD_LOAD
         md_bucket_1->load_fill_ballots_huge(my_tile, key, bucket_1_empty, bucket_1_tombstone, bucket_1_match);

         #else 

         md_bucket_1->load_fill_ballots(my_tile, key, bucket_1_empty, bucket_1_tombstone, bucket_1_match);

         #endif

         #if LARGE_BUCKET_MODS
         uint bucket_1_size = bucket_size - __popcll(bucket_1_empty | bucket_1_tombstone);
         #else
         uint bucket_1_size = bucket_size - __popc(bucket_1_empty | bucket_1_tombstone);
         #endif


         //return false;
         //printf("Reached alternate\n");

         //generic tile loop to perform operations.
         while (bucket_0_size != bucket_size || bucket_1_size != bucket_size){

            //check upserts
            #if LARGE_BUCKET_MODS
            if (__popcll(bucket_0_match) != 0){

               if (bucket_0_ptr->upsert_existing(my_tile, key, val, bucket_0_match) != -1){
                  return true;
               }

            }

            if (__popcll(bucket_1_match) != 0){

               if (bucket_1_ptr->upsert_existing(my_tile, key, val, bucket_1_match) != -1){
                  return true;
               }

            }

            #else

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

            #endif

            //check main insert

            if (bucket_0_size <= bucket_1_size){


               int tombstone_pos_0 = md_bucket_0->match_tombstone(my_tile, key, bucket_0_tombstone);

               if (tombstone_pos_0 != -1){

                  if (my_tile.thread_rank()  == (tombstone_pos_0 % partition_size)){

                     ADD_PROBE

                     ht_store_packed_pair(&bucket_0_ptr->slots[tombstone_pos_0], {key,val});

                     // #if ATOMIC_VERIFY

                     // if (!gallatin::utils::typed_atomic_write(&bucket_0_ptr->slots[tombstone_pos_0].key, tombstoneKey, key)){

                     //    uint16_t my_tag = md_bucket_0->get_tag(key);
                     //    printf("Failed to replace tombstone %u\n\n", my_tag);
                     // }

                     // #else
                     // ht_store(&bucket_0_ptr->slots[tombstone_pos_0].key, key);
                     // #endif
                     // ht_store(&bucket_0_ptr->slots[tombstone_pos_0].val, val);

                     __threadfence();

                  }

                  my_tile.sync();
                  return true;

               }

               int empty_pos_0 = md_bucket_0->match_empty(my_tile, key, bucket_0_empty);

               if (empty_pos_0 != -1){

                  if (my_tile.thread_rank()  == (empty_pos_0 % partition_size)){

                     ADD_PROBE

                     ht_store_packed_pair(&bucket_0_ptr->slots[empty_pos_0], {key,val});
                     
                     // #if ATOMIC_VERIFY
                     // if (!gallatin::utils::typed_atomic_write(&bucket_0_ptr->slots[empty_pos_0].key, defaultKey, key)){
                     //    printf("Failed to replace empty\n");
                     // }
                     // #else
                     // ht_store(&bucket_0_ptr->slots[empty_pos_0].key, key);

                     // #endif

                     // ht_store(&bucket_0_ptr->slots[empty_pos_0].val, val);

                     __threadfence();

                  }

                  my_tile.sync();
                  return true;


               }


            } else {

               int tombstone_pos_1 = md_bucket_1->match_tombstone(my_tile, key, bucket_1_tombstone);

               if (tombstone_pos_1 != -1){

                  if (my_tile.thread_rank()  == (tombstone_pos_1 % partition_size)){

                     ADD_PROBE

                     ht_store_packed_pair(&bucket_1_ptr->slots[tombstone_pos_1], {key,val});

                     // #if ATOMIC_VERIFY

                     // if (!gallatin::utils::typed_atomic_write(&bucket_1_ptr->slots[tombstone_pos_1].key, tombstoneKey, key)){

                     //    uint16_t my_tag = md_bucket_0->get_tag(key);
                     //    printf("Failed to replace tombstone %u\n\n", my_tag);
                     // }

                     // #else
                     // ht_store(&bucket_1_ptr->slots[tombstone_pos_1].key, key);
                     // #endif
         
                     // ht_store(&bucket_1_ptr->slots[tombstone_pos_1].val, val);

                     __threadfence();

                  }

                  my_tile.sync();
                  return true;

               }

               int empty_pos_1 = md_bucket_1->match_empty(my_tile, key, bucket_1_empty);

               if (empty_pos_1 != -1){

                  if (my_tile.thread_rank()  == (empty_pos_1 % partition_size)){

                     ADD_PROBE
                     
                     ht_store_packed_pair(&bucket_1_ptr->slots[empty_pos_1], {key,val});

                     // #if ATOMIC_VERIFY
                     // if (!gallatin::utils::typed_atomic_write(&bucket_1_ptr->slots[empty_pos_1].key, defaultKey, key)){
                     //    printf("Failed to replace empty\n");
                     // }
                     // #else
                     // ht_store(&bucket_1_ptr->slots[empty_pos_1].key, key);

                     // #endif

                     // ht_store(&bucket_1_ptr->slots[empty_pos_1].val, val);

                     __threadfence();

                  }

                  my_tile.sync();
                  return true;


               }
            
            }

            #if LARGE_MD_LOAD
               md_bucket_0->load_fill_ballots_huge(my_tile, key, bucket_0_empty, bucket_0_tombstone, bucket_0_match);

            #else 

               md_bucket_0->load_fill_ballots(my_tile, key, bucket_0_empty, bucket_0_tombstone, bucket_0_match);

            #endif

            #if LARGE_MD_LOAD
               md_bucket_1->load_fill_ballots_huge(my_tile, key, bucket_1_empty, bucket_1_tombstone, bucket_1_match);

            #else 

               md_bucket_1->load_fill_ballots(my_tile, key, bucket_1_empty, bucket_1_tombstone, bucket_1_match);

            #endif


            #if LARGE_BUCKET_MODS
            bucket_1_size = bucket_size - __popcll(bucket_1_empty | bucket_1_tombstone);
            bucket_0_size = bucket_size - __popcll(bucket_0_empty | bucket_0_tombstone);
            #else

            bucket_1_size = bucket_size - __popc(bucket_1_empty | bucket_1_tombstone);
            bucket_0_size = bucket_size - __popc(bucket_0_empty | bucket_0_tombstone);

            #endif


            
         }


         return false;


      }


      __device__ bool upsert_replace_internal(const tile_type & my_tile, const Key & key, const Val & val, uint64_t key_hash, uint64_t bucket_0){

         //uint64_t bucket_0 = hash(&key, sizeof(Key), seed) % n_buckets;
        

         md_bucket_type * md_bucket_0 = get_metadata(bucket_0);
         bucket_type * bucket_0_ptr = get_bucket_ptr(bucket_0);
        

         //first pass is the attempt to upsert/shortcut on primary
         //if this fails enter generic load loop

         //arguments for primary bucket are defined herer - needed for the primary upsert.
         //secondary come before the main non-shortcut loop - shorcut saves registers if possible.


         #if LARGE_BUCKET_MODS
         uint64_t bucket_0_empty;
         uint64_t bucket_0_tombstone;
         uint64_t bucket_0_match;
         #else

         uint32_t bucket_0_empty;
         uint32_t bucket_0_tombstone;
         uint32_t bucket_0_match;

         #endif


         //global load occurs here - if counting loads this is the spot for bucket 0.
         // #if COUNT_INSERT_PROBES
         // ADD_PROBE_BUCKET
         // #endif

         #if LARGE_MD_LOAD
         md_bucket_0->load_fill_ballots_huge(my_tile, key, bucket_0_empty, bucket_0_tombstone, bucket_0_match);

         #else 

         md_bucket_0->load_fill_ballots(my_tile, key, bucket_0_empty, bucket_0_tombstone, bucket_0_match);

         #endif

         // uint bucket_0_empty_small;
         // uint bucket_0_tombstone_small;
         // uint bucket_0_match_small;

         // md_bucket_0->load_fill_ballots_big(my_tile, key, bucket_0_empty_small, bucket_0_tombstone_small, bucket_0_match_small);

         // if (bucket_0_empty != bucket_0_empty_small){
         //    printf("Mismatch %u != %u\n", bucket_0_empty, bucket_0_empty_small);
         // }

         
         //bucket_0_ptr->load_fill_ballots_big(my_tile, key, bucket_0_empty, bucket_0_tombstone, bucket_0_match);

         //size is bucket_size - empty slots (empty + tombstone)
         //this saves an op over (bucket_size - __popcll(bucket_0_empty)) - __popcll(bucket_0_tombstone);

         #if LARGE_BUCKET_MODS
         uint bucket_0_size = bucket_size - __popcll(bucket_0_empty ); //| bucket_0_tombstone
         #else
         uint bucket_0_size = bucket_size - __popc(bucket_0_empty ); //| bucket_0_tombstone
         #endif

         //.75 shortcut for the moment.
         while (bucket_0_size < bucket_size*P2_MD_CUTOFF){


            //if (my_tile.thread_rank() == 0) printf("In shortcut loop %lu, %x ballot_map\n", bucket_0_size, bucket_0_empty);

            #if LARGE_BUCKET_MODS
            if (__popcll(bucket_0_match) != 0)
            #else
            if (__popc(bucket_0_match) != 0)
            #endif
            {

               if (bucket_0_ptr->upsert_existing(my_tile, key, val, bucket_0_match) != -1){
                  return true;
               }

               //match was observed but has changed - move on to tombstone.
               //because of lock other threads cannot interfere in this upsert other than deletion
               //due to stability + lock the key must not exist anymore so we can proceed with insertion.
               //__threadfence();
               // continue;

            }

            int tombstone_pos = md_bucket_0->match_tombstone(my_tile, key, bucket_0_tombstone);

            if (tombstone_pos != -1){

               if (my_tile.thread_rank()  == (tombstone_pos % partition_size)){

                  //temporarily make this atomic?

                  // if (!gallatin::utils::typed_atomic_write(&bucket_0_ptr->slots[tombstone_pos].key, tombstoneKey, key)){

                  //    uint16_t my_tag = md_bucket_0->get_tag(key);
                  //    printf("Failed to replace tombstone %u\n\n", my_tag);
                  // }
                  ADD_PROBE

                  ht_store_packed_pair(&bucket_0_ptr->slots[tombstone_pos], {key,val});

                  // #if ATOMIC_VERIFY

                  // if (!gallatin::utils::typed_atomic_write(&bucket_0_ptr->slots[tombstone_pos].key, tombstoneKey, key)){

                  //    uint16_t my_tag = md_bucket_0->get_tag(key);
                  //    printf("Failed to replace tombstone %u\n\n", my_tag);
                  // }

                  // #else
                  // ht_store(&bucket_0_ptr->slots[tombstone_pos].key, key);
                  // #endif
                  // ht_store(&bucket_0_ptr->slots[tombstone_pos].val, val);

                  __threadfence();

               }

               my_tile.sync();
               return true;

            }

            int empty_pos = md_bucket_0->match_empty(my_tile, key, bucket_0_empty);

            if (empty_pos != -1){

               if (my_tile.thread_rank()  == (empty_pos % partition_size)){


                  // if (!gallatin::utils::typed_atomic_write(&bucket_0_ptr->slots[empty_pos].key, defaultKey, key)){
                  //    printf("Failed to replace empty\n");
                  // }
                  ADD_PROBE

                  ht_store_packed_pair(&bucket_0_ptr->slots[empty_pos], {key,val});

                  // #if ATOMIC_VERIFY
                  // if (!gallatin::utils::typed_atomic_write(&bucket_0_ptr->slots[empty_pos].key, defaultKey, key)){
                  //    printf("Failed to replace empty\n");
                  // }
                  // #else
                  // ht_store(&bucket_0_ptr->slots[empty_pos].key, key);

                  // #endif
                  // ht_store(&bucket_0_ptr->slots[empty_pos].val, val);

                  __threadfence();

               }

               my_tile.sync();
               return true;


            }

            //reload
            #if LARGE_MD_LOAD
               md_bucket_0->load_fill_ballots_huge(my_tile, key, bucket_0_empty, bucket_0_tombstone, bucket_0_match);

            #else 

               md_bucket_0->load_fill_ballots(my_tile, key, bucket_0_empty, bucket_0_tombstone, bucket_0_match);

            #endif


            #if LARGE_BUCKET_MODS
            bucket_0_size = bucket_size - __popcll(bucket_0_empty); // | bucket_0_tombstone);
            #else
            bucket_0_size = bucket_size - __popc(bucket_0_empty);
            #endif

            
         }

         //setup for alternate
         bucket_0_size = bucket_size - __popcll(bucket_0_empty | bucket_0_tombstone); //| bucket_0_tombstone

         uint64_t bucket_1 = get_second_bucket(key_hash);
         md_bucket_type * md_bucket_1 = get_metadata(bucket_1);
         bucket_type * bucket_1_ptr = get_bucket_ptr(bucket_1);

         #if LARGE_BUCKET_MODS
         uint64_t bucket_1_empty;
         uint64_t bucket_1_tombstone;
         uint64_t bucket_1_match;
         #else
         uint32_t bucket_1_empty;
         uint32_t bucket_1_tombstone;
         uint32_t bucket_1_match;
         #endif
         // #if COUNT_INSERT_PROBES
         // ADD_PROBE_BUCKET
         // #endif

         #if LARGE_MD_LOAD
         md_bucket_1->load_fill_ballots_huge(my_tile, key, bucket_1_empty, bucket_1_tombstone, bucket_1_match);

         #else 

         md_bucket_1->load_fill_ballots(my_tile, key, bucket_1_empty, bucket_1_tombstone, bucket_1_match);

         #endif

         #if LARGE_BUCKET_MODS
         uint bucket_1_size = bucket_size - __popcll(bucket_1_empty | bucket_1_tombstone);
         #else
         uint bucket_1_size = bucket_size - __popc(bucket_1_empty | bucket_1_tombstone);
         #endif


         //return false;
         //printf("Reached alternate\n");

         //generic tile loop to perform operations.
         while (bucket_0_size != bucket_size || bucket_1_size != bucket_size){

            //check upserts
            #if LARGE_BUCKET_MODS
            if (__popcll(bucket_0_match) != 0){

               if (bucket_0_ptr->upsert_existing(my_tile, key, val, bucket_0_match) != -1){
                  return true;
               }

            }

            if (__popcll(bucket_1_match) != 0){

               if (bucket_1_ptr->upsert_existing(my_tile, key, val, bucket_1_match) != -1){
                  return true;
               }

            }

            #else

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

            #endif

            //check main insert

            if (bucket_0_size <= bucket_1_size){


               int tombstone_pos_0 = md_bucket_0->match_tombstone(my_tile, key, bucket_0_tombstone);

               if (tombstone_pos_0 != -1){

                  if (my_tile.thread_rank()  == (tombstone_pos_0 % partition_size)){

                     ADD_PROBE


                     ht_store_packed_pair(&bucket_0_ptr->slots[tombstone_pos_0], {key,val});


                     // #if ATOMIC_VERIFY

                     // if (!gallatin::utils::typed_atomic_write(&bucket_0_ptr->slots[tombstone_pos_0].key, tombstoneKey, key)){

                     //    uint16_t my_tag = md_bucket_0->get_tag(key);
                     //    printf("Failed to replace tombstone %u\n\n", my_tag);
                     // }

                     // #else
                     // ht_store(&bucket_0_ptr->slots[tombstone_pos_0].key, key);
                     // #endif
                     // ht_store(&bucket_0_ptr->slots[tombstone_pos_0].val, val);

                     __threadfence();

                  }

                  my_tile.sync();
                  return true;

               }

               int empty_pos_0 = md_bucket_0->match_empty(my_tile, key, bucket_0_empty);

               if (empty_pos_0 != -1){

                  if (my_tile.thread_rank()  == (empty_pos_0 % partition_size)){

                     ADD_PROBE

                     ht_store_packed_pair(&bucket_0_ptr->slots[empty_pos_0], {key,val});
                     
                     // #if ATOMIC_VERIFY
                     // if (!gallatin::utils::typed_atomic_write(&bucket_0_ptr->slots[empty_pos_0].key, defaultKey, key)){
                     //    printf("Failed to replace empty\n");
                     // }
                     // #else
                     // ht_store(&bucket_0_ptr->slots[empty_pos_0].key, key);

                     // #endif

                     // ht_store(&bucket_0_ptr->slots[empty_pos_0].val, val);

                     __threadfence();

                  }

                  my_tile.sync();
                  return true;


               }


            } else {

               int tombstone_pos_1 = md_bucket_1->match_tombstone(my_tile, key, bucket_1_tombstone);

               if (tombstone_pos_1 != -1){

                  if (my_tile.thread_rank()  == (tombstone_pos_1 % partition_size)){

                     ADD_PROBE

                     ht_store_packed_pair(&bucket_1_ptr->slots[tombstone_pos_1], {key,val});

                     // #if ATOMIC_VERIFY

                     // if (!gallatin::utils::typed_atomic_write(&bucket_1_ptr->slots[tombstone_pos_1].key, tombstoneKey, key)){

                     //    uint16_t my_tag = md_bucket_0->get_tag(key);
                     //    printf("Failed to replace tombstone %u\n\n", my_tag);
                     // }

                     // #else
                     // ht_store(&bucket_1_ptr->slots[tombstone_pos_1].key, key);
                     // #endif
         
                     // ht_store(&bucket_1_ptr->slots[tombstone_pos_1].val, val);

                     __threadfence();

                  }

                  my_tile.sync();
                  return true;

               }

               int empty_pos_1 = md_bucket_1->match_empty(my_tile, key, bucket_1_empty);

               if (empty_pos_1 != -1){

                  if (my_tile.thread_rank()  == (empty_pos_1 % partition_size)){

                     ADD_PROBE


                     ht_store_packed_pair(&bucket_1_ptr->slots[empty_pos_1], {key,val});
                     

                     // #if ATOMIC_VERIFY
                     // if (!gallatin::utils::typed_atomic_write(&bucket_1_ptr->slots[empty_pos_1].key, defaultKey, key)){
                     //    printf("Failed to replace empty\n");
                     // }
                     // #else
                     // ht_store(&bucket_1_ptr->slots[empty_pos_1].key, key);

                     // #endif

                     // ht_store(&bucket_1_ptr->slots[empty_pos_1].val, val);

                     __threadfence();

                  }

                  my_tile.sync();
                  return true;


               }
            
            }

            #if LARGE_MD_LOAD
               md_bucket_0->load_fill_ballots_huge(my_tile, key, bucket_0_empty, bucket_0_tombstone, bucket_0_match);

            #else 

               md_bucket_0->load_fill_ballots(my_tile, key, bucket_0_empty, bucket_0_tombstone, bucket_0_match);

            #endif

            #if LARGE_MD_LOAD
               md_bucket_1->load_fill_ballots_huge(my_tile, key, bucket_1_empty, bucket_1_tombstone, bucket_1_match);

            #else 

               md_bucket_1->load_fill_ballots(my_tile, key, bucket_1_empty, bucket_1_tombstone, bucket_1_match);

            #endif


            #if LARGE_BUCKET_MODS
            bucket_1_size = bucket_size - __popcll(bucket_1_empty | bucket_1_tombstone);
            bucket_0_size = bucket_size - __popcll(bucket_0_empty | bucket_0_tombstone);
            #else

            bucket_1_size = bucket_size - __popc(bucket_1_empty | bucket_1_tombstone);
            bucket_0_size = bucket_size - __popc(bucket_0_empty | bucket_0_tombstone);

            #endif


            
         }


         return false;


      }

      [[nodiscard]] __device__ bool find_with_reference(tile_type my_tile, Key key, Val & val){



         uint64_t key_hash = hash(&key, sizeof(Key), seed);
         uint64_t bucket_0 = get_first_bucket(key_hash);


         md_bucket_type * md_bucket_0 = get_metadata(bucket_0);

         bucket_type * bucket_0_ptr = get_bucket_ptr(bucket_0);

         // #if LARGE_BUCKET_MODS
         // uint64_t bucket_0_empty;
         // uint64_t bucket_0_tombstone;
         // uint64_t bucket_0_match;
         // #else
         // uint32_t bucket_0_empty;
         // uint32_t bucket_0_tombstone;
         // uint32_t bucket_0_match;
         // #endif

         //probe 1.


         if (md_bucket_0->query_md_and_bucket_large(my_tile, key, val, bucket_0_ptr)){

            return true;
         }

         // #if LARGE_BUCKET_MODS
         // if (bucket_size - __popcll(bucket_0_empty) < bucket_size*P2_MD_CUTOFF){

         //    unlock(my_tile, bucket_0);

         //    return false;

         // }

         // #else
         // if (bucket_size - __popc(bucket_0_empty) < bucket_size*P2_MD_CUTOFF){

         //    unlock(my_tile, bucket_0);

         //    return false;

         // }
         // #endif

         uint64_t bucket_1 = get_second_bucket(key_hash);

         md_bucket_type * md_bucket_1 = get_metadata(bucket_1);
         bucket_type * bucket_1_ptr = get_bucket_ptr(bucket_1);

         if (md_bucket_1->query_md_and_bucket_large(my_tile, key, val, bucket_1_ptr)){

            return true;
         }

         // if (bucket_0_ptr->query(my_tile, key, val)) return true;
         // if (bucket_1_ptr->query(my_tile, key, val)) return true;

         return false;

      }


      [[nodiscard]] __device__ bool find_with_reference_no_lock(tile_type my_tile, Key key, Val & val){



         uint64_t key_hash = hash(&key, sizeof(Key), seed);
         uint64_t bucket_0 = get_first_bucket(key_hash);

         //stall_lock(my_tile, bucket_0);

         md_bucket_type * md_bucket_0 = get_metadata(bucket_0);

         bucket_type * bucket_0_ptr = get_bucket_ptr(bucket_0);


         if (md_bucket_0->query_md_and_bucket_large(my_tile, key, val, bucket_0_ptr)){

            //unlock(my_tile, bucket_0);
            return true;
         }



         uint64_t bucket_1 = get_second_bucket(key_hash);

         md_bucket_type * md_bucket_1 = get_metadata(bucket_1);
         bucket_type * bucket_1_ptr = get_bucket_ptr(bucket_1);

         if (md_bucket_1->query_md_and_bucket_large(my_tile, key, val, bucket_1_ptr)){

            //unlock(my_tile, bucket_0);
            return true;
         }

         //unlock(my_tile, bucket_0);
         return false;

      }


      [[nodiscard]] __device__ packed_pair_type * find_pair(const tile_type my_tile, const Key key){



         uint64_t key_hash = hash(&key, sizeof(Key), seed);
         uint64_t bucket_0 = get_first_bucket(key_hash);

         md_bucket_type * md_bucket_0 = get_metadata(bucket_0);

         bucket_type * bucket_0_ptr = get_bucket_ptr(bucket_0);




         packed_pair_type * return_pair = md_bucket_0->query_md_and_bucket_pair(my_tile, key, bucket_0_ptr);

         if (return_pair != nullptr){
            return return_pair;
         }

         uint64_t bucket_1 = get_second_bucket(key_hash);

         md_bucket_type * md_bucket_1 = get_metadata(bucket_1);
         bucket_type * bucket_1_ptr = get_bucket_ptr(bucket_1);

         return_pair = md_bucket_1->query_md_and_bucket_pair(my_tile, key, bucket_1_ptr);

         return return_pair;

      }


      [[nodiscard]] __device__ packed_pair_type * find_pair_no_lock(const tile_type my_tile, const Key key){



         uint64_t key_hash = hash(&key, sizeof(Key), seed);
         uint64_t bucket_0 = get_first_bucket(key_hash);


         md_bucket_type * md_bucket_0 = get_metadata(bucket_0);

         bucket_type * bucket_0_ptr = get_bucket_ptr(bucket_0);



         packed_pair_type * return_pair = md_bucket_0->query_md_and_bucket_pair(my_tile, key, bucket_0_ptr);

         if (return_pair != nullptr){
            return return_pair;
         }

         uint64_t bucket_1 = get_second_bucket(key_hash);

         md_bucket_type * md_bucket_1 = get_metadata(bucket_1);
         bucket_type * bucket_1_ptr = get_bucket_ptr(bucket_1);

         return_pair = md_bucket_1->query_md_and_bucket_pair(my_tile, key, bucket_1_ptr);

         return return_pair;

      }



   

      __device__ bool remove(tile_type my_tile, Key key){


         uint64_t key_hash = hash(&key, sizeof(Key), seed);
         uint64_t bucket_0 = get_first_bucket(key_hash);
        
         //uint64_t bucket_0 = hash(&key, sizeof(Key), seed) % n_buckets;

         md_bucket_type * md_bucket_0 = get_metadata(bucket_0);

         bucket_type * bucket_0_ptr = get_bucket_ptr(bucket_0);

         stall_lock(my_tile, bucket_0);

         #if LARGE_BUCKET_MODS
         uint64_t bucket_0_empty;
         uint64_t bucket_0_tombstone;
         uint64_t bucket_0_match;
         #else
         uint32_t bucket_0_empty;
         uint32_t bucket_0_tombstone;
         uint32_t bucket_0_match;
         #endif

         //probe 1.

         #if LARGE_MD_LOAD
         md_bucket_0->load_fill_ballots_huge(my_tile, key, bucket_0_empty, bucket_0_tombstone, bucket_0_match);
         #else
         md_bucket_0->load_fill_ballots(my_tile, key, bucket_0_empty, bucket_0_tombstone, bucket_0_match);
         #endif

         int erase_index = bucket_0_ptr->erase_reference(my_tile, key, bucket_0_match);

         if (erase_index != -1){

            if (erase_index % partition_size == my_tile.thread_rank()){

               md_bucket_0->set_tombstone(my_tile, erase_index);

            }

            unlock(my_tile, bucket_0);

            return true;

         }
       

         uint64_t bucket_1 = get_second_bucket(key_hash);
         bucket_type * bucket_1_ptr = get_bucket_ptr(bucket_1);

          md_bucket_type * md_bucket_1 = get_metadata(bucket_1);
         // #if COUNT_INSERT_PROBES
         // ADD_PROBE_BUCKET
         // #endif

         #if LARGE_MD_LOAD
         md_bucket_1->load_fill_ballots_huge(my_tile, key, bucket_0_empty, bucket_0_tombstone, bucket_0_match);
         #else
         md_bucket_1->load_fill_ballots(my_tile, key, bucket_0_empty, bucket_0_tombstone, bucket_0_match);
         #endif

         erase_index = bucket_1_ptr->erase_reference(my_tile, key, bucket_0_match);

         if (erase_index != -1){


            if (erase_index % partition_size == my_tile.thread_rank()){

               md_bucket_1->set_tombstone(my_tile, erase_index);

            }


            unlock(my_tile, bucket_0);

            return true;

         }

         unlock(my_tile, bucket_0);

         return false;

      }


      __device__ bool remove_no_lock(tile_type my_tile, Key key){


         uint64_t key_hash = hash(&key, sizeof(Key), seed);
         uint64_t bucket_0 = get_first_bucket(key_hash);
        
         //uint64_t bucket_0 = hash(&key, sizeof(Key), seed) % n_buckets;

         md_bucket_type * md_bucket_0 = get_metadata(bucket_0);

         bucket_type * bucket_0_ptr = get_bucket_ptr(bucket_0);

         //stall_lock(my_tile, bucket_0);

         #if LARGE_BUCKET_MODS
         uint64_t bucket_0_empty;
         uint64_t bucket_0_tombstone;
         uint64_t bucket_0_match;
         #else
         uint32_t bucket_0_empty;
         uint32_t bucket_0_tombstone;
         uint32_t bucket_0_match;
         #endif

         //probe 1.

         #if LARGE_MD_LOAD
         md_bucket_0->load_fill_ballots_huge(my_tile, key, bucket_0_empty, bucket_0_tombstone, bucket_0_match);
         #else
         md_bucket_0->load_fill_ballots(my_tile, key, bucket_0_empty, bucket_0_tombstone, bucket_0_match);
         #endif

         int erase_index = bucket_0_ptr->erase_reference(my_tile, key, bucket_0_match);

         if (erase_index != -1){

            if (erase_index % partition_size == my_tile.thread_rank()){

               md_bucket_0->set_tombstone(my_tile, erase_index);

            }

            //unlock(my_tile, bucket_0);

            return true;

         }
       

         uint64_t bucket_1 = get_second_bucket(key_hash);
         bucket_type * bucket_1_ptr = get_bucket_ptr(bucket_1);

          md_bucket_type * md_bucket_1 = get_metadata(bucket_1);
         // #if COUNT_INSERT_PROBES
         // ADD_PROBE_BUCKET
         // #endif

         #if LARGE_MD_LOAD
         md_bucket_1->load_fill_ballots_huge(my_tile, key, bucket_0_empty, bucket_0_tombstone, bucket_0_match);
         #else
         md_bucket_1->load_fill_ballots(my_tile, key, bucket_0_empty, bucket_0_tombstone, bucket_0_match);
         #endif

         erase_index = bucket_1_ptr->erase_reference(my_tile, key, bucket_0_match);

         if (erase_index != -1){


            if (erase_index % partition_size == my_tile.thread_rank()){

               md_bucket_1->set_tombstone(my_tile, erase_index);

            }


            //unlock(my_tile, bucket_0);

            return true;

         }

         //unlock(my_tile, bucket_0);

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

      static std::string get_name(){
         return "p2_hashing_metadata";
      }

      __host__ void print_space_usage(){

         my_type * host_version = gallatin::utils::copy_to_host<my_type>(this);
            
         uint64_t capacity = host_version->n_buckets*(sizeof(bucket_type)+sizeof(md_bucket_type)) + (host_version->n_buckets-1)/8+1; 

         cudaFreeHost(host_version);

         printf("p2_metadata_table using %lu bytes\n", capacity);

      }


      __host__ void print_fill(){


         uint64_t * n_items;

         cudaMallocManaged((void **)&n_items, sizeof(uint64_t));

         n_items[0] = 0;

         my_type * host_version = gallatin::utils::copy_to_host<my_type>(this);

         uint64_t n_buckets = host_version->n_buckets;


         p2_md_get_fill_kernel<my_type, partition_size><<<(n_buckets*partition_size-1)/256+1,256>>>(this, n_buckets, n_items);

         cudaDeviceSynchronize();
         printf("fill: %lu/%lu = %f%%\n", n_items[0], n_buckets*bucket_size, 100.0*n_items[0]/(n_buckets*bucket_size));

         cudaFree(n_items);
         cudaFreeHost(host_version);



      }

      __host__ uint64_t get_fill(){


         uint64_t * n_items;

         cudaMallocManaged((void **)&n_items, sizeof(uint64_t));

         n_items[0] = 0;

         my_type * host_version = gallatin::utils::copy_to_host<my_type>(this);

         uint64_t n_buckets = host_version->n_buckets;


         p2_md_get_fill_kernel<my_type, partition_size><<<(n_buckets*partition_size-1)/256+1,256>>>(this, n_buckets, n_items);

         cudaDeviceSynchronize();
   
         uint64_t return_items = n_items[0];

         cudaFree(n_items);
         cudaFreeHost(host_version);

         return return_items;


      }

      __device__ uint64_t get_bucket_fill(tile_type my_tile, uint64_t bucket){


         md_bucket_type * md_bucket = get_metadata(bucket);

         #if LARGE_BUCKET_MODS
         uint64_t bucket_empty;
         uint64_t bucket_tombstone;
         uint64_t bucket_match;
         #else

         uint32_t bucket_empty;
         uint32_t bucket_tombstone;
         uint32_t bucket_match;

         #endif


         //global load occurs here - if counting loads this is the spot for bucket 0.
         // #if COUNT_INSERT_PROBES
         // ADD_PROBE_BUCKET
         // #endif

         #if LARGE_MD_LOAD
         md_bucket->load_fill_ballots_huge(my_tile, defaultKey, bucket_empty, bucket_tombstone, bucket_match);

         return bucket_size-__popcll(bucket_empty)-__popcll(bucket_tombstone);

         #else 

         md_bucket->load_fill_ballots(my_tile, defaultKey, bucket_empty, bucket_tombstone, bucket_match);

         return bucket_size-__popc(bucket_empty) - __popc(bucket_tombstone);

         #endif



      }


   };

template <typename T>
constexpr T generate_p2_md_tombstone(uint64_t offset) {
  return (~((T) 0)) - offset;
};

template <typename T>
constexpr T generate_p2_md_sentinel() {
  return ((T) 0);
};


// template <typename Key, Key sentinel, Key tombstone, typename Val, Val defaultVal, Val tombstoneVal, uint partition_size, uint bucket_size>
  
template <typename Key, typename Val, uint tile_size, uint bucket_size>
using md_p2_generic = typename warpSpeed::tables::p2_metadata_table<Key,
                                    generate_p2_md_sentinel<Key>(),
                                    generate_p2_md_tombstone<Key>(0),
                                    Val,
                                    generate_p2_md_sentinel<Val>(),
                                    generate_p2_md_tombstone<Val>(0),
                                    tile_size,
                                    bucket_size>;




} //namespace wrappers

}  // namespace ht_project

#endif  // GPU_BLOCK_