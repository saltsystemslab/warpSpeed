#ifndef WARPSPEED_DOUBLE_META
#define WARPSPEED_DOUBLE_META

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

#define META_MAX_PROBES 20


#define COUNT_INSERT_PROBES 1

#define LARGE_MD_LOAD 1


#define ATOMIC_VERIFY 0

//modifications required to use buckets > 32
#define LARGE_BUCKET_MODS 0



//cache protocol
//query cache
//on success add to pin?
//need delete from potential buckets implementation - need to download warpcore...
//buidld with primary p2bht first.



namespace warpSpeed {

namespace tables {


   //load 8 tags and pack them
   __device__ packed_tags double_load_multi_tags(const uint16_t * start_of_bucket){


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
   __global__ void double_md_get_fill_kernel(HT * metadata_table, uint64_t n_buckets, uint64_t * item_count){


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
   __global__ void init_double_md_table_kernel(table * hash_table){

      uint64_t tid = gallatin::utils::get_tid();

      hash_table->init_bucket_and_locks(tid);
      

   }



   template <typename Key, Key defaultKey, Key tombstoneKey, typename Val, Val defaultVal, Val tombstoneVal, uint partition_size, uint bucket_size>
   struct double_metadata_table {


      using my_type = double_metadata_table<Key, defaultKey, tombstoneKey, Val, defaultVal, tombstoneVal, partition_size, bucket_size>;


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

         init_double_md_table_kernel<my_type><<<(ext_n_buckets-1)/256+1,256>>>(device_version);

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

      __device__ uint64_t get_stride(uint64_t hash){
         return (hash >> 32);
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


      __device__ bool query_internal(tile_type my_tile, Key key, Val & val, uint64_t bucket_primary, uint64_t step){


         //uint64_t bucket_primary = hash(&key, sizeof(Key), seed) % n_buckets_primary;
         //uint64_t step = hash(&key, sizeof(Key), seed+1);


         for (int i = 0; i < META_MAX_PROBES; i++){


            uint64_t bucket_index = (bucket_primary + step*i) % n_buckets;

            md_bucket_type * md_bucket = get_metadata(bucket_index);
            bucket_type * bucket_ptr = get_bucket_ptr(bucket_index);
      


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
            md_bucket->load_fill_ballots_huge(my_tile, key, bucket_empty, bucket_tombstone, bucket_match);

            #else 

            md_bucket->load_fill_ballots(my_tile, key, bucket_empty, bucket_tombstone, bucket_match);

            #endif

            if (bucket_ptr->query_match(my_tile, key, val, bucket_match)){
               return true;
            }


            //shortcutting
            if (__popc(bucket_empty) > 0) return false;


         }

         return false;


      }

      __device__ packed_pair_type * query_packed_reference(tile_type my_tile, Key key, uint64_t bucket_primary, uint64_t step){


         //uint64_t bucket_primary = hash(&key, sizeof(Key), seed) % n_buckets_primary;
         //uint64_t step = hash(&key, sizeof(Key), seed+1);


         for (int i = 0; i < META_MAX_PROBES; i++){


            uint64_t bucket_index = (bucket_primary + step*i) % n_buckets;

            md_bucket_type * md_bucket = get_metadata(bucket_index);
            bucket_type * bucket_ptr = get_bucket_ptr(bucket_index);
      


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
            ADD_PROBE_BUCKET
            // #endif

            #if LARGE_MD_LOAD
            md_bucket->load_fill_ballots_huge(my_tile, key, bucket_empty, bucket_tombstone, bucket_match);

            #else 

            md_bucket->load_fill_ballots(my_tile, key, bucket_empty, bucket_tombstone, bucket_match);

            #endif


            packed_pair_type * val_loc = bucket_ptr->query_pair_ballot(my_tile, key, bucket_match);

            if (val_loc != nullptr){
               return val_loc;
            } 

            //shortcutting
            if (__popc(bucket_empty) > 0) return nullptr;


         }

         return nullptr;


      }


      __device__ bool remove_internal(tile_type my_tile, Key key, uint64_t bucket_primary, uint64_t step){


         //uint64_t bucket_primary = hash(&key, sizeof(Key), seed) % n_buckets_primary;
         //uint64_t step = hash(&key, sizeof(Key), seed+1);


         for (int i = 0; i < META_MAX_PROBES; i++){


            uint64_t bucket_index = (bucket_primary + step*i) % n_buckets;

            md_bucket_type * md_bucket = get_metadata(bucket_index);
            bucket_type * bucket_ptr = get_bucket_ptr(bucket_index);
      


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
            md_bucket->load_fill_ballots_huge(my_tile, key, bucket_empty, bucket_tombstone, bucket_match);

            #else 

            md_bucket->load_fill_ballots(my_tile, key, bucket_empty, bucket_tombstone, bucket_match);

            #endif

            int erased_index = bucket_ptr->erase_reference(my_tile, key, bucket_match);

            if (erased_index != -1){

               if (my_tile.thread_rank() == 0){
                  md_bucket->set_tombstone(erased_index);
               }

               my_tile.sync();

               return true;

            }

            //shortcutting
            if (__popc(bucket_empty) > 0) return false;


         }

         return false;


      }


      __device__ bool upsert_function(const tile_type & my_tile, const Key & key, const Val & val, void (*replace_func)(packed_pair_type *, Key, Val)){


         uint64_t key_hash = hash(&key, sizeof(Key), seed);
         uint64_t bucket_0 = get_first_bucket(key_hash);
         uint64_t step = get_stride(key_hash);
         
         stall_lock(my_tile, bucket_0);


         packed_pair_type * existing_loc = query_packed_reference(my_tile, key, bucket_0, step);

         if (existing_loc != nullptr){


            if (my_tile.thread_rank() == 0){

               //ht_store(&existing_loc->val, val);

               replace_func(existing_loc, key, val);
              __threadfence();

            }

            //this syncs.
            unlock(my_tile, bucket_0);
            //my_tile.sync();


            return true;
         }     

         bool return_val = upsert_replace_internal(my_tile, key, val, bucket_0, step);

         unlock(my_tile, bucket_0);

         return return_val;

      }

      __device__ bool upsert_function_no_lock(const tile_type & my_tile, const Key & key, const Val & val, void (*replace_func)(packed_pair_type *, Key, Val)){


         uint64_t key_hash = hash(&key, sizeof(Key), seed);
         uint64_t bucket_0 = get_first_bucket(key_hash);
         uint64_t step = get_stride(key_hash);
         
     


         packed_pair_type * existing_loc = query_packed_reference(my_tile, key, bucket_0, step);

         if (existing_loc != nullptr){


            if (my_tile.thread_rank() == 0){

               //ht_store(&existing_loc->val, val);

               replace_func(existing_loc, key, val);
              __threadfence();

            }

            //this syncs.
           
            //my_tile.sync();


            return true;
         }     

         bool return_val = upsert_replace_internal(my_tile, key, val, bucket_0, step);



         return return_val;

      }



      __device__ bool upsert_replace(const tile_type & my_tile, const Key & key, const Val & val){


         uint64_t key_hash = hash(&key, sizeof(Key), seed);
         uint64_t bucket_0 = get_first_bucket(key_hash);
         uint64_t step = get_stride(key_hash);
         
         stall_lock(my_tile, bucket_0);


         packed_pair_type * existing_loc = query_packed_reference(my_tile, key, bucket_0, step);

         if (existing_loc != nullptr){


            if (my_tile.thread_rank() == 0){

               ht_store(&existing_loc->val, val);
              __threadfence();

            }

            //this syncs.
            unlock(my_tile, bucket_0);
            //my_tile.sync();


            return true;
         }     

         bool return_val = upsert_replace_internal(my_tile, key, val, bucket_0, step);

         unlock(my_tile, bucket_0);

         return return_val;

      }

      __device__ bool upsert_no_lock(const tile_type & my_tile, const Key & key, const Val & val){


         uint64_t key_hash = hash(&key, sizeof(Key), seed);
         uint64_t bucket_0 = get_first_bucket(key_hash);
         uint64_t step = get_stride(key_hash);
         


         packed_pair_type * existing_loc = query_packed_reference(my_tile, key, bucket_0, step);

         if (existing_loc != nullptr){


            if (my_tile.thread_rank() == 0){

               ht_store(&existing_loc->val, val);
              __threadfence();

            }

            //this syncs.
            //my_tile.sync();


            return true;
         }     

         bool return_val = upsert_replace_internal(my_tile, key, val, bucket_0, step);

         return return_val;

      }

      __device__ bool upsert_replace_internal(const tile_type & my_tile, const Key & key, const Val & val, uint64_t bucket_primary, uint64_t step){

         //uint64_t bucket_0 = hash(&key, sizeof(Key), seed) % n_buckets;

         for (int i = 0; i < META_MAX_PROBES; i++){

            uint64_t bucket_index = (bucket_primary + step*i) % n_buckets;
            md_bucket_type * md_bucket = get_metadata(bucket_index);
            bucket_type * bucket_ptr = get_bucket_ptr(bucket_index);


            #if LARGE_BUCKET_MODS
               uint64_t bucket_empty;
               uint64_t bucket_tombstone;
               uint64_t bucket_match;
            #else
               uint32_t bucket_empty;
               uint32_t bucket_tombstone;
               uint32_t bucket_match;

            #endif


            #if LARGE_MD_LOAD
            md_bucket->load_fill_ballots_huge(my_tile, key, bucket_empty, bucket_tombstone, bucket_match);

            #else 

            md_bucket->load_fill_ballots(my_tile, key, bucket_empty, bucket_tombstone, bucket_match);

            #endif


            int result = md_bucket->match_empty(my_tile, key, bucket_empty);

            if (result != -1){

               ht_store_packed_pair(&bucket_ptr->slots[result], {key, val});

               // ht_store(&bucket_ptr->slots[result].key, key);
               // ht_store(&bucket_ptr->slots[result].val, val);

               return true;
            }

            result = md_bucket->match_tombstone(my_tile, key, bucket_tombstone);

            if (result != -1){

               ht_store_packed_pair(&bucket_ptr->slots[result], {key, val});

               return true;
            }


         }

         return false;

      }

   [[nodiscard]] __device__ bool find_with_reference(tile_type my_tile, Key key, Val & val){


         //return false;


         uint64_t key_hash = hash(&key, sizeof(Key), seed);
         uint64_t bucket_primary = get_first_bucket(key_hash);
         uint64_t step = get_stride(key_hash);

         return query_internal(my_tile, key, val, bucket_primary, step);

   }

   [[nodiscard]] __device__ packed_pair_type * find_pair(tile_type my_tile, Key key){


         //return false;


         uint64_t key_hash = hash(&key, sizeof(Key), seed);
         uint64_t bucket_primary = get_first_bucket(key_hash);
         uint64_t step = get_stride(key_hash);


         //stall_lock(my_tile, bucket_primary);

         packed_pair_type * val_location = query_packed_reference(my_tile, key, bucket_primary, step);

         //unlock(my_tile, bucket_primary);

         return val_location;

   }

   [[nodiscard]] __device__ packed_pair_type * find_pair_no_lock(tile_type my_tile, Key key){


         //return false;


         uint64_t key_hash = hash(&key, sizeof(Key), seed);
         uint64_t bucket_primary = get_first_bucket(key_hash);
         uint64_t step = get_stride(key_hash);


         

         packed_pair_type * val_location = query_packed_reference(my_tile, key, bucket_primary, step);


         return val_location;

   }


   [[nodiscard]] __device__ bool find_with_reference_no_lock(tile_type my_tile, Key key, Val & val){


         //return false;


         uint64_t key_hash = hash(&key, sizeof(Key), seed);
         uint64_t bucket_primary = get_first_bucket(key_hash);
         uint64_t step = get_stride(key_hash);



         packed_pair_type * val_location = query_packed_reference(my_tile, key, bucket_primary, step);

         if (val_location == nullptr){
            //unlock(my_tile, bucket_primary);
            return false;

         }

         val = hash_table_load(&val_location->val);
         __threadfence();

         //unlock(my_tile, bucket_primary);
         return true;

   }

   

      __device__ bool remove(tile_type my_tile, Key key){



         uint64_t key_hash = hash(&key, sizeof(Key), seed);
         uint64_t bucket_primary = get_first_bucket(key_hash);
         uint64_t step = get_stride(key_hash);


         stall_lock(my_tile, bucket_primary);


         bool result = remove_internal(my_tile, key, bucket_primary, step);


         unlock(my_tile, bucket_primary);

         return result;


      }

      __device__ bool remove_no_lock(tile_type my_tile, Key key){



         uint64_t key_hash = hash(&key, sizeof(Key), seed);
         uint64_t bucket_primary = get_first_bucket(key_hash);
         uint64_t step = get_stride(key_hash);


         bool result = remove_internal(my_tile, key, bucket_primary, step);



         return result;


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
         return "double_hashing_metadata";
      }

      __host__ void print_space_usage(){

         my_type * host_version = gallatin::utils::copy_to_host<my_type>(this);
            
         uint64_t capacity = host_version->n_buckets*(sizeof(bucket_type)+sizeof(md_bucket_type)) + (host_version->n_buckets-1)/8+1; 

         cudaFreeHost(host_version);

         printf("double_metadata_table using %lu bytes\n", capacity);

      }


      __host__ void print_fill(){


         uint64_t * n_items;

         cudaMallocManaged((void **)&n_items, sizeof(uint64_t));

         n_items[0] = 0;

         my_type * host_version = gallatin::utils::copy_to_host<my_type>(this);

         uint64_t n_buckets = host_version->n_buckets;


         double_md_get_fill_kernel<my_type, partition_size><<<(n_buckets*partition_size-1)/256+1,256>>>(this, n_buckets, n_items);

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


         double_md_get_fill_kernel<my_type, partition_size><<<(n_buckets*partition_size-1)/256+1,256>>>(this, n_buckets, n_items);

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
constexpr T generate_double_md_tombstone(uint64_t offset) {
  return (~((T) 0)) - offset;
};

template <typename T>
constexpr T generate_double_md_sentinel() {
  return ((T) 0);
};


// template <typename Key, Key sentinel, Key tombstone, typename Val, Val defaultVal, Val tombstoneVal, uint partition_size, uint bucket_size>
  
template <typename Key, typename Val, uint tile_size, uint bucket_size>
using md_double_generic = typename warpSpeed::tables::double_metadata_table<Key,
                                    generate_double_md_sentinel<Key>(),
                                    generate_double_md_tombstone<Key>(0),
                                    Val,
                                    generate_double_md_sentinel<Val>(),
                                    generate_double_md_tombstone<Val>(0),
                                    tile_size,
                                    bucket_size>;




} //namespace wrappers

}  // namespace ht_project

#endif  // GPU_BLOCK_