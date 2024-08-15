#ifndef GALLATIN_COOP_CHAINING_HASH
#define GALLATIN_COOP_CHAINING_HASH


#include <cuda.h>
#include <cuda_runtime_api.h>

//alloc utils needed for easy host_device transfer
#include <gallatin/allocators/global_allocator.cuh>


//murmurhash
#include <gallatin/allocators/murmurhash.cuh>

#include <gallatin/data_structs/ds_utils.cuh>

#include <gallatin/data_structs/callocable.cuh>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>
#include <hashing_project/helpers/probe_counts.cuh>
#include <hashing_project/helpers/ht_load.cuh>


#define COUNT_CHAINING_NEXT_LOAD 0

namespace cg = cooperative_groups;

namespace hashing_project {

namespace tables {


   template <typename ht_type>
   __global__ void chaining_table_fill_buffers(ht_type * table){


      uint64_t tid = gallatin::utils::get_tid();

      if (tid >= table->nblocks) return;


      auto block_ptr = &table->pointer_list[tid];

      table->attach_block(block_ptr);

   }

   // template <typename Key, typename Val>
   // __device__ ht_pair<Key, Val> load_chain_packed_pair(ht_pair<Key, Val> * pair_to_load){
   //    asm volatile ("trap;");
   // }


   // template<>
   // __device__ ht_pair<uint32_t, uint32_t> load_chain_packed_pair(ht_pair<uint32_t, uint32_t> * pair_to_load){

   //       //load 8 tags and pack them

   //          ht_pair<uint32_t,uint32_t> loaded_pair;

   //          asm volatile("ld.gpu.acquire.v2.u32 {%0,%1}, [%2];" : "=r"(loaded_pair.key), "=r"(loaded_pair.val) : "l"(&pair_to_load));
            
   //          return loaded_pair;


   // }

   // template<>
   // __device__ ht_pair<uint64_t, uint64_t> load_chain_packed_pair(ht_pair<uint64_t, uint64_t> * pair_to_load){

   //       //load 8 tags and pack them

   //       ht_pair<uint64_t,uint64_t> loaded_pair;

   //       asm volatile("ld.gpu.acquire.v2.u64 {%0,%1}, [%2];" : "=l"(loaded_pair.key), "=l"(loaded_pair.val) : "l"(&pair_to_load));
         
   //       return loaded_pair;


   // }


   template <typename ht_type>
   __global__ void calculate_chain_kernel(ht_type * table, uint64_t * max, uint64_t * avg, uint64_t nblocks){


      uint64_t tid = gallatin::utils::get_tid();

      if (tid >= nblocks) return;

      table->calculate_chain_length(max, avg, tid);

   }



   template <typename ht_type>
   __global__ void free_chains_kernel(ht_type * table, uint64_t nblocks){

      uint64_t tid = gallatin::utils::get_tid();

      if (tid >= nblocks) return;

      table->free_chain(tid);

   }


   template <typename ht_type>
   __global__ void count_n_chains(ht_type * table, uint64_t nblocks, uint64_t * count){
      uint64_t tid = gallatin::utils::get_tid();

      if (tid >= nblocks) return;

      uint64_t my_len = table->count_chain_length(tid);

      atomicAdd((unsigned long long int *)count, (unsigned long long int) my_len);


   }



   template <typename Key, Key defaultKey, Key tombstoneKey, typename Val, Val defaultVal, Val tombstoneVal, uint partition_size, uint bucket_size>
   struct coop_chaining_block {


      //static_assert(size >= partition_size, "size must be at least as large as team size");
      //static_assert((size % partition_size) == 0, "team size must be clean divisor of size"); 




      using pair_type = ht_pair<Key, Val>;

      using my_type = coop_chaining_block<Key, defaultKey, tombstoneKey, Val, defaultVal, tombstoneVal, partition_size, bucket_size>;
      //using my_type = coop_chaining_block<Key, Val, size, partition_size>;
     
      using tile_type = cg::thread_block_tile<partition_size>;


      static const uint64_t n_traversals = ((bucket_size-1)/partition_size+1)*partition_size;


      pair_type slots[bucket_size-1];


      my_type * next;

      __device__ void init(cg::thread_block_tile<partition_size> team){

         pair_type sentinel_pair{defaultKey, defaultVal};
         //next points to nullptr
         atomicExch((unsigned long long int *)&next, 0ULL);

         for (int i = team.thread_rank(); i < bucket_size-1; i+=partition_size){

            //typed_atomic_exchange(&keys[i], defaultKey);

            slots[i] = sentinel_pair;

         }

         __threadfence();

         team.sync();
      }

      __device__ pair_type load_packed_pair(int index){
            //load 8 tags and pack them


            //pair_type loaded_pair;

            // pair_type loaded_pair;

            // loaded_pair.key = hash_table_load(&slots[index].key);
            // loaded_pair.val = hash_table_load(&slots[index].val);

            // return loaded_pair;

            return ht_load_packed_pair<ht_pair, Key, Val>(&slots[index]);

            // asm volatile("ld.gpu.acquire.v2.u64 {%0,%1}, [%2];" : "=l"(loaded_pair.key), "=l"(loaded_pair.val) : "l"(&slots[index]));
            
            // return loaded_pair;

      }



      __device__ bool insert(cg::thread_block_tile<partition_size> my_tile, Key insertKey, Val insertVal){

         for (uint i = my_tile.thread_rank(); i < n_traversals; i+=my_tile.size()){


            ADD_PROBE_ADJUSTED
            //uint offset = i - my_tile.thread_rank();

            bool key_match = (i < bucket_size-1);

            //Key loaded_key;

            bool exist_ballot = false;
            bool tombstone_ballot = false;

            if (key_match){

               Key loaded_key = hash_table_load(&slots[i].key);

               exist_ballot = (loaded_key == defaultKey);

               tombstone_ballot = (loaded_key == tombstoneKey);
               
            }

            auto my_tile_ballot = my_tile.ballot(exist_ballot);

            //while threads observe empty, pick leader and attempt swap.
            while (my_tile_ballot){

               auto leader = __ffs(my_tile_ballot)-1; 

               bool success = false;

               if (leader == my_tile.thread_rank()){

                  //this thread is leader for this iteration
                  //no need to recheck, I observed previously.
                  ADD_PROBE
                  if (typed_atomic_write(&slots[i].key, defaultKey, insertKey)){
                     ht_store(&slots[i].val,insertVal);

                     success = true;
      
                  }


               }

               success = my_tile.ballot(success);

               if (success) return true;

               //unset bit tested this round.
               my_tile_ballot &= (~(1U << leader));


            }

            my_tile_ballot = my_tile.ballot(tombstone_ballot);
            while (my_tile_ballot){

               auto leader = __ffs(my_tile_ballot)-1; 

               bool success = false;

               if (leader == my_tile.thread_rank()){

                  //this thread is leader for this iteration
                  //no need to recheck, I observed previously.
                  ADD_PROBE
                  if (typed_atomic_write(&slots[i].key, tombstoneKey, insertKey)){
                     ht_store(&slots[i].val,insertVal);

                     success = true;
      
                  }


               }

               success = my_tile.ballot(success);

               if (success) return true;

               //unset bit tested this round.
               my_tile_ballot &= (~(1U << leader));


            }


         }

         return false;

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

            bool key_match = (i < bucket_size-1);

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
                  ADD_PROBE
                  ballot = typed_atomic_write(&slots[i].key, defaultKey, ext_key);
                  if (ballot){

                     ht_store(&slots[i].val, ext_val);
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
                  ADD_PROBE
                  ballot = typed_atomic_write(&slots[i].key, tombstoneKey, ext_key);

                  if (ballot){

                     //loop and wait on tombstone val to be done.

                     // Val loaded_val = hash_table_load(&slots[i].val);

                     // while(loaded_val != tombstoneVal){

                     //    //this may be an issue if a stored value is legitimately a tombstone - need special logic in delete?
                     //    loaded_val = hash_table_load(&slots[i].val);
                     //    __threadfence();
                     // }

                     // __threadfence();

                     ht_store(&slots[i].val, ext_val);


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
            bool key_match = (i < bucket_size-1) && (SET_BIT_MASK(i) & upsert_mapping);


            //early drop if loaded_key is gone
            // bool ballot = key_match 


            auto ballot_result = my_tile.ballot(key_match);

            while (ballot_result){

               bool ballot = false;

               const auto leader = __ffs(ballot_result)-1;

               if (leader == my_tile.thread_rank()){

                  ADD_PROBE
                  ballot = typed_atomic_write(&slots[i].key, ext_key, ext_key);
                  if (ballot){

                     ht_store(&slots[i].val, ext_val);
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



      // __device__ bool query(const cg::thread_block_tile<partition_size> & my_tile, Key ext_key, Val & return_val){


      //    for (uint i = my_tile.thread_rank(); i < n_traversals; i+=my_tile.size()){

      //       uint offset = i - my_tile.thread_rank();

      //       bool valid = i < bucket_size;

      //       bool found_ballot = false;

      //       Val loaded_val;

      //       if (valid){
      //          Key loaded_key = hash_table_load(&slots[i].key);

      //          found_ballot = (loaded_key == ext_key);

      //          if (found_ballot){
      //             loaded_val = hash_table_load(&slots[i].val);
      //          }
      //       }


      //       int found = __ffs(my_tile.ballot(found_ballot))-1;

      //       if (found == -1) continue;

      //       return_val = my_tile.shfl(loaded_val, found);

      //       return true;



      //    }


      //    return false;

      // }


      __device__ pair_type * query_packed_reference(const cg::thread_block_tile<partition_size> & my_tile, Key ext_key){


         for (uint i = my_tile.thread_rank(); i < n_traversals; i+=my_tile.size()){

            ADD_PROBE_ADJUSTED

            uint offset = i - my_tile.thread_rank();

            bool valid = i < bucket_size-1;

            bool found_ballot = false;

            pair_type * return_val = nullptr;

            if (valid){



               Key loaded_key = hash_table_load(&slots[i].key);

               found_ballot = (loaded_key == ext_key);

               if (found_ballot){
                  return_val = &slots[i];
               }
            }


            int found = __ffs(my_tile.ballot(found_ballot))-1;

            if (found == -1) continue;

            return_val = my_tile.shfl(return_val, found);

            return return_val;



         }


         return nullptr;

      }

      __device__ pair_type query_packed(const cg::thread_block_tile<partition_size> & my_tile, Key ext_key){


         for (uint i = my_tile.thread_rank(); i < n_traversals; i+=my_tile.size()){

            uint offset = i - my_tile.thread_rank();

            bool valid = i < bucket_size-1;

            bool found_ballot = false;

            pair_type return_val;

            if (valid){

               return_val = load_packed_pair(i);

               //Key loaded_key = hash_table_load(&slots[i].key);

               found_ballot = (return_val.key == ext_key);

               // if (found_ballot){
               //    return_val = &slots[i];
               // }
            }


            int found = __ffs(my_tile.ballot(found_ballot))-1;

            if (found == -1) continue;

            return_val = my_tile.shfl(return_val, found);

            return return_val;



         }


         return pair_type{defaultKey, defaultVal};

      }


      // __device__ bool erase(cg::thread_block_tile<partition_size> my_tile, Key ext_key){


      //    for (uint i = my_tile.thread_rank(); i < n_traversals; i+=my_tile.size()){

      //       uint offset = i - my_tile.thread_rank();

      //       bool valid = i < bucket_size;

      //       bool found_ballot = false;

      //       Val loaded_val;

      //       if (valid){
      //          Key loaded_key = hash_table_load(&slots[i].key);

      //          found_ballot = (loaded_key == ext_key);

      //       }

      //       uint ballot_result = my_tile.ballot(found_ballot);

      //       while (ballot_result){

      //          bool ballot = false;

      //          const auto leader = __ffs(ballot_result)-1;

      //          if (leader == my_tile.thread_rank()){


      //             ballot = typed_atomic_write(&slots[i].key, ext_key, tombstoneKey);
      //             if (ballot){

      //                //force store
      //                ht_store(&slots[i].val, tombstoneVal);
      //                //typed_atomic_exchange(&slots[i].val, ext_val);
      //             }
      //          }

     

      //          //if leader succeeds return
      //          if (my_tile.ballot(ballot)){
      //             return true;
      //          }
                  

      //             //if we made it here no successes, decrement leader
      //             ballot_result  ^= 1UL << leader;

      //             //printf("Stalling in insert_into_bucket keys\n");

      //       }

      //    }



      //    return false;
      // }


   };

   template <typename Key, Key defaultKey, Key tombstoneKey, typename Val, Val defaultVal, Val tombstoneVal, uint partition_size, uint bucket_size>
   struct chaining_table{

      using my_type = chaining_table<Key, defaultKey, tombstoneKey, Val, defaultVal, tombstoneVal, partition_size, bucket_size>;

      uint64_t nslots;

      uint64_t nblocks;

      uint64_t seed;

      using tile_type = cg::thread_block_tile<partition_size>;

      using block_type = coop_chaining_block<Key, defaultKey, tombstoneKey, Val, defaultVal, tombstoneVal, partition_size, bucket_size>;

      block_type ** pointer_list;

      uint64_t * locks;

      using packed_pair_type = ht_pair<Key, Val>;


      static __host__ my_type * generate_on_device(uint64_t ext_nslots, uint64_t ext_seed){

         my_type * host_version = gallatin::utils::get_host_version<my_type>();

         host_version->seed = ext_seed;
         host_version->nslots = ext_nslots;

         block_type ** ext_pointer_list;

         host_version->nblocks = (ext_nslots-1)/(bucket_size) +1;


         host_version->locks = gallatin::utils::get_device_version<uint64_t>(host_version->nblocks);

         cudaMemset(host_version->locks, 0, sizeof(uint64_t)*(host_version->nblocks));


         //host_version->defaultKey = ext_defaultKey;

         cudaMalloc((void **)&ext_pointer_list, host_version->nblocks*sizeof(block_type *));

         //set all slots to nullptr.
         cudaMemset(ext_pointer_list, 0, host_version->nblocks*sizeof(block_type *));

         host_version->pointer_list = ext_pointer_list;

         return gallatin::utils::move_to_device(host_version);


      }

      static __host__ my_type * generate_on_device_prealloc(uint64_t ext_nslots, Key ext_defaultKey, uint64_t ext_seed){

         my_type * host_version = gallatin::utils::get_host_version<my_type>();

         host_version->seed = ext_seed;
         host_version->nslots = ext_nslots;

         block_type ** ext_pointer_list;

         host_version->nblocks = (ext_nslots-1)/bucket_size +1;

         //host_version->defaultKey = ext_defaultKey;

         host_version->locks = gallatin::utils::get_device_version<uint64_t>( (host_version->nblocks-1)/64+1);

         cudaMemset(host_version->locks, 0, sizeof(uint64_t)*((host_version->nblocks-1)/64+1));


         cudaMalloc((void **)&ext_pointer_list, host_version->nblocks*sizeof(block_type *));

         //set all slots to nullptr.
         cudaMemset(ext_pointer_list, 0, host_version->nblocks*sizeof(block_type *));

         host_version->pointer_list = ext_pointer_list;

         my_type * device_version =  gallatin::utils::move_to_device(host_version);

         chaining_table_fill_buffers<my_type><<<(host_version->nblocks-1)/256+1,256>>>(device_version);

         cudaDeviceSynchronize();

         return device_version;


      }

      static __host__ void free_on_device(my_type * device_version){


         //call kernel.

         my_type * host_version_copy = gallatin::utils::copy_to_host<my_type>(device_version);

         uint64_t nblocks = host_version_copy->nblocks;

         cudaFreeHost(host_version_copy);


         free_chains_kernel<my_type><<<(nblocks-1)/256+1,256>>>(device_version, nblocks);


         auto host_version = gallatin::utils::move_to_host(device_version);

         cudaFree(host_version->pointer_list);

         cudaFree(host_version->locks);

         cudaFreeHost(host_version);

      }

      __device__ void stall_lock(tile_type my_tile, uint64_t bucket){

         if (my_tile.thread_rank() == 0){
            stall_lock_one_thread(bucket);
         }

         my_tile.sync();

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
            //printf("Looping in key lock %lu\n", bucket);
         }
         while (atomicOr((unsigned long long int *)&locks[high], (unsigned long long int) SET_BIT_MASK(low)) & SET_BIT_MASK(low));

      }

      __device__ void unlock(tile_type my_tile, uint64_t bucket){

         if (my_tile.thread_rank() == 0){

            //printf("Unlocking %lu\n", bucket);
            unlock_bucket_one_thread(bucket);
            //printf("Unlocked %lu\n",bucket);

         }

         my_tile.sync();

      }

      __device__ void lock_key(tile_type my_tile, Key key){

         uint64_t my_slot = gallatin::hashers::MurmurHash64A(&key, sizeof(Key), seed) % nblocks;

         stall_lock(my_tile, my_slot);



      }


      __device__ uint64_t get_lock_bucket(tile_type my_tile, Key key){

         uint64_t my_slot = gallatin::hashers::MurmurHash64A(&key, sizeof(Key), seed) % nblocks;


         return my_slot;

      }

      __host__ uint64_t get_num_locks(){

         my_type * host_version = gallatin::utils::copy_to_host<my_type>(this);

         uint64_t nblocks = host_version->nblocks;

         cudaFreeHost(host_version);

         return nblocks;
      }

      __device__ void unlock_key(tile_type my_tile, Key key){

         uint64_t my_slot = gallatin::hashers::MurmurHash64A(&key, sizeof(Key), seed) % nblocks;

         unlock(my_tile, my_slot);

      }


      

      __device__ void free_chain(uint64_t tid){

         block_type * main_block;
         block_type * next_block;

         main_block = pointer_list[tid];

         while (main_block != nullptr){

            next_block = main_block->next;

            global_free(main_block);

            main_block = next_block;

         }
         
      }

      __device__ uint64_t count_chain_length(uint64_t tid){

         uint64_t n_chain = 0;

         block_type * main_block;
         block_type * next_block;

         main_block = pointer_list[tid];

         while (main_block != nullptr){

            n_chain+=1;
            main_block = main_block->next;

         }

         return n_chain;


      }


      __device__ void unlock_bucket_one_thread(uint64_t bucket){

         #if LOAD_CHEAP
         return;
         #endif

         ADD_PROBE
         
         uint64_t high = bucket/64;
         uint64_t low = bucket % 64;

         //if old is 0, SET_BIT_MASK & 0 is 0 - loop exit.
         atomicAnd((unsigned long long int *)&locks[high], (unsigned long long int) ~SET_BIT_MASK(low));

      }




      //format new block and attempt to atomicCAS
      __device__ bool attach_block(cg::thread_block_tile<partition_size> team, block_type ** block_ptr){


         block_type * new_block = nullptr;

         bool first = (team.thread_rank() == 0);

         if (first){

            new_block = (block_type *) gallatin::allocators::global_malloc(sizeof(block_type));


         }

         new_block = team.shfl(new_block, 0);
         
         if (new_block == nullptr) return false;

         new_block->init(team);

         if (first){

            ADD_PROBE
            if (atomicCAS((unsigned long long int *)block_ptr, 0ULL, (unsigned long long int ) new_block) != 0ULL){

               gallatin::allocators::global_free(new_block);

            }

         }

         //TODO - check performnace diff without this...
         team.sync();

         return true;

      }


      __device__ void calculate_chain_length(uint64_t * max_len, uint64_t * avg_len, uint64_t my_index){

         uint64_t my_length = 0;

         block_type * my_block = pointer_list[my_index];

         while (my_block != nullptr){
            my_length += 1;
            my_block = my_block->next;
         }

         atomicMax((unsigned long long int *)max_len, (unsigned long long int) my_length);
         atomicAdd((unsigned long long int *)avg_len, (unsigned long long int) my_length);

      }


      __device__ bool upsert_replace_internal(const cg::thread_block_tile<partition_size> & my_tile, const Key & newKey, const Val & newVal){

         uint64_t my_slot;
         block_type * my_block;
         block_type ** my_pointer_addr;


         my_slot = gallatin::hashers::MurmurHash64A(&newKey, sizeof(Key), seed) % nblocks;

         //attempt to read my slot

         #if COUNT_CHAINING_NEXT_LOAD
         ADD_PROBE_TILE
         #endif
         
         my_block = (block_type *) hash_table_load((uint64_t *)&pointer_list[my_slot]);
         
         //my_block = hash_table_load(&pointer_list[my_slot]);



         my_pointer_addr = &pointer_list[my_slot];


         while (true){

            //printf("Looping in insert\n");

            ADD_PROBE_BUCKET
            if (my_block == nullptr){
               //failure to find new segment
               if (!attach_block(my_tile, my_pointer_addr)) return false;

               my_block = my_pointer_addr[0];
               continue;
            }

            //otherwise, try to insert

            
            if (my_block->insert(my_tile, newKey, newVal)){
               return true;
            }

            //otherwise, move to next block.

            #if COUNT_CHAINING_NEXT_LOAD
            ADD_PROBE_TILE
            #endif

            my_pointer_addr = &my_block->next;

            my_block = (block_type *) hash_table_load((uint64_t *)&my_block->next);


         }



         return false;
      }


      __device__ bool upsert_no_lock(const tile_type & my_tile, const Key & key, const Val & val){

         uint64_t bucket_0 = gallatin::hashers::MurmurHash64A(&key, sizeof(Key), seed) % nblocks;

         //stall_lock(my_tile, bucket_0);


         //query

         //Val * stored_val = query_reference(tile_type my_tile, Key key){

         packed_pair_type * stored_copy = query_packed_reference(my_tile, key, bucket_0);

         if (stored_copy != nullptr){

            //upsert
            ht_store(&stored_copy->val, val);

            //unlock(my_tile, bucket_0);
            return true;

         }



         bool return_val = upsert_replace_internal(my_tile, key, val);

         //unlock(my_tile, bucket_0);

         return true;


      }


      __device__ void insert(cg::thread_block_tile<partition_size> my_tile, Key newKey, Val newVal){

         uint64_t my_slot;
         block_type * my_block;
         block_type ** my_pointer_addr;


         my_slot = gallatin::hashers::MurmurHash64A(&newKey, sizeof(Key), seed) % nblocks;

         //attempt to read my slot

         my_block = pointer_list[my_slot];

         my_pointer_addr = &pointer_list[my_slot];


         while (true){

            ADD_PROBE_BUCKET
            if (my_block == nullptr){
               //failure to find new segment
               if (!attach_block(my_tile, my_pointer_addr)) return;

               my_block = my_pointer_addr[0];
               continue;
            }

            //otherwise, try to insert

            if (my_block->insert(my_tile, newKey, newVal)){
               return;
            }

            //otherwise, move to next block.
            #if COUNT_CHAINING_NEXT_LOAD
            ADD_PROBE_TILE
            #endif
            my_pointer_addr = &my_block->next;
            my_block = my_block->next;



         }



         return;
      }

      __device__ bool upsert_replace(const tile_type & my_tile, const Key & key, const Val & val){

         uint64_t bucket_0 = gallatin::hashers::MurmurHash64A(&key, sizeof(Key), seed) % nblocks;

         stall_lock(my_tile, bucket_0);


         //query

         //Val * stored_val = query_reference(tile_type my_tile, Key key){

         packed_pair_type * stored_copy = query_packed_reference(my_tile, key, bucket_0);

         if (stored_copy != nullptr){

            //upsert
            ht_store(&stored_copy->val, val);

            unlock(my_tile, bucket_0);
            return true;

         }



         bool return_val = upsert_replace_internal(my_tile, key, val);

         unlock(my_tile, bucket_0);

         return true;


      }

      __device__ bool upsert_function(const tile_type & my_tile, const Key & key, const Val & val, void (*replace_func)(packed_pair_type *, Key, Val)){

         uint64_t bucket_0 = gallatin::hashers::MurmurHash64A(&key, sizeof(Key), seed) % nblocks;

         stall_lock(my_tile, bucket_0);


         //query

         //Val * stored_val = query_reference(tile_type my_tile, Key key){

         packed_pair_type * stored_copy = query_packed_reference(my_tile, key, bucket_0);

         if (stored_copy != nullptr){

            //upsert
            
            replace_func(stored_copy, key, val);

            unlock(my_tile, bucket_0);
            return true;

         }



         bool return_val = upsert_replace_internal(my_tile, key, val);

         unlock(my_tile, bucket_0);

         return true;


      }

      __device__ bool upsert_function_no_lock(const tile_type & my_tile, const Key & key, const Val & val, void (*replace_func)(packed_pair_type *, Key, Val)){

         uint64_t bucket_0 = gallatin::hashers::MurmurHash64A(&key, sizeof(Key), seed) % nblocks;

         //stall_lock(my_tile, bucket_0);


         //query

         //Val * stored_val = query_reference(tile_type my_tile, Key key){

         packed_pair_type * stored_copy = query_packed_reference(my_tile, key, bucket_0);

         if (stored_copy != nullptr){

            //upsert
            
            replace_func(stored_copy, key, val);

            //unlock(my_tile, bucket_0);
            return true;

         }



         bool return_val = upsert_replace_internal(my_tile, key, val);

         //unlock(my_tile, bucket_0);

         return true;


      }

      __device__ packed_pair_type * query_packed_reference(cg::thread_block_tile<partition_size> my_tile, Key queryKey, uint64_t my_slot){


         //uint64_t my_slot = gallatin::hashers::MurmurHash64A(&queryKey, sizeof(Key), seed) % nblocks;


         #if COUNT_CHAINING_NEXT_LOAD
         ADD_PROBE_TILE
         #endif

         block_type * my_block = (block_type *) hash_table_load<uint64_t>((uint64_t *)&pointer_list[my_slot]);

         packed_pair_type * return_val = nullptr;

         while (my_block != nullptr){

            
            return_val = my_block->query_packed_reference(my_tile, queryKey);

            if (return_val != nullptr) return return_val;

            #if COUNT_CHAINING_NEXT_LOAD
            ADD_PROBE_TILE
            #endif
            my_block = (block_type *) hash_table_load((uint64_t *)&my_block->next);

         }


         return nullptr;


      }


      __device__ packed_pair_type * find_pair(const tile_type & my_tile, const Key & key){

         uint64_t bucket_0 = gallatin::hashers::MurmurHash64A(&key, sizeof(Key), seed) % nblocks;

         stall_lock(my_tile, bucket_0);


         //query

         //Val * stored_val = query_reference(tile_type my_tile, Key key){

         packed_pair_type * stored_copy = query_packed_reference(my_tile, key, bucket_0);


         unlock(my_tile, bucket_0);
         
         return stored_copy;


      }

      __device__ packed_pair_type * find_pair_no_lock(const tile_type & my_tile, const Key & key){

         uint64_t bucket_0 = gallatin::hashers::MurmurHash64A(&key, sizeof(Key), seed) % nblocks;

         //query

         //Val * stored_val = query_reference(tile_type my_tile, Key key){

         packed_pair_type * stored_copy = query_packed_reference(my_tile, key, bucket_0);

         
         return stored_copy;


      }

      __device__ packed_pair_type query_packed(cg::thread_block_tile<partition_size> my_tile, Key queryKey, uint64_t my_slot){


         //uint64_t my_slot = gallatin::hashers::MurmurHash64A(&queryKey, sizeof(Key), seed) % nblocks;


         #if COUNT_CHAINING_NEXT_LOAD
         ADD_PROBE_TILE
         #endif

         block_type * my_block = (block_type *) hash_table_load<uint64_t>((uint64_t *)&pointer_list[my_slot]);

         packed_pair_type return_val;

         while (my_block != nullptr){

            ADD_PROBE_BUCKET
            return_val = my_block->query_packed(my_tile, queryKey);

            if (return_val.key == queryKey) return return_val;

            #if COUNT_CHAINING_NEXT_LOAD
            ADD_PROBE_TILE
            #endif
            my_block = (block_type *) hash_table_load((uint64_t *)&my_block->next);

         }


         return packed_pair_type{defaultKey, defaultVal};


      }

      // __device__ bool find_with_reference(cg::thread_block_tile<partition_size> my_tile, Key queryKey, Val & returnVal){

      //    uint64_t my_slot = gallatin::hashers::MurmurHash64A(&queryKey, sizeof(Key), seed) % nblocks;

      //    stall_lock(my_tile, my_slot);


      //    packed_pair_type queried_value = query_packed(my_tile, queryKey, my_slot);

      //    if (queried_value.key != queryKey){
      //       unlock(my_tile, my_slot);
      //       return false;
      //    }

      //    returnVal = queried_value.val;

      //    unlock(my_tile, my_slot);

      //    return true;

      // }

      //added nodiscard - lookups should always have the return value checked.
      [[nodiscard]] __device__ bool find_with_reference(cg::thread_block_tile<partition_size> my_tile, Key queryKey, Val & returnVal){

         uint64_t my_slot = gallatin::hashers::MurmurHash64A(&queryKey, sizeof(Key), seed) % nblocks;

         stall_lock(my_tile, my_slot);


         packed_pair_type * queried_value = query_packed_reference(my_tile, queryKey, my_slot);

         if (queried_value == nullptr){

            unlock(my_tile, my_slot);
            return false;

         }

         if (my_tile.thread_rank() == 0){
            returnVal = hash_table_load(&queried_value->val);
         }

         returnVal = my_tile.shfl(returnVal, 0);

         unlock(my_tile, my_slot);

         return true;

      }

      [[nodiscard]] __device__ bool find_with_reference_no_lock(cg::thread_block_tile<partition_size> my_tile, Key queryKey, Val & returnVal){

         uint64_t my_slot = gallatin::hashers::MurmurHash64A(&queryKey, sizeof(Key), seed) % nblocks;

         //stall_lock(my_tile, my_slot);

         packed_pair_type queried_value = query_packed(my_tile, queryKey, my_slot);

         if (queried_value.key != queryKey){
            //unlock(my_tile, my_slot);
            return false;
         }

         returnVal = queried_value.val;

         //unlock(my_tile, my_slot);

         return true;

      }


      __device__ bool remove(cg::thread_block_tile<partition_size> my_tile, Key removeKey){


         uint64_t my_slot = gallatin::hashers::MurmurHash64A(&removeKey, sizeof(Key), seed) % nblocks;

         stall_lock(my_tile, my_slot);

         packed_pair_type * queried_value = query_packed_reference(my_tile, removeKey, my_slot);

         //overwrite

         if (queried_value == nullptr){

            unlock(my_tile, my_slot);
            return false;

         }


         ADD_PROBE_TILE

         bool erased = false;

         if (my_tile.thread_rank() == 0){
            erased = gallatin::utils::typed_atomic_write(&queried_value->key, removeKey, tombstoneKey);
         }

         erased = my_tile.ballot(erased);

         if (erased){

            //ht_store(&queried_value->val, tombstoneVal);

            unlock(my_tile, my_slot);

            return true;
         }

         unlock(my_tile, my_slot);
         return false;


      }


      __device__ bool remove_no_lock(cg::thread_block_tile<partition_size> my_tile, Key removeKey){


         uint64_t my_slot = gallatin::hashers::MurmurHash64A(&removeKey, sizeof(Key), seed) % nblocks;

         //stall_lock(my_tile, my_slot);

         packed_pair_type * queried_value = query_packed_reference(my_tile, removeKey, my_slot);

         //overwrite

         if (queried_value == nullptr){

            //unlock(my_tile, my_slot);
            return false;

         }

         ADD_PROBE_TILE
         bool erased = false;

         if (my_tile.thread_rank() == 0){
            erased = gallatin::utils::typed_atomic_write(&queried_value->key, removeKey, tombstoneKey);
         }

         erased = my_tile.ballot(erased);

         if (erased){

            //ht_store(&queried_value->val, tombstoneVal);

            
            return true;
         }

         
         return false;


      }



      __host__ void print_chain_stats(){


         my_type * host_version = gallatin::utils::copy_to_host<my_type>(this);

         uint64_t nblocks = host_version->nblocks;

         uint64_t * max;
         uint64_t * avg;

         cudaMallocManaged((void ** )&max, sizeof(uint64_t));
         cudaMallocManaged((void **)&avg, sizeof(uint64_t));

         max[0] = 0;
         avg[0] = 0;

         calculate_chain_kernel<my_type><<<(nblocks-1)/256+1, 256>>>(this, max, avg,nblocks);

         cudaDeviceSynchronize();

         std::cout << "Chains - Max: " << max[0] << ", Avg: " << 1.0*avg[0]/nblocks << ", nblocks: " << nblocks << std::endl;

         cudaFree(max);
         cudaFree(avg);
      }

      static char * get_name(){
         return "chaining_table";
      }

      __host__ void print_space_usage(){

         my_type * host_version = gallatin::utils::copy_to_host<my_type>(this);
            
         uint64_t capacity = host_version->nblocks*sizeof(block_type *) + (host_version->nblocks-1)/8+1; 


         uint64_t nblocks = host_version->nblocks;

         uint64_t * chain_count;

         cudaMallocManaged((void **)&chain_count, sizeof(uint64_t));

         chain_count[0] = 0;

         cudaDeviceSynchronize();

         count_n_chains<my_type><<<(nblocks-1)/256+1,256>>>(this, nblocks, chain_count); 

         cudaDeviceSynchronize();

         capacity += chain_count[0]*sizeof(block_type);

         cudaFree(chain_count);

         cudaFreeHost(host_version);

         printf("chaining_hashing using %llu bytes\n", capacity);

      }

      __host__ void print_fill(){
         printf("Not yet implemented\n");
      }

   };

template <typename T>
constexpr T generate_chaining_tombstone(uint64_t offset) {
  return (~((T) 0)) - offset;
};

template <typename T>
constexpr T generate_chaining_sentinel() {
  return ((T) 0);
};

template <typename Key, typename Val, uint tile_size, uint bucket_size>
using chaining_generic = typename hashing_project::tables::chaining_table<Key,
                                    generate_chaining_sentinel<Key>(),
                                    generate_chaining_tombstone<Key>(0),
                                    Val,
                                    generate_chaining_sentinel<Val>(),
                                    generate_chaining_tombstone<Val>(0),
                                    tile_size,
                                    bucket_size>;



}


}


#endif //end of resizing_hash guard