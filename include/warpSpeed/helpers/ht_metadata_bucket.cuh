#ifndef HT_MD_BUCKET
#define HT_MD_BUCKET

#include <warpSpeed/helpers/ht_pairs.cuh>
#include <warpSpeed/helpers/probe_counts.cuh>
#include <warpSpeed/helpers/ht_load.cuh>

namespace warpSpeed {

  namespace tables {

   template <typename Key, Key defaultKey, Key tombstoneKey, Key holdingKey, typename Val, Val defaultVal, Val tombstoneVal, uint partition_size, uint bucket_size>
   struct hash_table_metadata_bucket {

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


      __device__ pair_type load_packed_pair(int index){
            //load 8 tags and pack them


            return ht_load_packed_pair<ht_pair, Key, Val>(&slots[index]);

            // pair_type loaded_pair;

            // asm volatile("ld.gpu.acquire.v2.u64 {%0,%1}, [%2];" : "=l"(loaded_pair.key), "=l"(loaded_pair.val) : "l"(&slots[index]));
            
            // return loaded_pair;

      }


      //insert based on match_ballots
      //makes 1 attempt - on first failure trigger reload - this is needed for load balancing.
      #if LARGE_BUCKET_MODS
      __device__ bool insert_ballots(cg::thread_block_tile<partition_size> my_tile, Key ext_key, Val ext_val, uint64_t empty_match, uint64_t tombstone_match)
      #else 
      __device__ bool insert_ballots(cg::thread_block_tile<partition_size> my_tile, Key ext_key, Val ext_val, uint32_t empty_match, uint32_t tombstone_match)
      #endif
      {


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

                  ballot = typed_atomic_write(&slots[i].key, defaultKey, holdingKey);
                  ADD_PROBE
                  if (ballot){

                     //ht_store(&slots[i].val, ext_val);

                     ht_store_packed_pair(&slots[i], {ext_key, ext_val});
                     __threadfence();
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

                  ballot = typed_atomic_write(&slots[i].key, tombstoneKey, holdingKey);
                  ADD_PROBE

                  if (ballot){

                     //loop and wait on tombstone val to be done.

                     // Val loaded_val = hash_table_load(&slots[i].val); 

                     // while(loaded_val != tombstoneVal){

                     //    //this may be an issue if a stored value is legitimately a tombstone - need special logic in delete?
                     //    loaded_val = hash_table_load(&slots[i].val); 
                     //    __threadfence();
                     // }

                     // __threadfence();

                     //ht_store(&slots[i].val, ext_val);
                     ht_store_packed_pair(&slots[i], {ext_key, ext_val});
                     __threadfence();


                  }

               }

               //if leader succeeds return
               if (my_tile.ballot(ballot_exists)){
                  return my_tile.ballot(ballot);
               }
                  

               //if we made it here no successes, decrement leader
               //ballot_result  ^= 1UL << leader;

               //printf("Stalling in insert_into_bucket keys\n");


            }

         }


         return false;

      }

      //attempt to insert into the table based on an existing mapping.

      #if LARGE_BUCKET_MODS
      __device__ int upsert_existing(cg::thread_block_tile<partition_size> my_tile, Key ext_key, Val ext_val, uint64_t upsert_mapping)
      #else
      __device__ int upsert_existing(cg::thread_block_tile<partition_size> my_tile, Key ext_key, Val ext_val, uint32_t upsert_mapping)
      #endif
      {



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


                  //ballot = typed_atomic_write(&slots[i].key, ext_key, ext_key);
                  //ballot = typed_atomic_write(&slots[i].key, defaultKey, ext_key);
                  ballot = (hash_table_load(&slots[i].key) == ext_key);
                  ADD_PROBE
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


      #if LARGE_BUCKET_MODS
      __device__ int upsert_existing_func(cg::thread_block_tile<partition_size> my_tile, Key ext_key, Val ext_val, uint64_t upsert_mapping, void (*replace_func)(pair_type *, Key, Val))
      #else
      __device__ int upsert_existing_func(cg::thread_block_tile<partition_size> my_tile, Key ext_key, Val ext_val, uint32_t upsert_mapping, void (*replace_func)(pair_type *, Key, Val))
      #endif
      {



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

                  ballot = (gallatin::utils::ld_acq(&slots[i].key) == ext_key);

                  if (ballot){

                     replace_func(&slots[i], ext_key, ext_val);

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
      // __device__ void load_fill_ballots(const cg::thread_block_tile<partition_size> & my_tile, const Key & upsert_key, __restrict__ uint & empty_match, __restrict__ uint & tombstone_match, __restrict__ uint & key_match){


      //    //wipe previous
      //    empty_match = 0U;
      //    tombstone_match = 0U;
      //    key_match = 0U;

      //    int my_count = 0;

      //    for (uint i = my_tile.thread_rank(); i < n_traversals; i+=my_tile.size()){

      //       //step in clean intervals of my_tile. 
      //       uint offset = i - my_tile.thread_rank();

      //       bool valid = i < bucket_size;

            

      //       bool found_empty = false;
      //       bool found_tombstone = false;
      //       bool found_exact = false;

      //       if (valid){

      //          Key loaded_key = gallatin::utils::ld_acq(&slots[i].key);

      //          found_empty = (loaded_key == defaultKey);
      //          found_tombstone = (loaded_key == tombstoneKey);
      //          found_exact = (loaded_key == upsert_key);

      //       }

      //       empty_match |= (my_tile.ballot(found_empty) << offset);
      //       tombstone_match |= (my_tile.ballot(found_tombstone) << offset);
      //       key_match |= (my_tile.ballot(found_exact) << offset);

      //       //if (empty_match || key_match) return;

      //       //if (__popcll(key_match | empty_match)) return;

      //    }

      //    return;

      // }


      // __device__ bool load_fill_ballots_upserts(const cg::thread_block_tile<partition_size> & my_tile, const Key & upsert_key, const Val & upsert_val, __restrict__ uint & empty_match, __restrict__ uint & tombstone_match, __restrict__ uint & key_match){


      //    //wipe previous
      //    empty_match = 0U;
      //    tombstone_match = 0U;
      //    key_match = 0U;

      //    int my_count = 0;

      //    for (uint i = my_tile.thread_rank(); i < n_traversals; i+=my_tile.size()){

      //       //step in clean intervals of my_tile. 
      //       uint offset = i - my_tile.thread_rank();

      //       bool valid = i < bucket_size;

            

      //       bool found_empty = false;
      //       bool found_tombstone = false;
      //       bool found_exact = false;

      //       if (valid){

      //          Key loaded_key = gallatin::utils::ld_acq(&slots[i].key);

      //          found_empty = (loaded_key == defaultKey);
      //          found_tombstone = (loaded_key == tombstoneKey);
      //          found_exact = (loaded_key == upsert_key);

      //       }

      //       empty_match |= (my_tile.ballot(found_empty) << offset);
      //       tombstone_match |= (my_tile.ballot(found_tombstone) << offset);
      //       key_match |= (my_tile.ballot(found_exact) << offset);





      //       //if (empty_match || key_match) return;

      //       int leader = __ffs(key_match)-1;

      //       bool ballot = false;

      //       if (leader == my_tile.thread_rank()){

      //          if (gallatin::utils::typed_atomic_write(&slots[i].key, upsert_key, upsert_key)){

      //             gallatin::utils::st_rel(&slots[i].val, upsert_val);
      //             ballot = true;


      //          }

      //       }

      //       //upserted.
      //       if (my_tile.ballot(ballot)) return true;

      //       leader = __ffs(key_match)-1;

      //       if (leader == my_tile.thread_rank() && i < bucket_size*.75){

      //          if (gallatin::utils::typed_atomic_write(&slots[i].key, defaultKey, upsert_key)){
      //             gallatin::utils::st_rel(&slots[i].val, upsert_val);
      //             ballot = true;
      //          }

      //       }

      //       if (my_tile.ballot(ballot)) return true;

      //    }

      //    return false;

      // }


     #if LARGE_BUCKET_MODS
      __device__ bool query_match(const cg::thread_block_tile<partition_size> & my_tile, Key ext_key, Val & return_val, uint64_t & match_bitmap)
      #else
      __device__ bool query_match(const cg::thread_block_tile<partition_size> & my_tile, Key ext_key, Val & return_val, uint32_t & match_bitmap)
      #endif
      {


         #if LARGE_BUCKET_MODS
         if (__popcll(match_bitmap) == 0) return false;
         #else
         if (__popc(match_bitmap) == 0) return false;
         #endif


         for (uint i = my_tile.thread_rank(); i < n_traversals; i+=my_tile.size()){

            bool valid = i < bucket_size;

            bool found_ballot = false;

            Val loaded_val;

            if (my_tile.ballot(match_bitmap & SET_BIT_MASK(i))){
               ADD_PROBE_TILE;
            }

            if (valid && (match_bitmap & SET_BIT_MASK(i))){


               pair_type loaded_pair = load_packed_pair(i);

               //Key loaded_key = hash_table_load(&slots[i].key); 

               found_ballot = (loaded_pair.key == ext_key);

               if (found_ballot){
                  loaded_val = loaded_pair.val;
                  //loaded_val = hash_table_load(&slots[i].val);
               }


            }


            int found = __ffs(my_tile.ballot(found_ballot))-1;

            if (found == -1) continue;

            return_val = my_tile.shfl(loaded_val, found);

            return true;

         }


         return false;



      }


      __device__ pair_type * query_pair_ballot(const cg::thread_block_tile<partition_size> & my_tile, const Key & read_key, __restrict__ uint & match_ballot){

         //Val * return_val = nullptr;

         int found = __ffs(match_ballot)-1;

         while (found != -1){

            bool found_flag = false;

            if (my_tile.thread_rank() == found % my_tile.size()){

               ADD_PROBE

               Key loaded_key = hash_table_load(&slots[found].key);

               found_flag = (loaded_key == read_key);


            }

            if (my_tile.ballot(found_flag)){
               return &slots[found];
            }

            match_ballot ^= SET_BIT_MASK(found);

            found = __ffs(match_ballot)-1;

         }

         
         return nullptr;


      }


      __device__ bool query(const cg::thread_block_tile<partition_size> & my_tile, Key ext_key, Val & return_val){


         for (uint i = my_tile.thread_rank(); i < n_traversals; i+=my_tile.size()){

            ADD_PROBE_ADJUSTED

            uint offset = i - my_tile.thread_rank();

            bool valid = i < bucket_size;

            bool found_ballot = false;

            Val loaded_val;

            if (valid){

               pair_type loaded_pair = load_packed_pair(i);

               //Key loaded_key = hash_table_load(&slots[i].key); 

               //found_ballot = (loaded_key == ext_key);
               found_ballot = (loaded_pair.key == ext_key);

               if (found_ballot){
                  loaded_val = loaded_pair.val;
                  //loaded_val = hash_table_load(&slots[i].val); 
               }
            }


            int found = __ffs(my_tile.ballot(found_ballot))-1;

            if (found == -1) continue;

            return_val = my_tile.shfl(loaded_val, found);

            return true;



         }


         return false;

      }

      __device__ int erase(cg::thread_block_tile<partition_size> my_tile, Key ext_key){


         for (uint i = my_tile.thread_rank(); i < n_traversals; i+=my_tile.size()){

            ADD_PROBE_ADJUSTED

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
                  ADD_PROBE
                  // if (ballot){

                     //force store
                     //ht_store(&slots[i].val, tombstoneVal);
                     //typed_atomic_exchange(&slots[i].val, ext_val);
                  //}
               }

     

               //if leader succeeds return
               if (my_tile.ballot(ballot)){
                  return my_tile.shfl(i, leader);
               }
                  

                  //if we made it here no successes, decrement leader
               ballot_result  ^= 1UL << leader;

                  //printf("Stalling in insert_into_bucket keys\n");

            }

         }



         return -1;
      }


      #if LARGE_BUCKET_MODS
      __device__ int erase_reference(cg::thread_block_tile<partition_size> my_tile, Key ext_key, uint64_t match_ballot)
      #else
      __device__ int erase_reference(cg::thread_block_tile<partition_size> my_tile, Key ext_key, uint32_t match_ballot)
      #endif
      {

         for (uint i = my_tile.thread_rank(); i < n_traversals; i+=my_tile.size()){

            //ADD_PROBE_ADJUSTED

            //uint offset = i - my_tile.thread_rank();

            bool valid = i < bucket_size;

            bool found_ballot = false;

            Val loaded_val;


            bool valid_load_check = valid && (SET_BIT_MASK(i) & match_ballot);

            if (my_tile.ballot(valid_load_check)){
               ADD_PROBE_ADJUSTED
            }

            if (valid_load_check){
               Key loaded_key = hash_table_load(&slots[i].key); 

               found_ballot = (loaded_key == ext_key);

            }

            uint ballot_result = my_tile.ballot(found_ballot);

            while (ballot_result){

               bool ballot = false;

               const auto leader = __ffs(ballot_result)-1;

               if (leader == my_tile.thread_rank()){


                  ballot = typed_atomic_write(&slots[i].key, ext_key, tombstoneKey);
                  ADD_PROBE
                  // if (ballot){

                  //    //force store
                  //    //gallatin::utils::st_rel(&slots[i].val, tombstoneVal);
                  //    //typed_atomic_exchange(&slots[i].val, ext_val);
                  // }
               }

     

               //if leader succeeds return
               if (my_tile.ballot(ballot)){
                  return my_tile.shfl(i, leader);
               }
                  

                  //if we made it here no successes, decrement leader
               ballot_result  ^= 1UL << leader;

                  //printf("Stalling in insert_into_bucket keys\n");

            }

         }



         return -1;
      }


   };

   template <typename Key, Key emptyKey, Key tombstoneKey, typename Val, Val tombstoneVal, uint partition_size, uint bucket_size>
   struct hash_table_metadata {

      using pair_type = ht_pair<Key,Val>;

      static const uint64_t n_traversals = ((bucket_size-1)/partition_size+1)*partition_size;


      uint16_t metadata[bucket_size];

      __device__ void init(){

         for (uint64_t i = 0; i < bucket_size; i++){
            metadata[i]= get_empty_tag();
         }

         __threadfence();

      }


      __device__ uint16_t get_tag(Key key){

         // uint16_t tombstone_tag = get_tombstone_tag();
         // uint16_t empty_tag = get_empty_tag();

         uint16_t key_tag = (uint16_t) key;

         while (key_tag == get_empty_tag() || key_tag == get_tombstone_tag()){
            key += 1;
            key_tag = (uint16_t) key;
         }


         // if (key_tag == tombstone_tag){
         //    printf("Bad tag %u == %u\n", key_tag, tombstone_tag);
         // }
         // if (key_tag == empty_tag){
         //    printf("Bad tag %u == %u\n", key_tag, empty_tag);
         // }

         return key_tag;

      }

      __device__ uint16_t set_tag(int index, uint16_t current_tag, uint16_t replace_tag){

         ADD_PROBE
         return gallatin::utils::typed_atomic_CAS(&metadata[index], current_tag, replace_tag);

      }

      constexpr __host__ __device__ uint16_t get_empty_tag(){

         return (uint16_t) emptyKey;

      }

      constexpr __host__ __device__ uint16_t get_tombstone_tag(){

         return (uint16_t) tombstoneKey;

      }


      __device__ bool replace_empty(int index, const Key & upsert_key){

         return set_tag(index, get_empty_tag(), get_tag(upsert_key)) == get_empty_tag();

      }

      __device__ bool replace_tombstone(int index, const Key & upsert_key){

         return set_tag(index, get_tombstone_tag(), get_tag(upsert_key)) == get_tombstone_tag();

      }


      __device__ void set_tombstone(int index){

         //didn't have this instruction so doing it manuage
         ADD_PROBE
         asm volatile("st.gpu.release.u16 [%0], %1;" :: "l"(&metadata[index]), "h"(get_tombstone_tag()) : "memory");
         __threadfence();
      }


      #if LARGE_BUCKET_MODS
      __device__ int match_empty(const cg::thread_block_tile<partition_size> & my_tile, const Key & upsert_key, const uint64_t & empty_match)
      #else
      __device__ int match_empty(const cg::thread_block_tile<partition_size> & my_tile, const Key & upsert_key, const uint32_t & empty_match)
      #endif
      {

         for (uint i = my_tile.thread_rank(); i < n_traversals; i+=my_tile.size()){

            uint offset = i - my_tile.thread_rank();


            bool key_match = (i < bucket_size);

            bool empty = false;

            if (key_match){
               empty = empty_match & SET_BIT_MASK(i);
            }

            auto ballot_result = my_tile.ballot(empty);

            while (ballot_result){

               bool ballot = false;

               const auto leader = __ffs(ballot_result)-1;

               if (leader == my_tile.thread_rank()){

                  ADD_PROBE

                  ballot = typed_atomic_write(&metadata[i], get_empty_tag(), get_tag(upsert_key));

               }

               if (my_tile.ballot(ballot)){ return my_tile.shfl(i, leader); }

               ballot_result  ^= 1UL << leader;

            }


         }

         return -1;


      }

      __device__ void set_tombstone(const cg::thread_block_tile<partition_size> & my_tile, const int written_addr){


         int leader = written_addr % partition_size;

         if (leader == my_tile.thread_rank()){
            set_tombstone(written_addr);
         }

         return;

      }

      #if LARGE_BUCKET_MODS
      __device__ int match_tombstone(const cg::thread_block_tile<partition_size> & my_tile, const Key & upsert_key, const uint64_t & empty_match)
      #else
      __device__ int match_tombstone(const cg::thread_block_tile<partition_size> & my_tile, const Key & upsert_key, const uint32_t & empty_match)
      #endif 
      {

         for (uint i = my_tile.thread_rank(); i < n_traversals; i+=my_tile.size()){

            uint offset = i - my_tile.thread_rank();


            bool key_match = (i < bucket_size);

            bool empty = false;

            if (key_match){
               empty = empty_match & SET_BIT_MASK(i);
            }

            auto ballot_result = my_tile.ballot(empty);

            while (ballot_result){

               bool ballot = false;

               const auto leader = __ffs(ballot_result)-1;

               if (leader == my_tile.thread_rank()){

                  ADD_PROBE

                  ballot = typed_atomic_write(&metadata[i], get_tombstone_tag(), get_tag(upsert_key));

               }

               if (my_tile.ballot(ballot)){ return my_tile.shfl(i, leader); }


               ballot_result  ^= 1UL << leader;

            }

         }

         return -1;


      }


      #if LARGE_BUCKET_MODS
      __device__ void load_fill_ballots(const cg::thread_block_tile<partition_size> & my_tile, const Key & upsert_key, __restrict__ uint64_t & empty_match, __restrict__ uint64_t & tombstone_match, __restrict__ uint64_t & key_match)
      #else
      __device__ void load_fill_ballots(const cg::thread_block_tile<partition_size> & my_tile, const Key & upsert_key, __restrict__ uint32_t & empty_match, __restrict__ uint32_t & tombstone_match, __restrict__ uint32_t & key_match)
      #endif
      {

         ADD_PROBE_TILE

         //wipe previous
         empty_match = 0U;
         tombstone_match = 0U;
         key_match = 0U;


         uint16_t key_tag = get_tag(upsert_key);

         for (uint i = my_tile.thread_rank(); i < n_traversals; i+=my_tile.size()){

            //step in clean intervals of my_tile. 
            uint offset = i - my_tile.thread_rank();

            bool valid = i < bucket_size;

            

            bool found_empty = false;
            bool found_tombstone = false;
            bool found_exact = false;

            if (valid){

               uint16_t loaded_key = hash_table_load(&metadata[i]);

               found_empty = (loaded_key == get_empty_tag());
               found_tombstone = (loaded_key == get_tombstone_tag());
               found_exact = (loaded_key == key_tag);

            }

            #if LARGE_BUCKET_MODS
            empty_match |= ( (uint64_t) my_tile.ballot(found_empty) << offset);
            tombstone_match |= ( (uint64_t) my_tile.ballot(found_tombstone) << offset);
            key_match |= ( (uint64_t) my_tile.ballot(found_exact) << offset);
            #else
            empty_match |= ( my_tile.ballot(found_empty) << offset);
            tombstone_match |= (  my_tile.ballot(found_tombstone) << offset);
            key_match |= (my_tile.ballot(found_exact) << offset);
            #endif

            //if (empty_match || key_match) return;

         }

         return;

      }

      #if LARGE_BUCKET_MODS
      __device__ void load_fill_ballots_big(const cg::thread_block_tile<partition_size> & my_tile, const Key & upsert_key, __restrict__ uint64_t & empty_match, __restrict__ uint64_t & tombstone_match, __restrict__ uint64_t & key_match)
      #else
      __device__ void load_fill_ballots_big(const cg::thread_block_tile<partition_size> & my_tile, const Key & upsert_key, __restrict__ uint32_t & empty_match, __restrict__ uint32_t & tombstone_match, __restrict__ uint32_t & key_match)
      #endif
      {


         ADD_PROBE_TILE

         //uint4 tags = load_multi_tags(metadata);


         uint64_t * md_as_uint64_t = (uint64_t *) metadata; 

         //wipe previous
         empty_match = 0U;
         tombstone_match = 0U;
         key_match = 0U;


         uint16_t key_tag = get_tag(upsert_key);

         for (uint i = my_tile.thread_rank(); i < (n_traversals-1)/4+1; i+=my_tile.size()){

            //step in clean intervals of my_tile. 
            uint offset = i - my_tile.thread_rank();

            bool valid_to_load = i < (bucket_size-1/4)+1;

            
            //maybe change these
            #if LARGE_BUCKET_MODS
            uint64_t local_empty = 0U;
            uint64_t local_tombstone = 0U;
            uint64_t local_match = 0U;
            #else
            uint32_t local_empty = 0U;
            uint32_t local_tombstone = 0U;
            uint32_t local_match = 0U;
            #endif



            if (valid_to_load){

               uint64_t loaded_key = hash_table_load(&md_as_uint64_t[i]);


               uint16_t * load_key_indexer = (uint16_t *) &loaded_key;

               for (uint j = 0; j < 4; j++){

                  uint16_t loaded_tag = load_key_indexer[j];

                  #if LARGE_BUCKET_MODS
                  uint64_t set_index = SET_BIT_MASK(i*4+j);
                  #else
                  uint32_t set_index = SET_BIT_MASK(i*4+j);
                  #endif

                  local_empty |= (loaded_tag==get_empty_tag())*set_index;
                  local_tombstone |= (loaded_tag==get_tombstone_tag())*set_index;
                  local_match |= (loaded_tag==key_tag)*set_index;

               }

            }

            #if LARGE_BUCKET_MODS
            //cg::reduce(tile, thread_sum, cg::plus<int>()) / length;
            empty_match |= cg::reduce(my_tile, local_empty, cg::plus<uint64_t>());
            tombstone_match |= cg::reduce(my_tile, local_tombstone, cg::plus<uint64_t>());
            key_match |= cg::reduce(my_tile, local_match, cg::plus<uint64_t>());

            #else

            empty_match |= cg::reduce(my_tile, local_empty, cg::plus<uint32_t>());
            tombstone_match |= cg::reduce(my_tile, local_tombstone, cg::plus<uint32_t>());
            key_match |= cg::reduce(my_tile, local_match, cg::plus<uint32_t>());

            #endif

            // empty_match |= (my_tile.ballot(found_empty) << offset);
            // tombstone_match |= (my_tile.ballot(found_tombstone) << offset);
            // key_match |= (my_tile.ballot(found_exact) << offset);

            //if (empty_match || key_match) return;

         }

         return;

      }


      #if LARGE_BUCKET_MODS
      __device__ void load_fill_ballots_huge(const cg::thread_block_tile<partition_size> & my_tile, const Key & upsert_key, __restrict__ uint64_t & empty_match, __restrict__ uint64_t & tombstone_match, __restrict__ uint64_t & key_match)
      #else
      __device__ void load_fill_ballots_huge(const cg::thread_block_tile<partition_size> & my_tile, const Key & upsert_key, __restrict__ uint32_t & empty_match, __restrict__ uint32_t & tombstone_match, __restrict__ uint32_t & key_match)
      #endif
      {


         ADD_PROBE_TILE

         //uint4 tags = double_load_multi_tags(metadata);


         //uint64_t * md_as_uint64_t = (uint64_t *) metadata; 

         //wipe previous
         empty_match = 0U;
         tombstone_match = 0U;
         key_match = 0U;

         uint16_t key_tag = get_tag(upsert_key);

         for (uint i = my_tile.thread_rank(); i < (n_traversals-1)/8+1; i+=my_tile.size()){

            //step in clean intervals of my_tile. 
            uint offset = i - my_tile.thread_rank();

            bool valid_to_load = i < (bucket_size-1/8)+1;

            
            //maybe change these
            #if LARGE_BUCKET_MODS
            uint64_t local_empty = 0U;
            uint64_t local_tombstone = 0U;
            uint64_t local_match = 0U;
            #else
            uint32_t local_empty = 0U;
            uint32_t local_tombstone = 0U;
            uint32_t local_match = 0U;
            #endif



            if (valid_to_load){


               packed_tags loaded_pair = load_multi_tags(&metadata[i*8]);

              

               uint16_t * load_key_indexer = (uint16_t *) &loaded_pair;

               for (uint j = 0; j < 8; j++){

                  uint16_t loaded_tag = load_key_indexer[j];

                  #if LARGE_BUCKET_MODS
                  uint64_t set_index = SET_BIT_MASK(i*8+j);
                  #else
                  uint32_t set_index = SET_BIT_MASK(i*8+j);
                  #endif

                  local_empty |= (loaded_tag==get_empty_tag())*set_index;
                  local_tombstone |= (loaded_tag==get_tombstone_tag())*set_index;
                  local_match |= (loaded_tag==key_tag)*set_index;

               }

               empty_match |= local_empty;
               tombstone_match |= local_tombstone;
               key_match |= local_match;

            }


            // empty_match |= (my_tile.ballot(found_empty) << offset);
            // tombstone_match |= (my_tile.ballot(found_tombstone) << offset);
            // key_match |= (my_tile.ballot(found_exact) << offset);

            //if (empty_match || key_match) return;

         }

         //final reduction

         #if LARGE_BUCKET_MODS
         //cg::reduce(tile, thread_sum, cg::plus<int>()) / length;
         empty_match |= cg::reduce(my_tile, empty_match, cg::plus<uint64_t>());
         tombstone_match |= cg::reduce(my_tile, tombstone_match, cg::plus<uint64_t>());
         key_match |= cg::reduce(my_tile, key_match, cg::plus<uint64_t>());

         #else

         empty_match |= cg::reduce(my_tile, empty_match, cg::plus<uint32_t>());
         tombstone_match |= cg::reduce(my_tile, tombstone_match, cg::plus<uint32_t>());
         key_match |= cg::reduce(my_tile, key_match, cg::plus<uint32_t>());

         #endif

         //printf("Done with load_huge\n");

         return;

      }




      #if LARGE_BUCKET_MODS
      __device__ void load_fill_ballots_big_query(const cg::thread_block_tile<partition_size> & my_tile, const Key & upsert_key, __restrict__ uint64_t & empty_match, __restrict__ uint64_t & tombstone_match, __restrict__ uint64_t & key_match)
      #else
      __device__ void load_fill_ballots_big_query(const cg::thread_block_tile<partition_size> & my_tile, const Key & upsert_key, __restrict__ uint32_t & empty_match, __restrict__ uint32_t & tombstone_match, __restrict__ uint32_t & key_match)
      #endif
      {


         ADD_PROBE_TILE


         uint64_t * md_as_uint64_t = (uint64_t *) metadata; 

         //wipe previous
         empty_match = 0U;
         tombstone_match = 0U;
         key_match = 0U;

         uint16_t key_tag = get_tag(upsert_key);

         for (uint i = my_tile.thread_rank(); i < (n_traversals-1)/4+1; i+=my_tile.size()){

            //step in clean intervals of my_tile. 
            uint offset = i - my_tile.thread_rank();

            bool valid_to_load = i < (bucket_size-1/4)+1;

            
            //maybe change these
            #if LARGE_BUCKET_MODS
            //uint64_t local_empty = 0U;
            uint64_t local_match = 0U;
            #else
            //uint32_t local_empty = 0U;
            uint32_t local_match = 0U;
            #endif



            if (valid_to_load){

               uint64_t loaded_key = hash_table_load(&md_as_uint64_t[i]);


               uint16_t * load_key_indexer = (uint16_t *) &loaded_key;

               for (uint j = 0; j < 4; j++){

                  uint16_t loaded_tag = load_key_indexer[j];

                  #if LARGE_BUCKET_MODS
                  uint64_t set_index = SET_BIT_MASK(i*4+j);
                  #else
                  uint32_t set_index = SET_BIT_MASK(i*4+j);
                  #endif

                  //local_empty |= (loaded_tag==get_empty_tag())*set_index;
                  local_match |= (loaded_tag==key_tag)*set_index;

               }

            }

            #if LARGE_BUCKET_MODS
            //cg::reduce(tile, thread_sum, cg::plus<int>()) / length;
            //empty_match |= cg::reduce(my_tile, local_empty, cg::plus<uint64_t>());

            key_match |= cg::reduce(my_tile, local_match, cg::plus<uint64_t>());

            #else

            //empty_match |= cg::reduce(my_tile, local_empty, cg::plus<uint32_t>());
            key_match |= cg::reduce(my_tile, local_match, cg::plus<uint32_t>());

            #endif


            // int leader = __ffs(my_tile.ballot(local_match))-1;


            // if (leader != -1){
            //    key_match = my_tile.shfl(local_match, leader);
            // }
            


            // empty_match |= (my_tile.ballot(found_empty) << offset);
            // tombstone_match |= (my_tile.ballot(found_tombstone) << offset);
            // key_match |= (my_tile.ballot(found_exact) << offset);

            //if (empty_match || key_match) return;

         }

         return;

      }


      template <typename bucket_type>
      __device__ bool query_md_and_bucket(const cg::thread_block_tile<partition_size> & my_tile, const Key & upsert_key, Val & val, bucket_type * primary_bucket)
      {


         ADD_PROBE_TILE


         uint64_t * md_as_uint64_t = (uint64_t *) metadata; 

         //wipe previous

         uint16_t key_tag = get_tag(upsert_key);

         for (uint i = my_tile.thread_rank(); i < (n_traversals-1)/4+1; i+=my_tile.size()){

            //step in clean intervals of my_tile. 
            uint offset = i - my_tile.thread_rank();

            bool valid_to_load = i < (bucket_size-1/4)+1;

            
            //maybe change these
            #if LARGE_BUCKET_MODS
            //uint64_t local_empty = 0U;
            uint64_t local_match = 0U;
            #else
            //uint32_t local_empty = 0U;
            uint32_t local_match = 0U;
            #endif



            if (valid_to_load){



               uint64_t loaded_key = hash_table_load(&md_as_uint64_t[i]);


               uint16_t * load_key_indexer = (uint16_t *) &loaded_key;




               for (uint j = 0; j < 4; j++){

                  bool found = false;

                  uint16_t loaded_tag = load_key_indexer[j];

                  // #if LARGE_BUCKET_MODS
                  // uint64_t set_index = SET_BIT_MASK(i*4+j);
                  // #else
                  // uint32_t set_index = SET_BIT_MASK(i*4+j);
                  // #endif

                  if (loaded_tag == key_tag){

                     ADD_PROBE

                     //Key loaded_key = hash_table_load(&primary_bucket->slots[i*4+j].key);

                     auto loaded_pair = ht_load_packed_pair(&primary_bucket->slots[i*4+j]);

                     if (loaded_pair.key == upsert_key){

                        found = true;

                        val = loaded_pair.val;
                        //hash_table_load(&primary_bucket->slots[i*4+j].val);

                     }

                  }

                  int leader = __ffs(my_tile.ballot(found))-1;

                  if (leader != -1){
                     val = my_tile.shfl(val, leader);
                     return true;
                  }
                  
               }

            }


         }

         return false;

      }


      template <typename bucket_type>
      __device__ bool query_md_and_bucket_large(const cg::thread_block_tile<partition_size> & my_tile, const Key & upsert_key, Val & val, bucket_type * primary_bucket)
      {


         ADD_PROBE_TILE


         //uint64_t * md_as_uint64_t = (uint64_t *) metadata; 

         //wipe previous

         uint16_t key_tag = get_tag(upsert_key);

         for (uint i = my_tile.thread_rank(); i < (n_traversals-1)/8+1; i+=my_tile.size()){

            //step in clean intervals of my_tile. 
            uint offset = i - my_tile.thread_rank();

            bool valid_to_load = i < (bucket_size-1/8)+1;



            if (valid_to_load){

               packed_tags loaded_pair = load_multi_tags(&metadata[i*8]);

               //uint64_t loaded_key = hash_table_load(&md_as_uint64_t[i]);


               uint16_t * load_key_indexer = (uint16_t *) &loaded_pair;




               for (uint j = 0; j < 8; j++){

                  bool found = false;

                  uint16_t loaded_tag = load_key_indexer[j];

                  // #if LARGE_BUCKET_MODS
                  // uint64_t set_index = SET_BIT_MASK(i*4+j);
                  // #else
                  // uint32_t set_index = SET_BIT_MASK(i*4+j);
                  // #endif

                  if (loaded_tag == key_tag){

                     ADD_PROBE

                     //Key loaded_key = hash_table_load(&primary_bucket->slots[i*8+j].key);
                     auto loaded_pair = ht_load_packed_pair(&primary_bucket->slots[i*8+j]);

                     if (loaded_pair.key == upsert_key){

                        found = true;

                        val = loaded_pair.val;
                        // hash_table_load(&primary_bucket->slots[i*8+j].val);

                     }

                  }

                  int leader = __ffs(my_tile.ballot(found))-1;
                  
                  if (leader != -1){
                     val = my_tile.shfl(val, leader);
                     return true;
                  }
                  
               }

            } else {

               for (uint64_t j = 0; j < 8; j++){
                  bool found = false;
                  int leader = __ffs(my_tile.ballot(found))-1;

                  if (leader != -1){
                     val = my_tile.shfl(val, leader);
                     return true;
                  }
               }


            }


         }

         return false;

      }

      template <typename bucket_type>
      __device__ bool delete_md_and_bucket(const cg::thread_block_tile<partition_size> & my_tile, const Key & upsert_key, bucket_type * primary_bucket)
      {


         ADD_PROBE_TILE


         uint64_t * md_as_uint64_t = (uint64_t *) metadata; 

         //wipe previous


         uint16_t key_tag = get_tag(upsert_key);

         for (uint i = my_tile.thread_rank(); i < (n_traversals-1)/4+1; i+=my_tile.size()){

            //step in clean intervals of my_tile. 
            uint offset = i - my_tile.thread_rank();

            bool valid_to_load = i < (bucket_size-1/4)+1;

            
            //maybe change these
            #if LARGE_BUCKET_MODS
            //uint64_t local_empty = 0U;
            uint64_t local_match = 0U;
            #else
            //uint32_t local_empty = 0U;
            uint32_t local_match = 0U;
            #endif



            if (valid_to_load){

               uint64_t loaded_key = hash_table_load(&md_as_uint64_t[i]);


               uint16_t * load_key_indexer = (uint16_t *) &loaded_key;




               for (uint j = 0; j < 4; j++){

                  bool found = false;

                  uint16_t loaded_tag = load_key_indexer[j];

                  // #if LARGE_BUCKET_MODS
                  // uint64_t set_index = SET_BIT_MASK(i*4+j);
                  // #else
                  // uint32_t set_index = SET_BIT_MASK(i*4+j);
                  // #endif

                  if (loaded_tag == key_tag){

                     //delete it.

                     Key loaded_key = hash_table_load(&primary_bucket->slots[i*4+j].key);

                     if (loaded_key == upsert_key){

                        //ht_store(&primary_bucket->slots[i*4+j].val, tombstoneVal);
                        //__threadfence();
                        ht_store(&primary_bucket->slots[i*4+j].key, tombstoneKey);

                        //__threadfence();

                        set_tombstone(i*4+j);

                        __threadfence();

                        found = true;

                     }

                  }

                  int leader = __ffs(my_tile.ballot(found))-1;

                  if (leader != -1){
                     return true;
                  }
                  
               }

            }


         }

         return false;

      }


      template <typename bucket_type>
      __device__ pair_type * query_md_and_bucket_pair(const cg::thread_block_tile<partition_size> & my_tile, const Key & upsert_key, bucket_type * primary_bucket)
      {


         ADD_PROBE_TILE


         //uint64_t * md_as_uint64_t = (uint64_t *) metadata; 

         //wipe previous


         uint16_t key_tag = get_tag(upsert_key);

         for (uint i = my_tile.thread_rank(); i < (n_traversals-1)/8+1; i+=my_tile.size()){

            //step in clean intervals of my_tile. 
            uint offset = i - my_tile.thread_rank();

            bool valid_to_load = i < (bucket_size-1/8)+1;

            
            //maybe change these
            #if LARGE_BUCKET_MODS
            //uint64_t local_empty = 0U;
            uint64_t local_match = 0U;
            #else
            //uint32_t local_empty = 0U;
            uint32_t local_match = 0U;
            #endif



            if (valid_to_load){

               packed_tags loaded_pair = load_multi_tags(&metadata[i*8]);

               //uint64_t loaded_key = hash_table_load(&md_as_uint64_t[i]);


               uint16_t * load_key_indexer = (uint16_t *) &loaded_pair;




               for (uint j = 0; j < 8; j++){

                  bool found = false;

                  uint16_t loaded_tag = load_key_indexer[j];

                  // #if LARGE_BUCKET_MODS
                  // uint64_t set_index = SET_BIT_MASK(i*4+j);
                  // #else
                  // uint32_t set_index = SET_BIT_MASK(i*4+j);
                  // #endif

                  if (loaded_tag == key_tag){

                     ADD_PROBE

                     Key loaded_key = hash_table_load(&primary_bucket->slots[i*8+j].key);

                     if (loaded_key == upsert_key){

                        found = true;

                        //val = hash_table_load(&primary_bucket->slots[i*8+j].val);

                     }

                  }

                  int leader = __ffs(my_tile.ballot(found))-1;

                  if (leader != -1){

                     int index = my_tile.shfl(i*8+j, leader);

                     return &primary_bucket->slots[index];
                  }
                  
               }

            }


         }

         return nullptr;

      }



   };


  }

}

#endif