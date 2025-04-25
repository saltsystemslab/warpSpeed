#ifndef HT_BUCKET
#define HT_BUCKET

#include <warpSpeed/helpers/ht_pairs.cuh>
#include <warpSpeed/helpers/probe_counts.cuh>
#include <warpSpeed/helpers/ht_load.cuh>

namespace warpSpeed {

  namespace tables {



    template <typename Key, Key defaultKey, Key tombstoneKey, Key holdingKey, typename Val, Val defaultVal, Val tombstoneVal, uint partition_size, uint bucket_size>
    struct hash_table_bucket {

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
                 bool ballot_exists = false;

                 const auto leader = __ffs(ballot_result)-1;


                 if (leader == my_tile.thread_rank() && empty_ballot){

                    ballot_exists = true;

                    ballot = typed_atomic_write(&slots[i].key, defaultKey, holdingKey);
                    ADD_PROBE
                    if (ballot){

                       ht_store_packed_pair(&slots[i], {ext_key, ext_val});
                       __threadfence();
                       //ht_store(&slots[i].val, ext_val);
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
                       ht_store_packed_pair(&slots[i], {ext_key, ext_val});
                       __threadfence();

                       //ht_store(&slots[i].val, ext_val);


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


                    //ballot = typed_atomic_write(&slots[i].key, ext_key, ext_key);
                    ballot = true;

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

        //attempt to insert into the table based on an existing mapping.
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


                    ballot = true;
                    replace_func(&slots[i], ext_key, ext_val);
                    ADD_PROBE

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

           //int my_count = 0;

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

              //if (empty_match || key_match) return;

              //if (__popc(key_match | empty_match)) return;

           }

           return;

        }


      __device__ bool load_fill_ballots_upserts(const cg::thread_block_tile<partition_size> & my_tile, const Key & upsert_key, const Val & upsert_val, __restrict__ uint & empty_match, __restrict__ uint & tombstone_match, __restrict__ uint & key_match){


           //wipe previous
           empty_match = 0U;
           tombstone_match = 0U;
           key_match = 0U;

           //int my_count = 0;

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

              empty_match |= (my_tile.ballot(found_empty) << offset);
              tombstone_match |= (my_tile.ballot(found_tombstone) << offset);
              key_match |= (my_tile.ballot(found_exact) << offset);





              //if (empty_match || key_match) return;

              int leader = __ffs(my_tile.ballot(found_exact))-1;

              bool ballot = false;

              if (leader == my_tile.thread_rank()){

                 ADD_PROBE

                 //if (gallatin::utils::typed_atomic_write(&slots[i].key, upsert_key, upsert_key)){

                    ht_store(&slots[i].val, upsert_val);
                    ballot = true;


                 //}

              }

              //upserted.
              if (my_tile.ballot(ballot)) return true;

              leader = __ffs(my_tile.ballot(found_empty))-1;

              if (leader == my_tile.thread_rank() && i < bucket_size*.75){

                 ADD_PROBE

                 if (gallatin::utils::typed_atomic_write(&slots[i].key, defaultKey, holdingKey)){
                    
                    ht_store_packed_pair(&slots[i], {upsert_key, upsert_val});
                    __threadfence();
                    //ht_store(&slots[i].val, upsert_val);
                    ballot = true;
                 }

              }

              if (my_tile.ballot(ballot)) return true;

           }

           return false;

        }

        __device__ bool load_fill_ballots_upserts_func(const cg::thread_block_tile<partition_size> & my_tile, const Key & upsert_key, const Val & upsert_val, __restrict__ uint & empty_match, __restrict__ uint & tombstone_match, __restrict__ uint & key_match, void (*replace_func)(pair_type *, Key, Val)){


           //wipe previous
           empty_match = 0U;
           tombstone_match = 0U;
           key_match = 0U;

           //int my_count = 0;

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

              empty_match |= (my_tile.ballot(found_empty) << offset);
              tombstone_match |= (my_tile.ballot(found_tombstone) << offset);
              key_match |= (my_tile.ballot(found_exact) << offset);





              //if (empty_match || key_match) return;

              int leader = __ffs(my_tile.ballot(found_exact))-1;

              bool ballot = false;

              if (leader == my_tile.thread_rank()){

                 ADD_PROBE
                 ballot = true;
                 replace_func(&slots[i], upsert_key, upsert_val);
              }

              //upserted.
              if (my_tile.ballot(ballot)) return true;

              leader = __ffs(my_tile.ballot(found_empty))-1;

              if (leader == my_tile.thread_rank() && i < bucket_size*.75){

                 ADD_PROBE

                 if (gallatin::utils::typed_atomic_write(&slots[i].key, defaultKey, holdingKey)){
                    
                    ht_store_packed_pair(&slots[i], {upsert_key, upsert_val});
                    __threadfence();
                    //ht_store(&slots[i].val, upsert_val);
                    ballot = true;
                 }

              }

              if (my_tile.ballot(ballot)) return true;

           }

           return false;

        }

        __device__ pair_type * query_pair(const cg::thread_block_tile<partition_size> & my_tile, Key ext_key, bool & found_empty){

         found_empty = false;

         for (uint i = my_tile.thread_rank(); i < n_traversals; i+=my_tile.size()){

            ADD_PROBE_ADJUSTED

            uint offset = i - my_tile.thread_rank();

            bool valid = i < bucket_size;

            bool found_ballot = false;

            Val loaded_val;

            if (valid){

               Key loaded_key = hash_table_load(&slots[i].key);

               //Key loaded_key = hash_table_load(&slots[i].key);

               found_ballot = (loaded_key == ext_key);

               if (loaded_key == defaultKey){
                  found_empty = true;
               }

               
            }


            int found = __ffs(my_tile.ballot(found_ballot))-1;

            found_empty = my_tile.ballot(found_empty);

            if (found == -1){

               if (found_empty){
                  return nullptr;
               }

               continue;
            }


            
            //found_empty = my_tile.ballot(found_empty);

            return &slots[found];



         }

         found_empty = my_tile.ballot(found_empty);


         return nullptr;

      }

        __device__ bool query(const cg::thread_block_tile<partition_size> & my_tile, Key ext_key, Val & return_val){


           for (uint i = my_tile.thread_rank(); i < n_traversals; i+=my_tile.size()){

              uint offset = i - my_tile.thread_rank();

              bool valid = i < bucket_size;

              bool found_ballot = false;

              Val loaded_val;

              ADD_PROBE_ADJUSTED

              if (valid){

                 pair_type loaded_pair = load_packed_pair(i);

                 //Key loaded_key = hash_table_load(&slots[i].key);

                 found_ballot = (loaded_pair.key == ext_key);

                 //found_ballot = (loaded_key == ext_key);

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
                    if (ballot){

                       //no store needed, future writes will supply value.
                       //ht_store(&slots[i].val, tombstoneVal);
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

  }

}

#endif