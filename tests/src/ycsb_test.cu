/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */



#define COUNT_PROBES 0

#define LOAD_CHEAP 0

#include <argparse/argparse.hpp>

#include <gallatin/allocators/global_allocator.cuh>

#include <gallatin/allocators/timer.cuh>

#include <warpcore/single_value_hash_table.cuh>
#include <gallatin/allocators/timer.cuh>

#include <bght/p2bht.hpp>
#include <bght/bcht.hpp>
#include <bght/iht.hpp>

#include <hashing_project/cache.cuh>

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <assert.h>
#include <chrono>
#include <openssl/rand.h>

#include <filesystem>

namespace fs = std::filesystem;


// #include <hashing_project/table_wrappers/p2_wrapper.cuh>
// #include <hashing_project/table_wrappers/dummy_ht.cuh>
// #include <hashing_project/table_wrappers/iht_wrapper.cuh>
#include <hashing_project/table_wrappers/warpcore_wrapper.cuh>
#include <hashing_project/tables/p2_hashing_external.cuh>
#include <hashing_project/tables/p2_hashing_inverted.cuh>
#include <hashing_project/tables/p2_hashing_internal.cuh>
#include <hashing_project/tables/double_hashing.cuh>
#include <hashing_project/tables/iht_p2.cuh>
#include <hashing_project/tables/chaining.cuh>
#include <hashing_project/tables/p2_hashing_metadata.cuh>
#include <hashing_project/tables/iht_p2_metadata.cuh>
#include <hashing_project/tables/iht_p2_metadata_full.cuh>
#include <hashing_project/tables/cuckoo.cuh>
#include <hashing_project/tables/double_hashing_metadata.cuh>



#include <iostream>
#include <locale>


//thrust stuff.
#include <thrust/shuffle.h>
#include <thrust/random.h>

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

using namespace gallatin::allocators;



#define MEASURE_FAILS 1

#define MEASURE_COUNTERS 0

#define MEASURE_INDEPENDENT 1

#define PRINT_THROUGHPUT_ONLY 1


#define DATA_TYPE uint64_t


#define LARGE_MD_LOAD 1
#define LARGE_BUCKET_MODS 0

struct ycsb_load_type {


   bool * is_insert;
   uint64_t * keys;
   uint64_t * values;
   uint64_t n_items;

};

class ycsb_pair {
public:

   uint64_t is_insert;
   uint64_t key;
   uint64_t val;

    // Function to read data from a file stream
    friend std::ifstream& operator>>(std::ifstream& in, ycsb_pair& pair) {
        in >> pair.is_insert >> pair.key >> pair.val;
        return in;
    }
};

ycsb_load_type load_ycsb(std::string filename){

   std::vector<bool> is_insert;
   std::vector<uint64_t> keys;
   std::vector<uint64_t> values;

   uint64_t n_items = 0;

   std::ifstream file(filename);

   if(!file.is_open()){
      std::cerr << "Could not open file " << filename << std::endl;
   }

   ycsb_pair p;

   while (file >> p){
      is_insert.push_back(p.is_insert);
      keys.push_back(p.key);
      values.push_back(p.val);
      n_items+=1;
   }

   file.close();

   printf("Loaded %lu items\n", n_items);

   //copy out.


   //bool * vector_data = is_insert.data();

   bool * insert_status = gallatin::utils::get_host_version<bool>(n_items);
   //cudaMemcpy(insert_status, vector_data, sizeof(bool)*n_items, cudaMemcpyHostToHost);

   for (uint64_t i = 0; i < n_items; i++){
      insert_status[i] = is_insert[i];
   }

   uint64_t * keys_host = gallatin::utils::get_host_version<uint64_t>(n_items);
   cudaMemcpy(keys_host, keys.data(), sizeof(uint64_t)*n_items, cudaMemcpyHostToHost);

   uint64_t * values_host = gallatin::utils::get_host_version<uint64_t>(n_items);
   cudaMemcpy(values_host, values.data(), sizeof(uint64_t)*n_items, cudaMemcpyHostToHost);


   return {insert_status, keys_host, values_host, n_items};

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

   //printf("Generation done\n");
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

template <typename ht_type, uint tile_size>
__global__ void ycsb_kernel_cuckoo(ht_type * table, bool * op_type, uint64_t * key_buffer, uint64_t * val_buffer, uint64_t n_keys, uint64_t * misses, uint64_t * counters, bool cheap_insert, bool load_phase){


   auto thread_block = cg::this_thread_block();

   cg::thread_block_tile<tile_size> my_tile = cg::tiled_partition<tile_size>(thread_block);


   uint64_t tid = gallatin::utils::get_tile_tid(my_tile);

   if (tid >= n_keys) return;

   // if (__popc(my_tile.ballot(1)) != 32){
   //    printf("Bad tile size\n");
   // }


   bool is_insert = op_type[tid];
   uint64_t my_key = key_buffer[tid];
   uint64_t my_val = val_buffer[tid];

   if (is_insert){

      #if MEASURE_COUNTERS
      if (load_phase && my_tile.thread_rank() == 0){
         atomicAdd(&counters[0], 1ULL);
      }

      if (!load_phase && my_tile.thread_rank() == 0){
         atomicAdd(&counters[1], 1ULL);
      }
      #endif


      if (!table->upsert_replace(my_tile, my_key, my_val)){

         #if MEASURE_FAILS
         if (my_tile.thread_rank() == 0){

            atomicAdd((unsigned long long int *)&misses[0], 1ULL);
            //printf("Init upsert failed for %lu\n", my_key);
         }
         #endif
      
      }

   } else {

      #if MEASURE_COUNTERS
      if (my_tile.thread_rank() == 0){
         atomicAdd(&counters[3], 1ULL);
      }
      #endif


      // if (!table->find_with_reference(my_tile, my_key, my_val)){

      //    //table->upsert_replace(my_tile, my_key, my_key);

      //    //table->find_with_reference(my_tile, my_key, my_val);


      //    #if MEASURE_FAILS
      //    if (my_tile.thread_rank() == 0){

      //       atomicAdd((unsigned long long int *)&misses[1], 1ULL);
      //       //printf("Init upsert failed for %lu\n", my_key);
      //    }
      //    #endif
         
      // }

   }


}

template <typename ht_type, uint tile_size>
__global__ void ycsb_kernel(ht_type * table, bool * op_type, uint64_t * key_buffer, uint64_t * val_buffer, uint64_t n_keys, uint64_t * misses, uint64_t * counters, bool cheap_insert, bool load_phase){


   auto thread_block = cg::this_thread_block();

   cg::thread_block_tile<tile_size> my_tile = cg::tiled_partition<tile_size>(thread_block);


   uint64_t tid = gallatin::utils::get_tile_tid(my_tile);

   if (tid >= n_keys) return;

   // if (__popc(my_tile.ballot(1)) != 32){
   //    printf("Bad tile size\n");
   // }


   bool is_insert = op_type[tid];
   uint64_t my_key = key_buffer[tid];
   uint64_t my_val = val_buffer[tid];

   if (is_insert){

      if (cheap_insert){

         auto pair = table->find_pair(my_tile, my_key);

         if (pair != nullptr){

                  #if MEASURE_COUNTERS
                  if (my_tile.thread_rank() == 0){
                     atomicAdd(&counters[2], 1ULL);
                  }
                  #endif

            if (my_tile.thread_rank() == 0){

               ADD_PROBE
               //ht_store(&pair->val, my_val);
            }

            my_tile.sync();

            return;
         }

      }

      #if MEASURE_COUNTERS
      if (load_phase && my_tile.thread_rank() == 0){
         atomicAdd(&counters[0], 1ULL);
      }

      if (!load_phase && my_tile.thread_rank() == 0){
         atomicAdd(&counters[1], 1ULL);
      }
      #endif


      if (!table->upsert_replace(my_tile, my_key, my_val)){

         #if MEASURE_FAILS
         if (my_tile.thread_rank() == 0){

            atomicAdd((unsigned long long int *)&misses[0], 1ULL);
            //printf("Init upsert failed for %lu\n", my_key);
         }
         #endif
      
      }

   } else {

      #if MEASURE_COUNTERS
      if (my_tile.thread_rank() == 0){
         atomicAdd(&counters[3], 1ULL);
      }
      #endif


      if (!table->find_with_reference(my_tile, my_key, my_val)){

         //table->upsert_replace(my_tile, my_key, my_key);

         //table->find_with_reference(my_tile, my_key, my_val);


         #if MEASURE_FAILS
         if (my_tile.thread_rank() == 0){

            atomicAdd((unsigned long long int *)&misses[1], 1ULL);
            //printf("Init upsert failed for %lu\n", my_key);
         }
         #endif
         
      }

   }


}


template <typename ht_type, uint tile_size>
__global__ void ycsb_kernel_cheap(ht_type * table, bool * op_type, uint64_t * key_buffer, uint64_t * val_buffer, uint64_t n_keys, uint64_t * misses, uint64_t * counters, bool cheap_insert, bool load_phase){


   auto thread_block = cg::this_thread_block();

   cg::thread_block_tile<tile_size> my_tile = cg::tiled_partition<tile_size>(thread_block);


   uint64_t tid = gallatin::utils::get_tile_tid(my_tile);

   if (tid >= n_keys) return;

   // if (__popc(my_tile.ballot(1)) != 32){
   //    printf("Bad tile size\n");
   // }


   bool is_insert = op_type[tid];
   uint64_t my_key = key_buffer[tid];
   uint64_t my_val = val_buffer[tid];

   if (is_insert){

      if (cheap_insert){

         auto pair = table->find_pair(my_tile, my_key);

         if (pair != nullptr){
   
            if (my_tile.thread_rank() == 0){
               ADD_PROBE
               //ht_store(&pair->val, my_val);
            }

            return;
         }

      }

      if (!table->upsert_replace(my_tile, my_key, my_val)){

         #if MEASURE_FAILS
         if (my_tile.thread_rank() == 0){

            atomicAdd((unsigned long long int *)&misses[0], 1ULL);
            //printf("Init upsert failed for %lu\n", my_key);
         }
         #endif
      
      }

   } else {
      

      if (!table->find_with_reference_no_lock(my_tile, my_key, my_val)){

         //table->upsert_replace(my_tile, my_key, my_key);

         //table->find_with_reference(my_tile, my_key, my_val);


         #if MEASURE_FAILS
         if (my_tile.thread_rank() == 0){

            atomicAdd((unsigned long long int *)&misses[1], 1ULL);
            //printf("Init upsert failed for %lu\n", my_key);
         }
         #endif
         
      }

   }


}

template <typename ht_type, uint tile_size>
__global__ void remove_kernel(ht_type * table, DATA_TYPE * insert_buffer, uint64_t n_keys, uint64_t * misses){


   auto thread_block = cg::this_thread_block();

   cg::thread_block_tile<tile_size> my_tile = cg::tiled_partition<tile_size>(thread_block);


   uint64_t tid = gallatin::utils::get_tile_tid(my_tile);

   if (tid >= n_keys) return;

   // if (__popc(my_tile.ballot(1)) != 32){
   //    printf("Bad tile size\n");
   // }


   uint64_t my_key = insert_buffer[tid];

   if (!table->remove(my_tile, my_key)){

      // table->upsert_replace(my_tile, my_key, my_key);

      // table->remove(my_tile, my_key);


      #if MEASURE_FAILS
      if (my_tile.thread_rank() == 0){

         atomicAdd((unsigned long long int *)&misses[2], 1ULL);
         //printf("Init upsert failed for %lu\n", my_key);
      }
      #endif
      
   } 

}



template <typename ht_type, uint tile_size>
__global__ void query_kernel(ht_type * table, DATA_TYPE * insert_buffer, uint64_t n_keys, uint64_t * misses){


   auto thread_block = cg::this_thread_block();

   cg::thread_block_tile<tile_size> my_tile = cg::tiled_partition<tile_size>(thread_block);


   uint64_t tid = gallatin::utils::get_tile_tid(my_tile);

   if (tid >= n_keys) return;

   // if (__popc(my_tile.ballot(1)) != 32){
   //    printf("Bad tile size\n");
   // }


   DATA_TYPE my_key = insert_buffer[tid];
   DATA_TYPE my_val;



   if (!table->find_with_reference(my_tile, my_key, my_val)){

      //table->upsert_replace(my_tile, my_key, my_key);

      //table->find_with_reference(my_tile, my_key, my_val);


      #if MEASURE_FAILS
      if (my_tile.thread_rank() == 0){

         atomicAdd((unsigned long long int *)&misses[1], 1ULL);
         //printf("Init upsert failed for %lu\n", my_key);
      }
      #endif
      
   } else {

      if (my_val != my_key){


        //table->find_with_reference(my_tile, my_key, my_val);


         atomicAdd((unsigned long long int *)&misses[1], 1ULL);
      }
   }


}


template <template<typename, typename, uint, uint> typename hash_table_type, uint tile_size, uint bucket_size>
__host__ void lf_test(ycsb_load_type load_data, ycsb_load_type run_data, std::string source_filename, bool cheap, bool cheap_insert){



   using ht_type = hash_table_type<DATA_TYPE, DATA_TYPE, tile_size, bucket_size>;


   //generate table and buffers
   uint64_t * misses;

   cudaMallocManaged((void **)&misses, sizeof(uint64_t)*4);

   cudaDeviceSynchronize();

   misses[0] = 0;
   misses[1] = 0;
   misses[2] = 0;
   misses[3] = 0;



   uint64_t * counters;

   cudaMallocManaged((void **)&counters, sizeof(uint64_t)*4);

   cudaDeviceSynchronize();

   counters[0] = 0;
   counters[1] = 0;
   counters[2] = 0;
   counters[3] = 0;

   #if COUNT_PROBES

   std::string filename = "results/ycsb_probe/"+source_filename + "/";


   filename = filename + ht_type::get_name() + ".txt";


   //printf("Writing to %s\n", filename.c_str());
   //write to output

   std::ofstream myfile;
   myfile.open (filename.c_str());
   myfile << "lf,insert,query,remove\n";


   #else

   std::string filename = "results/ycsb/"+source_filename + "/";
   //std::string filename = "results/ycsb/";

   if (cheap){
      filename = "results/ycsb_cheap/"+source_filename + "/";
      printf("Executing cheap, writing to %s\n", filename.c_str());
   }

   


   filename = filename + ht_type::get_name() + ".txt";

   printf("Writing to %s\n", filename.c_str());

   //printf("Writing to %s\n", filename.c_str());
   //write to output

   std::ofstream myfile;
   myfile.open (filename.c_str());
   myfile << "insert,query\n";

   #endif


   uint64_t n_indices = load_data.n_items;
  


   double avg_insert_throughput = 0;
   double avg_query_throughput = 0;
   double avg_delete_throughput = 0;


   printf("N_items %lu n_queries %lu\n", n_indices, run_data.n_items);

   ht_type * table = ht_type::generate_on_device(n_indices*1.2, 42);

   helpers::get_num_probes();

   uint64_t items_to_insert = n_indices;

   bool * insert_status = gallatin::utils::get_device_version<bool>(items_to_insert);

   uint64_t * device_keys = gallatin::utils::get_device_version<uint64_t>(items_to_insert);

   uint64_t * device_vals = gallatin::utils::get_device_version<uint64_t>(items_to_insert);

   //set original buffer

   cudaMemcpy(insert_status, load_data.is_insert, sizeof(bool)*items_to_insert, cudaMemcpyHostToDevice);
   cudaMemcpy(device_keys, load_data.keys, sizeof(uint64_t)*items_to_insert, cudaMemcpyHostToDevice);
   cudaMemcpy(device_vals, load_data.values, sizeof(uint64_t)*items_to_insert, cudaMemcpyHostToDevice);

   cudaDeviceSynchronize();

   gallatin::utils::timer insert_timer;

   ycsb_kernel<ht_type, tile_size><<<(items_to_insert*tile_size-1)/256+1,256>>>(table, insert_status, device_keys, device_vals, items_to_insert, misses, counters, false, true);

   insert_timer.sync_end();

   uint64_t insert_probes = helpers::get_num_probes();

   cudaDeviceSynchronize();

   cudaFree(insert_status);
   cudaFree(device_keys);
   cudaFree(device_vals);


   uint64_t n_queries = run_data.n_items;

   DATA_TYPE * device_queries = gallatin::utils::get_device_version<DATA_TYPE>(n_queries);


   insert_status = gallatin::utils::get_device_version<bool>(n_queries);

   device_keys = gallatin::utils::get_device_version<uint64_t>(n_queries);

   device_vals = gallatin::utils::get_device_version<uint64_t>(n_queries);

   cudaMemcpy(insert_status, run_data.is_insert, sizeof(bool)*n_queries, cudaMemcpyHostToDevice);
   cudaMemcpy(device_keys, run_data.keys, sizeof(uint64_t)*n_queries, cudaMemcpyHostToDevice);
   cudaMemcpy(device_vals, run_data.values, sizeof(uint64_t)*n_queries, cudaMemcpyHostToDevice);

   cudaDeviceSynchronize();

   gallatin::utils::timer query_timer;


   if (cheap){

      ycsb_kernel_cheap<ht_type, tile_size><<<(n_queries*tile_size-1)/256+1,256>>>(table, insert_status, device_keys, device_vals, n_queries, misses, counters, cheap_insert, false);

   } else {

      ycsb_kernel<ht_type, tile_size><<<(n_queries*tile_size-1)/256+1,256>>>(table, insert_status, device_keys, device_vals, n_queries, misses, counters, cheap_insert, false);

   }

   query_timer.sync_end();

   uint64_t query_probes = helpers::get_num_probes();

   cudaDeviceSynchronize();

   cudaFree(insert_status);
   cudaFree(device_keys);
   cudaFree(device_vals);


   //table->print_chain_stats();

   ht_type::free_on_device(table);

   // insert_timer.print_throughput("Inserted", items_to_insert);
   // query_timer.print_throughput("Queried", items_to_insert);
   // remove_timer.print_throughput("Removed", items_to_insert);

   #if COUNT_PROBES


   printf("Insert Probes %f phase probes %f\n", 1.0*insert_probes/items_to_insert, 1.0*query_probes/n_queries);
   //printf("Probes %llu %llu %llu\n", insert_probes, query_probes, remove_probes);
 
   #endif


   myfile << std::setprecision(12) << 1.0*items_to_insert/(insert_timer.elapsed()*1000000) << "," << 1.0*n_queries/(query_timer.elapsed()*1000000) << "\n";

   std::cout << std::setprecision(12) << 1.0*items_to_insert/(insert_timer.elapsed()*1000000) << "," << 1.0*n_queries/(query_timer.elapsed()*1000000) << "\n";


   printf("Misses: %lu %lu\n", misses[0], misses[1]);

   printf("Counters: Load Inserts: %lu Run Inserts: %lu Run shortcuts: %lu queries: %lu\n", counters[0], counters[1], counters[2], counters[3]);

   // misses[0] = 0;
   // misses[1] = 0;
   // misses[2] = 0;
   cudaDeviceSynchronize();

   //cuckoo is not leaking memory oon device.
   //gallatin::allocators::print_global_stats();



   //double avg_throughput = (avg_insert_throughput+avg_query_throughput+avg_delete_throughput)/3;

   //printf("%u-%u Avg operations throughput %f\n", bucket_size, tile_size, avg_throughput);

   //printf("Misses: %lu %lu %lu\n", misses[0], misses[1], misses[2]);

   myfile.close();
 
  
   cudaFree(misses);
   cudaFree(counters);
   cudaDeviceSynchronize();

}



template <template<typename, typename, uint, uint> typename hash_table_type, uint tile_size, uint bucket_size>
__host__ void lf_test_cuckoo(ycsb_load_type load_data, ycsb_load_type run_data, std::string source_filename, bool cheap, bool cheap_insert){



   using ht_type = hash_table_type<DATA_TYPE, DATA_TYPE, tile_size, bucket_size>;


   //generate table and buffers
   uint64_t * misses;

   cudaMallocManaged((void **)&misses, sizeof(uint64_t)*4);

   cudaDeviceSynchronize();

   misses[0] = 0;
   misses[1] = 0;
   misses[2] = 0;
   misses[3] = 0;



   uint64_t * counters;

   cudaMallocManaged((void **)&counters, sizeof(uint64_t)*4);

   cudaDeviceSynchronize();

   counters[0] = 0;
   counters[1] = 0;
   counters[2] = 0;
   counters[3] = 0;

   #if COUNT_PROBES

   std::string filename = "results/ycsb_probe/"+source_filename + "/";


   filename = filename + ht_type::get_name() + ".txt";


   //printf("Writing to %s\n", filename.c_str());
   //write to output

   std::ofstream myfile;
   myfile.open (filename.c_str());
   myfile << "lf,insert,query,remove\n";


   #else

   std::string filename = "results/ycsb/"+source_filename + "/";
   //std::string filename = "results/ycsb/";

   if (cheap){
      filename = "results/ycsb_cheap/"+source_filename + "/";
      printf("Executing cheap, writing to %s\n", filename.c_str());
   }

   


   filename = filename + ht_type::get_name() + ".txt";

   printf("Writing to %s\n", filename.c_str());

   //printf("Writing to %s\n", filename.c_str());
   //write to output

   std::ofstream myfile;
   myfile.open (filename.c_str());
   myfile << "insert,query\n";

   #endif


   uint64_t n_indices = load_data.n_items;
  


   double avg_insert_throughput = 0;
   double avg_query_throughput = 0;
   double avg_delete_throughput = 0;


   printf("N_items %lu n_queries %lu\n", n_indices, run_data.n_items);

   ht_type * table = ht_type::generate_on_device(n_indices*1.2, 42);

   helpers::get_num_probes();

   uint64_t items_to_insert = n_indices;

   bool * insert_status = gallatin::utils::get_device_version<bool>(items_to_insert);

   uint64_t * device_keys = gallatin::utils::get_device_version<uint64_t>(items_to_insert);

   uint64_t * device_vals = gallatin::utils::get_device_version<uint64_t>(items_to_insert);

   //set original buffer

   cudaMemcpy(insert_status, load_data.is_insert, sizeof(bool)*items_to_insert, cudaMemcpyHostToDevice);
   cudaMemcpy(device_keys, load_data.keys, sizeof(uint64_t)*items_to_insert, cudaMemcpyHostToDevice);
   cudaMemcpy(device_vals, load_data.values, sizeof(uint64_t)*items_to_insert, cudaMemcpyHostToDevice);

   cudaDeviceSynchronize();

   gallatin::utils::timer insert_timer;

   ycsb_kernel_cuckoo<ht_type, tile_size><<<(items_to_insert*tile_size-1)/256+1,256>>>(table, insert_status, device_keys, device_vals, items_to_insert, misses, counters, false, true);

   insert_timer.sync_end();

   uint64_t insert_probes = helpers::get_num_probes();

   cudaDeviceSynchronize();

   cudaFree(insert_status);
   cudaFree(device_keys);
   cudaFree(device_vals);


   uint64_t n_queries = run_data.n_items;

   DATA_TYPE * device_queries = gallatin::utils::get_device_version<DATA_TYPE>(n_queries);


   insert_status = gallatin::utils::get_device_version<bool>(n_queries);

   device_keys = gallatin::utils::get_device_version<uint64_t>(n_queries);

   device_vals = gallatin::utils::get_device_version<uint64_t>(n_queries);

   cudaMemcpy(insert_status, run_data.is_insert, sizeof(bool)*n_queries, cudaMemcpyHostToDevice);
   cudaMemcpy(device_keys, run_data.keys, sizeof(uint64_t)*n_queries, cudaMemcpyHostToDevice);
   cudaMemcpy(device_vals, run_data.values, sizeof(uint64_t)*n_queries, cudaMemcpyHostToDevice);

   cudaDeviceSynchronize();

   gallatin::utils::timer query_timer;



   ycsb_kernel_cuckoo<ht_type, tile_size><<<(n_queries*tile_size-1)/256+1,256>>>(table, insert_status, device_keys, device_vals, n_queries, misses, counters, cheap_insert, false);


   query_timer.sync_end();

   uint64_t query_probes = helpers::get_num_probes();

   cudaDeviceSynchronize();

   cudaFree(insert_status);
   cudaFree(device_keys);
   cudaFree(device_vals);


   //table->print_chain_stats();

   ht_type::free_on_device(table);

   // insert_timer.print_throughput("Inserted", items_to_insert);
   // query_timer.print_throughput("Queried", items_to_insert);
   // remove_timer.print_throughput("Removed", items_to_insert);

   #if COUNT_PROBES

   //printf("Probes %llu %llu %llu\n", insert_probes, query_probes, remove_probes);
 
   myfile << std::setprecision(12) << 1.0*insert_probes/items_to_insert << "," << 1.0*query_probes/n_queries << "\n";

   std::cout << std::setprecision(12) << 1.0*insert_probes/items_to_insert << "," << 1.0*query_probes/n_queries << "\n";

   #else

   myfile << std::setprecision(12) << 1.0*items_to_insert/(insert_timer.elapsed()*1000000) << "," << 1.0*n_queries/(query_timer.elapsed()*1000000) << "\n";

   std::cout << std::setprecision(12) << 1.0*items_to_insert/(insert_timer.elapsed()*1000000) << "," << 1.0*n_queries/(query_timer.elapsed()*1000000) << "\n";

   #endif

   printf("Misses: %lu %lu\n", misses[0], misses[1]);

   printf("Counters: Load Inserts: %lu Run Inserts: %lu Run shortcuts: %lu queries: %lu\n", counters[0], counters[1], counters[2], counters[3]);

   // misses[0] = 0;
   // misses[1] = 0;
   // misses[2] = 0;
   cudaDeviceSynchronize();

   //cuckoo is not leaking memory oon device.
   //gallatin::allocators::print_global_stats();



   //double avg_throughput = (avg_insert_throughput+avg_query_throughput+avg_delete_throughput)/3;

   //printf("%u-%u Avg operations throughput %f\n", bucket_size, tile_size, avg_throughput);

   //printf("Misses: %lu %lu %lu\n", misses[0], misses[1], misses[2]);

   myfile.close();
 
  
   cudaFree(misses);
   cudaFree(counters);
   cudaDeviceSynchronize();

}


template <typename data>
__global__ void count_duplicates(data * data_array, uint64_t n_pairs, uint64_t * misses){

   uint64_t tid = gallatin::utils::get_tid();

   uint64_t first = tid/n_pairs;
   uint64_t second = tid % n_pairs;

   if (first >= n_pairs || second >= n_pairs) return;


   if (data_array[first] == data_array[second]){
      atomicAdd((unsigned long long int *)&misses, 1ULL);
   }

}

template <typename data>
__host__ void print_duplicates(data * data_array, uint64_t n_pairs){


   uint64_t * misses;

   cudaMallocManaged((void **)&misses, sizeof(uint64_t));

   misses[0] = 0;


   data * device_data = gallatin::utils::get_device_version<data>(n_pairs);

   cudaMemcpy(device_data, data_array, sizeof(data)*n_pairs, cudaMemcpyHostToDevice);

   cudaDeviceSynchronize();

   count_duplicates<<<(n_pairs*n_pairs-1)/512+1,512>>>(device_data, n_pairs, misses);

   cudaDeviceSynchronize();

   //printf("System has %llu duplicates\n", misses[0]);

   cudaFree(misses);
   cudaFree(device_data);

}


template<typename data, typename pair_type>
__global__ void setup_bght_pair(data * data_array, pair_type * pair_array, uint64_t n_pairs){


   uint64_t tid = gallatin::utils::get_tid();

   if (tid >= n_pairs) return;

   pair_array[tid].first = data_array[tid];
   pair_array[tid].second = data_array[tid];

} 


template <typename ht_type, typename data, uint tile_size>
__global__ void bght_query(ht_type table, data * query_data, uint64_t n_keys){

   auto thread_block = cg::this_thread_block();

   cg::thread_block_tile<tile_size> my_tile = cg::tiled_partition<tile_size>(thread_block);


   uint64_t tid = gallatin::utils::get_tile_tid(my_tile);

   if (tid >= n_keys) return;

   data key = query_data[tid];

   auto result = table.find(key, my_tile);

   //doesn't trigger?
   // if (result != key){

   //    //if (my_tile)
   //    printf("Missed\n");
   // }



}

template <template<typename, typename> typename hash_table_type, uint tile_size>
__host__ void lf_test_BGHT(uint64_t n_indices, DATA_TYPE * access_pattern, std::string ht_name){



   using ht_type = hash_table_type<DATA_TYPE, DATA_TYPE>;

   using pair_type = bght::padded_pair<DATA_TYPE, DATA_TYPE>;


   #if COUNT_PROBES

   std::string filename = "results/lf_probe_bght/";

   filename = filename + ht_name + ".txt";


   //printf("Writing to %s\n", filename.c_str());
   //write to output

   std::ofstream myfile;
   myfile.open (filename.c_str());
   myfile << "lf,insert,query\n";


   #else

   std::string filename = "results/lf_bght/";

   filename = filename + ht_name + ".txt";


   //printf("Writing to %s\n", filename.c_str());
   //write to output

   std::ofstream myfile;
   myfile.open (filename.c_str());
   myfile << "lf,insert,query\n";

   #endif


   for (int i = 1; i < 19; i++){

      

      double lf = .05*i;

      ht_type table(n_indices, 0U, ~0U);

      uint64_t items_to_insert = lf*n_indices;

      DATA_TYPE * device_data = gallatin::utils::get_device_version<DATA_TYPE>(items_to_insert);

      DATA_TYPE * query_data = gallatin::utils::get_device_version<DATA_TYPE>(items_to_insert);

      pair_type * device_pairs = gallatin::utils::get_device_version<pair_type>(items_to_insert);

      //set original buffer
      cudaMemcpy(device_data, access_pattern, sizeof(DATA_TYPE)*items_to_insert, cudaMemcpyHostToDevice);

      cudaDeviceSynchronize();

      setup_bght_pair<DATA_TYPE,pair_type><<<(items_to_insert-1)/256+1,256>>>(device_data, device_pairs, items_to_insert);


      #if COUNT_PROBES

      cudaDeviceSynchronize();

      bght::get_num_probes();

      #endif

      cudaDeviceSynchronize();




      gallatin::utils::timer insert_timer;

      bool inserted = table.insert(device_pairs, device_pairs+items_to_insert, 0);

      if (!inserted){
         printf("Failed to insert BGHT keys\n");
      }

      insert_timer.sync_end();

      #if COUNT_PROBES
      auto insert_probes = bght::get_num_probes();
      #endif


      //query
      gallatin::utils::timer query_timer;

      //bght_query<ht_type, DATA_TYPE, tile_size><<<(items_to_insert*tile_size-1)/256+1,256>>>(table, device_data, items_to_insert);

      table.find(device_data, device_data+items_to_insert, query_data, 0);


      query_timer.sync_end();


      #if COUNT_PROBES
      auto query_probes = bght::get_num_probes();
      #endif

      DATA_TYPE * output = gallatin::utils::get_device_version<DATA_TYPE>(items_to_insert);

      // gallatin::utils::timer self_query_timer;

      // table.find(device_data, device_data+items_to_insert, output);


      // self_query_timer.sync_end();

      #if COUNT_PROBES

      myfile << lf << "," << std::setprecision(12) << 1.0*insert_probes/items_to_insert << "," << 1.0*query_probes/items_to_insert  << "\n";

      #else

      myfile << lf << "," << std::setprecision(12) << 1.0*items_to_insert/(insert_timer.elapsed()*1000000) << "," << 1.0*items_to_insert/(query_timer.elapsed()*1000000) << "\n";

      #endif

      insert_timer.print_throughput("Inserted", items_to_insert);
      query_timer.print_throughput("Queried", items_to_insert);
      //self_query_timer.print_throughput("Self Queried", items_to_insert);

      cudaFree(device_pairs);
      cudaFree(device_data);
      cudaFree(query_data);
      cudaFree(output);




   }

   myfile.close();


}



__host__ void execute_test(std::string table, std::string filename, bool cheap, bool cheap_insert){

   std::string load_fname = "../../caching/traces/" + filename+"-load.txt";
   std::string run_fname = "../../caching/traces/" +filename+"-run.txt";

   //load in files

   auto load_data = load_ycsb(load_fname);

   auto run_data = load_ycsb(run_fname);


   //auto access_pattern = generate_data<DATA_TYPE>(table_capacity);

   if (table == "p2"){

      lf_test<hashing_project::tables::p2_ext_generic, 8, 32>(load_data, run_data, filename, cheap, cheap_insert);

      //p2 p2MD double doubleMD iceberg icebergMD cuckoo chaining bght_p2 bght_cuckoo");


   } else if (table == "p2inv"){

      lf_test<hashing_project::tables::p2_inv_generic, 8, 32>(load_data, run_data, filename, cheap, cheap_insert);

   } else if (table == "p2MD"){

      lf_test<hashing_project::tables::md_p2_generic, 4, 32>(load_data, run_data, filename, cheap, cheap_insert);

   } else if (table == "double"){
      lf_test<hashing_project::tables::double_generic, 8, 8>(load_data, run_data, filename, cheap, cheap_insert);

   } else if (table == "doubleMD"){

      lf_test<hashing_project::tables::md_double_generic, 4, 32>(load_data, run_data, filename, cheap, cheap_insert);


   } else if (table == "iceberg"){

      lf_test<hashing_project::tables::iht_p2_generic, 8, 32>(load_data, run_data, filename, cheap, cheap_insert);
     
   } else if (table == "icebergMD"){

      lf_test<hashing_project::tables::iht_p2_metadata_full_generic, 4, 32>(load_data, run_data, filename, cheap, cheap_insert);

   }
   else if (table == "cuckoo") {
       lf_test_cuckoo<hashing_project::tables::cuckoo_generic, 4, 8>(load_data, run_data, filename, false, false);
   
   } 
   else if (table == "chaining"){

      init_global_allocator(30ULL*1024*1024*1024, 111);

      lf_test<hashing_project::tables::chaining_generic, 4, 8>(load_data, run_data, filename, cheap, cheap_insert);

      free_global_allocator();
   } else {
      throw std::runtime_error("Unknown table");
   }



   //cudaFreeHost(access_pattern);
}



int main(int argc, char** argv) {


   argparse::ArgumentParser program("ycsb_test");

   // program.add_argument("square")
   // .help("display the square of a given integer")
   // .scan<'i', int>();

   program.add_argument("--table", "-t")
   .required()
   .help("Specify table type. Options [p2 p2inv p2MD double doubleMD iceberg icebergMD cuckoo chaining bght_p2 bght_cuckoo");

   //program.add_argument("--capacity", "-c").required().scan<'u', uint64_t>().help("Number of slots in the table. Default is 100,000,000");

   program.add_argument("--filename", "-f")
   .required();

   program.add_argument("--cheap")
   .help("lockless queries")
   .flag();

   program.add_argument("--cheap_insert")
   .help("fast upsert for YCSB")
   .flag();

   try {
    program.parse_args(argc, argv);
   }
   catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return 1;
   }

   auto table = program.get<std::string>("--table");
   auto filename = program.get<std::string>("--filename");

   bool cheap = false;
   if (program["--cheap"] == true){
      cheap = true;
   }

   bool cheap_insert = false;
   if (program["--cheap_insert"] == true){
      cheap_insert = true;
   }
   //auo table_capacity = program.get<uint64_t>("--capacity");

   // uint64_t table_capacity;


   std::cout << "Running ycsb test with table " << table << " on YCSB benchmark " << filename << std::endl;

   // if (argc < 2){
   //    table_capacity = 100000000;
   // } else {
   //    table_capacity = std::stoull(argv[1]);
   // }

   // if (argc < 3){
   //    n_rounds = 1000;
   // } else {
   //    n_rounds = std::stoull(argv[2]);
   // }


   if(fs::create_directory("results")){
    //std::cout << "Created a directory\n";
   } else {
    //std::cerr << "Failed to create a directory\n";
   }

   if(fs::create_directory("results/ycsb")){
    //std::cout << "Created a directory\n";
   } else {
    //std::cerr << "Failed to create a directory\n";
   }

   fs::create_directory("results/ycsb/"+filename);
   if (cheap){
      fs::create_directory("results/ycsb_cheap");

      fs::create_directory("results/ycsb_cheap/"+filename);
   }
 
   fs::create_directory("results/ycsb_probe");

   fs::create_directory("results/ycsb_probe/"+filename);

   execute_test(table, filename, cheap, cheap_insert);



   //print_duplicates<DATA_TYPE>(access_pattern, table_capacity);
   



   

   // lf_test<hashing_project::tables::p2_ext_generic, 4, 32>(load_data, run_data);
   // lf_test<hashing_project::tables::p2_ext_generic, 8, 32>(load_data, run_data);



   // lf_test<hashing_project::tables::md_p2_generic, 4, 32>(load_data, run_data);
   
   // lf_test<hashing_project::tables::md_double_generic, 4, 32>(load_data, run_data);



   
   // init_global_allocator(15ULL*1024*1024*1024, 111);

   // lf_test<hashing_project::tables::chaining_generic, 4, 8>(load_data, run_data);

   

   // free_global_allocator();

   // cudaDeviceSynchronize();

   // lf_test<hashing_project::tables::cuckoo_generic, 4, 8>(load_data, run_data);
   
   // lf_test<hashing_project::tables::cuckoo_generic, 8, 8>(load_data, run_data);
   

   // lf_test<hashing_project::tables::iht_p2_generic, 8, 32>(load_data, run_data);
   

   // lf_test<hashing_project::tables::iht_p2_metadata_full_generic, 4, 32>(load_data, run_data);

  


   // printf("Starting BGHT tests\n");

   // lf_test_BGHT<bght::bcht8, 8>(table_capacity, access_pattern, "bcht_8");

   // lf_test_BGHT<bght::bcht16, 16>(table_capacity, access_pattern, "bcht_16");
   
   // lf_test_BGHT<bght::bcht32, 32>(table_capacity, access_pattern, "bcht_32");


   // lf_test_BGHT<bght::p2bht8, 8>(table_capacity, access_pattern, "p2bht_8");
   
   // lf_test_BGHT<bght::p2bht16, 16>(table_capacity, access_pattern, "p2bht_16");
   
   // lf_test_BGHT<bght::p2bht32, 32>(table_capacity, access_pattern, "p2bht_32");



   //testing for all possible configurations.

   //test_all_combinations<hashing_project::tables::p2_ext_generic>(load_data, run_data);
   // test_all_combinations_md<hashing_project::tables::md_p2_generic>(table_capacity, access_pattern);

   //test_all_combinations<hashing_project::tables::double_generic>(table_capacity, access_pattern);
  
   // test_all_combinations_md<hashing_project::tables::md_double_generic>(table_capacity, access_pattern);

   // test_all_combinations_cuckoo<hashing_project::tables::cuckoo_generic>(table_capacity, access_pattern);
   
   // test_all_combinations<hashing_project::tables::iht_p2_generic>(table_capacity, access_pattern);
  
   // test_all_combinations_md<hashing_project::tables::iht_p2_metadata_full_generic>(table_capacity, access_pattern);

   // init_global_allocator(15ULL*1024*1024*1024, 111);

   // test_all_combinations<hashing_project::tables::chaining_generic>(table_capacity, access_pattern);
   

   // free_global_allocator();
   
  

   

   cudaDeviceReset();
   return 0;

}
