/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */

#define COUNT_PROBES 1

#define LOAD_CHEAP 0

#define MEASURE_INDEPENDENT 1

// #define COUNT_PROBES 1

#include <argparse/argparse.hpp>

#include <gallatin/allocators/global_allocator.cuh>

#include <gallatin/allocators/timer.cuh>

#include <warpcore/single_value_hash_table.cuh>
#include <gallatin/allocators/timer.cuh>

#include <bght/p2bht.hpp>

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
// #include <hashing_project/table_wrappers/warpcore_wrapper.cuh>
#include <hashing_project/tables/p2_hashing_external.cuh>
#include <hashing_project/tables/p2_hashing_internal.cuh>
//#include <hashing_project/tables/iht_double_hashing.cuh>
#include <hashing_project/tables/double_hashing.cuh>
#include <hashing_project/tables/iht_p2.cuh>
#include <hashing_project/tables/iht_p2_metadata.cuh>
#include <hashing_project/tables/chaining.cuh>
#include <hashing_project/tables/p2_hashing_metadata.cuh>
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



#define PRINT_THROUGHPUT_ONLY 1




#define DATA_TYPE uint64_t


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


template <typename ht_type, uint tile_size>
__global__ void insert_only_kernel(ht_type * table, DATA_TYPE * insert_buffer, uint64_t n_keys, uint64_t * misses){


   auto thread_block = cg::this_thread_block();

   cg::thread_block_tile<tile_size> my_tile = cg::tiled_partition<tile_size>(thread_block);


   uint64_t tid = gallatin::utils::get_tile_tid(my_tile);

   if (tid >= n_keys) return;

   // if (__popc(my_tile.ballot(1)) != 32){
   //    printf("Bad tile size\n");
   // }


   uint64_t my_key = insert_buffer[tid];

   if (!table->upsert_replace(my_tile, my_key, my_key)){

      //table->upsert_replace(my_tile, my_key, my_key);


      #if MEASURE_FAILS
      if (my_tile.thread_rank() == 0){

         atomicAdd((unsigned long long int *)&misses[0], 1ULL);
         //printf("Init upsert failed for %lu\n", my_key);
      }
      #endif
      
   }


}


template <typename ht_type, uint tile_size>
__global__ void sawtooth_kernel(ht_type * table, DATA_TYPE * item_buffer, uint64_t * opcodes, uint64_t n_keys, uint64_t * misses){

   auto thread_block = cg::this_thread_block();

   cg::thread_block_tile<tile_size> my_tile = cg::tiled_partition<tile_size>(thread_block);


   uint64_t tid = gallatin::utils::get_tile_tid(my_tile);

   if (tid >= n_keys*4) return;

   uint64_t my_key = item_buffer[tid];

   uint64_t my_code = opcodes[tid];

   // if (__popc(my_tile.ballot(1)) != 32){
   //    printf("Bad tile size\n");
   // }




   if (my_code == 0){
      //return;
      if (!table->upsert_replace(my_tile, my_key, my_key)){
         #if MEASURE_FAILS
         if (my_tile.thread_rank() == 0){
            atomicAdd((unsigned long long int *)&misses[1], 1ULL);
         }
         #endif

         //table->upsert_replace(my_tile, my_key, my_key);

      }
   } else if (my_code == 1){
      if (!table->remove(my_tile, my_key)){
         #if MEASURE_FAILS
         if (my_tile.thread_rank() == 0){
          atomicAdd((unsigned long long int *)&misses[2], 1ULL);
         }
         #endif
      }
   } else if (my_code == 2) {

      uint64_t found_val = 0;

      if (table->find_with_reference(my_tile, my_key, found_val)){
         if (found_val != my_key){
            #if MEASURE_FAILS
            if (my_tile.thread_rank() == 0){
             atomicAdd((unsigned long long int *)&misses[3], 1ULL);
            }
            #endif
            //printf("Bad key val pair: found %lu != %lu\n", found_val, my_key);
         }
      }
   }

}



template <typename ht_type, uint tile_size>
__global__ void sawtooth_insert(ht_type * table, DATA_TYPE * item_buffer, uint64_t * opcodes, uint64_t n_keys, uint64_t * misses){

   auto thread_block = cg::this_thread_block();

   cg::thread_block_tile<tile_size> my_tile = cg::tiled_partition<tile_size>(thread_block);


   uint64_t tid = gallatin::utils::get_tile_tid(my_tile);

   if (tid >= n_keys*3) return;

   uint64_t my_key = item_buffer[tid];

   uint64_t my_code = opcodes[tid];

   // if (__popc(my_tile.ballot(1)) != 32){
   //    printf("Bad tile size\n");
   // }




   if (my_code == 0){
      //return;
      if (!table->upsert_replace(my_tile, my_key, my_key)){

         table->upsert_replace(my_tile, my_key, my_key);
         #if MEASURE_FAILS
         if (my_tile.thread_rank() == 0){
            atomicAdd((unsigned long long int *)&misses[1], 1ULL);
         }
         #endif
      }
   } 

}


template <typename ht_type, uint tile_size>
__global__ void sawtooth_delete(ht_type * table, DATA_TYPE * item_buffer, uint64_t * opcodes, uint64_t n_keys, uint64_t * misses){

   auto thread_block = cg::this_thread_block();

   cg::thread_block_tile<tile_size> my_tile = cg::tiled_partition<tile_size>(thread_block);


   uint64_t tid = gallatin::utils::get_tile_tid(my_tile);

   if (tid >= n_keys*3) return;

   uint64_t my_key = item_buffer[tid];

   uint64_t my_code = opcodes[tid];

   // if (__popc(my_tile.ballot(1)) != 32){
   //    printf("Bad tile size\n");
   // }


   if (my_code == 1){
      if (!table->remove(my_tile, my_key)){
         #if MEASURE_FAILS
         if (my_tile.thread_rank() == 0){
          atomicAdd((unsigned long long int *)&misses[2], 1ULL);
         }
         #endif
      }
   } 

}


template <typename ht_type, uint tile_size>
__global__ void sawtooth_query(ht_type * table, DATA_TYPE * item_buffer, uint64_t * opcodes, uint64_t n_keys, uint64_t * misses){

   auto thread_block = cg::this_thread_block();

   cg::thread_block_tile<tile_size> my_tile = cg::tiled_partition<tile_size>(thread_block);


   uint64_t tid = gallatin::utils::get_tile_tid(my_tile);

   if (tid >= n_keys*3) return;

   uint64_t my_key = item_buffer[tid];

   uint64_t my_code = opcodes[tid];

   // if (__popc(my_tile.ballot(1)) != 32){
   //    printf("Bad tile size\n");
   // }

   //printf("Querying\n");

   if (my_code == 2) {

      DATA_TYPE found_val = 0;

      if (table->find_with_reference(my_tile, my_key, found_val)){
         if (found_val != my_key){
            #if MEASURE_FAILS
            if (my_tile.thread_rank() == 0){
             atomicAdd((unsigned long long int *)&misses[3], 1ULL);
            }
            #endif
            //printf("Bad key val pair: found %lu != %lu\n", found_val, my_key);
         }
      } else {

         if (my_tile.thread_rank() == 0){
             atomicAdd((unsigned long long int *)&misses[3], 1ULL);
         }

      }
   }

}


template <typename ht_type, uint tile_size>
__global__ void sawtooth_negative_query(ht_type * table, DATA_TYPE * item_buffer, uint64_t n_keys, uint64_t * misses){

   auto thread_block = cg::this_thread_block();

   cg::thread_block_tile<tile_size> my_tile = cg::tiled_partition<tile_size>(thread_block);


   uint64_t tid = gallatin::utils::get_tile_tid(my_tile);

   if (tid >= n_keys) return;

   uint64_t my_key = item_buffer[tid];

   // if (__popc(my_tile.ballot(1)) != 32){
   //    printf("Bad tile size\n");
   // }

   //printf("Querying\n");

   DATA_TYPE found_val = 0;

   if (table->find_with_reference(my_tile, my_key, found_val)){

      //should not find.

      if (my_tile.thread_rank() == 0){
         atomicAdd((unsigned long long int *)&misses[4], 1ULL);
      }
     
          

         //printf("Bad key val pair: found %lu != %lu\n", found_val, my_key);
      }

}




__global__ void set_opcode_kernel(uint64_t * opcodes, uint64_t offset_start, uint64_t codes_to_set, uint64_t code_value){

   uint64_t tid = gallatin::utils::get_tid();

   if (tid >= codes_to_set) return;

   opcodes[tid+offset_start] = code_value;

}


__host__ void set_opcodes(uint64_t * opcodes, uint64_t offset_start, uint64_t codes_to_set, uint64_t code_value){


   set_opcode_kernel<<<(codes_to_set-1)/256+1,256>>>(opcodes, offset_start, codes_to_set, code_value);

   cudaDeviceSynchronize();

}

//stolen shamelessy from https://stackoverflow.com/questions/43482488/how-to-format-a-number-with-thousands-separator-in-c-c
// struct separate_thousands : std::numpunct<char> {
//     char_type do_thousands_sep() const override { return ','; }  // separate with commas
//     string_type do_grouping() const override { return "\3"; } // groups of 3 digit
// };

template <template<typename, typename, uint, uint> typename hash_table_type, uint tile_size, uint bucket_size>
__host__ void sawtooth_test(uint64_t n_indices, double max_fill, double replacement_rate, DATA_TYPE * access_pattern, DATA_TYPE * negative_pattern, uint64_t n_loops){



   uint64_t keys_inserted = 0;
   uint64_t keys_deleted = 0;


   uint64_t keys_for_fill = n_indices*max_fill;

   uint64_t keys_per_round = n_indices*replacement_rate;


   using ht_type = hash_table_type<DATA_TYPE, DATA_TYPE, tile_size, bucket_size>;


   //generate table and buffers

   #if MEASURE_INDEPENDENT
   double insert_time = 0;
   double query_time = 0;
   double delete_time = 0;
   double negative_query_time = 0;


   uint64_t insert_probes = 0;
   uint64_t query_probes = 0;
   uint64_t delete_probes = 0;

   uint64_t negative_query_probes = 0;

   #else

   double total_time = 0;
   uint64_t total_probes = 0;

   #endif


   uint64_t * misses;

   cudaMallocManaged((void **)&misses, sizeof(uint64_t)*5);

   cudaDeviceSynchronize();

   misses[0] = 0;
   misses[1] = 0;
   misses[2] = 0;
   misses[3] = 0;
   misses[4] = 0;


   ht_type * table = ht_type::generate_on_device(n_indices, 42);


   DATA_TYPE * insert_buffer = gallatin::utils::get_device_version<DATA_TYPE>(keys_for_fill);


   //set original buffer
   cudaMemcpy(insert_buffer, access_pattern, sizeof(DATA_TYPE)*keys_for_fill, cudaMemcpyHostToDevice);


   helpers::get_num_probes();

   cudaDeviceSynchronize();

   printf("Starting test for %s, inserting %lu keys\n", ht_type::get_name(), keys_for_fill);;

   gallatin::utils::timer upsert_timing;

   insert_only_kernel<ht_type, tile_size><<<(keys_for_fill*tile_size-1)/256+1,256>>>(table, insert_buffer, keys_for_fill, misses);

   upsert_timing.sync_end();
   std::cout.imbue(std::locale(""));
   upsert_timing.print_throughput("Upserted", keys_for_fill);

   double original_upsert_time = upsert_timing.elapsed();

   uint64_t setup_probes = helpers::get_num_probes();

   #if !PRINT_THROUGHPUT_ONLY
   std::cout << "Upsert avg probes: " << 1.0*setup_probes/keys_for_fill << "\n";
   #endif

   keys_inserted+=keys_for_fill;

   //start loop

   cudaFree(insert_buffer);




   //create buffers
   DATA_TYPE * sawtooth_buffer = gallatin::utils::get_device_version<DATA_TYPE>(4*keys_per_round);

   DATA_TYPE * negative_buffer = gallatin::utils::get_device_version<DATA_TYPE>(keys_per_round);

   uint64_t * opcode_buffer = gallatin::utils::get_device_version<uint64_t>(4*keys_per_round);



   //

   #if !MEASURE_INDEPENDENT
   std::string filename = "results/sawtooth_combined_" + std::to_string(n_loops) + "/" + ht_type::get_name() + ".txt";


   printf("Writing to %s\n", filename.c_str());

   std::ofstream myfile;
   myfile.open (filename.c_str());
   myfile << "op perf\n";

   #endif

   //keys_deleted+=keys_per_round;

   for (uint64_t i =0; i < n_loops; i++){


      //setup buffers!

      cudaMemcpy(negative_buffer, negative_pattern+keys_deleted, sizeof(DATA_TYPE)*keys_per_round, cudaMemcpyHostToDevice);

      cudaMemcpy(sawtooth_buffer+keys_per_round, access_pattern+keys_deleted, sizeof(DATA_TYPE)*keys_per_round, cudaMemcpyHostToDevice);


      set_opcodes(opcode_buffer, keys_per_round, keys_per_round, 1ULL);


      keys_deleted+=keys_per_round;


      //first is old_queries - pull from keys in table      

      set_opcodes(opcode_buffer, 0ULL, keys_per_round, 2ULL);

      //cudaMemcpy(sawtooth_buffer+0ULL, access_pattern+keys_deleted, sizeof(DATA_TYPE)*keys_per_round, cudaMemcpyHostToDevice);


      //new version - pull from end of buffer
      cudaMemcpy(sawtooth_buffer+0ULL, access_pattern+keys_inserted-2*keys_per_round, sizeof(DATA_TYPE)*keys_per_round, cudaMemcpyHostToDevice);


      //and new insertions
      cudaMemcpy(sawtooth_buffer+keys_per_round+keys_per_round, access_pattern+keys_inserted, sizeof(DATA_TYPE)*keys_per_round, cudaMemcpyHostToDevice);

      keys_inserted+=keys_per_round;

      set_opcodes(opcode_buffer, 2*keys_per_round, keys_per_round, 0ULL);


      cudaMemcpy(sawtooth_buffer+3*keys_per_round, negative_pattern+i*keys_per_round, sizeof(DATA_TYPE)*keys_per_round, cudaMemcpyHostToDevice);
      set_opcodes(opcode_buffer, 3*keys_per_round, keys_per_round, 2ULL);

      //flush
      helpers::get_num_probes();

   
      //done with setup

      //printf("Staring round %lu\n", i);

      cudaDeviceSynchronize();

      #if MEASURE_INDEPENDENT

      gallatin::utils::timer query_timer;

      sawtooth_query<ht_type, tile_size><<<(3*keys_per_round*tile_size-1)/256+1,256>>>(table, sawtooth_buffer, opcode_buffer, keys_per_round, misses);

      query_timer.sync_end();

      query_time += query_timer.elapsed();

      //printf("Query done\n");

      query_probes += helpers::get_num_probes();
      cudaDeviceSynchronize();


      gallatin::utils::timer negative_timer;

      sawtooth_negative_query<ht_type, tile_size><<<(keys_per_round*tile_size-1)/256+1,256>>>(table, negative_buffer, keys_per_round, misses);

      negative_timer.sync_end();

      negative_query_time += negative_timer.elapsed();

      //printf("Query done\n");

      negative_query_probes += helpers::get_num_probes();
      cudaDeviceSynchronize();



      gallatin::utils::timer delete_timer;

      sawtooth_delete<ht_type, tile_size><<<(3*keys_per_round*tile_size-1)/256+1,256>>>(table, sawtooth_buffer, opcode_buffer, keys_per_round, misses);

      delete_timer.sync_end();

      delete_time += delete_timer.elapsed();

      // printf("Delete done\n");

      // printf("After delete round %lu load is %f\n", i, table->load());

      delete_probes += helpers::get_num_probes();
      cudaDeviceSynchronize();



      gallatin::utils::timer insert_timer;

      sawtooth_insert<ht_type, tile_size><<<(3*keys_per_round*tile_size-1)/256+1,256>>>(table, sawtooth_buffer, opcode_buffer, keys_per_round, misses);

      insert_timer.sync_end();

      insert_time += insert_timer.elapsed();

      //printf("Insert done\n");
      insert_probes += helpers::get_num_probes();
      cudaDeviceSynchronize();


      #else 



      gallatin::utils::timer sawtooth_timer;

      sawtooth_kernel<ht_type, tile_size><<<(4*keys_per_round*tile_size-1)/256+1,256>>>(table, sawtooth_buffer, opcode_buffer, keys_per_round, misses);

      sawtooth_timer.sync_end();

      //write round
      myfile << std::setprecision(12) << i << " " <<  1.0*4*keys_per_round/(sawtooth_timer.elapsed()*1000000) << "\n";


      total_time += sawtooth_timer.elapsed();

      total_probes += helpers::get_num_probes();
      cudaDeviceSynchronize();

      #endif





      //printf("Done with round %lu\n", i);


   }


   //
   //std::cout.imbue(std::locale(std::cout.getloc(), thousands.release()));
   #if !PRINT_THROUGHPUT_ONLY
   printf("Misses: Original: %lu upsert: %lu delete: %lu queries: %lu, negative: %lu\n", misses[0], misses[1], misses[2], misses[3], misses[4]);
   #endif


   cudaFree(misses);


   
   //write to output

   #if MEASURE_INDEPENDENT



   #if COUNT_PROBES

   std::string filename = "results/sawtooth_probe_" + std::to_string(n_loops) + "/" + ht_type::get_name() + ".txt";


   printf("Writing to %s\n", filename.c_str());

   std::ofstream myfile;
   myfile.open (filename.c_str());
   myfile << "op probes\n";

   myfile << "Upsert " << 1.0*insert_probes/(n_loops*keys_per_round) << "\n";
   myfile << "Query " << 1.0*query_probes/(n_loops*keys_per_round) << "\n";
   myfile << "Delete " << 1.0*delete_probes/(n_loops*keys_per_round) << "\n";
   myfile << "Negative " << 1.0*negative_query_probes/(n_loops*keys_per_round) << "\n";
  
   myfile.close();

   #else

   std::string filename = "results/sawtooth_" + std::to_string(n_loops) + "/" + ht_type::get_name() + ".txt";


   printf("Writing to %s\n", filename.c_str());

   std::ofstream myfile;
   myfile.open (filename.c_str());
   myfile << "op perf\n";
   //myfile << std::setprecision(12) << "Insertion " <<  1.0*keys_for_fill/(original_upsert_time*1000000) << "\n";
   myfile << std::setprecision(12) << "Upsert " <<  1.0*n_loops*keys_per_round/(insert_time*1000000) << "\n";
   myfile << std::setprecision(12) << "Query " <<  1.0*n_loops*keys_per_round/(query_time*1000000) << "\n";
   myfile << std::setprecision(12) << "Deletion " << 1.0*n_loops*keys_per_round/(delete_time*1000000) << "\n";
   myfile << std::setprecision(12) << "Negative " << 1.0*n_loops*keys_per_round/(negative_query_time*1000000) << "\n";
   myfile.close();

   #endif






   #if PRINT_THROUGHPUT_ONLY

   std::cout <<  1.0*n_loops*keys_per_round/insert_time << "\n";
   std::cout <<  1.0*n_loops*keys_per_round/query_time << "\n";
   std::cout <<  1.0*n_loops*keys_per_round/delete_time << "\n";


   #else
   std::cout << "Insert time " << insert_time << " throughput " << 1.0*n_loops*keys_per_round/insert_time << "\n";
   std::cout << "Query time " << query_time << " throughput " << 1.0*n_loops*keys_per_round/query_time << "\n";
   std::cout << "Delete time " << delete_time << " throughput " << 1.0*n_loops*keys_per_round/delete_time << "\n";
   std::cout << "Negative time " << negative_query_time << " throughput " << 1.0*n_loops*keys_per_round/negative_query_time << "\n";


   std::cout << "avg probes: insert " << 1.0*insert_probes/(n_loops*keys_per_round) << "\n";
   std::cout << "avg probes: query " << 1.0*query_probes/(n_loops*keys_per_round) << "\n";
   std::cout << "avg probes: delete " << 1.0*delete_probes/(n_loops*keys_per_round) << "\n";
   
   #endif

   #else

   std::cout.imbue(std::locale(""));
   printf("Done with %llu rounds in %f seconds, throughput %f\n", n_loops, total_time, 1.0*n_loops*4*keys_per_round/total_time);


   //write output


   //myfile << std::setprecision(12) << "Insertion " <<  1.0*keys_for_fill/(original_upsert_time*1000000) << "\n";
   //myfile << std::setprecision(12) << "throughput " <<  1.0*n_loops*3*keys_per_round/(total_time*1000000) << "\n";

   myfile.close();

   #endif

   cudaFree(sawtooth_buffer);
   cudaFree(opcode_buffer);
   cudaFree(negative_buffer);

   table->print_space_usage();

   ht_type::free_on_device(table);
   cudaDeviceSynchronize();

}


__host__ void execute_test(std::string table, uint64_t table_capacity, uint32_t n_rounds, double init_fill, double replacement_rate){

   uint64_t replacement_items = replacement_rate*(n_rounds+1)*table_capacity;

   auto access_pattern = generate_data<DATA_TYPE>(table_capacity+replacement_items);

   auto negative_pattern = generate_data<DATA_TYPE>(table_capacity+replacement_items);


   if (table == "p2"){

      sawtooth_test<hashing_project::tables::p2_ext_generic, 8, 32>(table_capacity, init_fill, replacement_rate, access_pattern, negative_pattern, n_rounds);

      //p2 p2MD double doubleMD iceberg icebergMD cuckoo chaining bght_p2 bght_cuckoo");


   } else if (table == "p2MD"){

      sawtooth_test<hashing_project::tables::md_p2_generic, 4, 32>(table_capacity, init_fill, replacement_rate, access_pattern, negative_pattern, n_rounds);

   } else if (table == "double"){
      sawtooth_test<hashing_project::tables::double_generic, 8, 8>(table_capacity, init_fill, replacement_rate, access_pattern, negative_pattern, n_rounds);

   } else if (table == "doubleMD"){

      sawtooth_test<hashing_project::tables::md_double_generic, 4, 32>(table_capacity, init_fill, replacement_rate, access_pattern, negative_pattern, n_rounds);


   } else if (table == "iceberg"){

      sawtooth_test<hashing_project::tables::iht_p2_generic, 8, 32>(table_capacity, init_fill, replacement_rate, access_pattern, negative_pattern, n_rounds);
     
   } else if (table == "icebergMD"){

      sawtooth_test<hashing_project::tables::iht_p2_metadata_full_generic, 4, 32>(table_capacity, init_fill, replacement_rate, access_pattern, negative_pattern, n_rounds);

   } else if (table == "cuckoo") {
       sawtooth_test<hashing_project::tables::cuckoo_generic, 4, 8>(table_capacity, init_fill, replacement_rate, access_pattern, negative_pattern, n_rounds);
   
   } else if (table == "chaining"){

      init_global_allocator(8ULL*1024*1024*1024, 111);

      sawtooth_test<hashing_project::tables::chaining_generic, 4, 8>(table_capacity, init_fill, replacement_rate, access_pattern, negative_pattern, n_rounds);

      free_global_allocator();
   } else {
      throw std::runtime_error("Unknown table");
   }



   cudaFreeHost(access_pattern);
   cudaFreeHost(negative_pattern);
}



int main(int argc, char** argv) {




   argparse::ArgumentParser program("scaling_test");

   // program.add_argument("square")
   // .help("display the square of a given integer")
   // .scan<'i', int>();

   program.add_argument("--table", "-t")
   .required()
   .help("Specify table type. Options [p2 p2MD double doubleMD iceberg icebergMD cuckoo chaining");

   program.add_argument("--capacity", "-c").required().scan<'u', uint64_t>().help("Number of slots in the table.");

   program.add_argument("--rounds", "-n").required().scan<'u', uint32_t>().help("Number of rounds to execute. Each round swaps out replacement_rate percentage of items.");

   program.add_argument("--init_fill", "-i").required().scan<'g', double>().help("Initial fill of the table before the sawtooth test. Fraction [0-1]");

   program.add_argument("--replacement_rate", "-r").required().scan<'g', double>().help("Percentage of items to swap out each round.");


   try {
    program.parse_args(argc, argv);
   }
   catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return 1;
   }

   auto table = program.get<std::string>("--table");
   auto table_capacity = program.get<uint64_t>("--capacity");

   uint32_t n_rounds = program.get<uint32_t>("--rounds");

   double init_fill = program.get<double>("--init_fill");

   double replacement_rate = program.get<double>("--replacement_rate");

   std::cout << "Running aging probe test with table " << table << " and " << table_capacity << " slots." << std::endl;


   execute_test(table, table_capacity, n_rounds, init_fill, replacement_rate);

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


   // if(fs::create_directory("results")){
   //  std::cout << "Created a directory\n";
   // } else {
   //  std::cerr << "Failed to create a directory\n";
   // }

   // if(fs::create_directory("results/sawtooth_" + std::to_string(n_rounds))){
   //  std::cout << "Created a directory\n";
   // } else {
   //  std::cerr << "Failed to create a directory\n";
   // }


   // #if !MEASURE_INDEPENDENT
   //    if(fs::create_directory("results/sawtooth_combined_" + std::to_string(n_rounds))){
   //     std::cout << "Created a directory\n";
   //    } else {
   //     std::cerr << "Failed to create a directory\n";
   //    }
   // #endif

   // #if COUNT_PROBES

   // if(fs::create_directory("results/sawtooth_probe_" + std::to_string(n_rounds))){
   //  std::cout << "Created a directory\n";
   // } else {
   //  std::cerr << "Failed to create a directory\n";
   // }

   // #endif

   // double init_fill = .85;
   // double replacement_rate = .01;

   // uint64_t replacement_items = replacement_rate*(n_rounds+1)*table_capacity;

   // auto access_pattern = generate_data<DATA_TYPE>(table_capacity+replacement_items);

   // auto negative_pattern = generate_data<DATA_TYPE>(table_capacity+replacement_items);


   // sawtooth_test<hashing_project::tables::md_p2_generic, 4, 32>(table_capacity, init_fill, replacement_rate, access_pattern, negative_pattern, n_rounds);

   // sawtooth_test<hashing_project::tables::p2_ext_generic, 8, 32>(table_capacity, init_fill, replacement_rate, access_pattern, negative_pattern, n_rounds);

   // sawtooth_test<hashing_project::tables::md_double_generic, 4, 32>(table_capacity, init_fill, replacement_rate, access_pattern, negative_pattern, n_rounds);

   // sawtooth_test<hashing_project::tables::cuckoo_generic, 4, 8>(table_capacity, init_fill, replacement_rate, access_pattern, negative_pattern, n_rounds);


   // init_global_allocator(8ULL*1024*1024*1024, 111);

   // sawtooth_test<hashing_project::tables::chaining_generic, 4, 8>(table_capacity, init_fill, replacement_rate, access_pattern, negative_pattern, n_rounds);

   // free_global_allocator();
   
   // sawtooth_test<hashing_project::tables::iht_p2_generic, 8, 32>(table_capacity, init_fill, replacement_rate, access_pattern, negative_pattern, n_rounds);
      
   
   // sawtooth_test<hashing_project::tables::iht_p2_metadata_full_generic, 4, 32>(table_capacity, init_fill, replacement_rate, access_pattern, negative_pattern, n_rounds);
   

   // sawtooth_test<hashing_project::wrappers::warpcore_wrapper, 8, 8>(table_capacity, init_fill, replacement_rate, access_pattern, negative_pattern, n_rounds); 

   // cudaFreeHost(access_pattern);

  


   cudaDeviceReset();
   return 0;

}
