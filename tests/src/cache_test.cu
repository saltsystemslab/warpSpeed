/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */




#include <argparse/argparse.hpp>

#include <gallatin/allocators/global_allocator.cuh>

#include <gallatin/allocators/timer.cuh>

#include <warpcore/single_value_hash_table.cuh>
#include <gallatin/allocators/timer.cuh>

#include <bght/p2bht.hpp>

#include <warpSpeed/helpers/cache.cuh>


#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <chrono>
#include <openssl/rand.h>

#include <fstream>
#include <locale>
#include <filesystem>

namespace fs = std::filesystem;



#include <warpSpeed/tables/p2_hashing_metadata.cuh>

#include <warpSpeed/tables/chaining.cuh>
#include <warpSpeed/tables/double_hashing.cuh>
#include <warpSpeed/tables/iht_p2.cuh>
#include <warpSpeed/tables/p2_hashing.cuh>

#include <warpSpeed/tables/iht_metadata.cuh>
#include <warpSpeed/tables/double_hashing_metadata.cuh>
#include <warpSpeed/tables/cuckoo.cuh>

#include <warpSpeed/helpers/zipf.cuh>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

using namespace gallatin::allocators;


#if GALLATIN_DEBUG_PRINTS
   #define TEST_BLOCK_SIZE 256
#else
   #define TEST_BLOCK_SIZE 256
#endif

template <typename T>
__host__ T * generate_uniform(uint64_t nitems){


   //malloc space

   T * vals;

   cudaMallocHost((void **)&vals, sizeof(T)*nitems);


   //          100,000,000
   uint64_t cap = 100000000ULL;

   for (uint64_t to_fill = 0; to_fill < nitems; to_fill+=0){

      uint64_t togen = (nitems - to_fill > cap) ? cap : nitems - to_fill;


      RAND_bytes((unsigned char *) (vals + to_fill), togen * sizeof(T));



      to_fill += togen;

      //printf("Generated %lu/%lu\n", to_fill, nitems);

   }

   //printf("Generation done\n");
   return vals;
}

template <typename T>
__host__ T * generate_data(uint64_t nitems, uint64_t host_items, bool zipfian, double alpha){

   if (zipfian){

      return generate_zipfian_values(nitems, host_items, alpha);

   } else {
      return generate_uniform<T>(nitems);
   }


}




template <typename cache_type, uint tile_size>
__global__ void cache_read_kernel(cache_type * cache, uint64_t n_indices, uint64_t * access_pattern, uint64_t n_ops){


   auto thread_block = cg::this_thread_block();

   cg::thread_block_tile<tile_size> my_tile = cg::tiled_partition<tile_size>(thread_block);


   uint64_t tid = gallatin::utils::get_tile_tid(my_tile);

   if (tid >= n_ops) return;

   uint64_t my_access = access_pattern[tid] % n_indices;

   //if (my_access == 0) my_access = 1;

   cache->read_item(my_tile, my_access);

   //if (my_tile.thread_rank() == 0 && tid % 100000 == 0) printf("Done with %lu\n", tid);

   //if (my_tile.thread_rank() == 0) printf("Done with %lu\n", tid);

   //cache->write_back_host<tile_size>(my_tile, my_access, 1);


   //if (my_tile.thread_rank() == 0 && tid % 1000000 == 0) printf("Done with %lu\n", tid);




}

//pull from blocks
//this kernel tests correctness, and outputs misses in a counter.
//works on actual pointers instead of uint64_t
//The correctness check is done by treating each allocation as a uint64_t and writing the tid
// if TID is not what is expected, we know that a double malloc has occurred.
template <template<typename, typename, uint, uint> typename hash_table_type, uint tile_size, uint bucket_size>
__host__ void cache_test(uint64_t host_items, uint64_t n_ops, uint64_t * data_pattern, bool zipfian){


   using cache_type = warpSpeed::ht_fifo_cache<hash_table_type, tile_size, bucket_size>;

   uint64_t * dev_data = gallatin::utils::get_device_version<uint64_t>(n_ops);

   cudaMemcpy(dev_data, data_pattern, sizeof(uint64_t)*n_ops, cudaMemcpyHostToDevice);

   //uint64_t * dev_data = gallatin::utils::copy_to_device<uint64_t>(access_data, n_ops);




   std::string filename = "results/cache";

   if (zipfian) {
      filename += "_zipfian/";
   } else {
      filename += "/";
   }

   filename += cache_type::get_name();

   filename += ".txt";

   //std::string filename = "results/cache/" + "test" + ".txt";

   //std::string filename = "results/cache/test.txt" + "booga";


   //printf("Writing to %s\n", filename.c_str());

   std::ofstream myfile;
   myfile.open (filename.c_str());
   myfile << "fill perf\n";

   cudaDeviceSynchronize();



   //first round - one percent


   uint64_t tiny_capacity = host_items*(.01);

   printf("Capacity: %lu\n", tiny_capacity);

   cache_type * tiny_cache = cache_type::generate_on_device(host_items, tiny_capacity, .85);

   cudaDeviceSynchronize();

   gallatin::utils::timer tiny_cache_timing;

   cache_read_kernel<cache_type, tile_size><<<(n_ops*tile_size -1)/256+1,256>>>(tiny_cache, host_items, dev_data, n_ops);

   tiny_cache_timing.sync_end();

   double tiny_duration = tiny_cache_timing.elapsed();

   myfile << .01 << " " << std::setprecision(12) << 1.0*n_ops/(tiny_duration*1000000) << "\n";

   tiny_cache->print_space_usage();
   
   cache_type::free_on_device(tiny_cache);





   //for (int i = 10; i < 11; i++){
   for (int i = 2; i <= 14; i++){

      uint64_t capacity = host_items*(.05*i);

      printf("Capacity: %lu\n", capacity);

      cache_type * cache = cache_type::generate_on_device(host_items, capacity, .85);

      cudaDeviceSynchronize();

      gallatin::utils::timer cache_timing;

      cache_read_kernel<cache_type, tile_size><<<(n_ops*tile_size -1)/256+1,256>>>(cache, host_items, dev_data, n_ops);

      cache_timing.sync_end();

      double duration = cache_timing.elapsed();

      myfile << .05*i << " " << std::setprecision(12) << 1.0*n_ops/(duration*1000000) << "\n";

      cache->print_space_usage();
   
      cache_type::free_on_device(cache);



   }

   myfile.close();

}


__host__ void execute_test(std::string table, uint64_t n_ops, uint64_t host_items, bool zipfian, double alpha){


   uint64_t * access_data = generate_data<uint64_t>(n_ops, host_items, zipfian, alpha);
   //auto access_pattern = generate_data<DATA_TYPE>(table_capacity);

   if (table == "p2"){

      cache_test<warpSpeed::tables::p2_ext_generic, 8, 32>(host_items, n_ops, access_data,zipfian);

      //p2 p2MD double doubleMD iceberg icebergMD cuckoo chaining bght_p2 bght_cuckoo");


   } else if (table == "p2MD"){

      cache_test<warpSpeed::tables::md_p2_generic, 4, 32>(host_items, n_ops, access_data,zipfian);

   } else if (table == "double"){
      cache_test<warpSpeed::tables::double_generic, 8, 8>(host_items, n_ops, access_data,zipfian);

   } else if (table == "doubleMD"){

      cache_test<warpSpeed::tables::md_double_generic, 4, 32>(host_items, n_ops, access_data,zipfian);


   } else if (table == "iceberg"){

      cache_test<warpSpeed::tables::iht_p2_generic, 8, 32>(host_items, n_ops, access_data,zipfian);
     
   } else if (table == "icebergMD"){

      cache_test<warpSpeed::tables::iht_metadata_generic, 4, 32>(host_items, n_ops, access_data,zipfian);

   } else if (table == "chaining"){

      init_global_allocator(16ULL*1024*1024*1024, 111);

      cache_test<warpSpeed::tables::chaining_generic, 4, 8>(host_items, n_ops, access_data,zipfian);

      free_global_allocator();
   } else {
      throw std::runtime_error("Unknown table");
   }



   cudaFreeHost(access_data);
}


int main(int argc, char** argv) {


   argparse::ArgumentParser program("cache_test");

   // program.add_argument("square")
   // .help("display the square of a given integer")
   // .scan<'i', int>();

   program.add_argument("--table", "-t")
   .required()
   .help("Specify table type. Options [p2 p2MD double doubleMD iceberg icebergMD chaining");

   program.add_argument("--n_queries", "-n").required().scan<'u', uint64_t>().help("Number of lookups to perform in the cache.");

   program.add_argument("--host_items", "-h").required().scan<'u', uint64_t>().help("Number of items in the test.");

   program.add_argument("--zipfian", "-z").flag().help("Use zipfian values. If not queries are uniform random.");

   program.add_argument("--alpha", "-a").scan<'g', double>().help("Alpha value for the zipfian generator, should be between 0-1.").default_value(.9);

   try {
    program.parse_args(argc, argv);
   }
   catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return 1;
   }

   auto table = program.get<std::string>("--table");
   auto n_queries = program.get<uint64_t>("--n_queries");

   auto host_items = program.get<uint64_t>("--host_items");

   bool zipfian = program.get<bool>("--zipfian");

   double alpha = program.get<double>("--alpha");

   // uint64_t host_items;

   // uint64_t n_ops;


   // if (argc < 2){
   //    host_items = 1000000;
   // } else {
   //    host_items = std::stoull(argv[1]);
   // }


   // if (argc < 3){
   //    n_ops = 1000;
   // } else {
   //    n_ops = std::stoull(argv[2]);
   // }

   if(fs::create_directory("results")){
    //std::cout << "Created a directory\n";
   } else {
    //std::cerr << "Failed to create a directory\n";
   }


   if (zipfian){
      fs::create_directory("results/cache_zipfian");
   }

   if(fs::create_directory("results/cache")){
    //std::cout << "Created a directory\n";
   } else {
    //std::cerr << "Failed to create a directory\n";
   }

   execute_test(table, n_queries, host_items, zipfian, alpha);

   //uint64_t * access_data = generate_data<uint64_t>(n_ops);


   //can't give up this space.
   // init_global_allocator(16ULL*1024*1024*1024, 111);

   
   
   // cache_test<warpSpeed::tables::chaining_generic, 4, 8>(host_items, n_ops, access_data);

   // free_global_allocator();

   // cache_test<warpSpeed::tables::md_double_generic, 4, 32>(host_items, n_ops, access_data);
  


   // // cache_test<warpSpeed::tables::p2_int_generic, 8, 32>(host_items, n_ops, access_data);
   // //cache_test<warpSpeed::tables::double_generic, 8, 8>(host_items, n_ops, access_data);
   // // cache_test<warpSpeed::tables::iht_p2_generic, 8, 32>(host_items, n_ops, access_data);
   // cache_test<warpSpeed::tables::p2_ext_generic, 8, 32>(host_items, n_ops, access_data);
   // cache_test<warpSpeed::tables::md_p2_generic, 4, 32>(host_items, n_ops, access_data);

   // cache_test<warpSpeed::tables::iht_metadata_generic, 4, 32>(host_items, n_ops, access_data);
   



   //free_global_allocator();


   cudaDeviceReset();
   return 0;

}
