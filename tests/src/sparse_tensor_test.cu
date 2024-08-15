/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */





#include <gallatin/allocators/global_allocator.cuh>

#include <gallatin/allocators/timer.cuh>

#include <warpcore/single_value_hash_table.cuh>
#include <gallatin/allocators/timer.cuh>

#include <bght/p2bht.hpp>

#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <chrono>
#include <openssl/rand.h>

#include <fstream>
#include <locale>
#include <filesystem>

namespace fs = std::filesystem;

// #include <hashing_project/table_wrappers/p2_wrapper.cuh>
// #include <hashing_project/table_wrappers/dummy_ht.cuh>
// #include <hashing_project/table_wrappers/iht_wrapper.cuh>

#include <hashing_project/tables/p2_hashing_metadata.cuh>
#include <hashing_project/tables/p2_hashing_internal.cuh>
#include <hashing_project/tables/chaining.cuh>
#include <hashing_project/tables/double_hashing.cuh>
#include <hashing_project/tables/iht_p2.cuh>
#include <hashing_project/tables/p2_hashing_external.cuh>
#include <hashing_project/tables/iht_p2_metadata.cuh>
#include <hashing_project/tables/iht_p2_metadata_full.cuh>
#include <hashing_project/tables/cuckoo.cuh>
#include <hashing_project/tables/double_hashing_metadata.cuh>
#include <cooperative_groups.h>

#include <hashing_project/helpers/tensor_contraction.cuh>

namespace cg = cooperative_groups;

using namespace gallatin::allocators;


#if GALLATIN_DEBUG_PRINTS
   #define TEST_BLOCK_SIZE 256
#else
   #define TEST_BLOCK_SIZE 256
#endif



int main(int argc, char** argv) {


   std::string input_file = "../dataset/nips.tns";

   uint64_t n_indices_output = 40000000ULL;
   //can't give up this space.
   // init_global_allocator(16ULL*1024*1024*1024, 111);

   printf("1 mode\n");

   tensor_contraction<4,1,hashing_project::tables::md_double_generic, 4, 32>(input_file, n_indices_output);

   tensor_contraction<4,1,hashing_project::tables::iht_p2_generic, 8, 32>(input_file, n_indices_output);
  

   tensor_contraction<4,1,hashing_project::tables::md_p2_generic, 4, 32>(input_file, n_indices_output);

   tensor_contraction<4,1,hashing_project::tables::double_generic, 4, 8>(input_file, n_indices_output);

      
   tensor_contraction<4,1,hashing_project::tables::p2_ext_generic, 8, 32>(input_file, n_indices_output);

   

   
   tensor_contraction<4,1,hashing_project::tables::iht_p2_metadata_full_generic, 4, 32>(input_file, n_indices_output);
   


   tensor_contraction<4,1,hashing_project::tables::chaining_generic, 4, 8>(input_file, n_indices_output, true);


   //3 mode

   printf("3 mode\n");
   tensor_contraction<4,3,hashing_project::tables::md_double_generic, 4, 32>(input_file, n_indices_output);

   tensor_contraction<4,3,hashing_project::tables::iht_p2_generic, 8, 32>(input_file, n_indices_output);
  

   tensor_contraction<4,3,hashing_project::tables::md_p2_generic, 4, 32>(input_file, n_indices_output);

   tensor_contraction<4,3,hashing_project::tables::double_generic, 4, 8>(input_file, n_indices_output);

      
   tensor_contraction<4,3,hashing_project::tables::p2_ext_generic, 8, 32>(input_file, n_indices_output);

   

   
   tensor_contraction<4,3,hashing_project::tables::iht_p2_metadata_full_generic, 4, 32>(input_file, n_indices_output);
   


   tensor_contraction<4,3,hashing_project::tables::chaining_generic, 4, 8>(input_file, n_indices_output);


   cudaDeviceReset();
   return 0;

}
