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

__host__ void execute_test(std::string table, std::string input_file, uint64_t n_indices_output){

   if (table == "p2"){

      //nips 2
      double first = tensor_contraction_nips_2<hashing_project::tables::p2_ext_generic, 8, 32>(input_file, n_indices_output);

      //nips 013
      double second = tensor_contraction_nips_013<hashing_project::tables::p2_ext_generic, 8, 32>(input_file, n_indices_output);

      printf("%s %f %f\n", table.c_str(), first, second);
      //p2 p2MD double doubleMD iceberg icebergMD cuckoo chaining bght_p2 bght_cuckoo");

   } else if (table == "p2MD"){

      double first = tensor_contraction_nips_2<hashing_project::tables::md_p2_generic, 4, 32>(input_file, n_indices_output);

      double second = tensor_contraction_nips_013<hashing_project::tables::md_p2_generic, 4, 32>(input_file, n_indices_output);

      printf("%s %f %f\n", table.c_str(), first, second);

   } else if (table == "double"){

      double first = tensor_contraction_nips_2<hashing_project::tables::double_generic, 4, 8>(input_file, n_indices_output);

      double second = tensor_contraction_nips_013<hashing_project::tables::double_generic, 4, 8>(input_file, n_indices_output);

      printf("%s %f %f\n", table.c_str(), first, second);
   } else if (table == "doubleMD"){

      double first = tensor_contraction_nips_2<hashing_project::tables::md_double_generic, 4, 32>(input_file, n_indices_output);

      double second = tensor_contraction_nips_013<hashing_project::tables::md_double_generic, 4, 32>(input_file, n_indices_output);

      printf("%s %f %f\n", table.c_str(), first, second);

   } else if (table == "iceberg"){

      double first = tensor_contraction_nips_2<hashing_project::tables::iht_p2_generic, 8, 32>(input_file, n_indices_output);
      
      double second = tensor_contraction_nips_013<hashing_project::tables::iht_p2_generic, 8, 32>(input_file, n_indices_output);
  
      printf("%s %f %f\n", table.c_str(), first, second);
   } else if (table == "icebergMD"){

      double first = tensor_contraction_nips_2<hashing_project::tables::iht_p2_metadata_full_generic, 4, 32>(input_file, n_indices_output);

      double second = tensor_contraction_nips_013<hashing_project::tables::iht_p2_metadata_full_generic, 4, 32>(input_file, n_indices_output);
 
      printf("%s %f %f\n", table.c_str(), first, second);
   } else if (table == "chaining"){

      double first = tensor_contraction_nips_2<hashing_project::tables::chaining_generic, 4, 8>(input_file, n_indices_output, true);

      double second = tensor_contraction_nips_013<hashing_project::tables::chaining_generic, 4, 8>(input_file, n_indices_output);

      printf("%s %f %f\n", table.c_str(), first, second);
   } else {
      throw std::runtime_error("Unknown table");
   }


}


int main(int argc, char** argv) {

   argparse::ArgumentParser program("sparse_tensor_test");

   // program.add_argument("square")
   // .help("display the square of a given integer")
   // .scan<'i', int>();

   program.add_argument("--table", "-t")
   .required()
   .help("Specify table type. Options [p2 p2MD double doubleMD iceberg icebergMD cuckoo chaining bght_p2 bght_cuckoo");

   program.add_argument("--tensor", "-r")
   .required()
   .help("Tensor for contraction. Tests both 1 mode and 3 mode contraction on the tensor. .mtx format\n");


   program.add_argument("--capacity", "-c").required().scan<'u', uint64_t>().help("Capacity of the output tensor");

   try {
    program.parse_args(argc, argv);
   }
   catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return 1;
   }

   auto table = program.get<std::string>("--table");
   auto input_file = program.get<std::string>("--tensor");
   uint64_t n_indices_output = program.get<uint64_t>("--capacity");

   execute_test(table, input_file, n_indices_output);
   //std::string input_file = "../dataset/nips.tns";

   //uint64_t n_indices_output = 40000000ULL;
   //can't give up this space.
   // init_global_allocator(16ULL*1024*1024*1024, 111);

  
   cudaDeviceReset();
   return 0;

}
