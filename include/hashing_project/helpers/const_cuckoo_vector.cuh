#ifndef SINGLE_VECTOR
#define SINGLE_VECTOR


#include <cuda.h>
#include <cuda_runtime_api.h>

//alloc utils needed for easy host_device transfer
#include <gallatin/allocators/global_allocator.cuh>

#include <vector>


namespace hashing_project {

namespace data_structs {


  template <typename Key>
  struct vector_pair {

    uint64_t hash;
    Key key;
    int bucket_id;


  };


  //"single-threaded" vector implementation
  //this is a simple tool to handle map-reduce parallelism
  //in CUDA using Gallatin. - reads, writes, and resizes are handled
  //lazily using one thread - no guarantee of correctness among CG.
  template <typename T, int n_items>
  struct cuckoo_vector {

    using my_type = cuckoo_vector<T, n_items>;

    uint64_t size;
    T data[n_items];

    //temporary pointer for managing incoming pointers


    __device__ void reset(T default_value){

      size = 0;

      // for (int i =0; i < n_items; i++){
      //   data[i] = default_value;
      // }

    }

    __device__ bool push_back(T new_item){

      uint64_t my_slot = size;

      size = size+1;

      if (size > n_items) return false;

      data[my_slot] = new_item;

      return true;


    }


    __device__ T& operator[](uint64_t index)
    {

       return data[index];
       
    }


    //static __host__ my_type * move_to_device

  };


}


}


#endif //end of queue name guard