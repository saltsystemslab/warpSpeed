#ifndef FIFO_RING_QUEUE
#define FIFO_RING_QUEUE


#include <cuda.h>
#include <cuda_runtime_api.h>

//alloc utils needed for easy host_device transfer
#include <gallatin/allocators/global_allocator.cuh>
#include <gallatin/allocators/alloc_utils.cuh>
#include <gallatin/data_structs/ds_utils.cuh>

namespace hashing_project {

namespace helpers {


  //basic form of queue using allocator
  //on instantiation on host or device, must be plugged into allocator.
  //This allows the queue to process memory dynamically.



  //Pipeline

  //insert op
  

  template <typename T>
  __global__ void init_ring_kernel(T * buffer, T default_value, uint64_t num_slots){

    uint64_t tid = gallatin::utils::get_tid();

    if (tid >= num_slots) return;

    buffer[tid] = default_value;

  }


  //For this project the ring buffer is going to be protected by the external system
  // so enqueue dequeue will always succeed and will not need extra protection.
  template <typename T, T default_value>
  struct fifo_queue {

    using my_type = fifo_queue<T, default_value>;


    uint64_t num_slots;
    T * buffer;

    uint64_t * locks;


    //int active_count;

    uint64_t enqueue_counter;

    //instantiate a queue on device.
    //currently does not pull from the allocator, but it totally should
    static __host__ my_type * generate_on_device(uint64_t ext_num_slots){

      my_type * host_version = gallatin::utils::get_host_version<my_type>();

      host_version->num_slots = ext_num_slots;


      T * ext_buffer;

      cudaMalloc((void **)&ext_buffer, sizeof(T)*ext_num_slots);

      uint64_t n_locks = (ext_num_slots-1)/64+1;

      uint64_t * ext_locks = gallatin::utils::get_device_version<uint64_t>(n_locks);

      cudaMemset(ext_locks, 0ULL, sizeof(uint64_t)*(n_locks));

      host_version->locks = ext_locks;

      init_ring_kernel<T><<<(ext_num_slots-1)/256+1,256>>>(ext_buffer, default_value, ext_num_slots);

      host_version->buffer = ext_buffer;
      host_version->enqueue_counter = 0;

      cudaDeviceSynchronize();

      return gallatin::utils::move_to_device<my_type>(host_version);


    }


    static __host__ void free_on_device(my_type * dev_queue){

      my_type * host_version = gallatin::utils::move_to_host(dev_queue);

      cudaFree(host_version->locks);
      cudaFree(host_version->buffer);
      cudaFreeHost(host_version);


    }

    __device__ bool is_empty_marker(T item){
      return item == default_value;
    }

    __device__ void enqueue_replace(T new_item, T & old_item){

      uint64_t enqueue_slot = atomicAdd((unsigned long long int *)&enqueue_counter, 1ULL) % num_slots;

      //stall_lock(enqueue_slot);

      old_item = typed_atomic_exchange(&buffer[enqueue_slot], new_item);

      return;

    }


    __device__ uint64_t stall_lock_enqueue_replace(T new_item, T & old_item){

      uint64_t enqueue_slot = atomicAdd((unsigned long long int *)&enqueue_counter, 1ULL) % num_slots;

      stall_lock(enqueue_slot);

      old_item = typed_atomic_exchange(&buffer[enqueue_slot], new_item);

      return enqueue_slot;


    }


    __device__ void stall_lock(uint64_t loc){


        uint64_t high = loc/64;
        uint64_t low = loc % 64;

        //if old is 0, SET_BIT_MASK & 0 is 0 - loop exit.

        while (atomicOr((unsigned long long int *)&locks[high], (unsigned long long int) SET_BIT_MASK(low)) & SET_BIT_MASK(low)){

          printf("Stalling on %llu\n", loc);

        }

    }

    __device__ void unlock(uint64_t loc){

        uint64_t high = loc/64;
        uint64_t low = loc % 64;

        //if old is 0, SET_BIT_MASK & 0 is 0 - loop exit.
        atomicAnd((unsigned long long int *)&locks[high], (unsigned long long int) ~SET_BIT_MASK(low));

    }


  };



}


}


#endif //end of queue name guard