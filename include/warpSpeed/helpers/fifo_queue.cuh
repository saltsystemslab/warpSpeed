#ifndef FIFO_RING_QUEUE
#define FIFO_RING_QUEUE


#include <cuda.h>
#include <cuda_runtime_api.h>

//alloc utils needed for easy host_device transfer
#include <gallatin/allocators/global_allocator.cuh>
#include <gallatin/allocators/alloc_utils.cuh>
#include <gallatin/data_structs/ds_utils.cuh>

namespace warpSpeed {

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


    //int active_count;

    uint64_t enqueue_counter;
    uint64_t dequeue_counter;

    //instantiate a queue on device.
    //currently does not pull from the allocator, but it totally should
    static __host__ my_type * generate_on_device(uint64_t ext_num_slots){

      my_type * host_version = gallatin::utils::get_host_version<my_type>();

      host_version->num_slots = ext_num_slots;


      T * ext_buffer;

      cudaMalloc((void **)&ext_buffer, sizeof(T)*ext_num_slots);

      init_ring_kernel<T><<<(ext_num_slots-1)/256+1,256>>>(ext_buffer, default_value, ext_num_slots);

      host_version->buffer = ext_buffer;
      host_version->enqueue_counter = 0;
      host_version->dequeue_counter = 0;

      cudaDeviceSynchronize();

      return gallatin::utils::move_to_device<my_type>(host_version);


    }


    static __host__ void free_on_device(my_type * dev_queue){

      my_type * host_version = gallatin::utils::move_to_host(dev_queue);

      cudaFree(host_version->buffer);
      cudaFreeHost(host_version);

    }

    __device__ bool enqueue(T new_item){




      
        // int slot_active_count = atomicAdd(&active_count, 1);

        // //this should be ok. double check this later.
        // //if (slot_active_count < 0)

        // //cycle to prevent overcount.
        // if (slot_active_count < 0){
        //   atomicSub(&active_count, 1);
        //   continue;
        // }


        // //can't enqueue, full.
        // if (slot_active_count >= num_slots){

        //   atomicSub(&active_count, 1);

        //   return false;
        // }

        //is slot_active_count + live_dequeue same as atomic?

        uint64_t enqueue_slot = atomicAdd((unsigned long long int *)&enqueue_counter,1ULL) % num_slots;

        //needs to loop.
        while(typed_atomic_CAS(&buffer[enqueue_slot], default_value, new_item) != default_value){
          printf("Failed to swap index %lu in for thread %lu\n", enqueue_slot, gallatin::utils::get_tid());
        }
        //  //throw error
        //  asm volatile("trap;"); 
        //  return false;
        // }

        return true;


    }

    //valid to make optional type?

    __device__ bool dequeue(T & return_val){

      //do these reads need to be atomic? 
      //I don't think so as these values don't change.
      //as queue doesn't change ABA not possible.


        //slot is valid!

        uint64_t dequeue_slot = atomicAdd((unsigned long long int *) &dequeue_counter, 1ULL) % num_slots;
        

        return_val = typed_atomic_exchange(&buffer[dequeue_slot], default_value);

        while (return_val == default_value){
          return_val = typed_atomic_exchange(&buffer[dequeue_slot], default_value);

          printf("Looping in dequeue\n");
        }

        return true;

    }


  };



}


}


#endif //end of queue name guard