#ifndef PROGRESS
#define PROGRESS

#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cmath>

#include "assert.h"
#include "stdio.h"

void display_progress(uint64_t current_item, uint64_t max_n_items, double update_interval){

    const int barwidth = 50;

    static uint64_t next_interval;
    static uint64_t interval_amount;

    if (current_item == 0){
      next_interval = 0;
      interval_amount = max_n_items*update_interval;
    }

    if (current_item == next_interval){

      next_interval+=interval_amount;

      std::cout << "[";
      int progress = barwidth*current_item/max_n_items;

      for (int j = 0; j < barwidth; j++){
            if (j < progress) std::cout << "=";
            else if (j == progress) std::cout << ">";
            else std::cout << " ";

      }

      double percent = 100.0*current_item/max_n_items;
      std::cout << "] " << current_item << "/" << max_n_items << " " << percent <<"%\r";
      std::cout.flush();

   }


}


void end_bar(uint64_t items_to_generate){

  const int barwidth = 50;
  std::cout << "[";

  for (int j = 0; j < barwidth; j++){
       std::cout << "=";

  }

  double percent = 100;
  std::cout << "] " << items_to_generate << "/" << items_to_generate << " " << percent <<"%\n";
  std::cout.flush();

}

#endif  // GPU_BLOCK_