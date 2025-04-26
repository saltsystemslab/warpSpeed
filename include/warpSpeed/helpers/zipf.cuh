#ifndef ZIPF
#define ZIPF

#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cmath>
#include <random>

#include <filesystem>
#include <iostream>
#include <fstream>

#include <warpSpeed/helpers/progress.cuh>

#include "assert.h"
#include "stdio.h"


namespace zipfian {

//=-------------------------------------------------------------------------=
//=  Author: Kenneth J. Christensen                                         =
//=          University of South Florida                                    =
//=          WWW: http://www.csee.usf.edu/~christen                         =
//=          Email: christen@csee.usf.edu                                   =
//=-------------------------------------------------------------------------=
//=  History: KJC (11/16/03) - Genesis (from genexp.c)                      =


//=========================================================================
//= Multiplicative LCG for generating uniform(0.0, 1.0) random numbers    =
//=   - x_n = 7^5*x_(n-1)mod(2^31 - 1)                                    =
//=   - With x seeded to 1 the 10000th x value should be 1043618065       =
//=   - From R. Jain, "The Art of Computer Systems Performance Analysis," =
//=     John Wiley & Sons, 1991. (Page 443, Figure 26.2)                  =
//=========================================================================
// double rand_val(int seed)
// {
//   const long  a =      16807;  // Multiplier
//   const long  m = 2147483647;  // Modulus
//   const long  q =     127773;  // m div a
//   const long  r =       2836;  // m mod a
//   static long x;               // Random int value
//   long        x_div_q;         // x divided by q
//   long        x_mod_q;         // x modulo q
//   long        x_new;           // New x value

//   // Set the seed if argument is non-zero and then return zero
//   if (seed > 0)
//   {
//     x = seed;
//     return(0.0);
//   }

//   // RNG using integer arithmetic
//   x_div_q = x / q;
//   x_mod_q = x % q;
//   x_new = (a * x_mod_q) - (r * x_div_q);
//   if (x_new > 0)
//     x = x_new;
//   else
//     x = x_new + m;

//   // Return a random value between 0.0 and 1.0
//   return((double) x / m);
// }

double rand_val(int seed)
{


  const double lower_bound = 0;
  const double upper_bound = 1;
  static std::uniform_real_distribution<double> dist(lower_bound,upper_bound);
  static std::mt19937 re;

  if (seed != 0){
    re.seed(seed);
  }
  
  return dist(re);



}


//improved zipfian from https://stackoverflow.com/questions/9983239/how-to-generate-zipf-distributed-numbers-efficiently
// credit to Masoud Kazemi
uint64_t zipf(double alpha, uint64_t n, bool reset=false)
{
  static int first = true;      // Static first time flag
  static uint64_t old_n;
  static double c = 0;          // Normalization constant
  static double *sum_probs;     // Pre-calculated sum of probabilities
  double z;                     // Uniform random number (0 < z < 1)
  uint64_t zipf_value = 0;               // Computed exponential value to be returned
  uint64_t   i;                     // Loop counter
  uint64_t low, high, mid;           // Binary-search bounds

  if (reset && old_n != n){
    printf("Starting new zipfian run\n");
    first = true;
  }

  // Compute normalization constant on first call only
  if (first == true)
  {
    old_n = n;
    printf("Calculating C...\n");
    for (i=1; i<=n; i++){
      c = c + (1.0 / pow((double) i, alpha));
      display_progress(i-1, n, .001);
    }

    c = 1.0 / c;
    end_bar(n);

    sum_probs = (double *) malloc((n+1)*sizeof(*sum_probs));
    sum_probs[0] = 0;
    printf("Assigning probabilities...\n");
    for (i=1; i<=n; i++) {
      sum_probs[i] = sum_probs[i-1] + c / pow((double) i, alpha);
      display_progress(i-1, n, .001);
    }
    first = false;
    end_bar(n);
  }

  // Pull a uniform random number (0 < z < 1)
  do
  {
    z = rand_val(0);
  }
  while ((z == 0) || (z == 1));



  // Map z to the value
  low = 1, high = n, mid;
  do {

    mid = floor((low+high)/2);
    if (sum_probs[mid] >= z && sum_probs[mid-1] < z) {
      zipf_value = mid;
      break;
    } else if (sum_probs[mid] >= z) {
      high = mid-1;
    } else {
      low = mid+1;
    }
  } while (low <= high);

  // Assert that zipf_value is between 1 and N
  assert((zipf_value >=1) && (zipf_value <= n));

  return(zipf_value);
}



} // namespace zipfian


__host__ uint64_t * generate_zipfian_values(uint64_t items_to_generate, uint64_t max_range, double alpha){

   uint64_t * data = gallatin::utils::get_host_version<uint64_t>(items_to_generate);

   std::string output_dir("../zipfian_data");


   std::string output_fname = output_dir + "/" + std::to_string(max_range) + "_" + std::to_string(items_to_generate) + "_" + std::to_string(alpha) + ".txt";

   printf("Generating %lu zipfian items with universe %lu\n", items_to_generate, max_range);


   if (std::filesystem::exists(output_fname)){
      std::cout << "Filename " << output_fname << " exists. Loading..." << std::endl;


      std::ifstream file(output_fname);

      uint64_t i = 0;

      uint64_t num;

      if (!file.is_open()) {
         std::cerr << "Error: Could not open file." << std::endl;
         return nullptr;
      }

      while (file >> num) {
         data[i] = num;
         i++;

         display_progress(i, items_to_generate, .001);
      }

      end_bar(items_to_generate);

      file.close();

      printf("\nDone, loaded %lu\n numbers\n", i);

      return data;

   } else {

      printf("No file found, generating...\n");

      int seed = (int) std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());

      zipfian::rand_val(seed);

      uint64_t zipfian_input = max_range;

      double barwidth=50;


      printf("Starting zipfian gen with alpha %f\n", alpha);

      for (uint64_t i = 0; i < items_to_generate; i++){
         data[i] = zipfian::zipf(alpha, zipfian_input, i==0);

         display_progress(i, items_to_generate, .01);


   }

   end_bar(items_to_generate);
   printf("\nDone\n");

   //write to file

   std::cout << "Opening file: " << output_fname << std::endl;

   std::filesystem::create_directory(output_dir);

   std::ofstream outfile(output_fname);

   if (!outfile.is_open()) {
     std::cerr << "Error opening file!" << std::endl;
     return nullptr;
   }

   for (uint64_t i = 0; i < items_to_generate; i++){
      outfile << data[i] << " ";
   }

   outfile.close();

   return data;

   }


}


#endif  // GPU_BLOCK_