#ifndef CONTRACTION
#define CONTRACTION


#include <fstream> 
#include <iostream> 
#include <string> 
#include <sstream>

#include <hashing_project/helpers/cuckoo_vector.cuh>
#include <hashing_project/tables/p2_hashing_metadata.cuh>

using namespace std;

//Dataset:
//Chicago taken from http://frostt.io/tensors/chicago-crime/, modified by SPARTA team.

//data storage


//there is one contraction dimension - this is a multimensional flattening.

//Hash table converts from flattening -> vector.

//for each slice in X, look up resulting contraction.


//COO object representation for reading in file.
//this is hard coded for 4d tensors and represents one data point.

//hash table upsert funcs
__device__ void per_value_accumulate(hashing_project::tables::ht_pair<uint64_t, uint64_t> * location, uint64_t key, uint64_t val){

	atomicAdd((unsigned long long int *)&location->val, (unsigned long long int) val);

}

__device__ void per_value_accumulate_external(hashing_project::tables::ht_pair<uint64_t, uint64_t> * location, uint64_t key, uint64_t val){


	uint64_t read_value = location->val;

	//domain specific - writes of 0 are never processed
	// if this triggers it means the read is potentially stale
	// and we should wait for a true value to be emplaced.
	while (read_value == 0){

		__threadfence();
		read_value = gallatin::utils::ld_acq(&location->val);
		//printf("Looping in read\n");

	}

	atomicAdd((unsigned long long int *)&location->val, (unsigned long long int) val);

}

//for cuckoo upsert - illegal to modify existing.
__device__ void upsert_do_nothing(hashing_project::tables::ht_pair<uint64_t, uint64_t> * location, uint64_t key, uint64_t val){

	return;

}

template <int n_dims>
struct COO
{
	uint dims[n_dims];
	uint64_t value;

	//parse
	__host__ COO(std::string input_string){

		stringstream ss(input_string);

		for (int i = 0; i < n_dims; i++){
			ss >> dims[i];
		}
		ss >> value;


		for (int i = 0; i < n_dims; i++){
			dims[i]--;
		}

	}

	//grab the lower dims.
	template <int alt_dims>
	__device__ void set_values(COO<alt_dims> other_COO){

		gallatin::utils::st_rel(&value, other_COO.value);

		for (int i = 0; i < n_dims; i++){

			int access_index = alt_dims-n_dims+i;

			dims[i] = other_COO.dims[access_index];

		}

	}

	__host__ __device__ COO(){
		return;
	}

};



template <typename COO_type>
__global__ void swap_dimension_kernel(COO_type * items, uint64_t n_items, int dim1, int dim2){

	uint64_t tid = gallatin::utils::get_tid();

	if (tid >= n_items) return;


	auto my_item = &items[tid];

	uint temp = my_item->dims[dim1];
	
	my_item->dims[dim1] = my_item->dims[dim2];

	my_item->dims[dim2] = temp;


}


template <int n_dims>
struct coo_matrix {

	COO<n_dims> * items;
	uint64_t n_items;

	uint maxDims [n_dims];

	coo_matrix(std::string filename){

		for (int i = 0; i < n_dims; i++){

			maxDims[i] = 0;

		}

		std::vector<COO<n_dims>> items_vec;

		ifstream file(filename);

		string line;

		if (file.is_open()){

			while (getline(file, line)){

				//parse

				COO<n_dims> string_as_COO(line);

				items_vec.push_back(string_as_COO); 

				//calculate limits

				for (int i = 0; i < n_dims; i++){

					if (string_as_COO.dims[i]+1 > maxDims[i]){
						maxDims[i] = string_as_COO.dims[i]+1;
					}

				}


			}

		}

		//all lines parsed, max dims set.

		n_items = items_vec.size();


		COO<n_dims> * cuda_items;

		cudaMalloc((void **)&cuda_items, n_items*sizeof(COO<n_dims>));

		items = cuda_items;

		cudaMemcpy(items, items_vec.data(), n_items*sizeof(COO<n_dims>), cudaMemcpyHostToDevice);

		cudaDeviceSynchronize();

		//vec auto destruct.
		return;




	}

	coo_matrix(const coo_matrix<n_dims> & other_matrix){

		for (int i = 0; i < n_dims; i++){

			maxDims[i] = other_matrix.maxDims[i];

		}

		n_items = other_matrix.n_items;

		COO<n_dims> * cuda_items;

		cudaMalloc((void **)&cuda_items, n_items*sizeof(COO<n_dims>));

		items = cuda_items;

		cudaMemcpy(items, other_matrix.items, n_items*sizeof(COO<n_dims>), cudaMemcpyHostToDevice);

		//vec auto destruct.
		return;




	}

	coo_matrix(){
		for (int i = 0; i < n_dims; i++){
			maxDims[i] = 0;
		}

		n_items = 0;
		items = nullptr;
	}

	//calculate merge_dimensions
	//this builds out the matrix max dimensions.
	template<int old_dimensions>
	void set_dimensions(const coo_matrix<old_dimensions> & x_mat, const coo_matrix<old_dimensions> & y_mat, int dimensions_to_merge){

		int dimensions_extracted = old_dimensions-dimensions_to_merge;

		for (int i =0; i < dimensions_extracted; i++){

			maxDims[i] = x_mat.maxDims[i];

		}

		for (int i = 0; i < dimensions_extracted; i++){
			maxDims[i+dimensions_extracted] = y_mat.maxDims[dimensions_to_merge+i]; 
		}


	}

	~coo_matrix(){

		// if (items != nullptr){
		// 	printf("Items is %lx\n", items);
		// 	cudaFree(items);
		// }
		
	}

	//swap dimensions in the matrix.
	__host__ void swap_dims(int dim1, int dim2){

		swap_dimension_kernel<COO<n_dims>><<<(n_items-1)/512+1, 512>>>(items, n_items, dim1, dim2);

		uint temp = maxDims[dim1];

		maxDims[dim1] = maxDims[dim2];
		maxDims[dim2] = temp;

	}

	__device__ uint64_t get_ht_key(uint64_t tid, int starting_dim, int dims_to_add){

		uint64_t rolling_hash = 1;

		//dummy example
		//10x10 matrix, processing point 1,1
		//should get 11.

		for (int i = starting_dim+dims_to_add-1; i >= starting_dim; i--){

			rolling_hash = rolling_hash*maxDims[i];
			rolling_hash += items[tid].dims[i];

		}

		return rolling_hash;

	}

	__host__ void print_dims(){

		for (int i = 0; i < n_dims; i++){
			printf("%u ", maxDims[i]);
		}

		printf("\n");
		

	}

	__host__ uint64_t get_max_dimensionality(){

		uint64_t output_dim = 1;

		for (int i = 0; i < n_dims; i++){
			output_dim*=maxDims[i];
		}

		return output_dim;

	}

	template<int lhs_dim, int rhs_dim>
	__device__ uint64_t get_output_key(COO<lhs_dim> & lhs, int contraction_dims, COO<rhs_dim> & rhs){

		uint64_t rolling_hash = 1;

		for (int i =0; i < lhs_dim-contraction_dims; i++){

			rolling_hash = rolling_hash*maxDims[i];
			rolling_hash += lhs.dims[i];

		}

		for (int i = 0; i < rhs_dim; i++){
			rolling_hash = rolling_hash*maxDims[i+lhs_dim-contraction_dims];
			rolling_hash += rhs.dims[i];
		}


		//calculate rhs
		//maxdims output acceleartead by 

		return rolling_hash;
	}

};


template <typename ht_type, typename lower_COO, typename vector_type, int n_dims, int contraction_dims, uint tile_size>
__global__ void convert_to_ht(ht_type * indirection_table, coo_matrix<n_dims> * conversion_matrix, uint64_t n_items){

   auto thread_block = cg::this_thread_block();

   cg::thread_block_tile<tile_size> my_tile = cg::tiled_partition<tile_size>(thread_block);


   uint64_t tid = gallatin::utils::get_tile_tid(my_tile);

   if (tid >= n_items) return;

   uint64_t hash_key = conversion_matrix->get_ht_key(tid, 0, contraction_dims);


   uint64_t key_lock_id = indirection_table->get_lock_bucket(my_tile, hash_key);

   indirection_table->stall_lock(my_tile, key_lock_id);

   uint64_t address;

   if (indirection_table->find_with_reference_no_lock(my_tile, hash_key, address)){


   	if (my_tile.thread_rank() == 0){

   		vector_type * allocation = (vector_type *) address;

	   	lower_COO lower_values;

	   	lower_values.set_values(conversion_matrix->items[tid]);

	   	allocation->push_back(lower_values);

   	}


   	my_tile.sync();

   	indirection_table->unlock(my_tile, key_lock_id);

   } else {

   	//get new_array
   	uint64_t vector_as_alloc;

   	if (my_tile.thread_rank() == 0){

   	vector_type * allocation = (vector_type *) gallatin::allocators::global_malloc(sizeof(vector_type));

   	if (allocation == nullptr){
   		printf("Bad alloc\n");
   	}

   	allocation->init(2);

   	lower_COO lower_values;

   	lower_values.set_values(conversion_matrix->items[tid]);

   	allocation->push_back(lower_values);

   	__threadfence();

   	vector_as_alloc = (uint64_t) allocation;

   	}

   	vector_as_alloc = my_tile.shfl(vector_as_alloc, 0);

   	indirection_table->upsert_no_lock(my_tile, hash_key, vector_as_alloc);


   	indirection_table->unlock(my_tile, key_lock_id);

   }

   // if (my_tile.thread_rank() == 0){
   // 	printf("Done with %llu\n", tid);
   // }

}


//variant for tables that cannot fuse ops.
// that's just cuckoo for now.
template <typename ht_type, typename lower_COO, typename vector_type, int n_dims, int contraction_dims, uint tile_size>
__global__ void convert_to_ht_individual(ht_type * indirection_table, coo_matrix<n_dims> * conversion_matrix, uint64_t n_items){

   auto thread_block = cg::this_thread_block();

   cg::thread_block_tile<tile_size> my_tile = cg::tiled_partition<tile_size>(thread_block);


   uint64_t tid = gallatin::utils::get_tile_tid(my_tile);

   if (tid >= n_items) return;

   uint64_t hash_key = conversion_matrix->get_ht_key(tid, 0, contraction_dims);


   uint64_t key_lock_id = indirection_table->get_lock_bucket(my_tile, hash_key);

   //indirection_table->stall_lock(my_tile, key_lock_id);

   uint64_t address;

   if (indirection_table->find_with_reference(my_tile, hash_key, address)){


   	if (my_tile.thread_rank() == 0){

   		vector_type * allocation = (vector_type *) address;

	   	lower_COO lower_values;

	   	lower_values.set_values(conversion_matrix->items[tid]);

	   	allocation->push_back(lower_values);

   	}


   	my_tile.sync();

   	indirection_table->unlock(my_tile, key_lock_id);

   } else {

   	//get new_array
   	uint64_t vector_as_alloc;

   	if (my_tile.thread_rank() == 0){

   	vector_type * allocation = (vector_type *) gallatin::allocators::global_malloc(sizeof(vector_type));

   	if (allocation == nullptr){
   		printf("Bad alloc\n");
   	}

   	allocation->init(2);

   	__threadfence();

   	vector_as_alloc = (uint64_t) allocation;

   	}

   	vector_as_alloc = my_tile.shfl(vector_as_alloc, 0);

   	indirection_table->upsert_function(my_tile, hash_key, vector_as_alloc, &upsert_do_nothing);

   	__threadfence();


   	uint64_t final_vector;
   	
   	bool found = indirection_table->find_with_reference(my_tile, hash_key, final_vector);

   	//add key / delete double alloc.
   	if (my_tile.thread_rank() == 0){


   		if (!found){
   			printf("inserted vector not found!\n");
   		}

   		if (final_vector != vector_as_alloc){
   			//free old vector.

   			vector_type * allocation = (vector_type *) vector_as_alloc;

   			allocation->free_vector();

   			gallatin::allocators::global_free(allocation);

   			__threadfence();

   		}

   		vector_type * stored_vector = (vector_type *) final_vector;

   		lower_COO lower_values;

	   	lower_values.set_values(conversion_matrix->items[tid]);

	   	stored_vector->push_back(lower_values);



   	}

   }

   // if (my_tile.thread_rank() == 0){
   // 	printf("Done with %llu\n", tid);
   // }

}





template <typename ht_type, typename vector_type, int n_dims, int contraction_dims, int output_dims, uint tile_size>
__global__ void contract(coo_matrix<n_dims> * x_mat, ht_type * y_mat, ht_type * accumulator, coo_matrix<output_dims> * output){

	auto thread_block = cg::this_thread_block();

   cg::thread_block_tile<tile_size> my_tile = cg::tiled_partition<tile_size>(thread_block);


   uint64_t tid = gallatin::utils::get_tile_tid(my_tile);

   if (tid >= x_mat->n_items) return;

   uint64_t hash_key = x_mat->get_ht_key(tid, n_dims-contraction_dims, contraction_dims);

   uint64_t allocation;

   if (!y_mat->find_with_reference_no_lock(my_tile, hash_key, allocation)) return;


   vector_type * vector = (vector_type *) allocation;

   auto lhs = x_mat->items[tid];


   for (int i = 0; i < vector->size; i++){

   	auto rhs = vector->data[i];


   	uint64_t merged_key = output->get_output_key(lhs, contraction_dims, rhs);

   	uint64_t value = rhs.value*lhs.value;

   	// uint64_t lock = accumulator->get_lock_bucket(my_tile, merged_key);

   	//disables accumulation.

   	//accumulator->stall_lock(my_tile, lock);


   	// if (accumulator->find_with_reference_no_lock(my_tile, merged_key, value)){
   	// 	value += rhs.value*lhs.value;
   	// } else {
   	// 	value = rhs.value*lhs.value;
   	// }

   	//value = rhs.value+lhs.value;

   	//accumulator->upsert_no_lock(my_tile, merged_key, value);

   	//accumulator->unlock(my_tile, lock);

   	auto location = accumulator->find_pair_no_lock(my_tile, merged_key);

   	if (location != nullptr){

   		if (my_tile.thread_rank() == 0){
   			per_value_accumulate_external(location, merged_key, value);
   		}


   		
   	} else {
   		accumulator->upsert_function(my_tile, merged_key, value, &per_value_accumulate);
   	}


   	//accumulator->upsert_function(my_tile, merged_key, value, &per_value_accumulate);
   	
   	my_tile.sync();


   }


}


struct data_pair {
	uint64_t key;
	uint64_t value;

};

struct local_accumulator {

	data_pair data[2048];


};


template <int n_dims, int contraction_dims>
__global__ void determine_header_kernel(coo_matrix<n_dims> * x_mat, uint64_t n_items, uint64_t * n_unique_headers, uint64_t * header_start){


	uint64_t tid = gallatin::utils::get_tid();

	if (tid >= n_items) return;

	uint64_t uncontracted_dims = x_mat->get_ht_key(tid, 0, n_dims-contraction_dims);

	if (tid != 0){


   	uint64_t prev_dims = x_mat->get_ht_key(tid-1, 0, n_dims-contraction_dims);

   	if (prev_dims != uncontracted_dims) return;

	}

	//all indices left are valid.

	uint64_t my_index = atomicAdd((unsigned long long int *)n_unique_headers, 1ULL);

	header_start[my_index] = tid;



}

template <typename ht_type, typename vector_type, int n_dims, int contraction_dims, int output_dims, uint tile_size>
__global__ void local_contract(coo_matrix<n_dims> * x_mat, ht_type * y_mat, coo_matrix<output_dims> * output){

	__shared__ local_accumulator acc;


	uint64_t starting_index = blockIdx.x;

	auto thread_block = cg::this_thread_block();

   cg::thread_block_tile<tile_size> my_tile = cg::tiled_partition<tile_size>(thread_block);


   uint64_t team_tid = threadIdx.x/tile_size;

   uint64_t team_step_size = (blockDim.x-1)/tile_size+1;

   if (starting_index >= x_mat->n_items) return;


   uint64_t uncontracted_dims = x_mat->get_ht_key(starting_index, 0, n_dims-contraction_dims);

   if (starting_index != 0){

   	//check if previous index is conflict...
   	//if so drop.
   	//how long does this take?
   	uint64_t prev_dims = x_mat->get_ht_key(starting_index-1, 0, n_dims-contraction_dims);

   	if (prev_dims != uncontracted_dims) return;

   }


   uint64_t currrent_index = starting_index+team_tid;

   while (x_mat->get_ht_key(currrent_index, 0, n_dims-contraction_dims) == uncontracted_dims){


   	uint64_t hash_key = x_mat->get_ht_key(currrent_index, n_dims-contraction_dims, contraction_dims);

   	uint64_t allocation;

	   if (!y_mat->find_with_reference_no_lock(my_tile, hash_key, allocation)) return;

	   vector_type * vector = (vector_type *) allocation;

	   auto lhs = x_mat->items[currrent_index];

	   currrent_index += team_step_size;


	   for (int i = 0; i < vector->size; i++){

	   	auto rhs = vector->data[i];

	   	uint64_t merged_key = output->get_output_key(lhs, contraction_dims, rhs);

	   	uint64_t value = rhs.value*lhs.value;

	   	if (my_tile.thread_rank() == 0){


	   		atomicAdd((unsigned long long int *)&acc.data[merged_key % 1024], (unsigned long long int) value);
	   	}
	   	

   	}



	}

   //separate.


}


template <int n_dims, int contraction_dims, template<typename, typename, uint, uint> typename hash_table_type, uint tile_size, uint bucket_size>
__host__ double tensor_contraction(std::string filename, uint64_t accumulator_nslots, bool table_uses_allocator=false){

	constexpr int leftover_dims = n_dims-contraction_dims;
	constexpr int output_dims = leftover_dims*2;
	using lower_COO = COO<leftover_dims>;

	using output_COO = COO<output_dims>;

	using vector_type = hashing_project::data_structs::cuckoo_vector<lower_COO>;
	using ht_type = hash_table_type<uint64_t, uint64_t, tile_size, bucket_size>;




	using output_mat_type = coo_matrix<output_dims>;


	if (table_uses_allocator){
		//if the table uses the allocator the allocator-table ratio is different - give more space to allocator
		gallatin::allocators::init_global_allocator(24ULL*1024*1024*1024, 111);
	} else {
		gallatin::allocators::init_global_allocator(16ULL*1024*1024*1024, 111);
	}
	


	coo_matrix<n_dims> x_mat(filename);

	x_mat.swap_dims(2, 3);

	coo_matrix<n_dims> y_mat(x_mat);


	//shuffle dimensions - bring the last n_dimensions forward.

	//this works! basically a backwards memcpy.

	int stride = n_dims-contraction_dims;

	for (int i =0; i < contraction_dims; i++){

		//dim we want to step into.
		int swap_dim = i;
		int alt_swap_dim = i+stride;

		y_mat.swap_dims(swap_dim, alt_swap_dim);

	}


	output_mat_type output_mat;

	output_mat.set_dimensions(x_mat, y_mat, contraction_dims);

	printf("Xmat: %u %u %u %u\n", x_mat.maxDims[0], x_mat.maxDims[1], x_mat.maxDims[2], x_mat.maxDims[3]);
	printf("Ymat: %u %u %u %u\n", y_mat.maxDims[0], y_mat.maxDims[1], y_mat.maxDims[2], y_mat.maxDims[3]);

	printf("Output mat: ");
	output_mat.print_dims();
	//now construct table.

	//table needs enough slots for the dimensions covered?
	//90\% load will never be exceeded if we use n_items.
	uint64_t n_slots= y_mat.n_items*1.10;

	ht_type * indirection_table = ht_type::generate_on_device(n_slots, 999);


	coo_matrix<n_dims> * y_mat_device = gallatin::utils::get_device_version<coo_matrix<n_dims>>();
	cudaMemcpy(y_mat_device, &y_mat, sizeof(coo_matrix<n_dims>), cudaMemcpyHostToDevice);


	coo_matrix<n_dims> * x_mat_device = gallatin::utils::get_device_version<coo_matrix<n_dims>>();
	cudaMemcpy(x_mat_device, &x_mat, sizeof(coo_matrix<n_dims>), cudaMemcpyHostToDevice);

	coo_matrix<output_dims> * output_mat_device = gallatin::utils::get_device_version<coo_matrix<output_dims>>();
	cudaMemcpy(output_mat_device, &output_mat, sizeof(coo_matrix<output_dims>), cudaMemcpyHostToDevice);


	cudaDeviceSynchronize();


	printf("%s starting\n", ht_type::get_name().c_str());


	gallatin::utils::timer convert_timing;

	convert_to_ht<ht_type, lower_COO, vector_type, n_dims, contraction_dims, tile_size><<<(y_mat.n_items*tile_size-1)/256+1,256>>>(indirection_table, y_mat_device, y_mat.n_items);

	convert_timing.sync_end();

	convert_timing.print_throughput("Converted", y_mat.n_items);

	indirection_table->print_fill();


	//construct accumulator:

	ht_type * accumulator = ht_type::generate_on_device(accumulator_nslots, 444);

	//accumulator gathers uncontrolled indices

	//store final output as full tensors in vector. inputCOO + outputCOO yields high dimension output.

	//run 4 way tensor hash - yield exact key that maps to value.

	// uint64_t * n_unique_headers;

	// uint64_t * header_start = gallatin::utils::get_device_version<uint64_t>(x_mat.n_items);

	//determine the slipping points?


	// cudaMallocManaged((void **)&n_unique_headers, sizeof(uint64_t));

	// n_unique_headers[0] = 0;



	//cudaDeviceSynchronize();



	// gallatin::utils::timer alt_comp_timing;

	// determine_header_kernel<n_dims, contraction_dims><<<(x_mat.n_items-1)/256+1, 256>>>(x_mat_device, x_mat.n_items, n_unique_headers, header_start);


	// //local_contract<ht_type, vector_type, n_dims, contraction_dims, output_dims, tile_size><<<x_mat.n_items*256,256>>>(x_mat_device, indirection_table, output_mat_device);


	// alt_comp_timing.sync_end();

	// alt_comp_timing.print_throughput("Alt loaded", x_mat.n_items);

	// printf("%lu headers\n", n_unique_headers[0]);

	cudaDeviceSynchronize();


	gallatin::utils::timer contract_timing;


	contract<ht_type, vector_type, n_dims, contraction_dims, output_dims, tile_size><<<(x_mat.n_items*tile_size-1)/256+1,256>>>(x_mat_device, indirection_table, accumulator, output_mat_device);


	contract_timing.sync_end();


	contract_timing.print_throughput("contracted", x_mat.n_items);
	cudaDeviceSynchronize();

	accumulator->print_fill();
	//run kernel.

	ht_type::free_on_device(indirection_table);

	ht_type::free_on_device(accumulator);


	gallatin::allocators::free_global_allocator();

	return convert_timing.elapsed()+contract_timing.elapsed();


}


template <template<typename, typename, uint, uint> typename hash_table_type, uint tile_size, uint bucket_size>
__host__ double tensor_contraction_nips_2(std::string filename, uint64_t accumulator_nslots, bool table_uses_allocator=false){

	constexpr int n_dims = 4;
	constexpr int contraction_dims = 1;

	constexpr int leftover_dims = n_dims-contraction_dims;
	constexpr int output_dims = leftover_dims*2;
	using lower_COO = COO<leftover_dims>;

	using output_COO = COO<output_dims>;

	using vector_type = hashing_project::data_structs::cuckoo_vector<lower_COO>;
	using ht_type = hash_table_type<uint64_t, uint64_t, tile_size, bucket_size>;




	using output_mat_type = coo_matrix<output_dims>;


	if (table_uses_allocator){
		//if the table uses the allocator the allocator-table ratio is different - give more space to allocator
		gallatin::allocators::init_global_allocator(24ULL*1024*1024*1024, 111);
	} else {
		gallatin::allocators::init_global_allocator(16ULL*1024*1024*1024, 111);
	}
	


	coo_matrix<n_dims> x_mat(filename);

	//0 1 2 3 -> 0 1 3 2
	x_mat.swap_dims(2, 3);

	coo_matrix<n_dims> y_mat(x_mat);


	//shuffle dimensions - bring the last n_dimensions forward.

	//this works! basically a backwards memcpy.

	int stride = n_dims-contraction_dims;

	//swap to 2 0 1 3
	for (int i =0; i < contraction_dims; i++){

		//dim we want to step into.
		int swap_dim = i;
		int alt_swap_dim = i+stride;

		y_mat.swap_dims(swap_dim, alt_swap_dim);

	}


	output_mat_type output_mat;

	output_mat.set_dimensions(x_mat, y_mat, contraction_dims);

	printf("Xmat: %u %u %u %u\n", x_mat.maxDims[0], x_mat.maxDims[1], x_mat.maxDims[2], x_mat.maxDims[3]);
	printf("Ymat: %u %u %u %u\n", y_mat.maxDims[0], y_mat.maxDims[1], y_mat.maxDims[2], y_mat.maxDims[3]);

	printf("Output mat: ");
	output_mat.print_dims();
	//now construct table.

	//table needs enough slots for the dimensions covered?
	//90\% load will never be exceeded if we use n_items.
	uint64_t n_slots= y_mat.n_items*1.10;

	ht_type * indirection_table = ht_type::generate_on_device(n_slots, 999);


	coo_matrix<n_dims> * y_mat_device = gallatin::utils::get_device_version<coo_matrix<n_dims>>();
	cudaMemcpy(y_mat_device, &y_mat, sizeof(coo_matrix<n_dims>), cudaMemcpyHostToDevice);


	coo_matrix<n_dims> * x_mat_device = gallatin::utils::get_device_version<coo_matrix<n_dims>>();
	cudaMemcpy(x_mat_device, &x_mat, sizeof(coo_matrix<n_dims>), cudaMemcpyHostToDevice);

	coo_matrix<output_dims> * output_mat_device = gallatin::utils::get_device_version<coo_matrix<output_dims>>();
	cudaMemcpy(output_mat_device, &output_mat, sizeof(coo_matrix<output_dims>), cudaMemcpyHostToDevice);


	cudaDeviceSynchronize();


	printf("%s starting\n", ht_type::get_name().c_str());


	gallatin::utils::timer convert_timing;

	convert_to_ht<ht_type, lower_COO, vector_type, n_dims, contraction_dims, tile_size><<<(y_mat.n_items*tile_size-1)/256+1,256>>>(indirection_table, y_mat_device, y_mat.n_items);

	convert_timing.sync_end();

	convert_timing.print_throughput("Converted", y_mat.n_items);

	indirection_table->print_fill();


	//construct accumulator:

	ht_type * accumulator = ht_type::generate_on_device(accumulator_nslots, 444);

	//accumulator gathers uncontrolled indices

	//store final output as full tensors in vector. inputCOO + outputCOO yields high dimension output.

	//run 4 way tensor hash - yield exact key that maps to value.

	// uint64_t * n_unique_headers;

	// uint64_t * header_start = gallatin::utils::get_device_version<uint64_t>(x_mat.n_items);

	//determine the slipping points?


	// cudaMallocManaged((void **)&n_unique_headers, sizeof(uint64_t));

	// n_unique_headers[0] = 0;



	//cudaDeviceSynchronize();



	// gallatin::utils::timer alt_comp_timing;

	// determine_header_kernel<n_dims, contraction_dims><<<(x_mat.n_items-1)/256+1, 256>>>(x_mat_device, x_mat.n_items, n_unique_headers, header_start);


	// //local_contract<ht_type, vector_type, n_dims, contraction_dims, output_dims, tile_size><<<x_mat.n_items*256,256>>>(x_mat_device, indirection_table, output_mat_device);


	// alt_comp_timing.sync_end();

	// alt_comp_timing.print_throughput("Alt loaded", x_mat.n_items);

	// printf("%lu headers\n", n_unique_headers[0]);

	cudaDeviceSynchronize();


	gallatin::utils::timer contract_timing;


	contract<ht_type, vector_type, n_dims, contraction_dims, output_dims, tile_size><<<(x_mat.n_items*tile_size-1)/256+1,256>>>(x_mat_device, indirection_table, accumulator, output_mat_device);


	contract_timing.sync_end();


	contract_timing.print_throughput("contracted", x_mat.n_items);
	cudaDeviceSynchronize();

	accumulator->print_fill();
	//run kernel.

	ht_type::free_on_device(indirection_table);

	ht_type::free_on_device(accumulator);


	gallatin::allocators::free_global_allocator();

	return convert_timing.elapsed()+contract_timing.elapsed();


}

template <template<typename, typename, uint, uint> typename hash_table_type, uint tile_size, uint bucket_size>
__host__ double tensor_contraction_nips_013(std::string filename, uint64_t accumulator_nslots, bool table_uses_allocator=false){

	constexpr int n_dims = 4;
	constexpr int contraction_dims = 3;


	constexpr int leftover_dims = n_dims-contraction_dims;
	constexpr int output_dims = leftover_dims*2;
	using lower_COO = COO<leftover_dims>;

	using output_COO = COO<output_dims>;

	using vector_type = hashing_project::data_structs::cuckoo_vector<lower_COO>;
	using ht_type = hash_table_type<uint64_t, uint64_t, tile_size, bucket_size>;




	using output_mat_type = coo_matrix<output_dims>;


	if (table_uses_allocator){
		//if the table uses the allocator the allocator-table ratio is different - give more space to allocator
		gallatin::allocators::init_global_allocator(24ULL*1024*1024*1024, 111);
	} else {
		gallatin::allocators::init_global_allocator(16ULL*1024*1024*1024, 111);
	}
	


	coo_matrix<n_dims> x_mat(filename);

	//0 1 2 3 -> 2 0 1 3
	x_mat.swap_dims(0, 2);
	x_mat.swap_dims(1,2);

	coo_matrix<n_dims> y_mat(x_mat);


	//shuffle dimensions - bring the last n_dimensions forward.

	//this works! basically a backwards memcpy.

	int stride = n_dims-contraction_dims;

	for (int i =0; i < contraction_dims; i++){

		//dim we want to step into.
		int swap_dim = i;
		int alt_swap_dim = i+stride;

		y_mat.swap_dims(swap_dim, alt_swap_dim);

	}


	output_mat_type output_mat;

	output_mat.set_dimensions(x_mat, y_mat, contraction_dims);

	printf("Xmat: %u %u %u %u\n", x_mat.maxDims[0], x_mat.maxDims[1], x_mat.maxDims[2], x_mat.maxDims[3]);
	printf("Ymat: %u %u %u %u\n", y_mat.maxDims[0], y_mat.maxDims[1], y_mat.maxDims[2], y_mat.maxDims[3]);

	printf("Output mat: ");
	output_mat.print_dims();
	//now construct table.

	//table needs enough slots for the dimensions covered?
	//90\% load will never be exceeded if we use n_items.
	uint64_t n_slots= y_mat.n_items*1.10;

	ht_type * indirection_table = ht_type::generate_on_device(n_slots, 999);


	coo_matrix<n_dims> * y_mat_device = gallatin::utils::get_device_version<coo_matrix<n_dims>>();
	cudaMemcpy(y_mat_device, &y_mat, sizeof(coo_matrix<n_dims>), cudaMemcpyHostToDevice);


	coo_matrix<n_dims> * x_mat_device = gallatin::utils::get_device_version<coo_matrix<n_dims>>();
	cudaMemcpy(x_mat_device, &x_mat, sizeof(coo_matrix<n_dims>), cudaMemcpyHostToDevice);

	coo_matrix<output_dims> * output_mat_device = gallatin::utils::get_device_version<coo_matrix<output_dims>>();
	cudaMemcpy(output_mat_device, &output_mat, sizeof(coo_matrix<output_dims>), cudaMemcpyHostToDevice);


	cudaDeviceSynchronize();


	printf("%s starting\n", ht_type::get_name().c_str());


	gallatin::utils::timer convert_timing;

	convert_to_ht<ht_type, lower_COO, vector_type, n_dims, contraction_dims, tile_size><<<(y_mat.n_items*tile_size-1)/256+1,256>>>(indirection_table, y_mat_device, y_mat.n_items);

	convert_timing.sync_end();

	convert_timing.print_throughput("Converted", y_mat.n_items);

	indirection_table->print_fill();


	//construct accumulator:

	ht_type * accumulator = ht_type::generate_on_device(accumulator_nslots, 444);

	//accumulator gathers uncontrolled indices

	//store final output as full tensors in vector. inputCOO + outputCOO yields high dimension output.

	//run 4 way tensor hash - yield exact key that maps to value.

	// uint64_t * n_unique_headers;

	// uint64_t * header_start = gallatin::utils::get_device_version<uint64_t>(x_mat.n_items);

	//determine the slipping points?


	// cudaMallocManaged((void **)&n_unique_headers, sizeof(uint64_t));

	// n_unique_headers[0] = 0;



	//cudaDeviceSynchronize();



	// gallatin::utils::timer alt_comp_timing;

	// determine_header_kernel<n_dims, contraction_dims><<<(x_mat.n_items-1)/256+1, 256>>>(x_mat_device, x_mat.n_items, n_unique_headers, header_start);


	// //local_contract<ht_type, vector_type, n_dims, contraction_dims, output_dims, tile_size><<<x_mat.n_items*256,256>>>(x_mat_device, indirection_table, output_mat_device);


	// alt_comp_timing.sync_end();

	// alt_comp_timing.print_throughput("Alt loaded", x_mat.n_items);

	// printf("%lu headers\n", n_unique_headers[0]);

	cudaDeviceSynchronize();


	gallatin::utils::timer contract_timing;


	contract<ht_type, vector_type, n_dims, contraction_dims, output_dims, tile_size><<<(x_mat.n_items*tile_size-1)/256+1,256>>>(x_mat_device, indirection_table, accumulator, output_mat_device);


	contract_timing.sync_end();


	contract_timing.print_throughput("contracted", x_mat.n_items);
	cudaDeviceSynchronize();

	accumulator->print_fill();
	//run kernel.

	ht_type::free_on_device(indirection_table);

	ht_type::free_on_device(accumulator);


	gallatin::allocators::free_global_allocator();

	return convert_timing.elapsed()+contract_timing.elapsed();


}




template <int n_dims, int contraction_dims, template<typename, typename, uint, uint> typename hash_table_type, uint tile_size, uint bucket_size>
__host__ void tensor_contraction_individual_ops(std::string filename, uint64_t accumulator_nslots){

	constexpr int leftover_dims = n_dims-contraction_dims;
	constexpr int output_dims = leftover_dims*2;
	using lower_COO = COO<leftover_dims>;

	using output_COO = COO<output_dims>;

	using vector_type = hashing_project::data_structs::cuckoo_vector<lower_COO>;
	using ht_type = hash_table_type<uint64_t, uint64_t, tile_size, bucket_size>;




	using output_mat_type = coo_matrix<output_dims>;


	gallatin::allocators::init_global_allocator(16ULL*1024*1024*1024, 111);


	coo_matrix<n_dims> x_mat(filename);

	x_mat.swap_dims(2, 3);

	coo_matrix<n_dims> y_mat(x_mat);


	//shuffle dimensions - bring the last n_dimensions forward.

	//this works! basically a backwards memcpy.

	int stride = n_dims-contraction_dims;

	for (int i =0; i < contraction_dims; i++){

		//dim we want to step into.
		int swap_dim = i;
		int alt_swap_dim = i+stride;

		y_mat.swap_dims(swap_dim, alt_swap_dim);

	}


	output_mat_type output_mat;

	output_mat.set_dimensions(x_mat, y_mat, contraction_dims);

	printf("Xmat: %u %u %u %u\n", x_mat.maxDims[0], x_mat.maxDims[1], x_mat.maxDims[2], x_mat.maxDims[3]);
	printf("Ymat: %u %u %u %u\n", y_mat.maxDims[0], y_mat.maxDims[1], y_mat.maxDims[2], y_mat.maxDims[3]);

	printf("Output mat: ");
	output_mat.print_dims();
	//now construct table.

	//table needs enough slots for the dimensions covered?
	//90\% load will never be exceeded if we use n_items.
	uint64_t n_slots= y_mat.n_items*1.10;

	ht_type * indirection_table = ht_type::generate_on_device(n_slots, 999);


	coo_matrix<n_dims> * y_mat_device = gallatin::utils::get_device_version<coo_matrix<n_dims>>();
	cudaMemcpy(y_mat_device, &y_mat, sizeof(coo_matrix<n_dims>), cudaMemcpyHostToDevice);


	coo_matrix<n_dims> * x_mat_device = gallatin::utils::get_device_version<coo_matrix<n_dims>>();
	cudaMemcpy(x_mat_device, &x_mat, sizeof(coo_matrix<n_dims>), cudaMemcpyHostToDevice);

	coo_matrix<output_dims> * output_mat_device = gallatin::utils::get_device_version<coo_matrix<output_dims>>();
	cudaMemcpy(output_mat_device, &output_mat, sizeof(coo_matrix<output_dims>), cudaMemcpyHostToDevice);


	cudaDeviceSynchronize();

	printf("%s starting\n", ht_type::get_name().c_str());

	gallatin::utils::timer convert_timing;

	convert_to_ht_individual<ht_type, lower_COO, vector_type, n_dims, contraction_dims, tile_size><<<(y_mat.n_items*tile_size-1)/256+1,256>>>(indirection_table, y_mat_device, y_mat.n_items);

	convert_timing.sync_end();

	convert_timing.print_throughput("Converted", y_mat.n_items);


	indirection_table->print_fill();

	//construct accumulator:

	ht_type * accumulator = ht_type::generate_on_device(accumulator_nslots, 444);

	//accumulator gathers uncontrolled indices

	//store final output as full tensors in vector. inputCOO + outputCOO yields high dimension output.

	//run 4 way tensor hash - yield exact key that maps to value.

	cudaDeviceSynchronize();


	gallatin::utils::timer contract_timing;


	contract<ht_type, vector_type, n_dims, contraction_dims, output_dims, tile_size><<<(x_mat.n_items*tile_size-1)/256+1,256>>>(x_mat_device, indirection_table, accumulator, output_mat_device);


	contract_timing.sync_end();


	contract_timing.print_throughput("contracted", x_mat.n_items);
	cudaDeviceSynchronize();

	accumulator->print_fill();
	//run kernel.

	ht_type::free_on_device(indirection_table);

	ht_type::free_on_device(accumulator);


	gallatin::allocators::free_global_allocator();


}


//multidimensional HT
// template <typename HT>
// __host__ HT * move_to_ht(COO * coo_table)

// template <typename HT>
// __host__ void tensor_contract(HT * tensor1, HT * tensor2)


#endif