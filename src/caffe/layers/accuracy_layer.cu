#include <algorithm>
#include <functional>
#include <utility>
#include <vector>
#include <cub/cub.cuh>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void accuracy_gpu_kernel(const int n, Dtype* bottom_data, 
    const Dtype* bottom_label, int* correct, int* count, const int num_labels, 
    const int top_k, const int ignore_label) {
	// this loops over N images and returns the correct index
  CUDA_KERNEL_LOOP(index, n) {
  	    // Specialize BlockLoad, BlockStore, and BlockRadixSort collective types
    typedef cub::BlockLoad<
        Dtype*, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_TRANSPOSE> BlockLoadT;
    typedef cub::BlockRadixSort<
        Dtype, 1, ITEMS_PER_THREAD, int> BlockRadixSortT;

	// Allocate type-safe, repurposable shared memory for collectives
    __shared__ union {
        typename BlockLoadT::TempStorage       load; 
        typename BlockRadixSortT::TempStorage  sort;
    } temp_storage; 

    // Obtain this block's segment of consecutive keys (blocked across threads)
    Dtype thread_keys[ITEMS_PER_THREAD];
    int thread_values[ITEMS_PER_THREAD];
    // cub::CountingInputIterator<Dtype*> itr(ITEMS_PER_THREAD);

    int block_offset = blockIdx.x * (BLOCK_THREADS * ITEMS_PER_THREAD);   
    BlockLoadT(temp_storage.load).Load(bottom_data + block_offset, thread_keys);

    __syncthreads();    // Barrier for smem reuse

  	int this_label = bottom_label[index];
  	if (this_label != ignore_label){

	    // Collectively sort the keys (not really, this is just a serial sort for now,
	    // but I can add more threads later)
	    BlockRadixSortT(temp_storage.sort).Sort(thread_keys, thread_values);
	    __syncthreads();    // Barrier for smem reuse

	    int this_correct = 0;
	    // if any of the top_k sorted values agree with the bottom_label, it increments correct
	    // everyone increments count if their bottom_label wasn't ignore
	    for (int i=0; i<top_k; i++){
	    	if (thread_values[i] == this_label){
	    		this_correct += 1;
	    	}
	    }
	    if (this_correct){
	    	correct[index] = 1;
	    }
	    count[index] = 1;
	  }
	// now we might as well do the reduction??
  }
}

// template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
// __global__ void accuracy_gpu_kernel(const int n, const double* bottom_data, 
//     const double* bottom_label, int* correct, int* count, const int num_labels, 
//     const int top_k, const int ignore_label) {
// 	// this loops over N images and returns the correct index
//   CUDA_KERNEL_LOOP(index, n) {
//   	    // Specialize BlockLoad, BlockStore, and BlockRadixSort collective types
//     typedef cub::BlockLoad<
//         int*, BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_LOAD_TRANSPOSE> BlockLoadT;
//     typedef cub::BlockRadixSort<
//         int, 1, ITEMS_PER_THREAD> BlockRadixSortT;

// 	// Allocate type-safe, repurposable shared memory for collectives
//     __shared__ union {
//         typename BlockLoadT::TempStorage       load; 
//         typename BlockRadixSortT::TempStorage  sort;
//     } temp_storage; 

//     // Obtain this block's segment of consecutive keys (blocked across threads)
//     double thread_keys[ITEMS_PER_THREAD];
//     cub::ArgIndexInputIterator<double*> thread_values(thread_keys);

//     int block_offset = blockIdx.x * (BLOCK_THREADS * ITEMS_PER_THREAD);   
//     BlockLoadT(temp_storage.load).Load(bottom_data + block_offset, thread_keys);

//     __syncthreads();    // Barrier for smem reuse

//   	int this_label = bottom_label[index];
//   	if (this_label != ignore_label)

// 	    // Collectively sort the keys (not really, this is just a serial sort for now,
// 	    // but I can add more threads later)
// 	    BlockRadixSortT(temp_storage.sort).Sort(thread_keys, thread_values);
// 	    __syncthreads();    // Barrier for smem reuse

// 	    int this_label = bottom_label[index];
// 	    int this_correct = 0;
// 	    // if any of the top_k sorted values agree with the bottom_label, it increments correct
// 	    // everyone increments count if their bottom_label wasn't ignore
// 	    for (int i, i++, i<top_k){
// 	    	if (thread_values[i] == this_label){
// 	    		this_correct += 1;
// 	    	}
// 	    }
// 	    if (this_correct){
// 	    	correct[index] = 1;
// 	    }
// 	    count[index] = 1;
// 	// now we might as well do the reduction??
//   }
// }
// template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
// void accuracy_gpu_kernel<BLOCK_THREADS, ITEMS_PER_THREAD>(const int n, const float* bottom_data, 
//     const float* bottom_label, int* correct, int* count, const int num_labels, 
//     const int top_k, const int ignore_label);

// template <int BLOCK_THREADS, int ITEMS_PER_THREAD> 
// void accuracy_gpu_kernel(const int n, const float* bottom_data, 
//     const float* bottom_label, int* correct, int* count, const int num_labels, 
//     const int top_k, const int ignore_label) {
// 	return accuracy_gpu_kernel<BLOCK_THREADS, ITEMS_PER_THREAD>(const int n, const float* bottom_data, 
//     const float* bottom_label, int* correct, int* count, const int num_labels, 
//     const int top_k, const int ignore_label);
// }

// template <int BLOCK_THREADS, int ITEMS_PER_THREAD> 
// void accuracy_gpu_kernel<BLOCK_THREADS, ITEMS_PER_THREAD>(const int n, const double* bottom_data, 
//     const double* bottom_label, int* correct, int* count, const int num_labels, 
//     const int top_k, const int ignore_label);

// template <int BLOCK_THREADS, int ITEMS_PER_THREAD> 
// void accuracy_gpu_kernel(const int n, const double* bottom_data, 
//     const double* bottom_label, int* correct, int* count, const int num_labels, 
//     const int top_k, const int ignore_label) {
// 	return accuracy_gpu_kernel<BLOCK_THREADS, ITEMS_PER_THREAD>(const int n, const double* bottom_data, 
//     const double* bottom_label, int* correct, int* count, const int num_labels, 
//     const int top_k, const int ignore_label);
// }

template <typename Dtype>
void AccuracyLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype* bottom_data = bottom[0]->mutable_gpu_data();
  const Dtype* bottom_label = bottom[1]->gpu_data();
  const int dim = bottom[0]->count() / outer_num_;
  // const int num_labels = bottom[0]->shape(label_axis_);
  if (!has_ignore_label_){
  	ignore_label_ = -1;
  }

  const int num_kernels = outer_num_*inner_num_;
  int* d_correct;
  int* d_count;
  size_t score_size = sizeof(int) * num_kernels;
  const int top_k = top_k_;
  const int ignore_label = ignore_label_;
  cudaMalloc(&d_correct, score_size);
  cudaMalloc(&d_count, score_size);
  const int num_threads = CAFFE_CUDA_NUM_THREADS;
  // NOLINT_NEXT_LINE(whitespace/operators)
  const int num_labels = 60;
  accuracy_gpu_kernel<Dtype, num_threads, num_labels><<<CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS>>>(num_kernels,
                             	bottom_data, bottom_label, 
                             	d_correct, d_count, num_labels, top_k, ignore_label);
  CUDA_POST_KERNEL_CHECK;

  Dtype* accuracy = top[0]->mutable_gpu_data();

	int *d_count_sum;
	cudaMalloc(&d_count_sum, sizeof(int));
	void *d_temp_storage = NULL;
	size_t temp_storage_bytes = 0;
	cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_correct, accuracy, num_kernels);

	// Allocate temporary storage
	cudaMalloc(&d_temp_storage, temp_storage_bytes);
	// sum the correct guesses into accuracy
	cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_correct, accuracy, num_kernels);

	// get the total number of counts
	cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_count, d_count_sum, num_kernels);

	// divide correct guesses by counts
	accuracy[0] /= d_count_sum[0];

}

INSTANTIATE_LAYER_GPU_FUNCS(AccuracyLayer);

}  // namespace caffe