#include <algorithm>
#include <functional>
#include <utility>
#include <vector>
#include <cub/cub.cuh>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void argmax_gpu_kernel(const int n, const Dtype* bottom_data, 
    const int num_labels, const size_t top_k, Dtype* top_indices, Dtype* top_values) {
  // this loops over N images and returns the correct index
  CUDA_KERNEL_LOOP(index, n) {
        // Specialize BlockLoad, BlockStore, and BlockRadixSort collective types
    // we read in num_labels items
    typedef cub::BlockLoad<
        int*, BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_LOAD_TRANSPOSE> BlockLoadT;
    // we write out top_k items
    typedef cub::BlockStore<
        int*, BLOCK_THREADS, top_k, BLOCK_LOAD_TRANSPOSE> BlockStoreT;
    typedef cub::BlockRadixSort<
        int, 1, ITEMS_PER_THREAD> BlockRadixSortT;

  // Allocate type-safe, repurposable shared memory for collectives
    __shared__ union {
        typename BlockLoadT::TempStorage       load; 
        typename BlockStoreT::TempStorage      store; 
        typename BlockRadixSortT::TempStorage  sort;
    } temp_storage; 

    // Obtain this block's segment of consecutive keys (blocked across threads)
    Dtype thread_keys[ITEMS_PER_THREAD];
    cub::ArgIndexInputIterator<Dtype*> thread_values(thread_keys);

    int block_offset = blockIdx.x * (BLOCK_THREADS * ITEMS_PER_THREAD);   
    BlockLoadT(temp_storage.load).Load(bottom_data + block_offset, thread_keys);


    __syncthreads();    // Barrier for smem reuse

    // Collectively sort the keys (not really, this is just a serial sort for now,
    // but I can add more threads later)
    BlockRadixSortT(temp_storage.sort).Sort(thread_keys, thread_values);
    __syncthreads();    // Barrier for smem reuse

    // write only the top k values to top_indices
    BlockStoreT(temp_storage.store).Load(top_indices + block_offset, thread_values);

    // write back the values if asked
    BlockStoreT(temp_storage.store).Load(top_values + block_offset, thread_keys);
  }
}

template <typename Dtype>
void ArgMaxLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_indices = top[0]->mutable_gpu_data();
  Dtype* top_values = NULL;
  const int num_labels = bottom[0]->shape(label_axis_);
  if (out_max_val_) {
    top_values = top[1]->mutable_gpu_data();
  }

  const int num_kernels = outer_num_*inner_num_;

  // NOLINT_NEXT_LINE(whitespace/operators)
  argmax_gpu_kernel<Dtype, CAFFE_CUDA_NUM_THREADS, num_labels><<<CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS>>>(num_kernels, bottom_data, num_labels, 
                              top_k_, top_indices, top_values);
  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_LAYER_GPU_FUNCS(ArgMaxLayer);

}  // namespace caffe
