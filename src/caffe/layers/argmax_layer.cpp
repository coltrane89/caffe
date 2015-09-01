#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void ArgMaxLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  out_max_val_ = this->layer_param_.argmax_param().out_max_val();
  top_k_ = this->layer_param_.argmax_param().top_k();
  CHECK_GE(top_k_, 1) << " top k must not be less than 1.";
  CHECK_LE(top_k_, bottom[0]->shape(1))
      << "top_k must be less than or equal to the number of classes.";
  if (out_max_val_){
    CHECK_EQ(top.size(), 2) << "two top layers must be defined if out_max_val is true";
  }
}

template <typename Dtype>
void ArgMaxLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  label_axis_ = 1;
  outer_num_ = bottom[0]->count(0, label_axis_);
  inner_num_ = bottom[0]->count(label_axis_ + 1);
  top[0]->Reshape(bottom[0]->num(), top_k_, bottom[0]->shape(2),
      bottom[0]->shape(3));
  if (out_max_val_) {
    // Produces max_ind and max_val
    top[1]->Reshape(bottom[0]->num(), top_k_, bottom[0]->shape(2),
      bottom[0]->shape(3));
  }
}

template <typename Dtype>
void ArgMaxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_indices = top[0]->mutable_cpu_data();
  Dtype* top_values = NULL;
  if (out_max_val_) {
    top_values = top[1]->mutable_cpu_data();
  }
  const int num_labels = bottom[0]->shape(label_axis_);
  const int dim = bottom[0]->count() / outer_num_;
  // i*j vectors made/sorted/copied
  for (int i = 0; i < outer_num_; ++i) {
    // j vectors made/sorted/copied
    for (int j = 0; j < inner_num_; ++j) {
      std::vector<std::pair<Dtype, int> > bottom_data_vector;  
      // k value, index pairs made
      for (int k = 0; k < num_labels; ++k) {
        bottom_data_vector.push_back(std::make_pair(
            bottom_data[i * dim + k * inner_num_ + j], k));
      }
      std::partial_sort(
          bottom_data_vector.begin(), bottom_data_vector.begin() + top_k_,
          bottom_data_vector.end(), std::greater<std::pair<Dtype, int> >());
      // k indices copied to top
      for (int k = 0; k < top_k_; ++k) {
        top_indices[(i * top_k_ + k) * inner_num_ + j] = bottom_data_vector[k].second;
      }
      if (out_max_val_) {
        // k values copied to top
        for (int k = 0; j < top_k_; ++k) {
          top_values[(i * top_k_ + k) * inner_num_ + j] = bottom_data_vector[k].first;
        }
      }
    }
  }
}

INSTANTIATE_CLASS(ArgMaxLayer);
REGISTER_LAYER_CLASS(ArgMax);

}  // namespace caffe
