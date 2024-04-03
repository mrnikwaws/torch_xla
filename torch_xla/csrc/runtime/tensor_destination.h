#ifndef XLA_CLIENT_TENSOR_DESTINATION_H_
#define XLA_CLIENT_TENSOR_DESTINATION_H_

#include <ATen/Tensor.h>
#include <torch/csrc/lazy/core/metrics.h>

#include <vector>

#include "torch_xla/csrc/dtype.h"
#include "torch_xla/csrc/runtime/debug_macros.h"
#include "xla/literal.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace torch_xla {
namespace runtime {

// Owns a contiguous block of data with the shape and layout matching `shape()`.
class TensorDestination {
 public:
  TensorDestination(std::string device) : device_(std::move(device)){};

  virtual void* data() const = 0;

  virtual const xla::Shape& shape() const = 0;

  const std::string& device() const { return device_; }

  virtual std::vector<int64_t> byte_strides() const {
    std::vector<int64_t> byte_strides(shape().dimensions_size());
    XLA_CHECK_OK(
        xla::ShapeUtil::ByteStrides(shape(), absl::MakeSpan(byte_strides)));
    return byte_strides;
  }

  virtual void resize(const xla::Shape& new_shape) = 0;

  virtual std::vector<int64_t> dimensions() const {
    auto dimensions = shape().dimensions();
    return {dimensions.begin(), dimensions.end()};
  }

  virtual xla::PrimitiveType primitive_type() const {
    return shape().element_type();
  }

 private:
  std::string device_;
};

class AtenDestination : public TensorDestination {
 public:
  AtenDestination(at::Tensor tensor)
      : TensorDestination(tensor.device().str()), 
        shape_(xla::ShapeUtil::MakeShape(XlaTypeFromTorchType(tensor.scalar_type()),tensor.sizes())) {
    at::ScalarType target_torch_type = TorchTypeFromXlaType(primitive_type());
    if (target_torch_type != tensor.type().scalarType()) {
      TORCH_LAZY_COUNTER("AtenDestinationDowncasts", 1);
    }

    // TODO(ysiraichi): check, first, if tensor lives in a device that the
    // current PjRt client has access. If so, we don't need to go through the
    // CPU.
    tensor_ = std::move(
        tensor.to(at::TensorOptions().device(at::kCPU).dtype(target_torch_type),
                  /*non_blocking=*/false,
                  /*copy=*/false, at::MemoryFormat::Contiguous));
  }

  void resize(const xla::Shape& new_shape) override {
    // Update shape
    shape_ = new_shape;

    at::ScalarType target_torch_type = TorchTypeFromXlaType(primitive_type());
    auto shape = new_shape.dimensions();

    // Ensure type has not changed
    tensor_ = std::move(
      tensor_.to(at::TensorOptions().device(at::kCPU).dtype(target_torch_type),
                /*non_blocking=*/false,
                /*copy=*/false, at::MemoryFormat::Contiguous));    

    // Resize
    tensor_.resize_({shape.begin(),shape.end()});
  }

  void* data() const override { return tensor_.mutable_data_ptr(); }

  const xla::Shape& shape() const override { return shape_; }

  std::vector<int64_t> byte_strides() const override {
    std::vector<int64_t> strides;
    for (auto& stride : tensor_.strides()) {
      strides.push_back(stride * tensor_.itemsize());
    }
    return strides;
  }

  std::vector<int64_t> dimensions() const override {
    auto sizes = tensor_.sizes();
    return {sizes.begin(), sizes.end()};
  }

 private:
  mutable at::Tensor tensor_;
  xla::Shape shape_;
};

}  // namespace runtime
}  // namespace torch_xla

#endif  // XLA_CLIENT_TENSOR_DESTINATION_H_
