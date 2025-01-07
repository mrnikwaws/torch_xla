#ifndef AOT_COMPILE_PJRT_AOT_MODEL_
#define AOT_COMPILE_PJRT_AOT_MODEL_

#include <torch/torch.h>
#include <ATen/ATen.h>

#include "absl/types/span.h"

#include "torch_xla/csrc/xla_backend_impl.h"

#include <string>
#include <memory>
#include <vector>
#include <mutex>

namespace torch_xla {

namespace runtime {
    class PjRtComputationClient;
}

namespace aot_compile {

/*
 * Simple interface to load and run from a compile + serialized PJRT model
 */
class PjRtAotModel {
public:

    PjRtAotModel( const std::string& compiled_model );

    bool LoadParameters(const std::vector<at::Tensor>& parameters);

    std::vector<at::Tensor> Execute(const std::vector<at::Tensor>& args);

private:

    bool initialize();

    torch::lazy::BackendImplInterface* backend_;
    std::unique_ptr<torch_xla::runtime::PjRtComputationClient> xla_pjrt_client_;
    std::unique_ptr<torch::lazy::BackendDevice> be_device_;
    torch_xla::runtime::ComputationClient::ComputationPtr computation_ptr_;
    std::vector<torch::lazy::BackendDataPtr> functional_arguments_;
};

} // namespace aot_compile
} // namespace torch_xla

#endif // AOT_COMPILE_PJRT_AOT_MODEL_