#include "torch_xla/csrc/aot_compile/aot_model.h"

#include "torch_xla/csrc/runtime/tf_logging.h"

#include "torch_xla/csrc/aten_xla_bridge.h"
#include "torch_xla/csrc/ops/xla_ops.h"
#include "torch_xla/csrc/xla_backend_impl.h"
#include "torch_xla/csrc/runtime/pjrt_computation_client.h"
#include "torch_xla/csrc/tensor_util.h"

namespace torch_xla {
namespace aot_compile {

PjRtAotModel::PjRtAotModel( const std::string& compiled_model ) {
    XLA_CHECK( initialize() );
    computation_ptr_ = xla_pjrt_client_->DeserializeComputation(compiled_model);
}

bool 
PjRtAotModel::LoadParameters(
    std::vector<at::Tensor>& parameters) {

    functional_arguments_.clear();
    functional_arguments_.reserve(parameters.size());

    // Crashes if an auto ref is used
    for( auto t : parameters ) {
        auto t_xla = t.to(at::Device(at::kXLA)); 
        auto xla_tensor_ptr = torch_xla::bridge::TryGetXlaTensor(t_xla);
        torch::lazy::BackendDataPtr dataptr = xla_tensor_ptr->GetXlaData();
        functional_arguments_.push_back(dataptr);        
    }
}

std::vector<at::Tensor> 
PjRtAotModel::Execute(const std::vector<at::Tensor>& aten_inputs) {
    auto start_size = functional_arguments_.size();

    std::vector<at::ScalarType> element_types;

    // Add value to me invoked
    for( auto& t : aten_inputs ) {
        auto t_xla = t.to(at::Device(at::kXLA)); 
        auto xla_tensor_ptr = torch_xla::bridge::TryGetXlaTensor(t_xla);
        torch::lazy::BackendDataPtr dataptr = xla_tensor_ptr->GetXlaData();
        functional_arguments_.push_back(dataptr);   
        element_types.push_back(t.scalar_type());     
    }

    auto outputs = backend_->ExecuteComputation( computation_ptr_, functional_arguments_, *be_device_);
    functional_arguments_.resize(start_size); // Discard args - keep parameters

    auto aten_tensors = torch_xla::XlaDataToTensors(outputs,element_types);

    // Since this is a local expect the return value optimizer to optimize
    return aten_tensors;
}

bool PjRtAotModel::initialize() {

    if( !torch_xla::InitXlaBackend() ) {
        VLOG(1) << "Torch XLA Initialization failed!";
        return false;
    }

    VLOG(1) << "Torch XLA Initialization succeeded!";

    backend_ = torch_xla::GetXlaBackendImpl();

    if( backend_ == nullptr ) {
        VLOG(1) << "Torch XLA backend get failed!";
        return false;        
    }

    auto client = std::make_unique<torch_xla::runtime::PjRtComputationClient>();
    xla_pjrt_client_ = std::move(client);

    VLOG(1) << "Torch XLA Backend obtained!";    

    auto unique_be_device = std::make_unique<torch::lazy::BackendDevice>(
        backend_->GetBackendDevice( at::Device(at::kXLA) ));
    be_device_ = std::move(unique_be_device);

    VLOG(1) << "Torch XLA Backend Device obtained!";

    return true;
}

} // namespace aot_compile
} // namespace torch_xla