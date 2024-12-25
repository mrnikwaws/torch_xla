#include "torch_xla/csrc/aot_compile/aot_model.h"

#include "torch_xla/csrc/runtime/tf_logging.h"

namespace torch_xla {
namespace aot_compile {


PjRtAotModel::PjRtAotModel( std::string_view compiled_model ) {

    XLA_CHECK( initialize() );

    computation_ptr_ = xla_pjrt_client_->DeserializeComputation(serialized_executable);
}

bool 
PjRtAotModel::LoadParameters(
    absl::span<at::Tensor> parameters, absl::span<int> param_ids) {

    functional_arguments_.clear();
    functional.reserve(parameters.size());

    for( auto& t : aten_inputs ) {
        auto t_xla = t.to(at::Device("xla")); 
        auto xla_tensor_ptr = torch_xla::bridge::TryGetXlaTensor(t_xla);
        torch::lazy::BackendDataPtr dataptr = xla_tensor_ptr->GetXlaData();
        functional_arguments_.push_back(dataptr);        
    }
}

std::vector<at::Tensor> 
PjRtAotModel::Execute(absl::span<at::Tensor> aten_inputs, std::vector<at::ScalarType> element_types) {
    auto start_size = functional_arguments_.size();

    // Add value to me invoked
    for( auto& t : aten_inputs ) {
        auto t_xla = t.to(at::Device("xla")); 
        auto xla_tensor_ptr = torch_xla::bridge::TryGetXlaTensor(t_xla);
        torch::lazy::BackendDataPtr dataptr = xla_tensor_ptr->GetXlaData();
        functional_arguments_.push_back(dataptr);        
    }

    auto outputs = backend->ExecuteComputation( computation_ptr_, functional_arguments_, be_device_);
    functional_arguments_.resize(start_size); // Discard args - keep parameters

    auto aten_tensors = torch_xla::XlaDataToTensors(outputs,element_types);

    // Since this is a local expect expect the return value optimizer to optimize
    return aten_tensors;
}

bool PjRtAotModel::initialize() {

    if( !torch_xla::InitXlaBackend() ) {
        VLOG(1) << "Torch XLA Initialization failed!";
        return false;
    }

    VLOG(1) << "Torch XLA Initialization succeeded!";

    backend_ = torch_xla::GetXlaBackendImpl();

    if( backend == nullptr ) {
        VLOG(1) << "Torch XLA backend get failed!";
        return false;        
    }

    xla_pjrt_client_ = std::make_unique<torch_xla::runtime::PjRtComputationClient>();

    VLOG(1) << "Torch XLA Backend obtained!";    

    auto unique_be_device = make_unique<torch::lazy::BackendDevice>(
        backend->GetBackendDevice( at::Device(device_name_) ))
    be_device_ = std::move(unique_be_device);

    VLOG(1) << "Torch XLA Backend Device obtained!";    
}

} // namespace aot_compile
} // namespace torch_xla