#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <torch/torch.h>
#include <ATen/ATen.h>
#include <torch/csrc/tensor/python_tensor.h>
#include <torch/csrc/jit/runtime/interpreter.h>

#include "torch_xla/csrc/aten_xla_bridge.h"
#include "torch_xla/csrc/ops/xla_ops.h"
#include "torch_xla/csrc/xla_backend_impl.h"
#include "torch_xla/csrc/runtime/pjrt_computation_client.h"
#include "torch_xla/csrc/tensor_util.h"

#include "xla/client/client_library.h"
#include "xla/pjrt/pjrt_api.h"
#include "xla/pjrt/pjrt_c_api_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/python/pjrt_ifrt/pjrt_client.h"
#include "xla/service/hlo.pb.h"
#include "xla/hlo/builder/xla_computation.h"


#include <iostream>
#include <memory>
#include <fstream>
#include <filesystem>

namespace se = stream_executor;

int main(int argc, char* argv[]) {
    if( argc != 2 ) {
        std::cout << "Usage: neuron_xla_client_test <hlo proto>" << std::endl;
        return 1;
    }

    // Odd code due to libneuronpjrt.so callbacks into the libneuronxla python package
    Py_Initialize();

    PyObject *sys = PyImport_ImportModule("sys");
    PyObject *path = PyObject_GetAttrString(sys, "path");

    PyList_Append(path,PyUnicode_FromWideChar(L"venv",-1));

    PyObject* module_name = PyUnicode_DecodeFSDefault("libneuronxla");
    PyObject* py_module = PyImport_Import(module_name);

    if (!py_module) {
        std::cerr << "Failed to import libneuronxla!" << std::endl;
        return 1;
    }

    // Plugin load
    auto status_or_plugin = pjrt::LoadPjrtPlugin("neuron", "./libneuronpjrt.so");

    if (!status_or_plugin.ok()) {
       std::cerr << "Load plugin for NEURON failed" << std::endl;
       return 1;
    }

    std::cout << "Neuron plugin loaded = " << status_or_plugin.value() << std::endl;

    auto status_or_c_api = pjrt::PjrtApi("neuron");

    if (!status_or_c_api.ok()) {
        std::cerr << "Get C API create failed!" << std::endl;
        return 1;
    }    

    const PJRT_Api* c_api = status_or_c_api.value();
    xla::PjRtCApiCompiler neuron_compiler(c_api);

    // Open previously lowered proto
    /* Python code:
    
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch_neuronx.experimental.profiler.v2_x.hlo_debug_syms import hlo_debug_syms, hlo_no_debug

        import json

        class Trivial(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.mul(x, 2)
        model = Trivial()

        example_inputs = torch.rand((4,4))

        print(model(example_inputs))

        output_string = hlo_debug_syms(model, [example_inputs], "pb2")

        with open("model.hlo.pb", "wb") as f:
            f.write(output_string)

        output_string = hlo_debug_syms(model, [example_inputs], "json")
        s = json.loads(output_string)

        with open("model.hlo.json", "w") as f:
            f.write(json.dumps(s, indent=2))
    */
    std::filesystem::path p{argv[1]};
    size_t file_size = std::filesystem::file_size(p);

    std::ifstream proto_file(argv[1], std::ifstream::binary);

    std::vector<char> buffer(file_size);
    proto_file.read(buffer.data(), file_size);

    std::string protobuf(buffer.begin(), buffer.end());

    std::cout << "Read protobuf of size=" << protobuf.size() << std::endl;

    xla::HloModuleProto hlo_proto;
    hlo_proto.ParseFromString(protobuf);

    // Create topology, a computation object from HLO, and options to invoke compile
    auto status_or_topology = xla::GetCApiTopology("neuron", "neuron_compile_topology");

    if (!status_or_topology.ok()) {
        std::cerr << "Get C API topology for compile failed! " << status_or_topology.status() << std::endl;
        return 1;
    }
    xla::XlaComputation comp(hlo_proto);
    xla::CompileOptions options;

    std::unique_ptr<xla::PjRtTopologyDescription> topology_desc = std::move(status_or_topology.value());

    // Client is *not* needed - set to nullptr
    auto executable_status_or = neuron_compiler.Compile(options, comp, *topology_desc, nullptr);

    if( !executable_status_or.ok() ) {
        std::cout << "Compile failed!" << std::endl;
        return 1;
    }

    auto executable_shared_ptr = std::move(executable_status_or.value());

    // Serialize the compiled executable
    auto status_or_serial = executable_shared_ptr->SerializeExecutable();

    if( !status_or_serial.ok() ) {
        std::cout << "Serialization of compiled artifacts faield" << std::endl;
        return 1;
    }

    // Save compiled exe (not loaded)
    std::string serialized_executable = status_or_serial.value();
    std::ofstream out_file("executable_save.xla.pt", std::ofstream::binary );

    out_file.write(serialized_executable.data(), serialized_executable.size());

    if( !out_file.good() ) {
        std::cout << "Executable save file failed!" << std::endl;
        return 1;
    } 

    std::cout << "Executable saved to file !" << std::endl;

    /** NOW load and run in torch_xla! **/
    // Initialize - note this need a chane in the PJRT registry (pjrt_registry.cc):
    /*
        } else if (device_type == "NEURON") {
            TF_VLOG(1) << "Initializing PjRt NEURON client...";
            auto status_or_api = pjrt::PjrtApi("NEURON");
            XLA_CHECK_OK(status_or_api.status());

            // Don't throw a cow if we already loaded the PJRT device
            if( !status_or_api.value() ) { 
            XLA_CHECK_OK(pjrt::LoadPjrtPlugin("NEURON", sys_util::GetEnvString(
                                                            env::kEnvNeuronLibraryPath,
                                                            "libneuronpjrt.so"))
                            .status());
            }

            client = std::move(xla::GetCApiClient("NEURON").value());
        }
    */
    if( !torch_xla::InitXlaBackend() ) {
        std::cout << "Torch XLA Initialization failed!" << std::endl;
        return 1;
    }

    std::cout << "Torch XLA Initialization succeeded!" << std::endl;
    torch::lazy::BackendImplInterface* backend = torch_xla::GetXlaBackendImpl();

    if( backend == nullptr ) {
        std::cout << "Torch XLA backend get failed!" << std::endl;
        return 1;        
    }

    torch_xla::runtime::PjRtComputationClient xla_pjrt_client;

    std::cout << "Torch XLA Backend obtained!" << std::endl;

    torch::lazy::BackendDevice be_device = backend->GetBackendDevice( at::Device("xla") );

    // Create input tensors - scaling factor and input tensor
    float scalar_value = 2.0;
    at::Tensor factor_cpu = at::tensor(scalar_value);
    at::Tensor input_tensor_cpu = at::rand({4, 4});

    auto factor = factor_cpu.to(at::Device("xla")); 
    auto input_tensor = input_tensor_cpu.to(at::Device("xla")); 

    std::cout << "Created input tensors on CPU" << std::endl;

    auto comp_ptr = xla_pjrt_client.DeserializeComputation(serialized_executable);

    std::cout << "PJRT Loaded Executable deserialized" << std::endl;

    auto aten_inputs = std::vector<at::Tensor>();

    aten_inputs.push_back(factor);
    aten_inputs.push_back(input_tensor);

    // Vector of args as BackendPtrs
    std::vector<torch::lazy::BackendDataPtr> arguments;

    // Show the user
    std::cout << "Inputs:" << std::endl;
    std::cout << std::endl;

    int i = 0;

    for( auto& t : aten_inputs ) {
        std::cout << "Input #" << i++ << std::endl;
        std::cout << t << std::endl;
        auto xla_tensor_ptr = torch_xla::bridge::TryGetXlaTensor(t);
        torch::lazy::BackendDataPtr dataptr = xla_tensor_ptr->GetXlaData();
        arguments.push_back(dataptr);        
    }

    std::cout << "Args ready, length = " << arguments.size() << std::endl;
    std::cout << "Compute expected args = " << comp_ptr->parameters_size() << std::endl;

    auto outputs = backend->ExecuteComputation( comp_ptr, arguments, be_device);
    std::vector<at::ScalarType> element_types;

    std::cout << "Ran execution, got " << outputs.size() << " results" << std::endl;

    for( int i = 0; i < outputs.size(); ++i) {
        element_types.push_back(at::kFloat);
    }
     
    auto aten_tensors = torch_xla::XlaDataToTensors(outputs,element_types);

    std::cout << "Outputs:" << std::endl;
    std::cout << std::endl;

    // Show the outputs are sane!
    for( auto t : aten_tensors ) {
        std::cout << t << std::endl;
    }

    std::cout << "SUCCESS!!" << std::endl;

    return 0;
}
