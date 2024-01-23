wit_bindgen::generate!({
    world: "interfaces",
    exports: {
        "wasi:nn/inference":WasiNnAdaptor ,
        "wasi:nn/graph":WasiNnAdaptor ,
    },
});

use exports::wasi::nn::{
    graph::{Error, ExecutionTarget, Graph, GraphBuilder, GraphEncoding, Guest as GraphGuest},
    inference::{GraphExecutionContext, Guest as InferenceGuest, TensorData},
    tensor::{Tensor, TensorType},
};
use wasi::logging::logging::{self, log};

struct WasiNnAdaptor;

// TODO(Iceber): Error and logs need to be optimized

impl GraphGuest for WasiNnAdaptor {
    fn load(
        builder: Vec<GraphBuilder>,
        encoding: GraphEncoding,
        target: ExecutionTarget,
    ) -> Result<Graph, Error> {
        let encoding = match encoding {
            GraphEncoding::Onnx => wasmcloud_nn::GraphEncoding::Onnx,
            GraphEncoding::Openvino => wasmcloud_nn::GraphEncoding::Openvino,
            GraphEncoding::Tensorflow => wasmcloud_nn::GraphEncoding::Tensorflow,
            GraphEncoding::Tensorflowlite => wasmcloud_nn::GraphEncoding::Tensorflowlite,
            GraphEncoding::Pytorch => wasmcloud_nn::GraphEncoding::Pytorch,
            GraphEncoding::Autodetect => wasmcloud_nn::GraphEncoding::Autodetect,
        };
        let target = match target {
            ExecutionTarget::Cpu => wasmcloud_nn::ExecutionTarget::Cpu,
            ExecutionTarget::Gpu => wasmcloud_nn::ExecutionTarget::Gpu,
            ExecutionTarget::Tpu => wasmcloud_nn::ExecutionTarget::Tpu,
        };

        let request = wasmcloud_nn::LoadRequest {
            builder,
            encoding,
            target,
        };
        let payload = match rmp_serde::to_vec_named(&request) {
            Ok(payload) => payload,
            Err(_) => return Err(Error::InvalidEncoding),
        };

        match wasmcloud::bus::host::call_sync(None, "wasmcloud:nn/Nn.Load", &payload) {
            Err(_) => Err(Error::InvalidArgument),
            Ok(result) => {
                match rmp_serde::from_slice::<Result<u32, wasmcloud_nn::ErrorCode>>(&result) {
                    Ok(gr) => match gr {
                        Ok(graph_id) => Ok(graph_id),
                        Err(_) => return Err(Error::InvalidEncoding),
                    },
                    Err(_) => return Err(Error::InvalidEncoding),
                }
            }
        }
    }

    fn load_by_name(name: String) -> Result<Graph, Error> {
        let payload = match rmp_serde::to_vec(&name) {
            Ok(payload) => payload,
            Err(e) => {
                log(
                    logging::Level::Error,
                    "serde in load by name",
                    format!("Error: {:?}", e).as_str(),
                );
                return Err(Error::InvalidEncoding);
            }
        };
        match wasmcloud::bus::host::call_sync(None, "wasmcloud:nn/Nn.LoadByName", &payload) {
            Ok(result) => {
                match rmp_serde::from_slice::<Result<u32, wasmcloud_nn::ErrorCode>>(&result) {
                    Ok(gr) => match gr {
                        Ok(graph_id) => Ok(graph_id),
                        Err(msg) => {
                            log(
                                logging::Level::Error,
                                "call after from slice in load by name",
                                format!("Error: {:?}", msg).as_str(),
                            );
                            return Err(Error::InvalidArgument);
                        }
                    },
                    Err(msg) => {
                        log(
                            logging::Level::Error,
                            "from slice in load by name",
                            format!("Error: {:?}", msg).as_str(),
                        );
                        return Err(Error::InvalidArgument);
                    }
                }
            }

            Err(err) => {
                log(
                    logging::Level::Error,
                    "call in load by name",
                    format!("Error: {:?}", err).as_str(),
                );
                Err(Error::InvalidArgument)
            }
        }
    }
}

impl InferenceGuest for WasiNnAdaptor {
    fn init_execution_context(graph: Graph) -> Result<GraphExecutionContext, Error> {
        let payload = match rmp_serde::to_vec(&graph) {
            Ok(payload) => payload,
            Err(_) => return Err(Error::InvalidEncoding),
        };
        match wasmcloud::bus::host::call_sync(
            None,
            "wasmcloud:nn/Nn.InitExecutionContext",
            &payload,
        ) {
            Err(_) => Err(Error::InvalidArgument),
            Ok(result) => {
                match rmp_serde::from_slice::<Result<u32, wasmcloud_nn::ErrorCode>>(&result) {
                    Ok(gr) => match gr {
                        Ok(ctx) => Ok(ctx),
                        Err(msg) => {
                            log(
                                logging::Level::Error,
                                "call after from slice in init execution context",
                                format!("Error: {:?}", msg).as_str(),
                            );
                            return Err(Error::InvalidArgument);
                        }
                    },
                    Err(msg) => {
                        log(
                            logging::Level::Error,
                            "from slice in load by name",
                            format!("Error: {:?}", msg).as_str(),
                        );
                        return Err(Error::InvalidArgument);
                    }
                }
            }
        }
    }

    fn set_input(ctx: GraphExecutionContext, index: u32, tensor: Tensor) -> Result<(), Error> {
        let tensor_type = match tensor.tensor_type {
            TensorType::Fp16 => wasmcloud_nn::TensorType::Fp16,
            TensorType::U8 => wasmcloud_nn::TensorType::U8,
            TensorType::Fp32 => wasmcloud_nn::TensorType::Fp32,
            TensorType::Fp64 => wasmcloud_nn::TensorType::Fp64,
            TensorType::Bf16 => wasmcloud_nn::TensorType::Bf16,
            TensorType::I32 => wasmcloud_nn::TensorType::I32,
            TensorType::I64 => wasmcloud_nn::TensorType::I64,
        };

        let request = wasmcloud_nn::SetInputRequest {
            context: ctx,
            index,
            tensor: wasmcloud_nn::Tensor {
                dimensions: tensor.dimensions,
                ty: tensor_type,
                data: tensor.data,
            },
        };
        let payload = match rmp_serde::to_vec_named(&request) {
            Ok(payload) => payload,
            Err(_) => return Err(Error::InvalidEncoding),
        };
        match wasmcloud::bus::host::call_sync(None, "wasmcloud:nn/Nn.SetInput", &payload) {
            Err(_) => Err(Error::InvalidArgument),
            Ok(result) => {
                match rmp_serde::from_slice::<Result<(), wasmcloud_nn::ErrorCode>>(&result) {
                    Ok(gr) => match gr {
                        Ok(()) => Ok(()),
                        Err(msg) => {
                            log(
                                logging::Level::Error,
                                "call after from slice in set input",
                                format!("Error: {:?}", msg).as_str(),
                            );
                            return Err(Error::InvalidArgument);
                        }
                    },
                    Err(msg) => {
                        log(
                            logging::Level::Error,
                            "from slice in load by name",
                            format!("Error: {:?}", msg).as_str(),
                        );
                        return Err(Error::InvalidArgument);
                    }
                }
            }
        }
    }

    fn compute(ctx: GraphExecutionContext) -> Result<(), Error> {
        let payload = match rmp_serde::to_vec(&ctx) {
            Ok(payload) => payload,
            Err(_) => return Err(Error::InvalidEncoding),
        };
        match wasmcloud::bus::host::call_sync(None, "wasmcloud:nn/Nn.Compute", &payload) {
            Err(_) => Err(Error::InvalidArgument),
            Ok(_) => Ok(()),
        }
    }

    fn get_output(ctx: GraphExecutionContext, index: u32) -> Result<TensorData, Error> {
        let request = wasmcloud_nn::GetOutputRequest {
            context: ctx,
            index,
        };
        let payload = match rmp_serde::to_vec_named(&request) {
            Ok(payload) => payload,
            Err(_) => return Err(Error::InvalidEncoding),
        };
        match wasmcloud::bus::host::call_sync(None, "wasmcloud:nn/Nn.GetOutput", &payload) {
            Err(_) => Err(Error::InvalidArgument),
            Ok(result) => {
                match rmp_serde::from_slice::<
                    Result<wasmcloud_nn::TensorData, wasmcloud_nn::ErrorCode>,
                >(&result)
                {
                    Ok(gr) => match gr {
                        Ok(data) => Ok(data),
                        Err(msg) => {
                            log(
                                logging::Level::Error,
                                "call after from slice in get output",
                                format!("Error: {:?}", msg).as_str(),
                            );
                            return Err(Error::InvalidArgument);
                        }
                    },
                    Err(msg) => {
                        log(
                            logging::Level::Error,
                            "from slice in get output",
                            format!("Error: {:?}", msg).as_str(),
                        );
                        return Err(Error::InvalidArgument);
                    }
                }
            }
        }
    }
}

mod wasmcloud_nn {
    pub type TensorDimensions = Vec<u32>;
    pub type TensorData = Vec<u8>;
    pub type GraphBuilderArray = Vec<GraphBuilder>;
    pub type GraphExecutionContext = u32;
    pub type GraphBuilder = Vec<u8>;

    #[derive(Debug, ::serde::Serialize, ::serde::Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct GetOutputRequest {
        pub context: GraphExecutionContext,
        pub index: u32,
    }

    #[derive(Debug, ::serde::Serialize, ::serde::Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct LoadRequest {
        pub builder: GraphBuilderArray,
        pub encoding: GraphEncoding,
        pub target: ExecutionTarget,
    }

    #[derive(Debug, ::serde::Serialize, ::serde::Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct SetInputRequest {
        pub context: GraphExecutionContext,
        pub index: u32,
        pub tensor: Tensor,
    }

    #[derive(Debug, ::serde::Serialize, ::serde::Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct Tensor {
        pub dimensions: TensorDimensions,
        pub ty: TensorType,
        pub data: TensorData,
    }

    #[derive(Debug, ::serde::Serialize, ::serde::Deserialize)]
    pub enum TensorType {
        Fp16,
        Fp32,
        Fp64,
        Bf16,
        U8,
        I32,
        I64,
    }

    #[derive(Debug, ::serde::Serialize, ::serde::Deserialize)]
    pub enum ExecutionTarget {
        Cpu,
        Gpu,
        Tpu,
    }

    #[derive(Debug, ::serde::Serialize, ::serde::Deserialize)]
    pub enum GraphEncoding {
        Openvino,
        Onnx,
        Tensorflow,
        Pytorch,
        Tensorflowlite,
        Autodetect,
        Other(String),
    }

    #[derive(Debug, ::serde::Serialize, ::serde::Deserialize)]
    pub enum ErrorCode {
        Success,
        InvalidArgument,
        InvalidEncoding,
        Timeout,
        RuntimeError,
        UnsupportedOperation,
        TooLarge,
        NotFound,
    }
}
