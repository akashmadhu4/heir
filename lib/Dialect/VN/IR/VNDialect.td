

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"


def VN_Dialect : Dialect {
    let name = "vn";
    let cppNamespace = "::mlir::heir::vn";
    let description = [{
        Virtual neuron dialect which basically decomposes linalg ops into dot products
    }];
    let useDefaultTypePrinterParser = 1;
}


