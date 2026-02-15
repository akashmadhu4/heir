#ifndef LIB_DIALECT_VN_TRANSFORMS_CONVERTLINALGTOVN_TD_
#define LIB_DIALECT_VN_TRANSFORMS_CONVERTLINALGTOVN_TD_
include "mlir/Pass/PassBase.td"
def ConvertLinalgToVN : Pass<"convert-linalg-to-vn", "mlir::ModuleOp"> {
  let summary = "Convert linalg operations to VN dialect";
  
  let description = [{
    This pass converts linalg.conv2d operations to VN dialect operations.
    
    The conversion decomposes convolutions into:
    1. Patch extraction using vn.extract_slice
    2. Element-wise multiplication using vn.mul
    3. Reduction using vn.reduce
    4. Merging results using vn.merge
  }];
  
  let dependentDialects = [
    "mlir::heir::vn::VNDialect",
    "mlir::arith::ArithDialect",
    "mlir::tensor::TensorDialect"
  ];
}
#endif  // LIB_DIALECT_VN_TRANSFORMS_CONVERTLINALGTOVN_TD_