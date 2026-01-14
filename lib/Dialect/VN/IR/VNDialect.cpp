#include "llvm/include/llvm/ADT/TypeSwitch.h"   
#include "lib/Dialect/VN/IR/VNDialect.h"
#include "mlir/include/mlir/IR/Builders.h"
#include "lib/Dialect/VN/IR/VNTypes.h"
#include "lib/Dialect/VN/IR/VNOps.h"

#include "mlir/include/mlir/IR/DialectImplementation.h"








#include "lib/Dialect/VN/IR/VNDialect.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "lib/Dialect/VN/IR/VNTypes.cpp.inc"
#define GET_OP_CLASSES
#include "lib/Dialect/VN/IR/VNOps.cpp.inc"




namespace mlir {
namespace heir {
namespace vn {


void VNDialect::initialize(){
    addTypes<
#define GET_TYPEDEF_LIST
#include "lib/Dialect/VN/IR/VNTypes.cpp.inc"
    >();
    addOperations<
#define GET_OP_LIST
#include "lib/Dialect/VN/IR/VNOps.cpp.inc"
    >();
}

}  // namespace vn
}  // namespace heir
}  // namespace mlir