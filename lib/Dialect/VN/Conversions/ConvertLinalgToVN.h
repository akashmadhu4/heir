#ifndef LIB_DIALECT_VN_TRANSFORMS_CONVERTLINALGTOVN_H_
#define LIB_DIALECT_VN_TRANSFORMS_CONVERTLINALGTOVN_H_

#include "mlir/include/mlir/Pass/Pass.h"


namespace mlir {

namespace heir {

namespace vn {

#define GEN_PASS_DECL_CONVERTLINALGTOVN
#include "lib/Dialect/VN/Conversions/ConvertLinalgToVN.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Dialect/VN/Conversions/ConvertLinalgToVN.h.inc"

}  // namespace vn
}  // namespace heir
}  // namespace mlir
#endif  // LIB_DIALECT_VN_TRANSFORMS_CONVERTLINALGTOVN_H_