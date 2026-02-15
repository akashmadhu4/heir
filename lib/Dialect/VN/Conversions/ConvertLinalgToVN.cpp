#include "lib/Dialect/VN/Conversions/ConvertLinalgToVN.h"


#include "lib/Dialect/VN/IR/VNDialect.h"
#include "lib/Dialect/VN/IR/VNOps.h"
#include "lib/Dialect/VN/IR/VNTypes.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/include/mlir/IR/PatternMatch.h"
#include "mlir/include/mlir/Transforms/DialectConversion.h"



namespace mlir {
namespace heir {
namespace vn {


#define GEN_PASS_DEF_CONVERTLINALGTOVN
#include "lib/Dialect/VN/Conversions/ConvertLinalgToVN.h.inc"


namespace {

struct Conv2DOutputDims {
  int64_t outHeight;
  int64_t outWidth;
};

Conv2DOutputDims computeConv2DOutputDims(
    ArrayRef<int64_t> inputShape,
    ArrayRef<int64_t> kernelShape,
    int64_t stride = 1,
    int64_t padding = 0) {

  int64_t inputHeight = inputShape[1];
  int64_t inputWidth = inputShape[2];
  int64_t kernelHeight = kernelShape[0];
  int64_t kernelWidth = kernelShape[1];
  
  int64_t outHeight = (inputHeight + 2 * padding - kernelHeight) / stride + 1;
  int64_t outWidth = (inputWidth + 2 * padding - kernelWidth) / stride + 1;
  
  return {outHeight, outWidth};

}


struct ConvertConv2DToVN : public OpConversionPattern<linalg::Conv2DNhwcHwcfOp>{

  public: 
    ConvertConv2DToVN(MLIRContext *ctx)
      :OpConversionPattern<linalg::Conv2DNhwcHwcfOp>(ctx){}
    
    LogicalResult matchAndRewrite(linalg::Conv2DNhwcHwcfOp op ,OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
      
      StringAttr strategyAttr = op->getAttrOfType<StringAttr>("vn_lowering_config");
      StringRef strategy = strategyAttr ? strategyAttr.getValue() : "vn-conv-naive";

      if (strategy == "vn-conv-naive") {
        return lowerNaive(op, adaptor, rewriter);
      } else if (strategy == "vn-conv-toeplitz") {// toeplitz{
        return lowerToeplitz(op, adaptor, rewriter);
      } else {
         return op->emitError("Unknown lowering strategy: ") << strategy;
      }
    }

  private:
  
    LogicalResult lowerToeplitz(linalg::Conv2DNhwcHwcfOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
      Location loc = op.getLoc();
      Value input = op.getInputs()[0];
      Value kernel = op.getInputs()[1];
      Value output = op.getOutputs()[0];
      
      auto inputType = cast<RankedTensorType>(input.getType());
      auto kernelType = cast<RankedTensorType>(kernel.getType());
      auto outputType = cast<RankedTensorType>(output.getType());

      ArrayRef<int64_t> inputShape = inputType.getShape();
      ArrayRef<int64_t> kernelShape = kernelType.getShape();
      
      int64_t H = inputShape[1];
      int64_t W = inputShape[2];
      int64_t Cin = inputShape[3];
      int64_t kH = kernelShape[0];
      int64_t kW = kernelShape[1];
      int64_t Cout = kernelShape[3];
      
      Conv2DOutputDims outDims = computeConv2DOutputDims(inputShape, kernelShape);
      Type elementType = inputType.getElementType();
      
      //assuming batch=1
      int64_t flatSize = H * W * Cin;
      auto flat1DType = RankedTensorType::get({flatSize}, elementType);
      
      SmallVector<ReassociationIndices> reassoc = {{0, 1, 2, 3}};
      Value flatInput = rewriter.create<tensor::CollapseShapeOp>(
          loc, flat1DType, input, reassoc);
      
      Value zeroScalar = rewriter.create<arith::ConstantOp>(
          loc, elementType, rewriter.getFloatAttr(elementType, 0.0));
      
      Value zeroTensor = rewriter.create<tensor::SplatOp>(
          loc, flat1DType, zeroScalar);
      
      SmallVector<Value> macResults;
      
      for (int64_t oc = 0; oc < Cout; ++oc) {
        for (int64_t oh = 0; oh < outDims.outHeight; ++oh) {
          for (int64_t ow = 0; ow < outDims.outWidth; ++ow) {
            Value toeplitzRow = zeroTensor;
            for (int64_t kh = 0; kh < kH; ++kh) {
              for (int64_t kw = 0; kw < kW; ++kw) {
                for (int64_t cin = 0; cin < Cin; ++cin) {
                  SmallVector<Value> kernelIndices = {
                    rewriter.create<arith::ConstantIndexOp>(loc, kh),
                    rewriter.create<arith::ConstantIndexOp>(loc, kw),
                    rewriter.create<arith::ConstantIndexOp>(loc, cin),
                    rewriter.create<arith::ConstantIndexOp>(loc, oc)
                  };
                  Value kernelElem = rewriter.create<tensor::ExtractOp>(
                      loc, kernel, kernelIndices);
                  
                  //flat index in input: (oh+kh)*W*Cin + (ow+kw)*Cin + cin
                  int64_t flatIdx = (oh + kh) * W * Cin + (ow + kw) * Cin + cin;
                  Value flatIdxVal = rewriter.create<arith::ConstantIndexOp>(loc, flatIdx);
                  toeplitzRow = rewriter.create<tensor::InsertOp>(
                      loc, kernelElem, toeplitzRow, ValueRange{flatIdxVal});
                }
              }
            }
            
            // vn.mul: element-wise multiply Toeplitz row with flat input
            Value mulResult = rewriter.create<MulOp>(
                loc, flat1DType, toeplitzRow, flatInput);
            
            // vn.reduce: sum to scalar
            Value macResult = rewriter.create<ReduceOp>(
                loc, elementType, mulResult);
            
            macResults.push_back(macResult);
          }
        }
      }
      
      // Merge all scalar results into the output tensor
      Value mergedResult = rewriter.create<MergeOp>(
          loc, outputType, macResults);
      
      rewriter.replaceOp(op, mergedResult);
      
      return success();
    }

    LogicalResult lowerNaive(linalg::Conv2DNhwcHwcfOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
      Location loc = op.getLoc();
      Value input = op.getInputs()[0];
      Value kernel = op.getInputs()[1];
      Value output = op.getOutputs()[0];
      
      auto inputType = cast<RankedTensorType>(input.getType());
      auto kernelType = cast<RankedTensorType>(kernel.getType());
      auto outputType = cast<RankedTensorType>(output.getType());

      ArrayRef<int64_t> inputShape = inputType.getShape();
      ArrayRef<int64_t> kernelShape = kernelType.getShape();
      
      int64_t batchSize = inputShape[0];
      int64_t kernelHeight = kernelShape[0];
      int64_t kernelWidth = kernelShape[1];
      int64_t inChannels = kernelShape[2];
      
      Conv2DOutputDims outDims = computeConv2DOutputDims(inputShape, kernelShape);

      Type elementType = inputType.getElementType();
    
      auto patchSliceType = RankedTensorType::get(
          {batchSize, kernelHeight, kernelWidth, inChannels}, elementType);
      
      int64_t outChannels = kernelShape[3];
      SmallVector<Value> kernelSlices;

      if (outChannels == 1) {
        kernelSlices.push_back(kernel);
      } else {
        auto kernelSliceType = RankedTensorType::get(
             {kernelHeight, kernelWidth, inChannels, 1}, elementType);
        for (int64_t oc = 0; oc < outChannels; ++oc) {
           Value kernelSlice = rewriter.create<tensor::ExtractSliceOp>(
              loc, kernelSliceType, kernel,
              /*offsets=*/ArrayRef<OpFoldResult>{
                rewriter.getIndexAttr(0),
                rewriter.getIndexAttr(0),
                rewriter.getIndexAttr(0),
                rewriter.getIndexAttr(oc)
              },
              /*sizes=*/ArrayRef<OpFoldResult>{
                rewriter.getIndexAttr(kernelHeight),
                rewriter.getIndexAttr(kernelWidth),
                rewriter.getIndexAttr(inChannels),
                rewriter.getIndexAttr(1)
              },
              /*strides=*/ArrayRef<OpFoldResult>{
                rewriter.getIndexAttr(1),
                rewriter.getIndexAttr(1),
                rewriter.getIndexAttr(1),
                rewriter.getIndexAttr(1)
              });
           kernelSlices.push_back(kernelSlice);
        }
      }
      
      SmallVector<Value> macResults;
      SmallVector<Value> inputPatches;
      for (int64_t oh = 0; oh < outDims.outHeight; ++oh) {
        for (int64_t ow = 0; ow < outDims.outWidth; ++ow) {
          Value patchSlice = rewriter.create<tensor::ExtractSliceOp>(
              loc, patchSliceType, input,
              /*offsets=*/ArrayRef<OpFoldResult>{
                rewriter.getIndexAttr(0),           // batch offset = 0
                rewriter.getIndexAttr(oh),          // height offset
                rewriter.getIndexAttr(ow),          // width offset
                rewriter.getIndexAttr(0)            // channel offset = 0
              },
              /*sizes=*/ArrayRef<OpFoldResult>{
                rewriter.getIndexAttr(batchSize),
                rewriter.getIndexAttr(kernelHeight),
                rewriter.getIndexAttr(kernelWidth),
                rewriter.getIndexAttr(inChannels)
              },
              /*strides=*/ArrayRef<OpFoldResult>{
                rewriter.getIndexAttr(1),
                rewriter.getIndexAttr(1),
                rewriter.getIndexAttr(1),
                rewriter.getIndexAttr(1)
              });
          inputPatches.push_back(patchSlice);
        }
      }
      for (int64_t oc = 0; oc < outChannels; ++oc) {
        for (size_t spatialIdx = 0; spatialIdx < inputPatches.size(); ++spatialIdx) {
            Value mulResult = rewriter.create<MulOp>(
                loc, patchSliceType, inputPatches[spatialIdx], kernelSlices[oc]);
            Value macResult = rewriter.create<ReduceOp>(
                loc, elementType, mulResult);
            
            macResults.push_back(macResult);
        }
      }
      
      Value mergedResult = rewriter.create<MergeOp>(
          loc, outputType, macResults);
      
      rewriter.replaceOp(op, mergedResult);
      
      return success();
    } 
};


struct ConvertLinalgToVNPass
    : public impl::ConvertLinalgToVNBase<ConvertLinalgToVNPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    
    patterns.add<ConvertConv2DToVN>(context);
    
    ConversionTarget target(*context);
    target.addLegalDialect<VNDialect, arith::ArithDialect,
                          tensor::TensorDialect>();
    target.addIllegalOp<linalg::Conv2DOp>();
    
    if (failed(applyPartialConversion(getOperation(), target,
                                     std::move(patterns)))) {
      signalPassFailure();
    }
  }
};



}

}
}
}