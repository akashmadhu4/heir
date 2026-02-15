module {
  func.func @conv2d_1c1f(%arg0: tensor<1x3x3x2xf32>, %arg1: tensor<2x2x2x1xf32>) -> tensor<1x2x2x1xf32> {
    %0 = tensor.empty() : tensor<1x2x2x1xf32>
    %extracted_slice = tensor.extract_slice %arg0[0, 0, 0, 0] [1, 2, 2, 2] [1, 1, 1, 1] : tensor<1x3x3x2xf32> to tensor<1x2x2x2xf32>
    %1 = vn.mul %extracted_slice, %arg1 : (tensor<1x2x2x2xf32>, tensor<2x2x2x1xf32>) -> tensor<1x2x2x2xf32>
    %2 = vn.reduce %1 : tensor<1x2x2x2xf32> -> f32
    %extracted_slice_0 = tensor.extract_slice %arg0[0, 0, 1, 0] [1, 2, 2, 2] [1, 1, 1, 1] : tensor<1x3x3x2xf32> to tensor<1x2x2x2xf32>
    %3 = vn.mul %extracted_slice_0, %arg1 : (tensor<1x2x2x2xf32>, tensor<2x2x2x1xf32>) -> tensor<1x2x2x2xf32>
    %4 = vn.reduce %3 : tensor<1x2x2x2xf32> -> f32
    %extracted_slice_1 = tensor.extract_slice %arg0[0, 1, 0, 0] [1, 2, 2, 2] [1, 1, 1, 1] : tensor<1x3x3x2xf32> to tensor<1x2x2x2xf32>
    %5 = vn.mul %extracted_slice_1, %arg1 : (tensor<1x2x2x2xf32>, tensor<2x2x2x1xf32>) -> tensor<1x2x2x2xf32>
    %6 = vn.reduce %5 : tensor<1x2x2x2xf32> -> f32
    %extracted_slice_2 = tensor.extract_slice %arg0[0, 1, 1, 0] [1, 2, 2, 2] [1, 1, 1, 1] : tensor<1x3x3x2xf32> to tensor<1x2x2x2xf32>
    %7 = vn.mul %extracted_slice_2, %arg1 : (tensor<1x2x2x2xf32>, tensor<2x2x2x1xf32>) -> tensor<1x2x2x2xf32>
    %8 = vn.reduce %7 : tensor<1x2x2x2xf32> -> f32
    %9 = vn.merge %2, %4, %6, %8 : (f32, f32, f32, f32) -> tensor<1x2x2x1xf32>
    return %9 : tensor<1x2x2x1xf32>
  }
}

