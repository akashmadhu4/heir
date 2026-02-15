func.func @conv2d_naive(%arg0: tensor<1x3x3x2xf32>, %arg1: tensor<2x2x2x1xf32>) -> tensor<1x2x2x1xf32> {
    %0 = tensor.empty() : tensor<1x2x2x1xf32>
    %result = linalg.conv_2d_nhwc_hwcf
      {vn_lowering_config = "vn-conv-naive"}
      ins(%arg0, %arg1 : tensor<1x3x3x2xf32>, tensor<2x2x2x1xf32>)
      outs(%0 : tensor<1x2x2x1xf32>)
      -> tensor<1x2x2x1xf32>
    return %result : tensor<1x2x2x1xf32>
}

// This function should fail if we try to lower it with the current implementation
func.func @conv2d_diagonal(%arg0: tensor<1x3x3x2xf32>, %arg1: tensor<2x2x2x1xf32>) -> tensor<1x2x2x1xf32> {
    %0 = tensor.empty() : tensor<1x2x2x1xf32>
    // This attribute should trigger the "not implemented" error
    %result = linalg.conv_2d_nhwc_hwcf
      {vn_lowering_config = "vn-conv-toeplitz"}
      ins(%arg0, %arg1 : tensor<1x3x3x2xf32>, tensor<2x2x2x1xf32>)
      outs(%0 : tensor<1x2x2x1xf32>)
      -> tensor<1x2x2x1xf32>
    return %result : tensor<1x2x2x1xf32>
}
