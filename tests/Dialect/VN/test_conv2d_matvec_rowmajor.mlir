func.func @conv2d_matvec(%arg0: tensor<1x3x3x1xf32>, %arg1: tensor<2x2x1x1xf32>) -> tensor<1x2x2x1xf32> {
    %0 = tensor.empty() : tensor<1x2x2x1xf32>
    %result = linalg.conv_2d_nhwc_hwcf
      {vn_lowering_config = "vn-conv-naive"}
      ins(%arg0, %arg1 : tensor<1x3x3x1xf32>, tensor<2x2x1x1xf32>)
      outs(%0 : tensor<1x2x2x1xf32>)
    return %result : tensor<1x2x2x1xf32>
}
