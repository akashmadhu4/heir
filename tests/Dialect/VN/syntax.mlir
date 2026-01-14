// RUN: heir-opt %s | FileCheck %s

module {

  // VN.add
  func.func @add(%a : !vn.vn1d<8 x f32>, %b : !vn.vn1d<8 x f32>)
      -> !vn.vn1d<8 x f32> {

    %0 = vn.add %a, %b
      : (!vn.vn1d<8 x f32>, !vn.vn1d<8 x f32>) -> !vn.vn1d<8 x f32>

    // CHECK-LABEL: func.func @add
    // CHECK: vn.add %{{.*}}, %{{.*}} : ({{.*}}) -> !vn.vn1d<8{{ *}}x{{ *}}f32>
    return %0 : !vn.vn1d<8 x f32>
  }

  // VN.mul
  func.func @mul(%a : !vn.vn1d<16 x i64>, %b : !vn.vn1d<16 x i64>)
      -> !vn.vn1d<16 x i64> {

    %0 = vn.mul %a, %b
      : (!vn.vn1d<16 x i64>, !vn.vn1d<16 x i64>) -> !vn.vn1d<16 x i64>

    // CHECK-LABEL: func.func @mul
    // CHECK: vn.mul %{{.*}}, %{{.*}} : ({{.*}}) -> !vn.vn1d<16{{ *}}x{{ *}}i64>
    return %0 : !vn.vn1d<16 x i64>
  }

  // VN.merge
  func.func @merge(%a : !vn.vn1d<4 x i32>, %b : !vn.vn1d<4 x i32>)
      -> !vn.vn1d<4 x i32> {

    %0 = vn.merge %a, %b
      : (!vn.vn1d<4 x i32>, !vn.vn1d<4 x i32>) -> !vn.vn1d<4 x i32>

    // CHECK-LABEL: func.func @merge
    // CHECK: vn.merge %{{.*}}, %{{.*}} : ({{.*}}) -> !vn.vn1d<4{{ *}}x{{ *}}i32>
    return %0 : !vn.vn1d<4 x i32>
  }

  // VN.reduce
  func.func @reduce(%a : !vn.vn1d<32 x f32>) -> f32 {

    %0 = vn.reduce %a : !vn.vn1d<32 x f32> -> f32

    // CHECK-LABEL: func.func @reduce
    // CHECK: vn.reduce %{{.*}} : {{.*}} -> f32
    return %0 : f32
  }
}
