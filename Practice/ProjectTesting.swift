//
//  ProjectTesting.swift
//  Swift4Tensorflow
//
//  Created by Alan Lee on 12/17/20.
//

import TensorFlow

func generalTest() {
    let x = Tensor<Float64>([[1, 2], [3, 4]])
    print(x * x)

    print(matmul(x, x))
}

