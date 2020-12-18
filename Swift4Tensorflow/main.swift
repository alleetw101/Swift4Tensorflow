//
//  main.swift
//  Swift4Tensorflow
//
//  Created by Alan Lee on 12/17/20.
//

import Foundation

print("Hello, World!")
// generalTest()

var sample_model = ModelTrainingWalkthrough()
sample_model.trainModel()
print(sample_model.trainingDataset)
