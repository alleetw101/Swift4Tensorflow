//
//  main.swift
//  Swift4Tensorflow
//
//  Created by Alan Lee on 12/17/20.
//

import Foundation

print("Hello, World!")

ModelTrainingWalkthroughRun()

func scratch() {
    //var PracticeScratch = ExampleClass()
    //print(PracticeScratch.exampleVariable)
    //PracticeScratch.exampleVariable = "String"
    //print(PracticeScratch.exampleVariable)
}

func ModelTrainingWalkthroughRun() {
    let modelTrainingWalkthrough = ModelTrainingWalkthrough(batchsize: 32, epochCount: 501)
    modelTrainingWalkthrough.trainModel()
    modelTrainingWalkthrough.testModel()
    // modelTrainingWalkthrough.visualizeTraining()
}
