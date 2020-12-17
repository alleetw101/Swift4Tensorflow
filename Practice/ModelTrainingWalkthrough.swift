//
//  ModelTrainingWalkthrough.swift
//  Swift4Tensorflow
//
//  Created by Alan Lee on 12/17/20.
//

import TensorFlow
import PythonKit
import Foundation

/// Model example from Tensorflow site
func modelTrainingWalkthrough() {
    let plt = Python.import("matplotlib.pyplot")
    
    // Function to download file (Doesn't work with provided csv link)
    func download(from sourceString: String, to destinationString: String) {
        let source = URL(string: sourceString)!
        let destination = URL(fileURLWithPath: destinationString)
        print(source)
        let data = try! Data.init(contentsOf: source)
        try! data.write(to: destination)
    }
    
    let trainDataFilename = "/Users/alan/Documents/Programming/Swift/Swift4Tensorflow/Practice/iris_training.csv"
//    download(from: "http://download.tensorflow.org/data/iris_training.csv", to: trainDataFilename)
    
    let f = Python.open(trainDataFilename)
    for _ in 0..<5 {
        print(Python.next(f).strip())
    }
    f.close()
    
    
}
