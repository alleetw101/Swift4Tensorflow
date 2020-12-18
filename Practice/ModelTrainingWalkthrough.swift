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
class ModelTrainingWalkthrough {
    let trainDataFilename = "/Users/alan/Documents/Programming/Swift/Swift4Tensorflow/Practice/iris_training.csv"
    var trainingDataset: [IrisBatch] {get {preProcessData()} set {}}
    
    init() {
    }
    
    func downloadData() {
        // Function to download file (Doesn't work with provided csv link)
        func download(from sourceString: String, to destinationString: String) {
            let source = URL(string: sourceString)!
            let destination = URL(fileURLWithPath: destinationString)
            print(source)
            let data = try! Data.init(contentsOf: source)
            try! data.write(to: destination)
        }
    
    //    download(from: "http://download.tensorflow.org/data/iris_training.csv", to: trainDataFilename)
    }
    
    func preProcessData() -> [IrisBatch] {
        // Examine csv
        let f = Python.open(trainDataFilename)
        for _ in 0..<5 {
            print(Python.next(f).strip())
        }
        f.close()
        
        // CSV column and label information
        let featureNames = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        let labelName = "species"
        let columnNames = featureNames + [labelName]
        let classNames = ["Iris setosa", "Iris versicolor", "Iris virginica"]
        
        // Creating dataset using Epochs API.
        let batchsize = 32
        
        // Initialize an IrisBatch dataset from a csv file
        func loadIrisDatasetFromCSV(contentsOf: String, hasHeader: Bool, featureColumns: [Int], labelColumns: [Int]) -> [IrisBatch] {
            let np = Python.import("numpy")
            
            let featuresNP = np.loadtxt(contentsOf, delimiter: ",", skiprows: hasHeader ? 1 : 0, usecols: featureColumns, dtype: Float.numpyScalarTypes.first!)
            guard let featuresTensor = Tensor<Float>(numpy: featuresNP) else {
                fatalError("np.loadtxt result can't be converted to Tensor")
            }
            
            let labelsNP = np.loadtxt(contentsOf, delimiter: ",", skiprows: hasHeader ? 1 : 0, usecols: labelColumns, dtype: Int32.numpyScalarTypes.first!)
            guard let labelsTensor = Tensor<Int32>(numpy: labelsNP) else {
                fatalError("np.loadtxt result can't be converted to Tensor")
            }
            
            return zip(featuresTensor.unstacked(), labelsTensor.unstacked()).map{IrisBatch(features: $0.0, labels: $0.1)}
        }
        return loadIrisDatasetFromCSV(contentsOf: trainDataFilename, hasHeader: true, featureColumns: [0, 1, 2, 3], labelColumns: [4])
    }
    
    func trainModel(batchsize: Int = 32) {
        let plt = Python.import("matplotlib.pyplot")
        let trainingEpochs = TrainingEpochs(samples: trainingDataset, batchSize: batchsize)

    }
}

// A batch of examples from the iris csv.
struct IrisBatch {
    // [batchsize, featureCount] tensor of features.
    let features: Tensor<Float> // TODO: Float32
    
    // [batchsize] tensor of labels.
    let labels: Tensor<Int32>
}

// Conform IrisBatch to Collatable so that it can be loaded into a training epoch
extension IrisBatch: Collatable {
    public init<BatchSamples: Collection>(collating samples: BatchSamples) where BatchSamples.Element == Self {
        // IrisBatch'es are collated by stacking their feature and label tensors along the batch axis to produce a single feature and label tensor
        features = Tensor<Float>(stacking: samples.map{$0.features})
        labels = Tensor<Int32>(stacking: samples.map{$0.labels})
    }
}
