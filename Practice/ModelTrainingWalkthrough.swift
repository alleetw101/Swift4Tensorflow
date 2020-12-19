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
///
/// Training dataset from http://download.tensorflow.org/data/iris_training.csv
class ModelTrainingWalkthrough {
    // Data file path
    let trainDataFilename = "/Users/alan/Documents/Programming/Swift/Swift4Tensorflow/Practice/iris_training.csv"
    
    // CSV column and label information
    let featureNames = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    let labelName = "species"
    let classNames = ["Iris setosa", "Iris versicolor", "Iris virginica"]
    
    // Training dataset list from preProcessData() from csv
    var trainingDataset: [IrisBatch] {get {preProcessData()} set {}}
    
    // Model
    var model = IrisModel()
    
    // Training stats
    var trainAccuracyResults: [Float] = []
    var trainLossResults: [Float] = []
    
    
    // Function to download file (Doesn't work with provided csv link)
    func downloadData() {
        func download(from sourceString: String, to destinationString: String) {
            let source = URL(string: sourceString)!
            let destination = URL(fileURLWithPath: destinationString)
            print(source)
            let data = try! Data.init(contentsOf: source)
            try! data.write(to: destination)
        }
    
    //    download(from: "http://download.tensorflow.org/data/iris_training.csv", to: trainDataFilename)
    }
    
    // Creating dataset using Epochs API.
    func preProcessData() -> [IrisBatch] {
        // Initialize an IrisBatch dataset from a csv file.
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
    
    func trainModel(batchsize: Int = 32, epochCount: Int = 500) {
        let trainingEpochs = TrainingEpochs(samples: trainingDataset, batchSize: batchsize)
        let columnNames = featureNames + [labelName]
        
        trainAccuracyResults = []
        trainLossResults = []
        
        let optimizer = SGD(for: model, learningRate: 0.01)
        
        func accuracy(predictions: Tensor<Int32>, truths: Tensor<Int32>) -> Float {
            return Tensor<Float>(predictions .== truths).mean().scalarized()
        }
        
        
        
    }
        
    func visualizeProcesses() {
        // Examine csv
        let f = Python.open(trainDataFilename)
        for _ in 0..<5 {
            print(Python.next(f).strip())
        }
        f.close()
        
        let trainingEpochs = TrainingEpochs(samples: trainingDataset, batchSize: 32)
        
        let firstTrainEpoch = trainingEpochs.next()!
        let firstTrainBatch = firstTrainEpoch.first!.collated
        let firstTrainFeatures = firstTrainBatch.features
        let firstTrainLabels = firstTrainBatch.labels
        
        let plt = Python.import("matplotlib.pyplot")
        let firstTrainFeaturesTransposed = firstTrainFeatures.transposed()
        let petalLengths = firstTrainFeaturesTransposed[2].scalars
        let sepalLengths = firstTrainFeaturesTransposed[0].scalars

        plt.scatter(petalLengths, sepalLengths, c: firstTrainLabels.array.scalars)
        plt.xlabel("Petal length")
        plt.ylabel("Sepal length")
        plt.show()
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

let hiddensize: Int = 10
struct IrisModel: Layer {
    var layer1 = Dense<Float>(inputSize: 4, outputSize: hiddensize, activation: relu)
    var layer2 = Dense<Float>(inputSize: hiddensize, outputSize: hiddensize, activation: relu)
    var layer3 = Dense<Float>(inputSize: hiddensize, outputSize: 3, activation: relu)
    
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        return input.sequenced(through: layer1, layer2, layer3)
    }
}
