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
    let testDataFilename = "/Users/alan/Documents/Programming/Swift/Swift4Tensorflow/Practice/iris_test.csv"
    
    // CSV column and label information
    let featureNames = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    let labelName = "species"
    let classNames = ["Iris setosa", "Iris versicolor", "Iris virginica"]
    
    // Training dataset list from preProcessData() from csv
    var trainingDataset: [IrisBatch] {get {preProcessData()} set {}}
    
    // Model
    var model = IrisModel()
    var batchSize: Int = 32
    var epochCount: Int = 501
    
    // Training stats
    var trainAccuracyResults: [Float] = []
    var trainLossResults: [Float] = []
    
    
    init (batchsize: Int = 32, epochCount: Int = 501) {
        self.batchSize = batchsize
        self.epochCount = epochCount
    }
    
    // Function to download file (Doesn't work with provided csv link)
    private func downloadData() {
        func download(from sourceString: String, to destinationString: String) {
            let source = URL(string: sourceString)!
            let destination = URL(fileURLWithPath: destinationString)
            print(source)
            let data = try! Data.init(contentsOf: source)
            try! data.write(to: destination)
        }
    
        // download(from: "http://download.tensorflow.org/data/iris_training.csv", to: trainDataFilename)
        // download(from: "http://download.tensorflow.org/data/iris_test.csv", to: testDataFilename)
    }
    
    // Initialize an IrisBatch dataset from a csv file.
    private func loadIrisDatasetFromCSV(contentsOf: String, hasHeader: Bool, featureColumns: [Int], labelColumns: [Int]) -> [IrisBatch] {
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
    
    // Creating dataset using Epochs API.
    private func preProcessData() -> [IrisBatch] {
        return loadIrisDatasetFromCSV(contentsOf: trainDataFilename, hasHeader: true, featureColumns: [0, 1, 2, 3], labelColumns: [4])
    }
    
    private func accuracy(predictions: Tensor<Int32>, truths: Tensor<Int32>) -> Float {
        return Tensor<Float>(predictions .== truths).mean().scalarized()
    }
    
    func trainModel() {
        let trainingEpochs = TrainingEpochs(samples: trainingDataset, batchSize: batchSize)
        
        trainAccuracyResults = []
        trainLossResults = []
        
        let optimizer = SGD(for: model, learningRate: 0.01)
        
        for (epochIndex, epoch) in trainingEpochs.prefix(epochCount).enumerated() {
            var epochLoss: Float = 0
            var epochAccuracy: Float = 0
            var batchCount: Int = 0
            
            for batchSamples in epoch {
                let batch = batchSamples.collated
                let (loss, grad) = valueWithGradient(at: model) { (model: IrisModel) -> Tensor<Float> in
                    let logits = model(batch.features)
                    return softmaxCrossEntropy(logits: logits, labels: batch.labels)
                }
                optimizer.update(&model, along: grad)
                
                let logits = model(batch.features)
                epochAccuracy += accuracy(predictions: logits.argmax(squeezingAxis: 1), truths: batch.labels)
                epochLoss += loss.scalarized()
                batchCount += 1
            }
            epochAccuracy /= Float(batchCount)
            epochLoss /= Float(batchCount)
            trainAccuracyResults.append(epochAccuracy)
            trainLossResults.append(epochLoss)
            if epochIndex % 50 == 0 {
                print("Epoch \(epochIndex): Loss: \(epochLoss), Accuracy: \(epochAccuracy)")
            }
        }
    }
    
    func testModel() {
        let testDataset = loadIrisDatasetFromCSV(contentsOf: testDataFilename, hasHeader: true, featureColumns: [0, 1, 2, 3], labelColumns: [4]).inBatches(of: batchSize)
        
        for batchSamples in testDataset {
            let batch = batchSamples.collated
            let logits = model(batch.features)
            let predictions = logits.argmax(squeezingAxis: 1)
            print("Test batch accuracy: \(accuracy(predictions: predictions, truths: batch.labels))")
        }
    }
        
    func visualizePreProcesses() {
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
    
    func visualizeTraining() {
        let plt = Python.import("matplotlib.pyplot")
        plt.figure(figsize: [12, 8])

        let accuracyAxes = plt.subplot(2, 1, 1)
        accuracyAxes.set_ylabel("Accuracy")
        accuracyAxes.plot(trainAccuracyResults)

        let lossAxes = plt.subplot(2, 1, 2)
        lossAxes.set_ylabel("Loss")
        lossAxes.set_xlabel("Epoch")
        lossAxes.plot(trainLossResults)

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
