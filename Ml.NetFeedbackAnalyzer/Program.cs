using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Api;

namespace ConsoleApp10
{
    class FeedBackTrainingData
    {
        [Column("0", "Label")]
        public bool IsGood { get; set; }

        [Column(ordinal: "1")]
        public string FeedBackText { get; set; }
    }

    class FeedBackPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool IsGood { get; set; }
    }

    class Program
    {
        private static List<FeedBackTrainingData> GetTrainingData()
        {
            var trainingData = new List<FeedBackTrainingData>
            {
                new FeedBackTrainingData {FeedBackText = "this is good", IsGood = true},
                new FeedBackTrainingData {FeedBackText = "this is not good", IsGood = true},
                new FeedBackTrainingData {FeedBackText = "this is horrible", IsGood = false},
                new FeedBackTrainingData {FeedBackText = "it very Average", IsGood = true},
                new FeedBackTrainingData {FeedBackText = "bad horrible", IsGood = false},
                new FeedBackTrainingData {FeedBackText = "well ok ok", IsGood = true},
                new FeedBackTrainingData {FeedBackText = "shitty terrible", IsGood = false},
                new FeedBackTrainingData {FeedBackText = "so nice", IsGood = true},
                new FeedBackTrainingData {FeedBackText = "cool nice", IsGood = true},
                new FeedBackTrainingData {FeedBackText = "sweet and nice", IsGood = true},
                new FeedBackTrainingData {FeedBackText = "nice and good", IsGood = true},
                new FeedBackTrainingData {FeedBackText = "very good", IsGood = true},
                new FeedBackTrainingData {FeedBackText = "quiet average", IsGood = true},
                new FeedBackTrainingData {FeedBackText = "god horrible", IsGood = false},
                new FeedBackTrainingData {FeedBackText = "average and ok", IsGood = true},
                new FeedBackTrainingData {FeedBackText = "bad and hell", IsGood = false},
                new FeedBackTrainingData {FeedBackText = "this nice but better can be done", IsGood = true},
                new FeedBackTrainingData {FeedBackText = "bad bad", IsGood = false},
                new FeedBackTrainingData {FeedBackText = "till now it looks nice", IsGood = true},
                new FeedBackTrainingData {FeedBackText = "shit", IsGood = false},
                new FeedBackTrainingData {FeedBackText = "oh this is shit", IsGood = false},
                new FeedBackTrainingData {FeedBackText = "sucks", IsGood = false},
                new FeedBackTrainingData {FeedBackText = "i know you sucks", IsGood = false}
            };

            return trainingData;
        }

        private static List<FeedBackTrainingData> GetTestData()
        {
            var testData = new List<FeedBackTrainingData>
            {
                new FeedBackTrainingData {FeedBackText = "good", IsGood = true},
                new FeedBackTrainingData {FeedBackText = "horrible terrible", IsGood = false},
                new FeedBackTrainingData {FeedBackText = "nice", IsGood = true},
                new FeedBackTrainingData {FeedBackText = "shit", IsGood = false},
                new FeedBackTrainingData {FeedBackText = "sweet", IsGood = true},
                new FeedBackTrainingData {FeedBackText = "average", IsGood = true},
                new FeedBackTrainingData {FeedBackText = "get lost", IsGood = false},
            };

            return testData;
        }

        static void Main(string[] args)
        {
            //Step 1: Create training data
            var trainingData = GetTrainingData();

            //Step 2: Transform our data to ML Context
            var mlContext = new MLContext();
            IDataView dataView = mlContext.CreateStreamingDataView(trainingData);

            //Step3: Define Workflow/pipeline.
            var pipeline = mlContext.Transforms.Text
                    .FeaturizeText(nameof(FeedBackTrainingData.FeedBackText), "Features")
                    .Append(mlContext.BinaryClassification.Trainers.FastTree(numLeaves: 50, numTrees: 50, minDatapointsInLeaves: 1));

            //Step5: Train the algorithm to create model.
            Console.WriteLine("Successfully trained algorithm using total '" + trainingData.Count + "' training data and Model is ready for use.");
            var model = pipeline.Fit(dataView);


            var testData = GetTestData();
            IDataView dataView1 = mlContext.CreateStreamingDataView(testData);

            var predictions = model.Transform(dataView1);
            var metrics = mlContext.BinaryClassification.Evaluate(predictions);


            Console.WriteLine("Model accuracy against test data is: " + Math.Round((metrics.Accuracy / 1) * 100, 2) + "%");

            // Step 7 :- use the model
            while (true)
            {
                Console.WriteLine(Environment.NewLine + "Enter a feedback to test:");
                var userInputString = Console.ReadLine();

                var algorithmPredictionFunction = model.MakePredictionFunction<FeedBackTrainingData, FeedBackPrediction>(mlContext);

                var feedbackInput = new FeedBackTrainingData { FeedBackText = userInputString };
                var predictedResult = algorithmPredictionFunction.Predict(feedbackInput);
                var userFeedback = predictedResult.IsGood ? "'good' feedback." : "'bad' feedback.";
                Console.WriteLine("Algorithm predicts given input as a " + userFeedback);
            }
        }
    }
}
