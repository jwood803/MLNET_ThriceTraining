using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using System;
using System.IO;

namespace MLNET_ThriceTraining
{
    class Program
    {
        private static string TRAIN_PATH = "./house_train.csv";
        private static string TEST_PATH = "./house_test.csv";
        private static string VALIDATION_PATH = "./house_validate.csv";
        private static string LABEL_NAME = "median_house_value";

        static void Main(string[] args)
        {
            var context = new MLContext();

            // Text loader from scratch
            //var textLoaderOptions = new TextLoader.Options
            //{
            //    Separators = new[] {','},
            //    HasHeader = true,
            //    Columns = new[]
            //    {
            //        new TextLoader.Column("Features", DataKind.Single, 0, 7),
            //        new TextLoader.Column("Label", DataKind.Single, 8),
            //        new TextLoader.Column("OceanProximity", DataKind.String, 9)
            //    }
            //};

            //var loader = context.Data.CreateTextLoader(textLoaderOptions);
            //var loaderData = loader.Load(TRAIN_PATH);

            // Text loader from inference
            var inference = context.Auto().InferColumns(TRAIN_PATH, hasHeader: true, separatorChar: ',', labelColumnIndex: 8);

            var loader = context.Data.CreateTextLoader(inference.TextLoaderOptions);

            var trainData = loader.Load(TRAIN_PATH);
            var validationData = loader.Load(VALIDATION_PATH);
            var testData = loader.Load(TEST_PATH);

            var experimentSettings = new RegressionExperimentSettings
            {
                MaxExperimentTimeInSeconds = 60,
                OptimizingMetric = RegressionMetric.RSquared
            };

            // First training (on train data)
            var experimentResult = context.Auto()
                .CreateRegressionExperiment(experimentSettings)
                .Execute(trainData, validationData, labelColumnName: LABEL_NAME);
            
            var firstPredictions = experimentResult.BestRun.Model.Transform(testData);
            var firstMetrics = context.Regression.Evaluate(firstPredictions, LABEL_NAME);
            PrintRSquared(firstMetrics);

            // Second training (on train + validation data)
            var trainPlusValidationData = loader.Load(new MultiFileSource(TRAIN_PATH, VALIDATION_PATH));
            var refitModel = experimentResult.BestRun.Estimator.Fit(trainPlusValidationData);
            IDataView predictionsRefitOnTrainPlusValidation = refitModel.Transform(testData);
            var secondMetrics = context.Regression.Evaluate(predictionsRefitOnTrainPlusValidation, LABEL_NAME);
            PrintRSquared(secondMetrics);

            // Third training (on train + test + validation data)
            var trainPlusValidationPlusTestDataView = loader.Load(new MultiFileSource(TRAIN_PATH, VALIDATION_PATH, TEST_PATH));
            var refitModelOnValidationData = experimentResult.BestRun.Estimator.Fit(trainPlusValidationPlusTestDataView);

            context.Model.Save(refitModelOnValidationData, trainData.Schema, "./house_model.zip");
        }

        public static void PrintRSquared(RegressionMetrics metrics)
        {
            Console.WriteLine($"R^2: {metrics.RSquared}");
            Console.Write(Environment.NewLine);
        }
    }
}
