using Decisions.ML;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Data;
using System.Collections.Generic;
using System.Linq;

namespace ConsoleApp
{
    public class MLModel
    {
        public ClassGenerator DataSetClass { get; set; }
        public ClassGenerator LabelClass { get; set; }
        MLTypesGenerator TypesGenerator { get; set; }

        public TransformerChain<Microsoft.ML.Transforms.KeyToValueMappingTransformer> Model { get; set; }
        public IDataView TrainingDataView { get; set; }
        MLFeature[] features;

        public MLModel(MLDataSet mLDataSet, string modelName)
        {
            DataSetClass = MLHelper.GenerateDataSetClass(mLDataSet.Features, modelName, $"{modelName}Namespace");
            LabelClass = MLHelper.GenerateLabelClass($"{modelName}Label", $"{modelName}LabelNamespace");
            this.features = mLDataSet.Features;

            List<object> generatedDataSet = CreateDataSet(mLDataSet);
            MLTypesGenerator typesGenerator = MLHelper.CreateTypesGenarator(DataSetClass, LabelClass);

            IDataView trainingDataView = MLHelper.GetDataView(typesGenerator, generatedDataSet);
            trainingDataView.Schema.ToList().Add(new DataViewSchema.Column());
            Model = Train(trainingDataView, "Label");
        }

        private List<object> CreateDataSet(MLDataSet mLDataSet)
        {
            List<object> generatedDataSet = new List<object>();

            foreach (MLDataRow row in mLDataSet.Rows)
            {
                object newSet = DataSetClass.GetInstance();
                foreach (var field in row.Data)
                {
                    newSet.GetType().GetField(field.Key).SetValue(newSet, field.Value);
                }
            }

            return generatedDataSet;

        }

        private TransformerChain<Microsoft.ML.Transforms.KeyToValueMappingTransformer> Train(IDataView trainingDataView, string mapValueToKey)
        {
            MLContext mlContext = new MLContext();

            List<string> fields = new List<string>();

            features.ToList()
                .Where(p => p.Name != mapValueToKey).ToList()
                .ForEach((p) =>
                    {
                        fields.Add(p.Name);
                    });

            var pipeline = mlContext.Transforms.Conversion.MapValueToKey(mapValueToKey)
                .Append(mlContext.Transforms.Concatenate("Features", fields.ToArray()))
                .AppendCacheCheckpoint(mlContext)
                .Append(mlContext.MulticlassClassification.Trainers.StochasticDualCoordinateAscent(labelColumnName: mapValueToKey, featureColumnName: "Features"))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            return pipeline.Fit(trainingDataView);
        }

        public object Predict(object example)
        {
            var predictionEngine = MLHelper.GetPredictionEngine(TypesGenerator, Model);
            var methodInfo = predictionEngine.GetType().GetMethod("Predict", new[] { DataSetClass.ClassType });

            var prediction = MLHelper.Predict(predictionEngine, DataSetClass, example);

            return prediction;
        }
    }
}
