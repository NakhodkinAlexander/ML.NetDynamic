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

        private string modelName = null;
        private string labelFieldName = null;

        public MLModel(MLFeature[] features, string modelName)
        {
            this.modelName = modelName;

            DataSetClass = MLHelper.GenerateDataSetClass(features, modelName, $"{modelName}Namespace");
            LabelClass = MLHelper.GenerateLabelClass($"{modelName}Label", $"{modelName}LabelNamespace");
            this.features = features;
        }

        public TransformerChain<Microsoft.ML.Transforms.KeyToValueMappingTransformer> Train(MLDataRow[] dataSetRows, string labelFieldName)
        {
            this.labelFieldName = labelFieldName;
            List<object> generatedDataSet = CreateDataSet(dataSetRows);
            TypesGenerator = MLHelper.CreateTypesGenarator(DataSetClass, LabelClass);

            IDataView trainingDataView = MLHelper.GetDataView(TypesGenerator, generatedDataSet);
            trainingDataView.Schema.ToList().Add(new DataViewSchema.Column());
            Model = Train(trainingDataView, labelFieldName);
            return Model;
        }

        private List<object> CreateDataSet(MLDataRow[] mLDataSet)
        {
            List<object> generatedDataSet = new List<object>();

            foreach (MLDataRow row in mLDataSet)
            {
                generatedDataSet.Add(ParseRowToObject(row));
            }

            return generatedDataSet;
        }

        public object ParseRowToObject(MLDataRow row)
        {
            object newSet = DataSetClass.GetInstance();
            foreach (var field in row.Data)
            {
                newSet.GetType().GetField(field.Key).SetValue(newSet, field.Value);
            }
            return newSet;
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

        public object Predict(MLDataRow example)
        {
            return Predict(ParseRowToObject(example));
        }
    }
}
