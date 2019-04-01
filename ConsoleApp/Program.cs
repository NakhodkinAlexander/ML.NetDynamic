using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using DynamicStructure;
using System.Collections;

namespace ConsoleApp
{
    class Program
    {
        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext();

            ClassGenerator classGenerator = new ClassGenerator("GeneratedIris", "CustomClass");
            classGenerator.AddField("SepalLength", typeof(float), System.CodeDom.MemberAttributes.Public);
            classGenerator.AddField("SepalWidth", typeof(float), System.CodeDom.MemberAttributes.Public);
            classGenerator.AddField("PetalLength", typeof(float), System.CodeDom.MemberAttributes.Public);
            classGenerator.AddField("PetalWidth", typeof(float), System.CodeDom.MemberAttributes.Public);
            classGenerator.AddField("Label", typeof(string), System.CodeDom.MemberAttributes.Public);
            classGenerator.Compile();

            List<object> generatedDataSet = new List<object>();

            dataset.ToList().ForEach((d) =>
            {
                generatedDataSet.Add(GetDynamicClass(d, classGenerator.GetInstance()));
            });

            var instance = classGenerator.GetInstance().GetType();
            DataViewGenerator listGenerator = new DataViewGenerator("ListIris", "CustomGenerator", instance, classGenerator.NamespaceName);
            var type = listGenerator.GeneratorType;
            var methodInfo = type.GetMethod("GetDataView");
            var dataView = methodInfo.Invoke(null, new object[] { generatedDataSet.ToList() });

            IDataView trainingDataView = (IDataView)dataView;
            trainingDataView.Schema.ToList().Add(new DataViewSchema.Column());

            var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label")
                .Append(mlContext.Transforms.Concatenate("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"))
                .AppendCacheCheckpoint(mlContext)
                .Append(mlContext.MulticlassClassification.Trainers.StochasticDualCoordinateAscent(labelColumnName: "Label", featureColumnName: "Features"))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));


            TransformerChain<Microsoft.ML.Transforms.KeyToValueMappingTransformer> model = pipeline.Fit(trainingDataView);

            var prediction = model.CreatePredictionEngine<IrisData, IrisPrediction>(mlContext).Predict(
                new IrisData()
                {
                    SepalLength = 5.9f,
                    SepalWidth = 3.0f,
                    PetalLength = 5.1f,
                    PetalWidth = 1.8f,
                });

            Console.WriteLine(prediction.PredictedLabels);

            Console.ReadLine();
        }

        static IList CreateList(Type t)
        {
            var listType = typeof(List<>);
            var constructedListType = listType.MakeGenericType(t);

            var instance = (IList)Activator.CreateInstance(constructedListType);
            return instance;
        }

        public class IrisData
        {
            public float SepalLength;

            public float SepalWidth;

            public float PetalLength;

            public float PetalWidth;

            public string Label;
        }

        public class IrisPrediction
        {
            [ColumnName("PredictedLabel")]
            public string PredictedLabels;
        }

        private static object GetDynamicClass(IrisData irisData, object newIris)
        {
            Type instanceType = newIris.GetType();
            irisData.GetType().GetFields().ToList().ForEach((field) =>
            {
                instanceType.GetField(field.Name).SetValue(newIris, field.GetValue(irisData));
            });

            return newIris;
        }

        private static readonly IrisData[] dataset = new IrisData[] {
            new IrisData() { SepalLength = 5.1f, SepalWidth = 3.5f, PetalLength = 1.4f, PetalWidth = 0.2f, Label = "Iris-setosa"},
            new IrisData() { SepalLength = 4.9f, SepalWidth = 3.0f, PetalLength = 1.4f, PetalWidth = 0.2f, Label = "Iris-setosa"},
            new IrisData() { SepalLength = 4.7f, SepalWidth = 3.2f, PetalLength = 1.3f, PetalWidth = 0.2f, Label = "Iris-setosa"},
            new IrisData() { SepalLength = 4.6f, SepalWidth = 3.1f, PetalLength = 1.5f, PetalWidth = 0.2f, Label = "Iris-setosa"},
            new IrisData() { SepalLength = 5.0f, SepalWidth = 3.6f, PetalLength = 1.4f, PetalWidth = 0.2f, Label = "Iris-setosa"},
            new IrisData() { SepalLength = 5.4f, SepalWidth = 3.9f, PetalLength = 1.7f, PetalWidth = 0.4f, Label = "Iris-setosa"},
            new IrisData() { SepalLength = 4.6f, SepalWidth = 3.4f, PetalLength = 1.4f, PetalWidth = 0.3f, Label = "Iris-setosa"},
            new IrisData() { SepalLength = 5.0f, SepalWidth = 3.4f, PetalLength = 1.5f, PetalWidth = 0.2f, Label = "Iris-setosa"},
            new IrisData() { SepalLength = 4.4f, SepalWidth = 2.9f, PetalLength = 1.4f, PetalWidth = 0.2f, Label = "Iris-setosa"},
            new IrisData() { SepalLength = 4.9f, SepalWidth = 3.1f, PetalLength = 1.5f, PetalWidth = 0.1f, Label = "Iris-setosa"},
            new IrisData() { SepalLength = 5.4f, SepalWidth = 3.7f, PetalLength = 1.5f, PetalWidth = 0.2f, Label = "Iris-setosa"},
            new IrisData() { SepalLength = 4.8f, SepalWidth = 3.4f, PetalLength = 1.6f, PetalWidth = 0.2f, Label = "Iris-setosa"},
            new IrisData() { SepalLength = 4.8f, SepalWidth = 3.0f, PetalLength = 1.4f, PetalWidth = 0.1f, Label = "Iris-setosa"},
            new IrisData() { SepalLength = 4.3f, SepalWidth = 3.0f, PetalLength = 1.1f, PetalWidth = 0.1f, Label = "Iris-setosa"},
            new IrisData() { SepalLength = 5.8f, SepalWidth = 4.0f, PetalLength = 1.2f, PetalWidth = 0.2f, Label = "Iris-setosa"},
            new IrisData() { SepalLength = 5.7f, SepalWidth = 4.4f, PetalLength = 1.5f, PetalWidth = 0.4f, Label = "Iris-setosa"},
            new IrisData() { SepalLength = 5.4f, SepalWidth = 3.9f, PetalLength = 1.3f, PetalWidth = 0.4f, Label = "Iris-setosa"},
            new IrisData() { SepalLength = 5.1f, SepalWidth = 3.5f, PetalLength = 1.4f, PetalWidth = 0.3f, Label = "Iris-setosa"},
            new IrisData() { SepalLength = 5.7f, SepalWidth = 3.8f, PetalLength = 1.7f, PetalWidth = 0.3f, Label = "Iris-setosa"},
            new IrisData() { SepalLength = 5.1f, SepalWidth = 3.8f, PetalLength = 1.5f, PetalWidth = 0.3f, Label = "Iris-setosa"},
            new IrisData() { SepalLength = 5.4f, SepalWidth = 3.4f, PetalLength = 1.7f, PetalWidth = 0.2f, Label = "Iris-setosa"},
            new IrisData() { SepalLength = 5.1f, SepalWidth = 3.7f, PetalLength = 1.5f, PetalWidth = 0.4f, Label = "Iris-setosa"},
            new IrisData() { SepalLength = 4.6f, SepalWidth = 3.6f, PetalLength = 1.0f, PetalWidth = 0.2f, Label = "Iris-setosa"},
            new IrisData() { SepalLength = 5.1f, SepalWidth = 3.3f, PetalLength = 1.7f, PetalWidth = 0.5f, Label = "Iris-setosa"},
            new IrisData() { SepalLength = 4.8f, SepalWidth = 3.4f, PetalLength = 1.9f, PetalWidth = 0.2f, Label = "Iris-setosa"},
            new IrisData() { SepalLength = 5.0f, SepalWidth = 3.0f, PetalLength = 1.6f, PetalWidth = 0.2f, Label = "Iris-setosa"},
            new IrisData() { SepalLength = 5.0f, SepalWidth = 3.4f, PetalLength = 1.6f, PetalWidth = 0.4f, Label = "Iris-setosa"},
            new IrisData() { SepalLength = 5.2f, SepalWidth = 3.5f, PetalLength = 1.5f, PetalWidth = 0.2f, Label = "Iris-setosa"},
            new IrisData() { SepalLength = 5.2f, SepalWidth = 3.4f, PetalLength = 1.4f, PetalWidth = 0.2f, Label = "Iris-setosa"},
            new IrisData() { SepalLength = 4.7f, SepalWidth = 3.2f, PetalLength = 1.6f, PetalWidth = 0.2f, Label = "Iris-setosa"},
            new IrisData() { SepalLength = 4.8f, SepalWidth = 3.1f, PetalLength = 1.6f, PetalWidth = 0.2f, Label = "Iris-setosa"},
            new IrisData() { SepalLength = 5.4f, SepalWidth = 3.4f, PetalLength = 1.5f, PetalWidth = 0.4f, Label = "Iris-setosa"},
            new IrisData() { SepalLength = 5.2f, SepalWidth = 4.1f, PetalLength = 1.5f, PetalWidth = 0.1f, Label = "Iris-setosa"},
            new IrisData() { SepalLength = 5.5f, SepalWidth = 4.2f, PetalLength = 1.4f, PetalWidth = 0.2f, Label = "Iris-setosa"},
            new IrisData() { SepalLength = 4.9f, SepalWidth = 3.1f, PetalLength = 1.5f, PetalWidth = 0.1f, Label = "Iris-setosa"},
            new IrisData() { SepalLength = 5.0f, SepalWidth = 3.2f, PetalLength = 1.2f, PetalWidth = 0.2f, Label = "Iris-setosa"},
            new IrisData() { SepalLength = 5.5f, SepalWidth = 3.5f, PetalLength = 1.3f, PetalWidth = 0.2f, Label = "Iris-setosa"},
            new IrisData() { SepalLength = 4.9f, SepalWidth = 3.1f, PetalLength = 1.5f, PetalWidth = 0.1f, Label = "Iris-setosa"},
            new IrisData() { SepalLength = 4.4f, SepalWidth = 3.0f, PetalLength = 1.3f, PetalWidth = 0.2f, Label = "Iris-setosa"},
            new IrisData() { SepalLength = 5.1f, SepalWidth = 3.4f, PetalLength = 1.5f, PetalWidth = 0.2f, Label = "Iris-setosa"},
            new IrisData() { SepalLength = 5.0f, SepalWidth = 3.5f, PetalLength = 1.3f, PetalWidth = 0.3f, Label = "Iris-setosa"},
            new IrisData() { SepalLength = 4.5f, SepalWidth = 2.3f, PetalLength = 1.3f, PetalWidth = 0.3f, Label = "Iris-setosa"},
            new IrisData() { SepalLength = 4.4f, SepalWidth = 3.2f, PetalLength = 1.3f, PetalWidth = 0.2f, Label = "Iris-setosa"},
            new IrisData() { SepalLength = 5.0f, SepalWidth = 3.5f, PetalLength = 1.6f, PetalWidth = 0.6f, Label = "Iris-setosa"},
            new IrisData() { SepalLength = 5.1f, SepalWidth = 3.8f, PetalLength = 1.9f, PetalWidth = 0.4f, Label = "Iris-setosa"},
            new IrisData() { SepalLength = 4.8f, SepalWidth = 3.0f, PetalLength = 1.4f, PetalWidth = 0.3f, Label = "Iris-setosa"},
            new IrisData() { SepalLength = 5.1f, SepalWidth = 3.8f, PetalLength = 1.6f, PetalWidth = 0.2f, Label = "Iris-setosa"},
            new IrisData() { SepalLength = 4.6f, SepalWidth = 3.2f, PetalLength = 1.4f, PetalWidth = 0.2f, Label = "Iris-setosa"},
            new IrisData() { SepalLength = 5.3f, SepalWidth = 3.7f, PetalLength = 1.5f, PetalWidth = 0.2f, Label = "Iris-setosa"},
            new IrisData() { SepalLength = 5.0f, SepalWidth = 3.3f, PetalLength = 1.4f, PetalWidth = 0.2f, Label = "Iris-setosa"},
            new IrisData() { SepalLength = 7.0f, SepalWidth = 3.2f, PetalLength = 4.7f, PetalWidth = 1.4f, Label = "Iris-versicolor"},
            new IrisData() { SepalLength = 6.4f, SepalWidth = 3.2f, PetalLength = 4.5f, PetalWidth = 1.5f, Label = "Iris-versicolor"},
            new IrisData() { SepalLength = 6.9f, SepalWidth = 3.1f, PetalLength = 4.9f, PetalWidth = 1.5f, Label = "Iris-versicolor"},
            new IrisData() { SepalLength = 5.5f, SepalWidth = 2.3f, PetalLength = 4.0f, PetalWidth = 1.3f, Label = "Iris-versicolor"},
            new IrisData() { SepalLength = 6.5f, SepalWidth = 2.8f, PetalLength = 4.6f, PetalWidth = 1.5f, Label = "Iris-versicolor"},
            new IrisData() { SepalLength = 5.7f, SepalWidth = 2.8f, PetalLength = 4.5f, PetalWidth = 1.3f, Label = "Iris-versicolor"},
            new IrisData() { SepalLength = 6.3f, SepalWidth = 3.3f, PetalLength = 4.7f, PetalWidth = 1.6f, Label = "Iris-versicolor"},
            new IrisData() { SepalLength = 4.9f, SepalWidth = 2.4f, PetalLength = 3.3f, PetalWidth = 1.0f, Label = "Iris-versicolor"},
            new IrisData() { SepalLength = 6.6f, SepalWidth = 2.9f, PetalLength = 4.6f, PetalWidth = 1.3f, Label = "Iris-versicolor"},
            new IrisData() { SepalLength = 5.2f, SepalWidth = 2.7f, PetalLength = 3.9f, PetalWidth = 1.4f, Label = "Iris-versicolor"},
            new IrisData() { SepalLength = 5.0f, SepalWidth = 2.0f, PetalLength = 3.5f, PetalWidth = 1.0f, Label = "Iris-versicolor"},
            new IrisData() { SepalLength = 5.9f, SepalWidth = 3.0f, PetalLength = 4.2f, PetalWidth = 1.5f, Label = "Iris-versicolor"},
            new IrisData() { SepalLength = 6.0f, SepalWidth = 2.2f, PetalLength = 4.0f, PetalWidth = 1.0f, Label = "Iris-versicolor"},
            new IrisData() { SepalLength = 6.1f, SepalWidth = 2.9f, PetalLength = 4.7f, PetalWidth = 1.4f, Label = "Iris-versicolor"},
            new IrisData() { SepalLength = 5.6f, SepalWidth = 2.9f, PetalLength = 3.6f, PetalWidth = 1.3f, Label = "Iris-versicolor"},
            new IrisData() { SepalLength = 6.7f, SepalWidth = 3.1f, PetalLength = 4.4f, PetalWidth = 1.4f, Label = "Iris-versicolor"},
            new IrisData() { SepalLength = 5.6f, SepalWidth = 3.0f, PetalLength = 4.5f, PetalWidth = 1.5f, Label = "Iris-versicolor"},
            new IrisData() { SepalLength = 5.8f, SepalWidth = 2.7f, PetalLength = 4.1f, PetalWidth = 1.0f, Label = "Iris-versicolor"},
            new IrisData() { SepalLength = 6.2f, SepalWidth = 2.2f, PetalLength = 4.5f, PetalWidth = 1.5f, Label = "Iris-versicolor"},
            new IrisData() { SepalLength = 5.6f, SepalWidth = 2.5f, PetalLength = 3.9f, PetalWidth = 1.1f, Label = "Iris-versicolor"},
            new IrisData() { SepalLength = 5.9f, SepalWidth = 3.2f, PetalLength = 4.8f, PetalWidth = 1.8f, Label = "Iris-versicolor"},
            new IrisData() { SepalLength = 6.1f, SepalWidth = 2.8f, PetalLength = 4.0f, PetalWidth = 1.3f, Label = "Iris-versicolor"},
            new IrisData() { SepalLength = 6.3f, SepalWidth = 2.5f, PetalLength = 4.9f, PetalWidth = 1.5f, Label = "Iris-versicolor"},
            new IrisData() { SepalLength = 6.1f, SepalWidth = 2.8f, PetalLength = 4.7f, PetalWidth = 1.2f, Label = "Iris-versicolor"},
            new IrisData() { SepalLength = 6.4f, SepalWidth = 2.9f, PetalLength = 4.3f, PetalWidth = 1.3f, Label = "Iris-versicolor"},
            new IrisData() { SepalLength = 6.6f, SepalWidth = 3.0f, PetalLength = 4.4f, PetalWidth = 1.4f, Label = "Iris-versicolor"},
            new IrisData() { SepalLength = 6.8f, SepalWidth = 2.8f, PetalLength = 4.8f, PetalWidth = 1.4f, Label = "Iris-versicolor"},
            new IrisData() { SepalLength = 6.7f, SepalWidth = 3.0f, PetalLength = 5.0f, PetalWidth = 1.7f, Label = "Iris-versicolor"},
            new IrisData() { SepalLength = 6.0f, SepalWidth = 2.9f, PetalLength = 4.5f, PetalWidth = 1.5f, Label = "Iris-versicolor"},
            new IrisData() { SepalLength = 5.7f, SepalWidth = 2.6f, PetalLength = 3.5f, PetalWidth = 1.0f, Label = "Iris-versicolor"},
            new IrisData() { SepalLength = 5.5f, SepalWidth = 2.4f, PetalLength = 3.8f, PetalWidth = 1.1f, Label = "Iris-versicolor"},
            new IrisData() { SepalLength = 5.5f, SepalWidth = 2.4f, PetalLength = 3.7f, PetalWidth = 1.0f, Label = "Iris-versicolor"},
            new IrisData() { SepalLength = 5.8f, SepalWidth = 2.7f, PetalLength = 3.9f, PetalWidth = 1.2f, Label = "Iris-versicolor"},
            new IrisData() { SepalLength = 6.0f, SepalWidth = 2.7f, PetalLength = 5.1f, PetalWidth = 1.6f, Label = "Iris-versicolor"},
            new IrisData() { SepalLength = 5.4f, SepalWidth = 3.0f, PetalLength = 4.5f, PetalWidth = 1.5f, Label = "Iris-versicolor"},
            new IrisData() { SepalLength = 6.0f, SepalWidth = 3.4f, PetalLength = 4.5f, PetalWidth = 1.6f, Label = "Iris-versicolor"},
            new IrisData() { SepalLength = 6.7f, SepalWidth = 3.1f, PetalLength = 4.7f, PetalWidth = 1.5f, Label = "Iris-versicolor"},
            new IrisData() { SepalLength = 6.3f, SepalWidth = 2.3f, PetalLength = 4.4f, PetalWidth = 1.3f, Label = "Iris-versicolor"},
            new IrisData() { SepalLength = 5.6f, SepalWidth = 3.0f, PetalLength = 4.1f, PetalWidth = 1.3f, Label = "Iris-versicolor"},
            new IrisData() { SepalLength = 5.5f, SepalWidth = 2.5f, PetalLength = 4.0f, PetalWidth = 1.3f, Label = "Iris-versicolor"},
            new IrisData() { SepalLength = 5.5f, SepalWidth = 2.6f, PetalLength = 4.4f, PetalWidth = 1.2f, Label = "Iris-versicolor"},
            new IrisData() { SepalLength = 6.1f, SepalWidth = 3.0f, PetalLength = 4.6f, PetalWidth = 1.4f, Label = "Iris-versicolor"},
            new IrisData() { SepalLength = 5.8f, SepalWidth = 2.6f, PetalLength = 4.0f, PetalWidth = 1.2f, Label = "Iris-versicolor"},
            new IrisData() { SepalLength = 5.0f, SepalWidth = 2.3f, PetalLength = 3.3f, PetalWidth = 1.0f, Label = "Iris-versicolor"},
            new IrisData() { SepalLength = 5.6f, SepalWidth = 2.7f, PetalLength = 4.2f, PetalWidth = 1.3f, Label = "Iris-versicolor"},
            new IrisData() { SepalLength = 5.7f, SepalWidth = 3.0f, PetalLength = 4.2f, PetalWidth = 1.2f, Label = "Iris-versicolor"},
            new IrisData() { SepalLength = 5.7f, SepalWidth = 2.9f, PetalLength = 4.2f, PetalWidth = 1.3f, Label = "Iris-versicolor"},
            new IrisData() { SepalLength = 6.2f, SepalWidth = 2.9f, PetalLength = 4.3f, PetalWidth = 1.3f, Label = "Iris-versicolor"},
            new IrisData() { SepalLength = 5.1f, SepalWidth = 2.5f, PetalLength = 3.0f, PetalWidth = 1.1f, Label = "Iris-versicolor"},
            new IrisData() { SepalLength = 5.7f, SepalWidth = 2.8f, PetalLength = 4.1f, PetalWidth = 1.3f, Label = "Iris-versicolor"},
            new IrisData() { SepalLength = 6.3f, SepalWidth = 3.3f, PetalLength = 6.0f, PetalWidth = 2.5f, Label = "Iris-virginica"},
            new IrisData() { SepalLength = 5.8f, SepalWidth = 2.7f, PetalLength = 5.1f, PetalWidth = 1.9f, Label = "Iris-virginica"},
            new IrisData() { SepalLength = 7.1f, SepalWidth = 3.0f, PetalLength = 5.9f, PetalWidth = 2.1f, Label = "Iris-virginica"},
            new IrisData() { SepalLength = 6.3f, SepalWidth = 2.9f, PetalLength = 5.6f, PetalWidth = 1.8f, Label = "Iris-virginica"},
            new IrisData() { SepalLength = 6.5f, SepalWidth = 3.0f, PetalLength = 5.8f, PetalWidth = 2.2f, Label = "Iris-virginica"},
            new IrisData() { SepalLength = 7.6f, SepalWidth = 3.0f, PetalLength = 6.6f, PetalWidth = 2.1f, Label = "Iris-virginica"},
            new IrisData() { SepalLength = 4.9f, SepalWidth = 2.5f, PetalLength = 4.5f, PetalWidth = 1.7f, Label = "Iris-virginica"},
            new IrisData() { SepalLength = 7.3f, SepalWidth = 2.9f, PetalLength = 6.3f, PetalWidth = 1.8f, Label = "Iris-virginica"},
            new IrisData() { SepalLength = 6.7f, SepalWidth = 2.5f, PetalLength = 5.8f, PetalWidth = 1.8f, Label = "Iris-virginica"},
            new IrisData() { SepalLength = 7.2f, SepalWidth = 3.6f, PetalLength = 6.1f, PetalWidth = 2.5f, Label = "Iris-virginica"},
            new IrisData() { SepalLength = 6.5f, SepalWidth = 3.2f, PetalLength = 5.1f, PetalWidth = 2.0f, Label = "Iris-virginica"},
            new IrisData() { SepalLength = 6.4f, SepalWidth = 2.7f, PetalLength = 5.3f, PetalWidth = 1.9f, Label = "Iris-virginica"},
            new IrisData() { SepalLength = 6.8f, SepalWidth = 3.0f, PetalLength = 5.5f, PetalWidth = 2.1f, Label = "Iris-virginica"},
            new IrisData() { SepalLength = 5.7f, SepalWidth = 2.5f, PetalLength = 5.0f, PetalWidth = 2.0f, Label = "Iris-virginica"},
            new IrisData() { SepalLength = 5.8f, SepalWidth = 2.8f, PetalLength = 5.1f, PetalWidth = 2.4f, Label = "Iris-virginica"},
            new IrisData() { SepalLength = 6.4f, SepalWidth = 3.2f, PetalLength = 5.3f, PetalWidth = 2.3f, Label = "Iris-virginica"},
            new IrisData() { SepalLength = 6.5f, SepalWidth = 3.0f, PetalLength = 5.5f, PetalWidth = 1.8f, Label = "Iris-virginica"},
            new IrisData() { SepalLength = 7.7f, SepalWidth = 3.8f, PetalLength = 6.7f, PetalWidth = 2.2f, Label = "Iris-virginica"},
            new IrisData() { SepalLength = 7.7f, SepalWidth = 2.6f, PetalLength = 6.9f, PetalWidth = 2.3f, Label = "Iris-virginica"},
            new IrisData() { SepalLength = 6.0f, SepalWidth = 2.2f, PetalLength = 5.0f, PetalWidth = 1.5f, Label = "Iris-virginica"},
            new IrisData() { SepalLength = 6.9f, SepalWidth = 3.2f, PetalLength = 5.7f, PetalWidth = 2.3f, Label = "Iris-virginica"},
            new IrisData() { SepalLength = 5.6f, SepalWidth = 2.8f, PetalLength = 4.9f, PetalWidth = 2.0f, Label = "Iris-virginica"},
            new IrisData() { SepalLength = 7.7f, SepalWidth = 2.8f, PetalLength = 6.7f, PetalWidth = 2.0f, Label = "Iris-virginica"},
            new IrisData() { SepalLength = 6.3f, SepalWidth = 2.7f, PetalLength = 4.9f, PetalWidth = 1.8f, Label = "Iris-virginica"},
            new IrisData() { SepalLength = 6.7f, SepalWidth = 3.3f, PetalLength = 5.7f, PetalWidth = 2.1f, Label = "Iris-virginica"},
            new IrisData() { SepalLength = 7.2f, SepalWidth = 3.2f, PetalLength = 6.0f, PetalWidth = 1.8f, Label = "Iris-virginica"},
            new IrisData() { SepalLength = 6.2f, SepalWidth = 2.8f, PetalLength = 4.8f, PetalWidth = 1.8f, Label = "Iris-virginica"},
            new IrisData() { SepalLength = 6.1f, SepalWidth = 3.0f, PetalLength = 4.9f, PetalWidth = 1.8f, Label = "Iris-virginica"},
            new IrisData() { SepalLength = 6.4f, SepalWidth = 2.8f, PetalLength = 5.6f, PetalWidth = 2.1f, Label = "Iris-virginica"},
            new IrisData() { SepalLength = 7.2f, SepalWidth = 3.0f, PetalLength = 5.8f, PetalWidth = 1.6f, Label = "Iris-virginica"},
            new IrisData() { SepalLength = 7.4f, SepalWidth = 2.8f, PetalLength = 6.1f, PetalWidth = 1.9f, Label = "Iris-virginica"},
            new IrisData() { SepalLength = 7.9f, SepalWidth = 3.8f, PetalLength = 6.4f, PetalWidth = 2.0f, Label = "Iris-virginica"},
            new IrisData() { SepalLength = 6.4f, SepalWidth = 2.8f, PetalLength = 5.6f, PetalWidth = 2.2f, Label = "Iris-virginica"},
            new IrisData() { SepalLength = 6.3f, SepalWidth = 2.8f, PetalLength = 5.1f, PetalWidth = 1.5f, Label = "Iris-virginica"},
            new IrisData() { SepalLength = 6.1f, SepalWidth = 2.6f, PetalLength = 5.6f, PetalWidth = 1.4f, Label = "Iris-virginica"},
            new IrisData() { SepalLength = 7.7f, SepalWidth = 3.0f, PetalLength = 6.1f, PetalWidth = 2.3f, Label = "Iris-virginica"},
            new IrisData() { SepalLength = 6.3f, SepalWidth = 3.4f, PetalLength = 5.6f, PetalWidth = 2.4f, Label = "Iris-virginica"},
            new IrisData() { SepalLength = 6.4f, SepalWidth = 3.1f, PetalLength = 5.5f, PetalWidth = 1.8f, Label = "Iris-virginica"},
            new IrisData() { SepalLength = 6.0f, SepalWidth = 3.0f, PetalLength = 4.8f, PetalWidth = 1.8f, Label = "Iris-virginica"},
            new IrisData() { SepalLength = 6.9f, SepalWidth = 3.1f, PetalLength = 5.4f, PetalWidth = 2.1f, Label = "Iris-virginica"},
            new IrisData() { SepalLength = 6.7f, SepalWidth = 3.1f, PetalLength = 5.6f, PetalWidth = 2.4f, Label = "Iris-virginica"},
            new IrisData() { SepalLength = 6.9f, SepalWidth = 3.1f, PetalLength = 5.1f, PetalWidth = 2.3f, Label = "Iris-virginica"},
            new IrisData() { SepalLength = 5.8f, SepalWidth = 2.7f, PetalLength = 5.1f, PetalWidth = 1.9f, Label = "Iris-virginica"},
            new IrisData() { SepalLength = 6.8f, SepalWidth = 3.2f, PetalLength = 5.9f, PetalWidth = 2.3f, Label = "Iris-virginica"},
            new IrisData() { SepalLength = 6.7f, SepalWidth = 3.3f, PetalLength = 5.7f, PetalWidth = 2.5f, Label = "Iris-virginica"},
            new IrisData() { SepalLength = 6.7f, SepalWidth = 3.0f, PetalLength = 5.2f, PetalWidth = 2.3f, Label = "Iris-virginica"},
            new IrisData() { SepalLength = 6.3f, SepalWidth = 2.5f, PetalLength = 5.0f, PetalWidth = 1.9f, Label = "Iris-virginica"},
            new IrisData() { SepalLength = 6.5f, SepalWidth = 3.0f, PetalLength = 5.2f, PetalWidth = 2.0f, Label = "Iris-virginica"},
            new IrisData() { SepalLength = 6.2f, SepalWidth = 3.4f, PetalLength = 5.4f, PetalWidth = 2.3f, Label = "Iris-virginica"},
            new IrisData() { SepalLength = 5.9f, SepalWidth = 3.0f, PetalLength = 5.1f, PetalWidth = 1.8f, Label = "Iris-virginica"}
    };
    }
}
