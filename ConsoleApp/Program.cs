using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using DynamicStructure;
using System.Collections;
using Decisions.ML;

namespace ConsoleApp
{
    class Program
    {
        static void Main(string[] args)
        {
            LoadData();

            MLModel mLModel = new MLModel(mLData.Features, "TestModelName");
            mLModel.Train(mLData.Rows, "Label");

            MLDataRow exampleRow = new MLDataRow();
            exampleRow.Data = new Dictionary<string, object>();
            exampleRow.Data.Add("SepalLength", 5.9f);
            exampleRow.Data.Add("SepalWidth", 3.0f);
            exampleRow.Data.Add("PetalLength", 5.1f);
            exampleRow.Data.Add("PetalWidth", 1.8f);
            exampleRow.Data.Add("Label", "Iris-setosa");

            dynamic dynamic = mLModel.Predict(exampleRow);

            Console.WriteLine(dynamic.PredictedLabels);

            Console.ReadLine();
        }

        private static object CopyObjectFields(object oldClass, object newClass)
        {
            Type instanceType = oldClass.GetType();
            instanceType.GetFields().ToList().ForEach((field) =>
            {
                newClass.GetType().GetField(field.Name).SetValue(newClass, field.GetValue(oldClass));
            });

            return newClass;
        }

        private static MLDataSet mLData = new MLDataSet()
        {
            Features = new MLFeature[]
            {
                new MLFeature(){ Name = "SepalLength", Type = typeof(float) },
                new MLFeature(){ Name = "SepalWidth", Type = typeof(float) },
                new MLFeature(){ Name = "PetalLength", Type = typeof(float) },
                new MLFeature(){ Name = "PetalWidth", Type = typeof(float) },
                new MLFeature(){ Name = "Label", Type = typeof(string) },
            }
        };

        private static void LoadData()
        {
            List<MLDataRow> mLDataRows = new List<MLDataRow>();

            MLDataRow newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 5.1f); 

            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 5.1f); newRow.Data.Add("SepalWidth", 3.5f); newRow.Data.Add("PetalLength", 1.4f); newRow.Data.Add("PetalWidth", 0.2f); newRow.Data.Add("Label", "Iris-setosa");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 4.9f); newRow.Data.Add("SepalWidth", 3.0f); newRow.Data.Add("PetalLength", 1.4f); newRow.Data.Add("PetalWidth", 0.2f); newRow.Data.Add("Label", "Iris-setosa");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 4.7f); newRow.Data.Add("SepalWidth", 3.2f); newRow.Data.Add("PetalLength", 1.3f); newRow.Data.Add("PetalWidth", 0.2f); newRow.Data.Add("Label", "Iris-setosa");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 4.6f); newRow.Data.Add("SepalWidth", 3.1f); newRow.Data.Add("PetalLength", 1.5f); newRow.Data.Add("PetalWidth", 0.2f); newRow.Data.Add("Label", "Iris-setosa");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 5.0f); newRow.Data.Add("SepalWidth", 3.6f); newRow.Data.Add("PetalLength", 1.4f); newRow.Data.Add("PetalWidth", 0.2f); newRow.Data.Add("Label", "Iris-setosa");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 5.4f); newRow.Data.Add("SepalWidth", 3.9f); newRow.Data.Add("PetalLength", 1.7f); newRow.Data.Add("PetalWidth", 0.4f); newRow.Data.Add("Label", "Iris-setosa");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 4.6f); newRow.Data.Add("SepalWidth", 3.4f); newRow.Data.Add("PetalLength", 1.4f); newRow.Data.Add("PetalWidth", 0.3f); newRow.Data.Add("Label", "Iris-setosa");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 5.0f); newRow.Data.Add("SepalWidth", 3.4f); newRow.Data.Add("PetalLength", 1.5f); newRow.Data.Add("PetalWidth", 0.2f); newRow.Data.Add("Label", "Iris-setosa");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 4.4f); newRow.Data.Add("SepalWidth", 2.9f); newRow.Data.Add("PetalLength", 1.4f); newRow.Data.Add("PetalWidth", 0.2f); newRow.Data.Add("Label", "Iris-setosa");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 4.9f); newRow.Data.Add("SepalWidth", 3.1f); newRow.Data.Add("PetalLength", 1.5f); newRow.Data.Add("PetalWidth", 0.1f); newRow.Data.Add("Label", "Iris-setosa");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 5.4f); newRow.Data.Add("SepalWidth", 3.7f); newRow.Data.Add("PetalLength", 1.5f); newRow.Data.Add("PetalWidth", 0.2f); newRow.Data.Add("Label", "Iris-setosa");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 4.8f); newRow.Data.Add("SepalWidth", 3.4f); newRow.Data.Add("PetalLength", 1.6f); newRow.Data.Add("PetalWidth", 0.2f); newRow.Data.Add("Label", "Iris-setosa");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 4.8f); newRow.Data.Add("SepalWidth", 3.0f); newRow.Data.Add("PetalLength", 1.4f); newRow.Data.Add("PetalWidth", 0.1f); newRow.Data.Add("Label", "Iris-setosa");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 4.3f); newRow.Data.Add("SepalWidth", 3.0f); newRow.Data.Add("PetalLength", 1.1f); newRow.Data.Add("PetalWidth", 0.1f); newRow.Data.Add("Label", "Iris-setosa");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 5.8f); newRow.Data.Add("SepalWidth", 4.0f); newRow.Data.Add("PetalLength", 1.2f); newRow.Data.Add("PetalWidth", 0.2f); newRow.Data.Add("Label", "Iris-setosa");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 5.7f); newRow.Data.Add("SepalWidth", 4.4f); newRow.Data.Add("PetalLength", 1.5f); newRow.Data.Add("PetalWidth", 0.4f); newRow.Data.Add("Label", "Iris-setosa");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 5.4f); newRow.Data.Add("SepalWidth", 3.9f); newRow.Data.Add("PetalLength", 1.3f); newRow.Data.Add("PetalWidth", 0.4f); newRow.Data.Add("Label", "Iris-setosa");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 5.1f); newRow.Data.Add("SepalWidth", 3.5f); newRow.Data.Add("PetalLength", 1.4f); newRow.Data.Add("PetalWidth", 0.3f); newRow.Data.Add("Label", "Iris-setosa");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 5.7f); newRow.Data.Add("SepalWidth", 3.8f); newRow.Data.Add("PetalLength", 1.7f); newRow.Data.Add("PetalWidth", 0.3f); newRow.Data.Add("Label", "Iris-setosa");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 5.1f); newRow.Data.Add("SepalWidth", 3.8f); newRow.Data.Add("PetalLength", 1.5f); newRow.Data.Add("PetalWidth", 0.3f); newRow.Data.Add("Label", "Iris-setosa");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 5.4f); newRow.Data.Add("SepalWidth", 3.4f); newRow.Data.Add("PetalLength", 1.7f); newRow.Data.Add("PetalWidth", 0.2f); newRow.Data.Add("Label", "Iris-setosa");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 5.1f); newRow.Data.Add("SepalWidth", 3.7f); newRow.Data.Add("PetalLength", 1.5f); newRow.Data.Add("PetalWidth", 0.4f); newRow.Data.Add("Label", "Iris-setosa");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 4.6f); newRow.Data.Add("SepalWidth", 3.6f); newRow.Data.Add("PetalLength", 1.0f); newRow.Data.Add("PetalWidth", 0.2f); newRow.Data.Add("Label", "Iris-setosa");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 5.1f); newRow.Data.Add("SepalWidth", 3.3f); newRow.Data.Add("PetalLength", 1.7f); newRow.Data.Add("PetalWidth", 0.5f); newRow.Data.Add("Label", "Iris-setosa");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 4.8f); newRow.Data.Add("SepalWidth", 3.4f); newRow.Data.Add("PetalLength", 1.9f); newRow.Data.Add("PetalWidth", 0.2f); newRow.Data.Add("Label", "Iris-setosa");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 5.0f); newRow.Data.Add("SepalWidth", 3.0f); newRow.Data.Add("PetalLength", 1.6f); newRow.Data.Add("PetalWidth", 0.2f); newRow.Data.Add("Label", "Iris-setosa");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 5.0f); newRow.Data.Add("SepalWidth", 3.4f); newRow.Data.Add("PetalLength", 1.6f); newRow.Data.Add("PetalWidth", 0.4f); newRow.Data.Add("Label", "Iris-setosa");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 5.2f); newRow.Data.Add("SepalWidth", 3.5f); newRow.Data.Add("PetalLength", 1.5f); newRow.Data.Add("PetalWidth", 0.2f); newRow.Data.Add("Label", "Iris-setosa");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 5.2f); newRow.Data.Add("SepalWidth", 3.4f); newRow.Data.Add("PetalLength", 1.4f); newRow.Data.Add("PetalWidth", 0.2f); newRow.Data.Add("Label", "Iris-setosa");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 4.7f); newRow.Data.Add("SepalWidth", 3.2f); newRow.Data.Add("PetalLength", 1.6f); newRow.Data.Add("PetalWidth", 0.2f); newRow.Data.Add("Label", "Iris-setosa");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 4.8f); newRow.Data.Add("SepalWidth", 3.1f); newRow.Data.Add("PetalLength", 1.6f); newRow.Data.Add("PetalWidth", 0.2f); newRow.Data.Add("Label", "Iris-setosa");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 5.4f); newRow.Data.Add("SepalWidth", 3.4f); newRow.Data.Add("PetalLength", 1.5f); newRow.Data.Add("PetalWidth", 0.4f); newRow.Data.Add("Label", "Iris-setosa");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 5.2f); newRow.Data.Add("SepalWidth", 4.1f); newRow.Data.Add("PetalLength", 1.5f); newRow.Data.Add("PetalWidth", 0.1f); newRow.Data.Add("Label", "Iris-setosa");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 5.5f); newRow.Data.Add("SepalWidth", 4.2f); newRow.Data.Add("PetalLength", 1.4f); newRow.Data.Add("PetalWidth", 0.2f); newRow.Data.Add("Label", "Iris-setosa");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 4.9f); newRow.Data.Add("SepalWidth", 3.1f); newRow.Data.Add("PetalLength", 1.5f); newRow.Data.Add("PetalWidth", 0.1f); newRow.Data.Add("Label", "Iris-setosa");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 5.0f); newRow.Data.Add("SepalWidth", 3.2f); newRow.Data.Add("PetalLength", 1.2f); newRow.Data.Add("PetalWidth", 0.2f); newRow.Data.Add("Label", "Iris-setosa");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 5.5f); newRow.Data.Add("SepalWidth", 3.5f); newRow.Data.Add("PetalLength", 1.3f); newRow.Data.Add("PetalWidth", 0.2f); newRow.Data.Add("Label", "Iris-setosa");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 4.9f); newRow.Data.Add("SepalWidth", 3.1f); newRow.Data.Add("PetalLength", 1.5f); newRow.Data.Add("PetalWidth", 0.1f); newRow.Data.Add("Label", "Iris-setosa");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 4.4f); newRow.Data.Add("SepalWidth", 3.0f); newRow.Data.Add("PetalLength", 1.3f); newRow.Data.Add("PetalWidth", 0.2f); newRow.Data.Add("Label", "Iris-setosa");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 5.1f); newRow.Data.Add("SepalWidth", 3.4f); newRow.Data.Add("PetalLength", 1.5f); newRow.Data.Add("PetalWidth", 0.2f); newRow.Data.Add("Label", "Iris-setosa");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 5.0f); newRow.Data.Add("SepalWidth", 3.5f); newRow.Data.Add("PetalLength", 1.3f); newRow.Data.Add("PetalWidth", 0.3f); newRow.Data.Add("Label", "Iris-setosa");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 4.5f); newRow.Data.Add("SepalWidth", 2.3f); newRow.Data.Add("PetalLength", 1.3f); newRow.Data.Add("PetalWidth", 0.3f); newRow.Data.Add("Label", "Iris-setosa");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 4.4f); newRow.Data.Add("SepalWidth", 3.2f); newRow.Data.Add("PetalLength", 1.3f); newRow.Data.Add("PetalWidth", 0.2f); newRow.Data.Add("Label", "Iris-setosa");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 5.0f); newRow.Data.Add("SepalWidth", 3.5f); newRow.Data.Add("PetalLength", 1.6f); newRow.Data.Add("PetalWidth", 0.6f); newRow.Data.Add("Label", "Iris-setosa");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 5.1f); newRow.Data.Add("SepalWidth", 3.8f); newRow.Data.Add("PetalLength", 1.9f); newRow.Data.Add("PetalWidth", 0.4f); newRow.Data.Add("Label", "Iris-setosa");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 4.8f); newRow.Data.Add("SepalWidth", 3.0f); newRow.Data.Add("PetalLength", 1.4f); newRow.Data.Add("PetalWidth", 0.3f); newRow.Data.Add("Label", "Iris-setosa");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 5.1f); newRow.Data.Add("SepalWidth", 3.8f); newRow.Data.Add("PetalLength", 1.6f); newRow.Data.Add("PetalWidth", 0.2f); newRow.Data.Add("Label", "Iris-setosa");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 4.6f); newRow.Data.Add("SepalWidth", 3.2f); newRow.Data.Add("PetalLength", 1.4f); newRow.Data.Add("PetalWidth", 0.2f); newRow.Data.Add("Label", "Iris-setosa");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 5.3f); newRow.Data.Add("SepalWidth", 3.7f); newRow.Data.Add("PetalLength", 1.5f); newRow.Data.Add("PetalWidth", 0.2f); newRow.Data.Add("Label", "Iris-setosa");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 5.0f); newRow.Data.Add("SepalWidth", 3.3f); newRow.Data.Add("PetalLength", 1.4f); newRow.Data.Add("PetalWidth", 0.2f); newRow.Data.Add("Label", "Iris-setosa");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 7.0f); newRow.Data.Add("SepalWidth", 3.2f); newRow.Data.Add("PetalLength", 4.7f); newRow.Data.Add("PetalWidth", 1.4f); newRow.Data.Add("Label", "Iris-versicoor");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 6.4f); newRow.Data.Add("SepalWidth", 3.2f); newRow.Data.Add("PetalLength", 4.5f); newRow.Data.Add("PetalWidth", 1.5f); newRow.Data.Add("Label", "Iris-versicoor");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 6.9f); newRow.Data.Add("SepalWidth", 3.1f); newRow.Data.Add("PetalLength", 4.9f); newRow.Data.Add("PetalWidth", 1.5f); newRow.Data.Add("Label", "Iris-versicoor");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 5.5f); newRow.Data.Add("SepalWidth", 2.3f); newRow.Data.Add("PetalLength", 4.0f); newRow.Data.Add("PetalWidth", 1.3f); newRow.Data.Add("Label", "Iris-versicoor");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 6.5f); newRow.Data.Add("SepalWidth", 2.8f); newRow.Data.Add("PetalLength", 4.6f); newRow.Data.Add("PetalWidth", 1.5f); newRow.Data.Add("Label", "Iris-versicoor");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 5.7f); newRow.Data.Add("SepalWidth", 2.8f); newRow.Data.Add("PetalLength", 4.5f); newRow.Data.Add("PetalWidth", 1.3f); newRow.Data.Add("Label", "Iris-versicoor");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 6.3f); newRow.Data.Add("SepalWidth", 3.3f); newRow.Data.Add("PetalLength", 4.7f); newRow.Data.Add("PetalWidth", 1.6f); newRow.Data.Add("Label", "Iris-versicoor");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 4.9f); newRow.Data.Add("SepalWidth", 2.4f); newRow.Data.Add("PetalLength", 3.3f); newRow.Data.Add("PetalWidth", 1.0f); newRow.Data.Add("Label", "Iris-versicoor");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 6.6f); newRow.Data.Add("SepalWidth", 2.9f); newRow.Data.Add("PetalLength", 4.6f); newRow.Data.Add("PetalWidth", 1.3f); newRow.Data.Add("Label", "Iris-versicoor");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 5.2f); newRow.Data.Add("SepalWidth", 2.7f); newRow.Data.Add("PetalLength", 3.9f); newRow.Data.Add("PetalWidth", 1.4f); newRow.Data.Add("Label", "Iris-versicoor");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 5.0f); newRow.Data.Add("SepalWidth", 2.0f); newRow.Data.Add("PetalLength", 3.5f); newRow.Data.Add("PetalWidth", 1.0f); newRow.Data.Add("Label", "Iris-versicoor");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 5.9f); newRow.Data.Add("SepalWidth", 3.0f); newRow.Data.Add("PetalLength", 4.2f); newRow.Data.Add("PetalWidth", 1.5f); newRow.Data.Add("Label", "Iris-versicoor");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 6.0f); newRow.Data.Add("SepalWidth", 2.2f); newRow.Data.Add("PetalLength", 4.0f); newRow.Data.Add("PetalWidth", 1.0f); newRow.Data.Add("Label", "Iris-versicoor");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 6.1f); newRow.Data.Add("SepalWidth", 2.9f); newRow.Data.Add("PetalLength", 4.7f); newRow.Data.Add("PetalWidth", 1.4f); newRow.Data.Add("Label", "Iris-versicoor");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 5.6f); newRow.Data.Add("SepalWidth", 2.9f); newRow.Data.Add("PetalLength", 3.6f); newRow.Data.Add("PetalWidth", 1.3f); newRow.Data.Add("Label", "Iris-versicoor");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 6.7f); newRow.Data.Add("SepalWidth", 3.1f); newRow.Data.Add("PetalLength", 4.4f); newRow.Data.Add("PetalWidth", 1.4f); newRow.Data.Add("Label", "Iris-versicoor");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 5.6f); newRow.Data.Add("SepalWidth", 3.0f); newRow.Data.Add("PetalLength", 4.5f); newRow.Data.Add("PetalWidth", 1.5f); newRow.Data.Add("Label", "Iris-versicoor");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 5.8f); newRow.Data.Add("SepalWidth", 2.7f); newRow.Data.Add("PetalLength", 4.1f); newRow.Data.Add("PetalWidth", 1.0f); newRow.Data.Add("Label", "Iris-versicoor");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 6.2f); newRow.Data.Add("SepalWidth", 2.2f); newRow.Data.Add("PetalLength", 4.5f); newRow.Data.Add("PetalWidth", 1.5f); newRow.Data.Add("Label", "Iris-versicoor");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 5.6f); newRow.Data.Add("SepalWidth", 2.5f); newRow.Data.Add("PetalLength", 3.9f); newRow.Data.Add("PetalWidth", 1.1f); newRow.Data.Add("Label", "Iris-versicoor");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 5.9f); newRow.Data.Add("SepalWidth", 3.2f); newRow.Data.Add("PetalLength", 4.8f); newRow.Data.Add("PetalWidth", 1.8f); newRow.Data.Add("Label", "Iris-versicoor");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 6.1f); newRow.Data.Add("SepalWidth", 2.8f); newRow.Data.Add("PetalLength", 4.0f); newRow.Data.Add("PetalWidth", 1.3f); newRow.Data.Add("Label", "Iris-versicoor");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 6.3f); newRow.Data.Add("SepalWidth", 2.5f); newRow.Data.Add("PetalLength", 4.9f); newRow.Data.Add("PetalWidth", 1.5f); newRow.Data.Add("Label", "Iris-versicoor");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 6.1f); newRow.Data.Add("SepalWidth", 2.8f); newRow.Data.Add("PetalLength", 4.7f); newRow.Data.Add("PetalWidth", 1.2f); newRow.Data.Add("Label", "Iris-versicoor");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 6.4f); newRow.Data.Add("SepalWidth", 2.9f); newRow.Data.Add("PetalLength", 4.3f); newRow.Data.Add("PetalWidth", 1.3f); newRow.Data.Add("Label", "Iris-versicoor");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 6.6f); newRow.Data.Add("SepalWidth", 3.0f); newRow.Data.Add("PetalLength", 4.4f); newRow.Data.Add("PetalWidth", 1.4f); newRow.Data.Add("Label", "Iris-versicoor");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 6.8f); newRow.Data.Add("SepalWidth", 2.8f); newRow.Data.Add("PetalLength", 4.8f); newRow.Data.Add("PetalWidth", 1.4f); newRow.Data.Add("Label", "Iris-versicoor");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 6.7f); newRow.Data.Add("SepalWidth", 3.0f); newRow.Data.Add("PetalLength", 5.0f); newRow.Data.Add("PetalWidth", 1.7f); newRow.Data.Add("Label", "Iris-versicoor");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 6.0f); newRow.Data.Add("SepalWidth", 2.9f); newRow.Data.Add("PetalLength", 4.5f); newRow.Data.Add("PetalWidth", 1.5f); newRow.Data.Add("Label", "Iris-versicoor");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 5.7f); newRow.Data.Add("SepalWidth", 2.6f); newRow.Data.Add("PetalLength", 3.5f); newRow.Data.Add("PetalWidth", 1.0f); newRow.Data.Add("Label", "Iris-versicoor");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 5.5f); newRow.Data.Add("SepalWidth", 2.4f); newRow.Data.Add("PetalLength", 3.8f); newRow.Data.Add("PetalWidth", 1.1f); newRow.Data.Add("Label", "Iris-versicoor");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 5.5f); newRow.Data.Add("SepalWidth", 2.4f); newRow.Data.Add("PetalLength", 3.7f); newRow.Data.Add("PetalWidth", 1.0f); newRow.Data.Add("Label", "Iris-versicoor");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 5.8f); newRow.Data.Add("SepalWidth", 2.7f); newRow.Data.Add("PetalLength", 3.9f); newRow.Data.Add("PetalWidth", 1.2f); newRow.Data.Add("Label", "Iris-versicoor");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 6.0f); newRow.Data.Add("SepalWidth", 2.7f); newRow.Data.Add("PetalLength", 5.1f); newRow.Data.Add("PetalWidth", 1.6f); newRow.Data.Add("Label", "Iris-versicoor");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 5.4f); newRow.Data.Add("SepalWidth", 3.0f); newRow.Data.Add("PetalLength", 4.5f); newRow.Data.Add("PetalWidth", 1.5f); newRow.Data.Add("Label", "Iris-versicoor");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 6.0f); newRow.Data.Add("SepalWidth", 3.4f); newRow.Data.Add("PetalLength", 4.5f); newRow.Data.Add("PetalWidth", 1.6f); newRow.Data.Add("Label", "Iris-versicoor");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 6.7f); newRow.Data.Add("SepalWidth", 3.1f); newRow.Data.Add("PetalLength", 4.7f); newRow.Data.Add("PetalWidth", 1.5f); newRow.Data.Add("Label", "Iris-versicoor");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 6.3f); newRow.Data.Add("SepalWidth", 2.3f); newRow.Data.Add("PetalLength", 4.4f); newRow.Data.Add("PetalWidth", 1.3f); newRow.Data.Add("Label", "Iris-versicoor");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 5.6f); newRow.Data.Add("SepalWidth", 3.0f); newRow.Data.Add("PetalLength", 4.1f); newRow.Data.Add("PetalWidth", 1.3f); newRow.Data.Add("Label", "Iris-versicoor");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 5.5f); newRow.Data.Add("SepalWidth", 2.5f); newRow.Data.Add("PetalLength", 4.0f); newRow.Data.Add("PetalWidth", 1.3f); newRow.Data.Add("Label", "Iris-versicoor");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 5.5f); newRow.Data.Add("SepalWidth", 2.6f); newRow.Data.Add("PetalLength", 4.4f); newRow.Data.Add("PetalWidth", 1.2f); newRow.Data.Add("Label", "Iris-versicoor");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 6.1f); newRow.Data.Add("SepalWidth", 3.0f); newRow.Data.Add("PetalLength", 4.6f); newRow.Data.Add("PetalWidth", 1.4f); newRow.Data.Add("Label", "Iris-versicoor");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 5.8f); newRow.Data.Add("SepalWidth", 2.6f); newRow.Data.Add("PetalLength", 4.0f); newRow.Data.Add("PetalWidth", 1.2f); newRow.Data.Add("Label", "Iris-versicoor");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 5.0f); newRow.Data.Add("SepalWidth", 2.3f); newRow.Data.Add("PetalLength", 3.3f); newRow.Data.Add("PetalWidth", 1.0f); newRow.Data.Add("Label", "Iris-versicoor");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 5.6f); newRow.Data.Add("SepalWidth", 2.7f); newRow.Data.Add("PetalLength", 4.2f); newRow.Data.Add("PetalWidth", 1.3f); newRow.Data.Add("Label", "Iris-versicoor");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 5.7f); newRow.Data.Add("SepalWidth", 3.0f); newRow.Data.Add("PetalLength", 4.2f); newRow.Data.Add("PetalWidth", 1.2f); newRow.Data.Add("Label", "Iris-versicoor");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 5.7f); newRow.Data.Add("SepalWidth", 2.9f); newRow.Data.Add("PetalLength", 4.2f); newRow.Data.Add("PetalWidth", 1.3f); newRow.Data.Add("Label", "Iris-versicoor");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 6.2f); newRow.Data.Add("SepalWidth", 2.9f); newRow.Data.Add("PetalLength", 4.3f); newRow.Data.Add("PetalWidth", 1.3f); newRow.Data.Add("Label", "Iris-versicoor");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 5.1f); newRow.Data.Add("SepalWidth", 2.5f); newRow.Data.Add("PetalLength", 3.0f); newRow.Data.Add("PetalWidth", 1.1f); newRow.Data.Add("Label", "Iris-versicoor");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 5.7f); newRow.Data.Add("SepalWidth", 2.8f); newRow.Data.Add("PetalLength", 4.1f); newRow.Data.Add("PetalWidth", 1.3f); newRow.Data.Add("Label", "Iris-versicoor");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 6.3f); newRow.Data.Add("SepalWidth", 3.3f); newRow.Data.Add("PetalLength", 6.0f); newRow.Data.Add("PetalWidth", 2.5f); newRow.Data.Add("Label", "Iris-virginia");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 5.8f); newRow.Data.Add("SepalWidth", 2.7f); newRow.Data.Add("PetalLength", 5.1f); newRow.Data.Add("PetalWidth", 1.9f); newRow.Data.Add("Label", "Iris-virginia");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 7.1f); newRow.Data.Add("SepalWidth", 3.0f); newRow.Data.Add("PetalLength", 5.9f); newRow.Data.Add("PetalWidth", 2.1f); newRow.Data.Add("Label", "Iris-virginia");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 6.3f); newRow.Data.Add("SepalWidth", 2.9f); newRow.Data.Add("PetalLength", 5.6f); newRow.Data.Add("PetalWidth", 1.8f); newRow.Data.Add("Label", "Iris-virginia");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 6.5f); newRow.Data.Add("SepalWidth", 3.0f); newRow.Data.Add("PetalLength", 5.8f); newRow.Data.Add("PetalWidth", 2.2f); newRow.Data.Add("Label", "Iris-virginia");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 7.6f); newRow.Data.Add("SepalWidth", 3.0f); newRow.Data.Add("PetalLength", 6.6f); newRow.Data.Add("PetalWidth", 2.1f); newRow.Data.Add("Label", "Iris-virginia");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 4.9f); newRow.Data.Add("SepalWidth", 2.5f); newRow.Data.Add("PetalLength", 4.5f); newRow.Data.Add("PetalWidth", 1.7f); newRow.Data.Add("Label", "Iris-virginia");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 7.3f); newRow.Data.Add("SepalWidth", 2.9f); newRow.Data.Add("PetalLength", 6.3f); newRow.Data.Add("PetalWidth", 1.8f); newRow.Data.Add("Label", "Iris-virginia");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 6.7f); newRow.Data.Add("SepalWidth", 2.5f); newRow.Data.Add("PetalLength", 5.8f); newRow.Data.Add("PetalWidth", 1.8f); newRow.Data.Add("Label", "Iris-virginia");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 7.2f); newRow.Data.Add("SepalWidth", 3.6f); newRow.Data.Add("PetalLength", 6.1f); newRow.Data.Add("PetalWidth", 2.5f); newRow.Data.Add("Label", "Iris-virginia");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 6.5f); newRow.Data.Add("SepalWidth", 3.2f); newRow.Data.Add("PetalLength", 5.1f); newRow.Data.Add("PetalWidth", 2.0f); newRow.Data.Add("Label", "Iris-virginia");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 6.4f); newRow.Data.Add("SepalWidth", 2.7f); newRow.Data.Add("PetalLength", 5.3f); newRow.Data.Add("PetalWidth", 1.9f); newRow.Data.Add("Label", "Iris-virginia");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 6.8f); newRow.Data.Add("SepalWidth", 3.0f); newRow.Data.Add("PetalLength", 5.5f); newRow.Data.Add("PetalWidth", 2.1f); newRow.Data.Add("Label", "Iris-virginia");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 5.7f); newRow.Data.Add("SepalWidth", 2.5f); newRow.Data.Add("PetalLength", 5.0f); newRow.Data.Add("PetalWidth", 2.0f); newRow.Data.Add("Label", "Iris-virginia");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 5.8f); newRow.Data.Add("SepalWidth", 2.8f); newRow.Data.Add("PetalLength", 5.1f); newRow.Data.Add("PetalWidth", 2.4f); newRow.Data.Add("Label", "Iris-virginia");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 6.4f); newRow.Data.Add("SepalWidth", 3.2f); newRow.Data.Add("PetalLength", 5.3f); newRow.Data.Add("PetalWidth", 2.3f); newRow.Data.Add("Label", "Iris-virginia");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 6.5f); newRow.Data.Add("SepalWidth", 3.0f); newRow.Data.Add("PetalLength", 5.5f); newRow.Data.Add("PetalWidth", 1.8f); newRow.Data.Add("Label", "Iris-virginia");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 7.7f); newRow.Data.Add("SepalWidth", 3.8f); newRow.Data.Add("PetalLength", 6.7f); newRow.Data.Add("PetalWidth", 2.2f); newRow.Data.Add("Label", "Iris-virginia");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 7.7f); newRow.Data.Add("SepalWidth", 2.6f); newRow.Data.Add("PetalLength", 6.9f); newRow.Data.Add("PetalWidth", 2.3f); newRow.Data.Add("Label", "Iris-virginia");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 6.0f); newRow.Data.Add("SepalWidth", 2.2f); newRow.Data.Add("PetalLength", 5.0f); newRow.Data.Add("PetalWidth", 1.5f); newRow.Data.Add("Label", "Iris-virginia");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 6.9f); newRow.Data.Add("SepalWidth", 3.2f); newRow.Data.Add("PetalLength", 5.7f); newRow.Data.Add("PetalWidth", 2.3f); newRow.Data.Add("Label", "Iris-virginia");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 5.6f); newRow.Data.Add("SepalWidth", 2.8f); newRow.Data.Add("PetalLength", 4.9f); newRow.Data.Add("PetalWidth", 2.0f); newRow.Data.Add("Label", "Iris-virginia");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 7.7f); newRow.Data.Add("SepalWidth", 2.8f); newRow.Data.Add("PetalLength", 6.7f); newRow.Data.Add("PetalWidth", 2.0f); newRow.Data.Add("Label", "Iris-virginia");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 6.3f); newRow.Data.Add("SepalWidth", 2.7f); newRow.Data.Add("PetalLength", 4.9f); newRow.Data.Add("PetalWidth", 1.8f); newRow.Data.Add("Label", "Iris-virginia");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 6.7f); newRow.Data.Add("SepalWidth", 3.3f); newRow.Data.Add("PetalLength", 5.7f); newRow.Data.Add("PetalWidth", 2.1f); newRow.Data.Add("Label", "Iris-virginia");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 7.2f); newRow.Data.Add("SepalWidth", 3.2f); newRow.Data.Add("PetalLength", 6.0f); newRow.Data.Add("PetalWidth", 1.8f); newRow.Data.Add("Label", "Iris-virginia");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 6.2f); newRow.Data.Add("SepalWidth", 2.8f); newRow.Data.Add("PetalLength", 4.8f); newRow.Data.Add("PetalWidth", 1.8f); newRow.Data.Add("Label", "Iris-virginia");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 6.1f); newRow.Data.Add("SepalWidth", 3.0f); newRow.Data.Add("PetalLength", 4.9f); newRow.Data.Add("PetalWidth", 1.8f); newRow.Data.Add("Label", "Iris-virginia");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 6.4f); newRow.Data.Add("SepalWidth", 2.8f); newRow.Data.Add("PetalLength", 5.6f); newRow.Data.Add("PetalWidth", 2.1f); newRow.Data.Add("Label", "Iris-virginia");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 7.2f); newRow.Data.Add("SepalWidth", 3.0f); newRow.Data.Add("PetalLength", 5.8f); newRow.Data.Add("PetalWidth", 1.6f); newRow.Data.Add("Label", "Iris-virginia");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 7.4f); newRow.Data.Add("SepalWidth", 2.8f); newRow.Data.Add("PetalLength", 6.1f); newRow.Data.Add("PetalWidth", 1.9f); newRow.Data.Add("Label", "Iris-virginia");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 7.9f); newRow.Data.Add("SepalWidth", 3.8f); newRow.Data.Add("PetalLength", 6.4f); newRow.Data.Add("PetalWidth", 2.0f); newRow.Data.Add("Label", "Iris-virginia");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 6.4f); newRow.Data.Add("SepalWidth", 2.8f); newRow.Data.Add("PetalLength", 5.6f); newRow.Data.Add("PetalWidth", 2.2f); newRow.Data.Add("Label", "Iris-virginia");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 6.3f); newRow.Data.Add("SepalWidth", 2.8f); newRow.Data.Add("PetalLength", 5.1f); newRow.Data.Add("PetalWidth", 1.5f); newRow.Data.Add("Label", "Iris-virginia");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 6.1f); newRow.Data.Add("SepalWidth", 2.6f); newRow.Data.Add("PetalLength", 5.6f); newRow.Data.Add("PetalWidth", 1.4f); newRow.Data.Add("Label", "Iris-virginia");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 7.7f); newRow.Data.Add("SepalWidth", 3.0f); newRow.Data.Add("PetalLength", 6.1f); newRow.Data.Add("PetalWidth", 2.3f); newRow.Data.Add("Label", "Iris-virginia");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 6.3f); newRow.Data.Add("SepalWidth", 3.4f); newRow.Data.Add("PetalLength", 5.6f); newRow.Data.Add("PetalWidth", 2.4f); newRow.Data.Add("Label", "Iris-virginia");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 6.4f); newRow.Data.Add("SepalWidth", 3.1f); newRow.Data.Add("PetalLength", 5.5f); newRow.Data.Add("PetalWidth", 1.8f); newRow.Data.Add("Label", "Iris-virginia");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 6.0f); newRow.Data.Add("SepalWidth", 3.0f); newRow.Data.Add("PetalLength", 4.8f); newRow.Data.Add("PetalWidth", 1.8f); newRow.Data.Add("Label", "Iris-virginia");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 6.9f); newRow.Data.Add("SepalWidth", 3.1f); newRow.Data.Add("PetalLength", 5.4f); newRow.Data.Add("PetalWidth", 2.1f); newRow.Data.Add("Label", "Iris-virginia");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 6.7f); newRow.Data.Add("SepalWidth", 3.1f); newRow.Data.Add("PetalLength", 5.6f); newRow.Data.Add("PetalWidth", 2.4f); newRow.Data.Add("Label", "Iris-virginia");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 6.9f); newRow.Data.Add("SepalWidth", 3.1f); newRow.Data.Add("PetalLength", 5.1f); newRow.Data.Add("PetalWidth", 2.3f); newRow.Data.Add("Label", "Iris-virginia");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 5.8f); newRow.Data.Add("SepalWidth", 2.7f); newRow.Data.Add("PetalLength", 5.1f); newRow.Data.Add("PetalWidth", 1.9f); newRow.Data.Add("Label", "Iris-virginia");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 6.8f); newRow.Data.Add("SepalWidth", 3.2f); newRow.Data.Add("PetalLength", 5.9f); newRow.Data.Add("PetalWidth", 2.3f); newRow.Data.Add("Label", "Iris-virginia");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 6.7f); newRow.Data.Add("SepalWidth", 3.3f); newRow.Data.Add("PetalLength", 5.7f); newRow.Data.Add("PetalWidth", 2.5f); newRow.Data.Add("Label", "Iris-virginia");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 6.7f); newRow.Data.Add("SepalWidth", 3.0f); newRow.Data.Add("PetalLength", 5.2f); newRow.Data.Add("PetalWidth", 2.3f); newRow.Data.Add("Label", "Iris-virginia");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 6.3f); newRow.Data.Add("SepalWidth", 2.5f); newRow.Data.Add("PetalLength", 5.0f); newRow.Data.Add("PetalWidth", 1.9f); newRow.Data.Add("Label", "Iris-virginia");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 6.5f); newRow.Data.Add("SepalWidth", 3.0f); newRow.Data.Add("PetalLength", 5.2f); newRow.Data.Add("PetalWidth", 2.0f); newRow.Data.Add("Label", "Iris-virginia");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 6.2f); newRow.Data.Add("SepalWidth", 3.4f); newRow.Data.Add("PetalLength", 5.4f); newRow.Data.Add("PetalWidth", 2.3f); newRow.Data.Add("Label", "Iris-virginia");
            mLDataRows.Add(newRow);
            newRow = new MLDataRow(); newRow.Data = new Dictionary<string, object>(); newRow.Data.Add("SepalLength", 5.9f); newRow.Data.Add("SepalWidth", 3.0f); newRow.Data.Add("PetalLength", 5.1f); newRow.Data.Add("PetalWidth", 1.8f); newRow.Data.Add("Label", "Iris-virginia");
            mLDataRows.Add(newRow);

            mLData.Rows = mLDataRows.ToArray();
        }
    }
}
