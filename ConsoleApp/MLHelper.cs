using Decisions.ML;
using Microsoft.Data.DataView;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;

namespace ConsoleApp
{
    public static class MLHelper
    {
        public static ClassGenerator GenerateDataSetClass(Type classType, string className, string classNamespace)
        {
            if (classType.GetTypeInfo().IsClass)
            {
                ClassGenerator classGenerator = new ClassGenerator(className, classNamespace);

                foreach (var item in classType.GetFields())
                {
                    classGenerator.AddField(item.Name, item.FieldType, System.CodeDom.MemberAttributes.Public);
                }
                classGenerator.Compile();

                return classGenerator;
            }
            else
            {
                throw new Exception("Type not a class");
            }
        }

        public static ClassGenerator GenerateDataSetClass(IEnumerable<MLFeature> features, string className, string classNamespace)
        {
            ClassGenerator classGenerator = new ClassGenerator(className, classNamespace);

            foreach (var item in features)
            {
                classGenerator.AddField(item.Name, item.Type, System.CodeDom.MemberAttributes.Public);
            }
            classGenerator.Compile();

            return classGenerator;
        }

        public static ClassGenerator GenerateLabelClass(string className, string classNamespace)
        {
            Dictionary<string, string> attributes = new Dictionary<string, string>();
            attributes.Add("ColumnNameAttribute", "PredictedLabel");
            ClassGenerator labelClassGenerator = new ClassGenerator(className, classNamespace);
            labelClassGenerator.AddField("PredictedLabels", typeof(string), System.CodeDom.MemberAttributes.Public, attributes);
            labelClassGenerator.Compile();

            return labelClassGenerator;
        }

        public static MLTypesGenerator CreateTypesGenarator(ClassGenerator classGenerator, ClassGenerator labelClassGenerator)
        {
            MLTypesGenerator typesGenerator = new MLTypesGenerator($"{classGenerator.className}TypesGnerator", $"{classGenerator.className}TypesGneratorNamespace", classGenerator, labelClassGenerator);
            return typesGenerator;
        }

        public static IDataView GetDataView(MLTypesGenerator typesGenerator, List<object> generatedDataSet)
        {
            var type = typesGenerator.GeneratorType;
            var methodInfo = type.GetMethod("GetDataView");
            var dataView = methodInfo.Invoke(null, new object[] { generatedDataSet.ToList() });

            return (IDataView)dataView;
        }

        public static object GetPredictionEngine(MLTypesGenerator typesGenerator, TransformerChain<Microsoft.ML.Transforms.KeyToValueMappingTransformer> model)
        {
            var methodInfo = typesGenerator.GeneratorType.GetMethod("GetPredictionEngine");
            var predictionEngine = methodInfo.Invoke(null, new object[] { model });

            return predictionEngine;
        }

        public static object Predict(object predictionEngine, ClassGenerator classGenerator, object example)
        {
            var methodInfo = predictionEngine.GetType().GetMethod("Predict", new[] { classGenerator.ClassType });
            var prediction = methodInfo.Invoke(predictionEngine, new object[] { example });

            return prediction;
        }
    }
}
