using Microsoft.CSharp;
using Microsoft.Data.DataView;
using System;
using System.CodeDom;
using System.CodeDom.Compiler;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;

namespace ConsoleApp
{
    public class MLTypesGenerator
    {
        private string namespaceName = null;
        private string className = null;

        private CSharpCodeProvider provider = new CSharpCodeProvider();
        private CompilerParameters compilerParameters = new CompilerParameters();
        private CodeCompileUnit compileUnit = new CodeCompileUnit();
        private CodeNamespace namespaces = null;
        private CodeTypeDeclaration customClass = null;
        private CompilerResults compilerResults = null;

        ClassGenerator datSetClass = null;
        ClassGenerator labelClass = null;

        public Type GeneratorType { get; private set; } = null;
        public string OutputDll { get; private set; } = null;

        public MLTypesGenerator(string className, string namespaceName, ClassGenerator datSetClass, ClassGenerator labelClass)
        {
            // Add an assembly reference.
            compilerParameters.ReferencedAssemblies.AddRange(new [] 
            {
                "System.dll",
                "System.Collections.dll",
                "System.Collections.Concurrent.dll",
                "System.Core.dll",
                "netstandard.dll",
                "System.Linq.dll",
                "System.Xml.dll",
                "System.Xml.Linq.dll",
                "System.Linq.Expressions.dll",
                "System.Linq.Parallel.dll",
                "System.Linq.Queryable.dll",
                "Microsoft.ML.Data.dll",
                "Microsoft.ML.Core.dll",
                "Microsoft.Data.DataView.dll",
                $"{datSetClass.namespaceName}.dll",
                $"{labelClass.namespaceName}.dll"
            });
            compilerParameters.GenerateExecutable = false;
            compilerParameters.GenerateInMemory = false;

            this.OutputDll = $"{namespaceName}.dll";
            compilerParameters.OutputAssembly = this.OutputDll;

            this.namespaceName = namespaceName;
            namespaces = new CodeNamespace(namespaceName);
            namespaces.Imports.Add(new CodeNamespaceImport("System"));
            namespaces.Imports.Add(new CodeNamespaceImport("System.Collections"));
            namespaces.Imports.Add(new CodeNamespaceImport("System.Collections.Generic"));
            namespaces.Imports.Add(new CodeNamespaceImport("System.Linq"));
            namespaces.Imports.Add(new CodeNamespaceImport("Microsoft.ML"));
            namespaces.Imports.Add(new CodeNamespaceImport("Microsoft.ML.Data"));
            namespaces.Imports.Add(new CodeNamespaceImport("Microsoft.Data.DataView"));
            namespaces.Imports.Add(new CodeNamespaceImport($"{datSetClass.namespaceName}"));
            namespaces.Imports.Add(new CodeNamespaceImport($"{labelClass.namespaceName}"));
            compileUnit.Namespaces.Add(namespaces);

            this.className = className;
            customClass = new CodeTypeDeclaration(className);
            customClass.IsClass = true;
            customClass.TypeAttributes = System.Reflection.TypeAttributes.Public;

            this.datSetClass = datSetClass;
            this.labelClass = labelClass;
            this.Compile();
        }

        private void AddDataViewGenerator()
        {
            CodeMemberMethod toStringMethod = new CodeMemberMethod();
            toStringMethod.Attributes =
                MemberAttributes.Public | MemberAttributes.Static;
            toStringMethod.Parameters.Add(new CodeParameterDeclarationExpression("List<object>", "listToParse"));

            toStringMethod.Name = "GetDataView";

            CodeSnippetExpression snippet1 = new CodeSnippetExpression("MLContext mlContext = new MLContext();");
            CodeSnippetExpression snippet2 = new CodeSnippetExpression($"IDataView trainingDataView = mlContext.Data.LoadFromEnumerable(listToParse.Select(d => ({this.datSetClass.ClassType.Name})d).ToList());");
            CodeExpressionStatement stmt1 = new CodeExpressionStatement(snippet1);
            CodeExpressionStatement stmt2 = new CodeExpressionStatement(snippet2);
            toStringMethod.Statements.Add(stmt1);
            toStringMethod.Statements.Add(stmt2);

            toStringMethod.ReturnType =
                new CodeTypeReference(typeof(IDataView));

  
            toStringMethod.Statements.Add(new CodeMethodReturnStatement(new CodeArgumentReferenceExpression($"trainingDataView")));
            customClass.Members.Add(toStringMethod);
        }

        private void AddPredictionEngineGenerator()
        {
            CodeMemberMethod toStringMethod = new CodeMemberMethod();
            toStringMethod.Attributes =
                MemberAttributes.Public | MemberAttributes.Static;
            toStringMethod.Parameters.Add(new CodeParameterDeclarationExpression("TransformerChain<Microsoft.ML.Transforms.KeyToValueMappingTransformer>", "model"));

            toStringMethod.Name = "GetPredictionEngine";

            CodeSnippetExpression snippet = new CodeSnippetExpression("MLContext mlContext = new MLContext();");
            CodeExpressionStatement stmt = new CodeExpressionStatement(snippet);
            toStringMethod.Statements.Add(stmt);

            toStringMethod.ReturnType =
                new CodeTypeReference($"PredictionEngine<{datSetClass.namespaceName}.{datSetClass.className}, {labelClass.namespaceName}.{labelClass.className}>");


            toStringMethod.Statements.Add(new CodeMethodReturnStatement(new CodeArgumentReferenceExpression($"model.CreatePredictionEngine<{datSetClass.namespaceName}.{datSetClass.className}, {labelClass.namespaceName}.{labelClass.className}>(mlContext)")));
            customClass.Members.Add(toStringMethod);
        }

        private void AddConstructor()
        {
            CodeConstructor constructor = new CodeConstructor();
            constructor.Attributes =
                MemberAttributes.Public | MemberAttributes.Final;

            customClass.Members.Add(constructor);
        }

        private void Compile()
        {
            AddDataViewGenerator();
            AddPredictionEngineGenerator();
            AddConstructor();
            namespaces.Types.Add(customClass);
            compilerResults = provider.CompileAssemblyFromDom(compilerParameters, compileUnit);
            if (compilerResults.Errors.Count > 0)
            {
                ApplicationException exception = new ApplicationException("Error building assembly");
                foreach (CompilerError ce in compilerResults.Errors)
                {
                    exception.Data.Add(ce.ToString(), ce);
                }
               
                throw exception;
            }
            Assembly assembly = compilerResults.CompiledAssembly;
            object instance = assembly.CreateInstance($"{namespaceName}.{className}", true, BindingFlags.Default, null, null, null, null);
            this.GeneratorType = instance.GetType();
        }

        public object GetInstance()
        {
            if (compilerResults != null)
            {
                Assembly assembly = compilerResults.CompiledAssembly;
                object instance = assembly.CreateInstance($"{namespaceName}.{className}", true, BindingFlags.Default, null, null, null, null);
                return instance;
            }
            else
            {
                throw new Exception("Assembly not compiled");
            }
        }

    }
}
