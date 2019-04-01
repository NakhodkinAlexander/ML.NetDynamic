using Microsoft.CSharp;
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
    public class ClassGenerator
    {
        private string namespaceName = null;
        private string className = null;

        private CSharpCodeProvider provider = new CSharpCodeProvider();
        private CompilerParameters compilerParameters = new CompilerParameters();
        private CodeCompileUnit compileUnit = new CodeCompileUnit();
        private CodeNamespace namespaces = null;
        private CodeTypeDeclaration customClass = null;
        private CompilerResults compilerResults = null;

        public Type ClassType { get; private set; } = null;
        public string NamespaceName { get; private set; } = null;

        public ClassGenerator(string className,string namespaceName, IEnumerable<string> references = null)
        {
            references?.ToList().ForEach((reference) =>
            {
                compilerParameters.ReferencedAssemblies.Add(reference);
            });
            compilerParameters.ReferencedAssemblies.Add("System.dll");
            compilerParameters.GenerateExecutable = false;
            compilerParameters.GenerateInMemory = false;

            this.NamespaceName = namespaceName;
            compilerParameters.OutputAssembly = $"{namespaceName}.dll"; ;

            this.namespaceName = namespaceName;
            namespaces = new CodeNamespace(namespaceName);
            compileUnit.Namespaces.Add(namespaces);

            this.className = className;
            customClass = new CodeTypeDeclaration(className);
            customClass.IsClass = true;
            customClass.TypeAttributes = System.Reflection.TypeAttributes.Public;
        }

        public void AddField(string fieldName, Type type, MemberAttributes memberAttributes = MemberAttributes.Private)
        {
            CodeMemberField field = new CodeMemberField(type.FullName, fieldName);
            field.Attributes = memberAttributes;
            customClass.Members.Add(field);
        }

        public void AddProperty(string propertyName, Type type)
        {
            this.AddField($"_{propertyName.ToLower()}", type);

            CodeMemberProperty propertyField = new CodeMemberProperty();
            propertyField.Attributes = MemberAttributes.Public;
            propertyField.Name = propertyName;
            propertyField.HasGet = true;
            propertyField.Type = new CodeTypeReference(type.FullName);
            propertyField.GetStatements.Add(new CodeMethodReturnStatement(
                new CodeMethodReferenceExpression(new CodeThisReferenceExpression(), $"_{propertyName.ToLower()}")));
            propertyField.SetStatements.Add(new CodeAssignStatement(
            new CodeFieldReferenceExpression(new CodeThisReferenceExpression(), $"_{propertyName.ToLower()}"), new CodePropertySetValueReferenceExpression()));
            customClass.Members.Add(propertyField);
        }

        public void AddConstructor()
        {
            // Declare the constructor
            CodeConstructor constructor = new CodeConstructor();
            constructor.Attributes =
                MemberAttributes.Public | MemberAttributes.Final;

            customClass.Members.Add(constructor);
        }


        public void Compile()
        {
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

                compilerResults = null;
                throw exception;
            }
            Assembly assembly = compilerResults.CompiledAssembly;
            object instance = assembly.CreateInstance($"{namespaceName}.{className}", true, BindingFlags.Default, null, null, null, null);
            this.ClassType = instance.GetType();
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
