using System;
using System.Collections.Generic;
using System.Dynamic;
using System.Linq;
using System.Threading.Tasks;

namespace DynamicStructure
{
    public class DynamicDictionary: DynamicObject
    {
        private Dictionary<string, object> dictionary = new Dictionary<string, object>();

        public override IEnumerable<string> GetDynamicMemberNames()
        {
            List<string> result = new List<string>();

            foreach (var item in dictionary)
            {
                result.Add(item.Key);
            }

            return result;
        }

        public bool AddField(string name, object value)
        {
            object tryToGet = null;
            if (!dictionary.TryGetValue(name , out tryToGet))
            {
                dictionary[name] = value;
                return true;
            }
            else
            {
                return false;
            }
        }

        public override bool TryCreateInstance(CreateInstanceBinder binder, object[] args, out object result)
        {
            return base.TryCreateInstance(binder, args, out result);
        }

        public override bool TryGetMember(
            GetMemberBinder binder, out object result)
        {
            string name = binder.Name.ToLower();

            return dictionary.TryGetValue(name, out result);
        }

        public override bool TrySetMember(
            SetMemberBinder binder, object value)
        {
            dictionary[binder.Name.ToLower()] = value;
            return true;
        }
    }

    public class Field
    {
        public Field(string name, Type type)
        {
            this.FieldName = name;
            this.FieldType = type;
        }

        public string FieldName;

        public Type FieldType;
    }
}
