using System;
using System.Collections.Generic;

namespace Decisions.ML
{
    public class MLDataSet
    {
        public MLFeature[] Features { get; set; }
        public MLDataRow[] Rows { get; set; }
    }

    public class MLDataRow
    {
        public string ID { get; set; }
        public Dictionary<string, object> Data { get; set; }
    }

    public class MLFeature
    {
        public string Name { get; set; }
        public Type Type { get; set; }
    }
}
