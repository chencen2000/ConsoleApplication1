using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConsoleApplication1
{
    class msml
    {
        static void Main(string[] args)
        {
            string s = System.IO.File.ReadAllText(@"C:\Users\qa\Source\Repos\chencen2000\ConsoleApplication1\ConsoleApplication1\data\iPhonsSize.json");
            List<Dictionary<string, object>> db = Newtonsoft.Json.JsonConvert.DeserializeObject<List<Dictionary<string, object>>>(s);
        }
    }
}
