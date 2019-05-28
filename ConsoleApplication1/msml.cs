﻿using Emgu.CV;
using Emgu.CV.ML;
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
            test();
        }
        static void test()
        {
            // load data
            string s = System.IO.File.ReadAllText(@"C:\Tools\avia\iphone_size_trainingdata.json");
            List<Dictionary<string, object>> db = Newtonsoft.Json.JsonConvert.DeserializeObject<List<Dictionary<string, object>>>(s);
            test_svm(db);
        }

        public static int predict_test(double height, double width)
        {
            int ret = 0;
            using (SVM model = new SVM())
            {
                model.Load(@"data\iphone_size_svm.xml");
                Matrix<float> data = new Matrix<float>(1, 2);
                data[0, 0] = (float)height;
                data[0, 1] = (float)width;
                float r = model.Predict(data);
                ret = (int)r;
            }
            return ret;
        }
        static void prep_test_data(List<Dictionary<string, object>> db, out Matrix<float> data)
        {
            data = new Matrix<float>(1, 2);
            data[0, 0] = 136.3014f;
            data[0, 1] = 66.31143f;
        }
        static void load_training_data(List<Dictionary<string,object>> db,  out Matrix<float> data, out Matrix<int> label)
        {
            int rows = db.Count;
            data = new Matrix<float>(rows, 2);
            label = new Matrix<int>(rows, 1);
            for (int i = 0; i < rows; i++)
            {
                Dictionary<string, object> r = db[i];
                data[i, 0] = (float)((double)r["height"]);
                data[i, 1] = (float)((double)r["width"]);
                int v;
                if(Int32.TryParse(r["label"].ToString(), out v))
                {
                    label[i, 0] = v;
                }
            }
        }
        static void test_svm(List<Dictionary<string, object>> db)
        {
            Matrix<float> data;
            Matrix<int> response;
            //ReadColorData(out data, out response);
            load_training_data(db, out data, out response);

            using (SVM model = new SVM())
            {
                //model.KernelType = SVM.SvmKernelType.Linear;
                model.SetKernel(SVM.SvmKernelType.Rbf);
                model.Type = SVM.SvmType.CSvc;
                model.C = 1;
                model.TermCriteria = new Emgu.CV.Structure.MCvTermCriteria(100, 0.00001);

                //SVMParams p = new SVMParams();
                //p.KernelType = Emgu.CV.ML.MlEnum.SVM_KERNEL_TYPE.LINEAR;
                //p.SVMType = Emgu.CV.ML.MlEnum.SVM_TYPE.C_SVC;
                //p.C = 1;
                //p.TermCrit = new MCvTermCriteria(100, 0.00001);
                bool ok = model.Train(data, Emgu.CV.ML.MlEnum.DataLayoutType.RowSample, response);
                Matrix<float> sample;
                //prep_test_data(null,out sample);
                //float r = model.Predict(sample);
                model.Save("iphone_size_svm.xml");
                predict_test(136.3014, 66.31143);
            }
        }

    }
}
