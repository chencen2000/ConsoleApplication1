using Emgu.CV;
using Emgu.CV.Features2D;
using Emgu.CV.Flann;
using Emgu.CV.ML;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using Emgu.CV.XFeatures2D;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConsoleApplication1
{
    class opencv
    {
        static void Main(string[] args)
        {
            // BOW
            //test();
            test_haar();
        }

        static void test()
        {
            int clusters = 500;
            SURF surf = new SURF(300);
            string dir1 = @"C:\Users\qa\Desktop\picture\iphone_icon";
            MCvTermCriteria term = new MCvTermCriteria(10000, 0.0001d);
            BOWKMeansTrainer bowtrainer = new BOWKMeansTrainer(clusters, term, 5, Emgu.CV.CvEnum.KMeansInitType.PPCenters);
            int count = 0;
            foreach (string s in System.IO.Directory.GetFiles(dir1))
            {
                Mat m = CvInvoke.Imread(s, Emgu.CV.CvEnum.ImreadModes.AnyColor);
                VectorOfKeyPoint kp = new VectorOfKeyPoint();
                Mat desc = new Mat();
                surf.DetectAndCompute(m, null, kp, desc, false);
                bowtrainer.Add(desc);
                count++;
            }
            using (FileStorage fs = new FileStorage("test.yaml", FileStorage.Mode.Write))
            {
                Mat m = new Mat();
                bowtrainer.Cluster(m);
                fs.Write(m, "vocab");
                fs.ReleaseAndGetString();
            }

            Mat voc = new Mat();
            bowtrainer.Cluster(voc);
            BFMatcher matcher = new BFMatcher(DistanceType.L2);
            BOWImgDescriptorExtractor bowDE = new BOWImgDescriptorExtractor(surf, matcher); //new FlannBasedMatcher(new LinearIndexParams(), new SearchParams()));
            bowDE.SetVocabulary(voc);
            Matrix<int> label = new Matrix<int>(count, 1);
            //Matrix<float> train_data = new Matrix<float>(count, 1);
            Mat train_data = new Mat();
            count = 0;
            foreach (string s in System.IO.Directory.GetFiles(dir1))
            {
                Mat m = CvInvoke.Imread(s, Emgu.CV.CvEnum.ImreadModes.AnyColor);
                MKeyPoint [] kp = surf.Detect(m);
                Mat desc = new Mat();
                bowDE.Compute(m, new VectorOfKeyPoint(kp), desc);
                train_data.PushBack(desc);
                if (string.Compare(System.IO.Path.GetFileNameWithoutExtension(s), "setting_icon") == 0)
                    label[count, 0] = 1;
                else label[count, 0] = 0;
                count++;
            }

            SVM svm = new SVM();
            svm.C = 312.5;
            svm.Gamma = 0.50625000000000009;
            svm.SetKernel(SVM.SvmKernelType.Rbf);
            svm.Type = SVM.SvmType.CSvc;
            svm.TermCriteria = new MCvTermCriteria(1, 1e-6);
            bool td = svm.Train(train_data, Emgu.CV.ML.MlEnum.DataLayoutType.RowSample, label);

            // test
            {
                Mat img = CvInvoke.Imread(@"C:\Users\qa\Desktop\picture\iphone_icon\phone_icon.jpg", Emgu.CV.CvEnum.ImreadModes.AnyColor);
                MKeyPoint[] kp = surf.Detect(img);
                Mat desc = new Mat();
                bowDE.Compute(img, new VectorOfKeyPoint(kp), desc);
                float classVal = svm.Predict(desc);
                float scoreVal = svm.Predict(desc, null, 1);
            }
        }
        static void test_haar()
        {
            CascadeClassifier haar_email = new CascadeClassifier(@"C:\Users\qa\Desktop\picture\haar\ios_settings_icon\cascade.xml");
            Mat img = CvInvoke.Imread(@"C:\Users\qa\Desktop\picture\test_02.jpg", Emgu.CV.CvEnum.ImreadModes.AnyColor);
            var b = img.ToImage<Bgr, Byte>();
            var email_rect = haar_email.DetectMultiScale(b, 1.1);
        }
    }
}
