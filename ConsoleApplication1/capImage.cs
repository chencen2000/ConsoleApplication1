using Accord;
using Accord.Imaging;
using Accord.Imaging.Filters;
using Accord.Imaging.Formats;
using Accord.MachineLearning;
using Accord.Math;
using Accord.Math.Distances;
using Accord.Statistics.Distributions.DensityKernels;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.UI;
using Emgu.CV.Util;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using Tesseract;

namespace ConsoleApplication1
{
    class capImage
    {
        static void Main(string[] args)
        {
            Mat img1 = CvInvoke.Imread(@"C:\Users\qa\Desktop\picture\save_10.jpg", Emgu.CV.CvEnum.ImreadModes.AnyColor);
            Mat img2 = CvInvoke.Imread(@"C:\Users\qa\Desktop\picture\save_11.jpg", Emgu.CV.CvEnum.ImreadModes.AnyColor);
            Rectangle[] blocks = detect_black_rectangle(img1.ToImage<Bgr, Byte>(), img2.ToImage<Bgr, Byte>());
            //Rectangle[] blocks = detect_blue_rectangle(img1.ToImage<Bgr, Byte>(), img2.ToImage<Bgr, Byte>());
            Rectangle rm = Rectangle.Empty;
            foreach(Rectangle r in blocks)
            {
                if (rm == Rectangle.Empty) rm = r;
                else rm = Rectangle.Union(rm, r);
            }
            Image<Bgr, Byte> i = img2.ToImage<Bgr, Byte>().GetSubRect(rm);
            i.Save("temp_2.jpg");
        }
        static Rectangle[] detect_blue_rectangle(Image<Bgr, Byte> img1, Image<Bgr, Byte> img2)
        {
            List<Rectangle> ret = new List<Rectangle>();
            if (img1.Size == img2.Size)
            {
                Image<Bgr, Byte> diff = img2.AbsDiff(img1);
                UMat uimage = new UMat();
                CvInvoke.CvtColor(diff, uimage, ColorConversion.Bgr2Gray);
                UMat pyrDown = new UMat();
                CvInvoke.PyrDown(uimage, pyrDown);
                CvInvoke.PyrUp(pyrDown, uimage);
                MCvScalar m1 = new MCvScalar();
                MCvScalar m2 = new MCvScalar();
                CvInvoke.MeanStdDev(uimage, ref m1, ref m2);
                Image<Gray, Byte> t = uimage.ToImage<Gray, Byte>().ThresholdBinary(new Gray(m1.V0 + m2.V0), new Gray(255));
                uimage = t.ToUMat();
                uimage.Save("temp_1.jpg");
                using (VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint())
                {
                    //Matrix<int> hierarchy = new Matrix<int>(1, contours.Size);
                    Mat hierarchy = new Mat();
                    CvInvoke.FindContours(uimage, contours, hierarchy, RetrType.External, ChainApproxMethod.ChainApproxNone);
                    int count = contours.Size;                    
                    for (int i = 0; i < count; i++)
                    {
                        RotatedRect rr = CvInvoke.MinAreaRect(contours[i]);
                        Rectangle r = rr.MinAreaRect();
                        System.Diagnostics.Trace.WriteLine(string.Format("rect={0}", r));
                        if (r.Width > 50 && r.Height > 50 && r.X >= 0 && r.Y >= 0)
                        {

                            System.Diagnostics.Trace.WriteLine(string.Format("[{1}]: rect={0}", r, i));
                            CvInvoke.Rectangle(diff, rr.MinAreaRect(), new MCvScalar(255, 255, 0, 0));

                            using (VectorOfPoint contour = contours[i])
                            using (VectorOfPoint approxContour = new VectorOfPoint())
                            {
                                CvInvoke.ApproxPolyDP(contour, approxContour, CvInvoke.ArcLength(contour, true) * 0.05, false);
                            }
                        }
                    }
                }
                diff.Save("temp_3.jpg");
            }
            return ret.ToArray();
        }

        static Rectangle[] detect_black_rectangle(Image<Bgr, Byte> img1, Image<Bgr, Byte> img2)
        {
            List<Rectangle> ret = new List<Rectangle>();
            if (img1.Size == img2.Size)
            {
                Image<Bgr, Byte> diff = img2.AbsDiff(img1);
                UMat uimage = new UMat();
                CvInvoke.CvtColor(diff, uimage, ColorConversion.Bgr2Gray);
                UMat pyrDown = new UMat();
                CvInvoke.PyrDown(uimage, pyrDown);
                CvInvoke.PyrUp(pyrDown, uimage);
                MCvScalar m1 = new MCvScalar();
                MCvScalar m2 = new MCvScalar();
                CvInvoke.MeanStdDev(uimage, ref m1, ref m2);
                Image<Gray, Byte> t = uimage.ToImage<Gray, Byte>().ThresholdBinary(new Gray(m1.V0+m2.V0), new Gray(255));
                uimage = t.ToUMat();
                uimage.Save("temp_1.jpg");

                using (VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint())
                {
                    CvInvoke.FindContours(uimage, contours, null, RetrType.External, ChainApproxMethod.ChainApproxNone);
                    int count = contours.Size;
                    for (int i = 0; i < count; i++)
                    {
                        RotatedRect rr = CvInvoke.MinAreaRect(contours[i]);
                        Rectangle r = rr.MinAreaRect();
                        
                        if (r.Width > 50 && r.Height > 50 && r.X>0 && r.Y>0)
                        {
                            System.Diagnostics.Trace.WriteLine(string.Format("rect={0}", r));
                            CvInvoke.Rectangle(diff, rr.MinAreaRect(), new MCvScalar(255, 255, 0, 0));
                            ret.Add(r);
                            using (VectorOfPoint contour = contours[i])
                            using (VectorOfPoint approxContour = new VectorOfPoint())
                            {
                                CvInvoke.ApproxPolyDP(contour, approxContour, CvInvoke.ArcLength(contour, true) * 0.05, false);
                            }
                        }
                    }
                }
                diff.Save("temp_3.jpg");
            }
            return ret.ToArray();
        }

        static void test()
        {
            System.Collections.Generic.Dictionary<string, object> data = new Dictionary<string, object>();
            Bitmap b1 = ImageDecoder.DecodeFromFile(@"C:\Users\qa\Desktop\picture\iphone_icon\scoll_down_selected_icon.jpg");
            var surf = new SpeededUpRobustFeaturesDetector(threshold: 0.0002f, octaves: 5, initial: 2);
            var descriptors = surf.Transform(b1);
            //List<SpeededUpRobustFeaturePoint> descriptors = new List<SpeededUpRobustFeaturePoint>(surf.Transform(b1));
            //double[][] features = descriptors.Apply(d => d.Descriptor);
            List<double[]> features = new List<double[]>();
            foreach(var d in descriptors)
            {
                features.Add(d.Descriptor);
            }
            data.Add("pos", features.ToArray());

            System.Web.Script.Serialization.JavaScriptSerializer jss = new System.Web.Script.Serialization.JavaScriptSerializer();
            string s = jss.Serialize(data);

            /*
            foreach(string s in System.IO.Directory.GetFiles(@"C:\Users\qa\Desktop\picture\iphone_icon"))
            {
                Bitmap b = new Bitmap(s);
                var surf = new SpeededUpRobustFeaturesDetector(threshold: 0.0002f, octaves: 5, initial: 2);
                var descriptors = surf.Transform(b);
                //double[][] features = descriptors.Apply(d => d.Descriptor);


            }
            */
        }
    }
}
