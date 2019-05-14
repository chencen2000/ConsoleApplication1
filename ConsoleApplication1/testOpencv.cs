using Accord.Imaging;
using Emgu.CV;
using Emgu.CV.Cuda;
using Emgu.CV.CvEnum;
using Emgu.CV.Features2D;
using Emgu.CV.Flann;
using Emgu.CV.ML;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using Emgu.CV.XFeatures2D;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using Tesseract;

namespace ConsoleApplication1
{
    class testOpencv
    {
        static void Main(string[] args)
        {
            //test_edge_detect_1();
            //check_apple_logo();
            //test_apple_logo_v3();
            //test_apple_logo();
            //test();
            //test_ocr();
            //check_image_similarity();
            //test_skelton();
            //pre_process();
            //extra_icon();
            //find_focused_item();
            //test_3();
            //test_1();
            //test();
            //test();
            //test_surf();
            //Image<Bgra, byte> img1 = new Image<Bgra, byte>(@"C:\Users\qa\Desktop\picture\menu_1.jpg");
            //Image<Bgra, byte> img2 = new Image<Bgra, byte>(@"C:\Users\qa\Desktop\picture\iphone_icon\scroll_left_icon.jpg");
            //long l;
            //Mat m = DrawMatches.Draw(img2.Mat, img1.Mat, out l);
            //m.Save("temp_1.jpg");
            //find_focused_menu();
            //Image<Bgra, byte> source = new Image<Bgra, byte>(@"C:\Users\qa\Desktop\picture\WIN_20180822_11_29_39_Pro.jpg");
            //source.Rotate(-2.3, new Bgra(0, 0, 0, 255), false).Save("temp_1.jpg");
            //source.Save("temp_1.jpg");
            //LineSegment2D v1 = new LineSegment2D(new Point(100, 100), new Point(100, 1000));
            //LineSegment2D v2 = new LineSegment2D(new Point(100, 1000), new Point(100, 100));
            //LineSegment2D v3 = new LineSegment2D(new Point(100, 100), new Point(1000, 100));
            //if (v1.Direction.X < 0 || v1.Direction.Y < 0)
            //{
            //    Point p1 = v1.P1;
            //    Point p2 = v1.P2;
            //    v1 = new LineSegment2D(p2, p1);
            //}
            //double a = v1.GetExteriorAngleDegree(v3);
            //a = v2.GetExteriorAngleDegree(v3);
        }
        static void check_image_similarity()
        {
            Mat b1 = CvInvoke.Imread(@"C:\Users\qa\Desktop\picture\save_06.jpg");
            Mat b2 = CvInvoke.Imread(@"C:\Users\qa\Desktop\picture\save_10.jpg");
            Mat s1 = new Mat();
            CvInvoke.AbsDiff(b1, b2, s1);
            Mat s1f = new Mat();
            s1.ConvertTo(s1f, DepthType.Cv32F);
            CvInvoke.Multiply(s1f, s1f, s1f);
            MCvScalar sum = CvInvoke.Sum(s1f);
            double sse = sum.V0 + sum.V1 + sum.V2;
            double mse = sse / ((double)b1.NumberOfChannels * b1.Total.ToInt32());
            double psnr = 10.0 * Math.Log10((255 * 255) / mse);
            Program.logIt(string.Format("{0}", psnr));
        }
        static void pre_process()
        {
            Image<Bgra, byte> source = new Image<Bgra, byte>(@"C:\Users\qa\Desktop\picture\save_100.jpg");
            UMat uimage = new UMat();
            CvInvoke.CvtColor(source, uimage, ColorConversion.Bgr2Gray);
            UMat pyrDown = new UMat();
            CvInvoke.PyrDown(uimage, pyrDown);
            CvInvoke.PyrUp(pyrDown, uimage);
            //CvInvoke.Threshold(uimage, pyrDown, 100, 255, ThresholdType.Binary);
            //uimage = pyrDown;
            double cannyThreshold = 180.0;
            double cannyThresholdLinking = 120.0;
            UMat cannyEdges = new UMat();
            CvInvoke.Canny(uimage, cannyEdges, cannyThreshold, cannyThresholdLinking);
            CvInvoke.GaussianBlur(cannyEdges, uimage, new Size(5, 5), 0);
            uimage.Save("temp_1.jpg");

            Mat disp = new Mat();
            CvInvoke.CvtColor(cannyEdges, disp, ColorConversion.Gray2Bgr);
            Image<Bgra, byte> rotated = null;
            {
                LineSegment2D[] lines = CvInvoke.HoughLinesP(
                   cannyEdges, //cannyEdges,
                   1, //Distance resolution in pixel-related units
                   Math.PI / 180.0, //Angle resolution measured in radians.
                   20, //threshold
                   150, //min Line width
                   10); //gap between lines
                LineSegment2D vl = new LineSegment2D(new Point(100, 100), new Point(100, 1000));
                //Matrix<float> angles = new Matrix<float>(lines.Length, 1);
                List<double> all_a= new List<double>();
                double ratio = 1.0;
                for (int i = 0; i < lines.Length; i++)
                {
                    CvInvoke.Line(disp, lines[i].P1, lines[i].P2, new MCvScalar(0, 0, 255));
                    //.Draw(lines[i], new Bgra(0, 0, 255, 255), 1);
                    int d_y = (lines[i].P1.Y - lines[i].P2.Y);
                    int d_x = (lines[i].P1.X - lines[i].P2.X);
                    double k = (double)d_y / d_x;
                    double t = Math.Abs(k);
                    double a = vl.GetExteriorAngleDegree(lines[i]);
                    a = Math.Atan(t) * (180 / Math.PI);
                    if (a > 45.0)
                    {
                        a = 90.0 - a;
                        if (k > 0) ratio = 1.0;
                        else ratio = -1.0;
                    }
                    else
                    {
                        if (k > 0) ratio = -1.0;
                        else ratio = 1.0;
                    }
                    //angles[i, 0] = Convert.ToSingle(a);
                    all_a.Add(a);
                    Program.logIt(string.Format("{0}-{1}, {3} len={2}", lines[i].P1, lines[i].P2, lines[i].Length, a));
                }
                //float af = 0.0f;
                double af = all_a.Average();
                rotated = source.Rotate(af * ratio, new Bgra(), true);
                rotated.Save("temp_1.jpg");
                //source.Save("temp_2.jpg");
                //source.Rotate(0.95f, new Bgra(), true).Save("temp_3.jpg");
                /*
                MCvTermCriteria term = new MCvTermCriteria();
                Matrix<int> label = new Matrix<int>(lines.Length, 1);
                CvInvoke.Kmeans(angles, 2, label, term, 2, KMeansInitType.RandomCenters);
                List<float> a0 = new List<float>();
                List<float> a1 = new List<float>();
                for (int i = 0; i < lines.Length; i++)
                {
                    if (label[i, 0] == 0) a0.Add(angles[i, 0]);
                    else if (label[i, 0] == 1) a1.Add(angles[i, 0]);
                }
                float a0_m = a0.Average();
                float a1_m = a1.Average();
                float af = 0.0f;
                if (Math.Abs(a0_m) < 45.0f) af = a0_m;
                else af = a1_m;
                rotated = source.Rotate(-af, new Bgra(), true);
                //CvInvoke.Imshow("a", rotated);
                //CvInvoke.WaitKey(0);
                //CvInvoke.DestroyAllWindows();
                rotated.Save("temp_1.jpg");
                */
            }
            disp.Save("temp_0.jpg");
            return;
            // 
            if (rotated != null)
            {
                uimage = new UMat();
                CvInvoke.CvtColor(rotated, uimage, ColorConversion.Bgr2Gray);
                pyrDown = new UMat();
                CvInvoke.PyrDown(uimage, pyrDown);
                CvInvoke.PyrUp(pyrDown, uimage);
                MCvScalar mean = new MCvScalar();
                MCvScalar std_dev = new MCvScalar();
                CvInvoke.MeanStdDev(uimage, ref mean, ref std_dev);
                CvInvoke.Threshold(uimage, pyrDown, mean.V0 + std_dev.V0, 255, ThresholdType.Binary);
                uimage = pyrDown;
                uimage.Save("temp_2.jpg");
                Rectangle roi = Rectangle.Empty;
                using (VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint())
                {
                    CvInvoke.FindContours(uimage, contours, null, RetrType.External, ChainApproxMethod.ChainApproxNone);
                    int count = contours.Size;
                    for (int i = 0; i < count; i++)
                    {
                        double a1 = CvInvoke.ContourArea(contours[i], false);
                        Rectangle r = CvInvoke.BoundingRectangle(contours[i]);
                        if (roi.IsEmpty) roi = r;
                        else roi = Rectangle.Union(roi, r);
                        //Program.logIt(string.Format("{0}", r));
                        //if (a1 > 1000.0)
                        //{
                        //    RotatedRect rr = CvInvoke.MinAreaRect(contours[i]);
                        //    double a2 = rr.Size.Height * rr.Size.Width;
                        //    double r = a1 / a2;
                        //    if (r > 0.9)
                        //    {
                        //        source.Draw(rr.MinAreaRect(), new Bgra(0, 0, 255, 255));
                        //        //rs.Add(rr.MinAreaRect());
                        //    }
                        //}
                    }
                }
                Image<Bgra, byte> copped = rotated.GetSubRect(roi);
                copped.Save("temp_3.jpg");
            }
        }
        static void extra_icon()
        {
            double margin_left = 0.07;
            double margin_width = 0.07;
            double margin_top = 0.034;
            double top_of_botton_line = 0.88;
            double margin_height = 0.04;
            double icon_width = 0.163;
            double icon_higth = 0.093;
            double icon_marging = 0.1;
            Image<Bgra, byte> source = new Image<Bgra, byte>(@"C:\Users\qa\Desktop\picture\test_02.jpg");
            double y = margin_top * source.Height;
            double w = icon_width * source.Width;
            double h = icon_higth * source.Height;
            double botton_y = top_of_botton_line * source.Height;
            int row = 1;
            int col = 1;
            while (y+h < botton_y)
            {
                col = 1;
                double x = margin_left * source.Width;
                while (x + w < source.Width)
                {
                    double xr = x - w * icon_marging;
                    double yr = y - h * icon_marging;
                    double wr = w + w * icon_marging * 2;
                    double hr = h + h * icon_marging * 2;
                    Rectangle r = new Rectangle((int)xr, (int)yr, (int)wr, (int)hr);
                    source.GetSubRect(r).Save(string.Format(@"icons\icon_{0}_{1}.jpg", row, col++));
                    source.Draw(r, new Bgra(0, 0, 255, 255));
                    x = x + w + margin_width * source.Width;
                }
                y = y + h + margin_height * source.Height;
                row++;
            }
            // dotton line
            {
                row = 0;
                col = 1;
                y = botton_y;
                double x = margin_left * source.Width;
                while (x + w < source.Width)
                {
                    double xr = x - w * icon_marging;
                    double yr = y - h * icon_marging;
                    double wr = w + w * icon_marging * 2;
                    double hr = h + h * icon_marging * 2;
                    Rectangle r = new Rectangle((int)xr, (int)yr, (int)wr, (int)hr);
                    source.GetSubRect(r).Save(string.Format(@"icons\icon_{0}_{1}.jpg", row, col++));
                    source.Draw(r, new Bgra(0, 0, 255, 255));
                    x = x + w + margin_width * source.Width;
                }
            }
            source.Save("temp_2.jpg");
        }
        static void find_focused_menu_item()
        {
            Image<Bgra, byte> source = new Image<Bgra, byte>(@"C:\Users\qa\Desktop\picture\menu_2.jpg");
            UMat uimage = new UMat();
            CvInvoke.CvtColor(source, uimage, ColorConversion.Bgr2Gray);
            UMat pyrDown = new UMat();
            CvInvoke.PyrDown(uimage, pyrDown);
            CvInvoke.PyrUp(pyrDown, uimage);
            double cannyThreshold = 180.0;
            double cannyThresholdLinking = 120.0;
            UMat cannyEdges = new UMat();
            CvInvoke.Canny(uimage, cannyEdges, cannyThreshold, cannyThresholdLinking);
            cannyEdges.Save("temp_1.jpg");

            List<Rectangle> rs = new List<Rectangle>();
            using (VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint())
            {
                CvInvoke.FindContours(cannyEdges, contours, null, RetrType.External, ChainApproxMethod.ChainApproxNone);
                int count = contours.Size;
                for (int i = 0; i < count; i++)
                {
                    double a1 = CvInvoke.ContourArea(contours[i], false);
                    //if (a1 > 1000.0)
                    {
                        RotatedRect rr = CvInvoke.MinAreaRect(contours[i]);
                        double a2 = rr.Size.Height * rr.Size.Width;
                        double r = a1 / a2;
                        if(r > 0.9)
                        {
                            source.Draw(rr.MinAreaRect(), new Bgra(255, 0, 0, 255));
                            rs.Add(rr.MinAreaRect());
                        }
                    }
                }
            }

            foreach (Rectangle r in rs)
                Program.logIt(string.Format("{0}", r));

            CvInvoke.Imshow("a", source);
            CvInvoke.WaitKey(0);
            CvInvoke.DestroyAllWindows();
        }
        static void find_focused_item()
        {
            Image<Bgra, byte> source = new Image<Bgra, byte>(@"C:\Users\qa\Desktop\picture\save_00.jpg");
            UMat uimage = new UMat();
            CvInvoke.CvtColor(source, uimage, ColorConversion.Bgr2Gray);
            UMat pyrDown = new UMat();
            CvInvoke.PyrDown(uimage, pyrDown);
            CvInvoke.PyrUp(pyrDown, uimage);
            double cannyThreshold = 180.0;
            double cannyThresholdLinking = 120.0;
            UMat cannyEdges = new UMat();
            CvInvoke.Canny(uimage, cannyEdges, cannyThreshold, cannyThresholdLinking);
            cannyEdges.Save("temp_1.jpg");

            List<Rectangle> rs = new List<Rectangle>();
            using (VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint())
            {
                CvInvoke.FindContours(cannyEdges, contours, null, RetrType.External, ChainApproxMethod.ChainApproxNone);
                int count = contours.Size;
                for (int i = 0; i < count; i++)
                {
                    double a1 = CvInvoke.ContourArea(contours[i], false);
                    if (a1 > 1000.0)
                    {
                        RotatedRect rr = CvInvoke.MinAreaRect(contours[i]);
                        double a2 = rr.Size.Height * rr.Size.Width;
                        double r = a1 / a2;
                        if (r > 0.9)
                        {
                            source.Draw(rr.MinAreaRect(), new Bgra(0, 0, 255, 255));
                            rs.Add(rr.MinAreaRect());
                        }
                    }
                }
            }

            foreach (Rectangle r in rs)
                Program.logIt(string.Format("{0}", r));

            source.Save("temp_0.jpg");
            //CvInvoke.Imshow("a", source);
            //CvInvoke.WaitKey(0);
            //CvInvoke.DestroyAllWindows();
        }
        static void test_3()
        {
            string src = @"C:\Tools\avia\Recog source\AP002-iphone6_Gray\1733.1.bmp";
            Mat b0 = CvInvoke.Imread(src, ImreadModes.Grayscale);
            Mat b1 = new Mat();

            //CvInvoke.GaussianBlur(b0, b1, new Size(3,3), 0, 0, BorderType.Default);
            //CvInvoke.Laplacian(b1, b0, DepthType.Cv16S, 3,1,0, BorderType.Replicate);
            //CvInvoke.ConvertScaleAbs(b0, b1, 1, 0);
            //b1.Save("temp_2.bmp");

            CvInvoke.GaussianBlur(b0, b1, new Size(3, 3), 0, 0, BorderType.Default);
            Mat dx = new Mat();
            Mat dy = new Mat();
            CvInvoke.Sobel(b1, dx, DepthType.Cv16S, 1, 0);
            CvInvoke.ConvertScaleAbs(dx, dx, 1, 0);
            CvInvoke.Sobel(b1, dy, DepthType.Cv16S, 0, 1);
            CvInvoke.ConvertScaleAbs(dy, dy, 1, 0);
            CvInvoke.AddWeighted(dx, 0.5, dy, 0.5, 0, b1);
            b1.Save("temp_2.bmp");

            DenseHistogram hist = new DenseHistogram(256, new RangeF(0.0f, 255.0f));
            hist.Calculate(new Image<Gray, Byte>[] { b1.ToImage<Gray, Byte>() }, true, null);
            float[] bins = hist.GetBinValues();
            int v = 0;
            float count = 0;
            for(int i = bins.Length - 1; i >= 0 && v==0; i--)
            {
                count += bins[i];
                float r = count / b1.Total.ToInt64();
                if (r > 0.002f)
                    v = i;
            }

            CvInvoke.Threshold(b1, b0, v, 255, ThresholdType.Binary);
            b0.Save("temp_3.bmp");


        }
        static void test_skelton()
        {
            Mat m = CvInvoke.Imread(@"C:\Users\qa\Desktop\picture\iphone_icon\app_switch_close_1.jpg", ImreadModes.Grayscale);
            Mat gm = new Mat();
            double f = CvInvoke.Threshold(m, gm, 0, 255, ThresholdType.Otsu);
            CvInvoke.BitwiseNot(gm, m);
            m.Save("temp_1.jpg");
            Mat element = new Mat();
            CvInvoke.GetStructuringElement(ElementShape.Cross, new Size(3, 3), new Point(-1, -1));
            bool done = false;
            gm = Mat.Ones(m.Rows, m.Cols, DepthType.Cv8U, 1);
            Mat temp = new Mat();
            Mat erode = new Mat();
            do
            {
                CvInvoke.Erode(m, erode, element, new Point(-1, -1), 1, BorderType.Constant, new MCvScalar(255));
                CvInvoke.Dilate(erode, temp, element, new Point(-1, -1), 1, BorderType.Constant, new MCvScalar(255));
                CvInvoke.Subtract(m, temp, temp);
                CvInvoke.BitwiseOr(gm, temp, gm);
                erode.CopyTo(m);
                done = (CvInvoke.CountNonZero(m) == 0);
            } while (!done);
            gm.Save("temp_2.jpg");
        }
        static void test_1()
        {
            Bitmap b1 = new Bitmap(@"C:\Users\qa\Desktop\picture\save_01.jpg");
            Bitmap b2 = new Bitmap(@"C:\Users\qa\Desktop\picture\save_06.jpg");
            Bitmap menu = null;
            {
                Image<Gray, Byte> bb1 = new Image<Gray, byte>(b1);
                Image<Gray, Byte> bb2 = new Image<Gray, byte>(b2);
                Mat diff = new Mat();
                CvInvoke.AbsDiff(bb1, bb2, diff);
                diff.Save("temp_1.jpg");
                Mat tmp = new Mat();
                CvInvoke.Threshold(diff, tmp, 0, 255, ThresholdType.Binary | ThresholdType.Otsu);
                tmp.Save("temp_2.jpg");
                Rectangle ret = Rectangle.Empty;
                using (VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint())
                {
                    diff = tmp;
                    CvInvoke.FindContours(diff, contours, null, RetrType.External, ChainApproxMethod.ChainApproxNone);
                    int count = contours.Size;
                    for (int i = 0; i < count; i++)
                    {
                        double d = CvInvoke.ContourArea(contours[i]);
                        if (d > 100.0)
                        {
                            Rectangle rect = CvInvoke.BoundingRectangle(contours[i]);
                            Program.logIt(string.Format("{0}: {1}", d, rect));
                            if (ret.IsEmpty) ret = rect;
                            else ret = Rectangle.Union(ret, rect);
                        }
                    }
                }
                Program.logIt(string.Format("{0}", ret));
                Image<Bgr, Byte> b = new Image<Bgr, byte>(b2);
                menu = b.GetSubRect(ret).Bitmap;
            }
            if (menu != null)
            {
                Image<Gray, Byte> bb1 = new Image<Gray, byte>(menu);
                Mat tmp = new Mat();
                CvInvoke.Threshold(bb1, tmp, 0, 255, ThresholdType.Binary | ThresholdType.Otsu);
                tmp.Save("temp_1.jpg");
                Rectangle ret = Rectangle.Empty;
                using (VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint())
                {
                    CvInvoke.FindContours(tmp, contours, null, RetrType.External, ChainApproxMethod.ChainApproxNone);
                    int count = contours.Size;
                    double m = 0.0;
                    for (int i = 0; i < count; i++)
                    {
                        double d = CvInvoke.ContourArea(contours[i]);
                        if (d > m)
                        {
                            m = d;
                            ret = CvInvoke.BoundingRectangle(contours[i]);
                            Program.logIt(string.Format("{0}: {1}", d, ret));
                        }
                    }
                }
                Image<Bgr, Byte> b = new Image<Bgr, byte>(menu);
                b.GetSubRect(ret).Save("temp_2.jpg");
            }
        }
        static void test_ocr()
        {
            using (TesseractEngine TE = new TesseractEngine("tessdata", "eng", EngineMode.TesseractOnly))
            {
                Bitmap b = new Bitmap(@"C:\test\avia\image_1.bmp");
                var p = TE.Process(b);
                string s = p.GetText();
                s = p.GetHOCRText(0);
            }
        }
        static void test()
        {
            string folder = @"C:\tools\avia\Recog source\AP001-iphone6_gold";
            string[] files = System.IO.Directory.GetFiles(folder, "*.bmp", SearchOption.AllDirectories);

            //string src = @"C:\tools\avia\Recog source\AP002-iphone6_Gray\1733.1.bmp";
            foreach (string src in files)
            {
                Mat b1 = CvInvoke.Imread(src);
                DenseHistogram h = new DenseHistogram(256, new RangeF(0f, 255f));
                h.Calculate(new Image<Gray, Byte>[] { b1.ToImage<Gray, Byte>() }, false, null);
                float count = 0.0f;
                float r = 0.0f;
                float[] binValues = h.GetBinValues();
                int value = -1;
                for (int i = 255; i >= 0 && r < 0.008f; i--)
                {
                    count += binValues[i];
                    r = count / b1.Total.ToInt64();
                    if (r > 0.008f)
                    {
                        value = i;
                        Program.logIt($"binValue: {value}");
                    }
                }
                // 
                Mat b0 = new Mat();
                CvInvoke.Threshold(b1, b0, value, 255, ThresholdType.Binary);
                b0.Save($@"test\{System.IO.Path.GetFileName(src)}");
            }
        }
        static void test_edge_detect_1()
        {
            string src = @"C:\tools\avia\Recog source\AP002-iphone6_Gray\1733.1.bmp";
            Mat b0 = CvInvoke.Imread(src);
            Mat b1 = new Mat();
            Mat b2 = new Mat();
            Size size = b0.Size;
            CvInvoke.Resize(b0, b1, new Size(size.Width / 10, size.Height / 10));
            CvInvoke.GaussianBlur(b1, b2, new Size(3, 3), 0);
            CvInvoke.MedianBlur(b2, b1, 3);
            Mat dx = new Mat(b1.Rows, b1.Cols, DepthType.Cv16S, 1);
            Mat dy = new Mat(b1.Rows, b1.Cols, DepthType.Cv16S, 1);
            CvInvoke.Sobel(b1, dx, DepthType.Cv16S, 1, 0);
            CvInvoke.Sobel(b1, dy, DepthType.Cv16S, 0, 1);
            MCvScalar stddev = new MCvScalar();
            MCvScalar mean = new MCvScalar();
            CvInvoke.MeanStdDev(b1, ref mean, ref stddev);
            //CvInvoke.MeanStdDev(dx, mean, stddev);
            //CvInvoke.MeanStdDev(dy, mean, stddev);
            //CvInvoke.Canny(dx, dy, b1,  , 90);
            b1.Save("temp_1.bmp");
        }
        static void test_edge_detect()
        {
            string src = @"C:\tools\avia\Recog source\AP002-iphone6_Gray\1733.1.bmp";
            Mat b1 = CvInvoke.Imread(src);
            Mat b2 = new Mat();
            Size size = b1.Size;
            CvInvoke.Resize(b1, b2, new Size(size.Width / 10, size.Height / 10));
            CvInvoke.GaussianBlur(b2, b1, new Size(3, 3), 0);
            CvInvoke.MedianBlur(b1, b2, 3);
            b1 = b2;
            size = b1.Size;
            // 
            Mat dx = new Mat(b1.Rows, b1.Cols, DepthType.Cv16S, 1);
            Mat dy = new Mat(b1.Rows, b1.Cols, DepthType.Cv16S, 1);
            CvInvoke.Sobel(b1, dx, DepthType.Cv16S, 1, 0);
            CvInvoke.Sobel(b1, dy, DepthType.Cv16S, 0, 1);
            //
            Matrix<Int16> dx_m = new Matrix<Int16>(dx.Rows, dx.Cols, dx.NumberOfChannels);
            dx.CopyTo(dx_m);
            Matrix<Int16> dy_m = new Matrix<Int16>(dy.Rows, dy.Cols, dy.NumberOfChannels);
            dy.CopyTo(dy_m);
            //Image<Gray, Int16> idx = dx.ToImage<Gray, Int16>();
            //Mat img = Mat.Zeros(b1.Rows, b1.Cols, DepthType.Cv32F, 1);
            Matrix<float> img_m = new Matrix<float>(b1.Rows, b1.Cols, b1.NumberOfChannels);
            
            float maxV = 0.0f;
            float data;
            for(int i=0; i < size.Height; i++)
            {
                for(int j=0; j < size.Width; j++)
                {
                    data = Math.Abs(dx_m.Data[i,j]) + Math.Abs(dy_m.Data[i,j]);
                    //img.ToImage<Gray, float>().Data[i, j, 0] = data;
                    img_m.Data[i, j] = data;
                    maxV = maxV < data ? data : maxV;
                }
            }
            //img.Save("temp_1.bmp");
            //b2.Save("temp_1.bmp");
            int bin_size = Math.Max((int)maxV, 256);
            RangeF r = new RangeF(0.0f, maxV);
            DenseHistogram h = new DenseHistogram(bin_size, r);
            h.Calculate(new Matrix<float>[] { img_m }, false, null);
            float[] binValues = h.GetBinValues();
            float th = 0.8f * size.Width * size.Height;
            float sum = 0.0f;
            int the_low_value = 0;
            for(int i = 0; i < binValues.Length; i++)
            {
                sum += binValues[i];
                if (sum > th)
                {
                    the_low_value = i;
                    break;
                }
            }
            double the_high_value = 2.5 * the_low_value;
            b1 = CvInvoke.Imread(src);
            b1.ToImage<Gray, byte>().Canny(the_low_value, the_high_value).Save("temp_1.bmp");
        }
        static void check_apple_logo()
        {
            Mat b0 = CvInvoke.Imread(@"data\Apple_logo.png", ImreadModes.Grayscale);
            Image<Gray, Byte> img = b0.ToImage<Gray, Byte>();
            Image<Gray, Byte> img_logo = img.Not();
            b0 = CvInvoke.Imread(@"C:\Tools\avia\Recog source\AP001-iphone6_gold\0123.1.bmp", ImreadModes.Grayscale);
            img = b0.ToImage<Gray, Byte>();
            //img = img.Resize(0.1, Inter.Cubic);
            CvInvoke.EqualizeHist(img, b0);
            b0.Save("temp_1.bmp");
            Mat mean = new Mat();
            Mat stddev = new Mat();
            img = img.Rotate(-90.0, new Gray(0));
            CvInvoke.MeanStdDev(img.Mat, mean, stddev);
            img = img.ThresholdBinary(new Gray(127), new Gray(255));
            img.Save("temp_1.bmp");
            img = img.Erode(3);
            img.Save("temp_2.bmp");
            Rectangle ret = Rectangle.Empty;
            using (VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint())
            {
                CvInvoke.FindContours(img, contours, null, RetrType.External, ChainApproxMethod.ChainApproxNone);
                int count = contours.Size;
                for (int i = 0; i < count; i++)
                {
                    double d = CvInvoke.ContourArea(contours[i]);
                    if (d > 10000.0)
                    {
                        Rectangle rect = CvInvoke.BoundingRectangle(contours[i]);
                        Program.logIt(string.Format("{0}: {1}", d, rect));
                        if (ret.IsEmpty) ret = rect;
                        else ret = Rectangle.Union(ret, rect);
                    }
                }
            }
            ret.Inflate(100, 100);
            img = img.GetSubRect(ret);
            img.Save("temp_4.bmp");
        }
        static void test_apple_logo_v3()
        {
            Mat b0 = CvInvoke.Imread(@"data\Apple_logo.png", ImreadModes.Grayscale);
            Mat b1 = new Mat();
            CvInvoke.BitwiseNot(b0, b1);
            b0 = CvInvoke.Imread(@"temp_3.bmp", ImreadModes.Grayscale);
            Mat b2 = new Mat();
            CvInvoke.Resize(b0, b2, new Size(b0.Size.Width / 10, b0.Size.Height / 10));
            b2.Save("temp_1.bmp");
            Rectangle rect = Rectangle.Empty;
            // found logo rect
            if (rect.IsEmpty)
            {
                CvInvoke.Threshold(b2, b0, 250, 255, ThresholdType.Binary);
                using (VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint())
                {
                    CvInvoke.FindContours(b0, contours, null, RetrType.External, ChainApproxMethod.ChainApproxNone);
                    int count = contours.Size;
                    for (int i = 0; i < count; i++)
                    {
                        double d = CvInvoke.ContourArea(contours[i]);
                        if (d > 100.0)
                        {
                            Rectangle r = CvInvoke.BoundingRectangle(contours[i]);
                            Program.logIt(string.Format("{0}: {1}", d, r));
                            if (rect.IsEmpty) rect = r;
                            else rect = Rectangle.Union(rect, r);
                        }
                    }
                }
            }
            if (!rect.IsEmpty)
            {
                rect.Inflate(50, 50);
                Image<Gray, Byte> img2 = b2.ToImage<Gray, Byte>();
                Image<Gray, Byte> img2p = img2.GetSubRect(rect);
                Image<Gray, float> img2s = img2p.Sobel(0, 1, 3).Add(img2p.Sobel(1, 0, 3)).AbsDiff(new Gray(0.0));
                img2s.Save("temp_2.bmp");

                Image<Gray, Byte> img1 = b1.ToImage<Gray, Byte>();
                //Image<Gray, Byte> img2p = img2.GetSubRect(rect);
                Image<Gray, float> img1s = img1.Sobel(0, 1, 3).Add(img1.Sobel(1, 0, 3)).AbsDiff(new Gray(0.0));
                img1s.Save("temp_1.bmp");

                double[] rates = new double[] { 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.225, 0.2125,0.2, 0.175,0.15,0.1 };
                foreach(double r in rates)
                {
                    Image<Gray, Byte> i = img1.Resize(r, Inter.Cubic);
                    if (img2p.Size.Height > i.Size.Height && img2p.Size.Width>i.Size.Width)
                    {
                        img1s= i.Sobel(0, 1, 3).Add(i.Sobel(1, 0, 3)).AbsDiff(new Gray(0.0));
                        Emgu.CV.Image<Emgu.CV.Structure.Gray, float> t_Result = img2s.MatchTemplate(img1s, TemplateMatchingType.CcoeffNormed);
                        double[] t_MinValues, t_maxValues;
                        Point[] t_MinLocations, t_MaxLocations;
                        t_Result.MinMax(out t_MinValues, out t_maxValues, out t_MinLocations, out t_MaxLocations);
                        Program.logIt($"r={r}, max={t_maxValues[0]},  loc={t_MaxLocations[0]}");
                    }
                }

            }
        }
        static void test_apple_logo_v2()
        {
            //SIFT d = new SIFT();
            //KAZE d = new KAZE();
            ORBDetector d = new ORBDetector();

            Mat b0 = CvInvoke.Imread(@"data\Apple_logo.png", ImreadModes.Grayscale);
            Mat b1 = new Mat();
            CvInvoke.BitwiseNot(b0, b1);

            MKeyPoint[] kp1 = d.Detect(b1);
            Mat desc1 = new Mat();
            d.Compute(b1, new VectorOfKeyPoint(kp1), desc1);

            b0 = CvInvoke.Imread(@"temp_3.bmp", ImreadModes.Grayscale);
            Mat b2 = new Mat();
            CvInvoke.Resize(b0, b2, new Size(b0.Size.Width / 10, b0.Size.Height / 10));
            MKeyPoint[] kp2 = d.Detect(b2);
            Mat desc2 = new Mat();
            d.Compute(b2, new VectorOfKeyPoint(kp2), desc2);

            //FlannBasedMatcher fbm = new FlannBasedMatcher(new KdTreeIndexParams(), new SearchParams());
            //fbm.Add(desc1);
            //fbm.KnnMatch(desc2, matches, 2, null);
            VectorOfDMatch vm = new VectorOfDMatch();
            BFMatcher bf = new BFMatcher(DistanceType.Hamming, true);
            bf.Add(desc1);
            bf.Match(desc2, vm);
            //Matrix<DepthType.Cv8U> mask = new Matrix<DepthType.Cv8U>(vm.Size,1);
            //Mat mask = new Mat(vm.Size, 1, DepthType.Cv8U, 1);
            //mask.SetTo(new MCvScalar(0));
            VectorOfVectorOfDMatch matches = new VectorOfVectorOfDMatch();
            for(int i=0;i<vm.Size; i++)
            {
                MDMatch m = vm[i];
                matches.Push(new VectorOfDMatch(new MDMatch[] { m }));
                if (m.Distance<20)
                {

                }
            }
            //bf.KnnMatch(desc2, matches, 2);
            /*
            Mat mask = new Mat(matches.Size, 1, DepthType.Cv8U, 1);
            mask.SetTo(new MCvScalar(255));
            Features2DToolbox.VoteForUniqueness(matches, 0.9, mask);
            int nonZeroCount = CvInvoke.CountNonZero(mask);
            PointF[] pts = null;
            {
                nonZeroCount = Features2DToolbox.VoteForSizeAndOrientation(new VectorOfKeyPoint(kp1), new VectorOfKeyPoint(kp2), matches, mask, 1.5, 1);
                if (nonZeroCount > 4)
                {
                    Mat homography = Features2DToolbox.GetHomographyMatrixFromMatchedFeatures(new VectorOfKeyPoint(kp1),
                               new VectorOfKeyPoint(kp2), matches, mask, 2);
                    if (homography != null)
                    {
                        Rectangle rect = new Rectangle(Point.Empty, b1.Size);
                        pts = new PointF[]
                        {
                  new PointF(rect.Left, rect.Bottom),
                  new PointF(rect.Right, rect.Bottom),
                  new PointF(rect.Right, rect.Top),
                  new PointF(rect.Left, rect.Top)
                        };
                        pts = CvInvoke.PerspectiveTransform(pts, homography);

                        //Point[] points = Array.ConvertAll<PointF, Point>(pts, Point.Round);
                        //using (VectorOfPoint vp = new VectorOfPoint(points))
                        //{
                        //    CvInvoke.Polylines(result, vp, true, new MCvScalar(255, 0, 0, 255), 5);
                        //}

                    }
                }
            }
            */
            Mat result = new Mat();
            Features2DToolbox.DrawMatches(b1, new VectorOfKeyPoint(kp1), b2, new VectorOfKeyPoint(kp2), matches, result, new MCvScalar(255, 255, 255), new MCvScalar(255, 255, 255), null);
            //if (pts != null)
            //{
            //    System.Drawing.Point[] pps = Array.ConvertAll<PointF, Point>(pts, Point.Round);
            //    using (VectorOfPoint vp = new VectorOfPoint(pps))
            //    {
            //        CvInvoke.Polylines(result, vp, true, new MCvScalar(255, 0, 0, 255), 5);
            //    }
            //}
            result.Save("temp_4.bmp");
        }
        static void test_apple_logo()
        {
            //SIFT d = new SIFT();
            KAZE d = new KAZE();
            //ORBDetector d = new ORBDetector();

            Mat b0 = CvInvoke.Imread(@"data\Apple_logo.png", ImreadModes.Grayscale);
            Mat b1 = new Mat();
            CvInvoke.BitwiseNot(b0, b1);
            
            MKeyPoint[] kp1 = d.Detect(b1);
            Mat desc1 = new Mat();
            d.Compute(b1, new VectorOfKeyPoint(kp1), desc1);

            b0 = CvInvoke.Imread(@"temp_3.bmp", ImreadModes.Grayscale);
            Mat b2 = new Mat();
            CvInvoke.Resize(b0, b2, new Size(b0.Size.Width / 10, b0.Size.Height / 10));
            MKeyPoint[] kp2 = d.Detect(b2);
            Mat desc2 = new Mat();
            d.Compute(b2, new VectorOfKeyPoint(kp2), desc2);

            VectorOfVectorOfDMatch matches = new VectorOfVectorOfDMatch();
            FlannBasedMatcher fbm = new FlannBasedMatcher(new KdTreeIndexParams(), new SearchParams());
            fbm.Add(desc1);
            fbm.KnnMatch(desc2, matches, 2, null);

            Mat mask = new Mat(matches.Size, 1, DepthType.Cv8U, 1);
            mask.SetTo(new MCvScalar(255));
            Features2DToolbox.VoteForUniqueness(matches, 0.9, mask);
            int nonZeroCount = CvInvoke.CountNonZero(mask);
            PointF[] pts = null;
            {
                nonZeroCount = Features2DToolbox.VoteForSizeAndOrientation(new VectorOfKeyPoint(kp1), new VectorOfKeyPoint(kp2), matches, mask, 1.5, 1);
                if (nonZeroCount > 4)
                {
                    Mat homography = Features2DToolbox.GetHomographyMatrixFromMatchedFeatures(new VectorOfKeyPoint(kp1),
                               new VectorOfKeyPoint(kp2), matches, mask, 2);
                    if (homography != null)
                    {
                        Rectangle rect = new Rectangle(Point.Empty, b1.Size);
                        pts = new PointF[]
                        {
                  new PointF(rect.Left, rect.Bottom),
                  new PointF(rect.Right, rect.Bottom),
                  new PointF(rect.Right, rect.Top),
                  new PointF(rect.Left, rect.Top)
                        };
                        pts = CvInvoke.PerspectiveTransform(pts, homography);

                        //Point[] points = Array.ConvertAll<PointF, Point>(pts, Point.Round);
                        //using (VectorOfPoint vp = new VectorOfPoint(points))
                        //{
                        //    CvInvoke.Polylines(result, vp, true, new MCvScalar(255, 0, 0, 255), 5);
                        //}

                    }
                }
            }

            Mat result = new Mat();
            Features2DToolbox.DrawMatches(b1, new VectorOfKeyPoint(kp1), b2, new VectorOfKeyPoint(kp2), matches, result, new MCvScalar(255, 255, 255), new MCvScalar(255, 255, 255), mask);
            if (pts != null)
            {
                System.Drawing.Point[] pps = Array.ConvertAll<PointF, Point>(pts, Point.Round);
                using (VectorOfPoint vp = new VectorOfPoint(pps))
                {
                    CvInvoke.Polylines(result, vp, true, new MCvScalar(255, 0, 0, 255), 5);
                }
            }
            result.Save("temp_4.bmp");
        }
    }
}

