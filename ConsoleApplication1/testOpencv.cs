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

namespace ConsoleApplication1
{
    public static class DrawMatches
    {
        public static void FindMatch(Mat modelImage, Mat observedImage, out long matchTime, out VectorOfKeyPoint modelKeyPoints, out VectorOfKeyPoint observedKeyPoints, VectorOfVectorOfDMatch matches, out Mat mask, out Mat homography)
        {
            int k = 2;
            double uniquenessThreshold = 0.8;
            double hessianThresh = 300;

            Stopwatch watch;
            homography = null;

            modelKeyPoints = new VectorOfKeyPoint();
            observedKeyPoints = new VectorOfKeyPoint();

#if !__IOS__
            if (CudaInvoke.HasCuda)
            {
                CudaSURF surfCuda = new CudaSURF((float)hessianThresh);
                using (GpuMat gpuModelImage = new GpuMat(modelImage))
                //extract features from the object image
                using (GpuMat gpuModelKeyPoints = surfCuda.DetectKeyPointsRaw(gpuModelImage, null))
                using (GpuMat gpuModelDescriptors = surfCuda.ComputeDescriptorsRaw(gpuModelImage, null, gpuModelKeyPoints))
                using (CudaBFMatcher matcher = new CudaBFMatcher(DistanceType.L2))
                {
                    surfCuda.DownloadKeypoints(gpuModelKeyPoints, modelKeyPoints);
                    watch = Stopwatch.StartNew();

                    // extract features from the observed image
                    using (GpuMat gpuObservedImage = new GpuMat(observedImage))
                    using (GpuMat gpuObservedKeyPoints = surfCuda.DetectKeyPointsRaw(gpuObservedImage, null))
                    using (GpuMat gpuObservedDescriptors = surfCuda.ComputeDescriptorsRaw(gpuObservedImage, null, gpuObservedKeyPoints))
                    //using (GpuMat tmp = new GpuMat())
                    //using (Stream stream = new Stream())
                    {
                        matcher.KnnMatch(gpuObservedDescriptors, gpuModelDescriptors, matches, k);

                        surfCuda.DownloadKeypoints(gpuObservedKeyPoints, observedKeyPoints);

                        mask = new Mat(matches.Size, 1, DepthType.Cv8U, 1);
                        mask.SetTo(new MCvScalar(255));
                        Features2DToolbox.VoteForUniqueness(matches, uniquenessThreshold, mask);

                        int nonZeroCount = CvInvoke.CountNonZero(mask);
                        if (nonZeroCount >= 4)
                        {
                            nonZeroCount = Features2DToolbox.VoteForSizeAndOrientation(modelKeyPoints, observedKeyPoints,
                               matches, mask, 1.5, 20);
                            if (nonZeroCount >= 4)
                                homography = Features2DToolbox.GetHomographyMatrixFromMatchedFeatures(modelKeyPoints,
                                   observedKeyPoints, matches, mask, 2);
                        }
                    }
                    watch.Stop();
                }
            }
            else
#endif
            {
                //using (UMat uModelImage = modelImage.ToUMat(AccessType.Read))
                //using (UMat uObservedImage = observedImage.ToUMat(AccessType.Read))
                UMat uModelImage = modelImage.GetUMat(AccessType.Read);
                UMat uObservedImage = observedImage.GetUMat(AccessType.Read);
                {
                    ////Brisk surfCPU = new Brisk();
                    Brisk surfCPU = new Brisk();
                    //SURF surfCPU = new SURF(hessianThresh);
                    //Freak surfCPU = new Freak();
                    //SIFT surfCPU = new SIFT();
                    //extract features from the object image
                    UMat modelDescriptors = new UMat();
                    surfCPU.DetectAndCompute(modelImage, null, modelKeyPoints, modelDescriptors, false);
                    //surfCPU.Detect(modelImage, modelKeyPoints);
                    //surfCPU.Compute(modelImage, modelKeyPoints, modelDescriptors);

                    watch = Stopwatch.StartNew();

                    // extract features from the observed image
                    UMat observedDescriptors = new UMat();
                    surfCPU.DetectAndCompute(observedImage, null, observedKeyPoints, observedDescriptors, false);
                    BFMatcher matcher = new BFMatcher(DistanceType.L2);
                    matcher.Add(modelDescriptors);

                    matcher.KnnMatch(observedDescriptors, matches, k, null);
                    mask = new Mat(matches.Size, 1, DepthType.Cv8U, 1);
                    mask.SetTo(new MCvScalar(255));
                    Features2DToolbox.VoteForUniqueness(matches, uniquenessThreshold, mask);

                    int nonZeroCount = CvInvoke.CountNonZero(mask);
                    if (nonZeroCount >= 4)
                    {
                        nonZeroCount = Features2DToolbox.VoteForSizeAndOrientation(modelKeyPoints, observedKeyPoints,
                           matches, mask, 1.5, 20);
                        if (nonZeroCount >= 4)
                            homography = Features2DToolbox.GetHomographyMatrixFromMatchedFeatures(modelKeyPoints,
                               observedKeyPoints, matches, mask, 2);
                    }

                    watch.Stop();
                }
            }
            matchTime = watch.ElapsedMilliseconds;
        }

        /// <summary>
        /// Draw the model image and observed image, the matched features and homography projection.
        /// </summary>
        /// <param name="modelImage">The model image</param>
        /// <param name="observedImage">The observed image</param>
        /// <param name="matchTime">The output total time for computing the homography matrix.</param>
        /// <returns>The model image and observed image, the matched features and homography projection.</returns>
        public static Mat Draw(Mat modelImage, Mat observedImage, out long matchTime)
        {
            Mat homography;
            VectorOfKeyPoint modelKeyPoints;
            VectorOfKeyPoint observedKeyPoints;
            using (VectorOfVectorOfDMatch matches = new VectorOfVectorOfDMatch())
            {
                Mat mask;
                FindMatch(modelImage, observedImage, out matchTime, out modelKeyPoints, out observedKeyPoints, matches,
                   out mask, out homography);

                //Draw the matched keypoints
                Mat result = new Mat();
                Features2DToolbox.DrawMatches(modelImage, modelKeyPoints, observedImage, observedKeyPoints,
                   matches, result, new MCvScalar(255, 255, 255), new MCvScalar(255, 255, 255), mask);

                #region draw the projected region on the image

                if (homography != null)
                {
                    //draw a rectangle along the projected model
                    Rectangle rect = new Rectangle(Point.Empty, modelImage.Size);
                    PointF[] pts = new PointF[]
                    {
                  new PointF(rect.Left, rect.Bottom),
                  new PointF(rect.Right, rect.Bottom),
                  new PointF(rect.Right, rect.Top),
                  new PointF(rect.Left, rect.Top)
                    };
                    pts = CvInvoke.PerspectiveTransform(pts, homography);

                    Point[] points = Array.ConvertAll<PointF, Point>(pts, Point.Round);
                    using (VectorOfPoint vp = new VectorOfPoint(points))
                    {
                        CvInvoke.Polylines(result, vp, true, new MCvScalar(255, 0, 0, 255), 5);
                    }

                }

                #endregion

                return result;

            }
        }

        public static void test(Mat img1, Mat img2)
        {
            UMat uimg1 = img1.GetUMat(AccessType.Read);
            UMat uimg2 = img2.GetUMat(AccessType.Read);
            SURF surfCPU = new SURF(300);
            VectorOfKeyPoint img2_kp = new VectorOfKeyPoint();
            UMat img2_desc = new UMat();
            surfCPU.DetectRaw(img2, img2_kp);
            surfCPU.Compute(img2, img2_kp, img2_desc);
            VectorOfKeyPoint img1_kp = new VectorOfKeyPoint();
            UMat img1_desc = new UMat();
            surfCPU.DetectRaw(img1, img1_kp);
            surfCPU.Compute(img1, img1_kp, img1_desc);
            BFMatcher matcher = new BFMatcher(DistanceType.L2Sqr);
            matcher.Add(img2_desc);
            VectorOfVectorOfDMatch matches = new VectorOfVectorOfDMatch();
            matcher.KnnMatch(img1_desc, matches, 2, null);
            Mat mask = new Mat(matches.Size, 1, DepthType.Cv8U, 1);
            mask.SetTo(new MCvScalar(255));
            Features2DToolbox.VoteForUniqueness(matches, 0.8, mask);
            int nonZeroCount = CvInvoke.CountNonZero(mask);
            if (nonZeroCount >= 4)
            {
                nonZeroCount = Features2DToolbox.VoteForSizeAndOrientation(img2_kp, img1_kp, matches, mask, 1.5, 20);
                if (nonZeroCount >= 4)
                {
                    Mat result = new Mat();
                    Features2DToolbox.DrawMatches(img2, img2_kp, img1, img1_kp,
                       matches, result, new MCvScalar(255, 255, 255), new MCvScalar(255, 255, 255), mask);

                    Mat homography = Features2DToolbox.GetHomographyMatrixFromMatchedFeatures(img2_kp, img1_kp, matches, mask, 2);
                    if (homography != null)
                    {
                        Rectangle rect = new Rectangle(Point.Empty, img2.Size);
                        PointF[] pts = new PointF[]
                        {
                  new PointF(rect.Left, rect.Bottom),
                  new PointF(rect.Right, rect.Bottom),
                  new PointF(rect.Right, rect.Top),
                  new PointF(rect.Left, rect.Top)
                        };
                        pts = CvInvoke.PerspectiveTransform(pts, homography);

                        Point[] points = Array.ConvertAll<PointF, Point>(pts, Point.Round);
                        using (VectorOfPoint vp = new VectorOfPoint(points))
                        {
                            CvInvoke.Polylines(result, vp, true, new MCvScalar(255, 0, 0, 255), 5);
                        }

                    }

                    CvInvoke.Imshow("a", result);
                    CvInvoke.WaitKey(0);
                    CvInvoke.DestroyAllWindows();
                }
            }
        }
    }
    class testOpencv
    {
        static void Main(string[] args)
        {
            //check_image_similarity();
            //test_skelton();
            //pre_process();
            //extra_icon();
            //find_focused_item();
            //test_3();
            test_1();
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
            Image<Bgr, byte> img1 = new Image<Bgr, byte>(@"C:\Users\qa\Desktop\picture\scroll_lect.jpg");
            Image<Gray, byte> g = img1.Convert<Gray, byte>();
            g = g.ThresholdBinary(new Gray(127), new Gray(255));
            g.Save("temp_1.jpg");
            MCvMoments m = g.GetMoments(true);

            VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint();
            CvInvoke.FindContours(g, contours, null, RetrType.External, ChainApproxMethod.ChainApproxNone);
            int count = contours.Size;
            for(int i=0; i < count; i++)
            {
                double a = CvInvoke.ContourArea(contours[i]);
                bool b = CvInvoke.IsContourConvex(contours[i]);
            }
            Mat h = new Mat();
            CvInvoke.FindContours(g, contours, h, RetrType.Tree, ChainApproxMethod.ChainApproxSimple);
            Matrix<Int32> he = new Matrix<int>(h.Rows, h.Cols, h.NumberOfChannels);
            h.CopyTo(he);
            count = contours.Size;
            for (int i = 0; i < count; i++)
            {
                VectorOfPoint ps = contours[i];
                double a = CvInvoke.ContourArea(contours[i]);
                bool b = CvInvoke.IsContourConvex(contours[i]);
            }
        }
        static void test_surf()
        {
            Mat img1 = CvInvoke.Imread(@"C:\Users\qa\Desktop\picture\save_10.jpg", Emgu.CV.CvEnum.ImreadModes.Grayscale);
            Mat img2 = CvInvoke.Imread(@"C:\Users\qa\Desktop\picture\iphone_icon\setting_icon.jpg", Emgu.CV.CvEnum.ImreadModes.Grayscale);
            SURF surf = new SURF(300);
            VectorOfKeyPoint kp1 = new VectorOfKeyPoint();
            VectorOfKeyPoint kp2 = new VectorOfKeyPoint();
            UMat d1 = new UMat();
            UMat d2 = new UMat();
            surf.DetectAndCompute(img1, null, kp1, d1, false);
            surf.DetectAndCompute(img2, null, kp2, d2, false);
            VectorOfVectorOfDMatch matches = new VectorOfVectorOfDMatch();
            BFMatcher matcher = new BFMatcher(DistanceType.L2);
            matcher.Add(d1);
            matcher.KnnMatch(d2, matches, 2, null);
            double max_dist = 0; double min_dist = 100;
            //-- Quick calculation of max and min distances between keypoints  
            for (int i = 0; i < matches.Size; i++)
            {
                VectorOfDMatch m = matches[i];
                if (m.Size > 1)
                {
                    double dist = m[1].Distance;
                    if (dist < min_dist) min_dist = dist;
                    if (dist > max_dist) max_dist = dist;
                }
            }
            Program.logIt(string.Format("max={0} min={1}", max_dist, min_dist));
            //
            //-- Draw only "good" matches (i.e. whose distance is less than 0.6*max_dist )  
            //-- PS.- radiusMatch can also be used here.  
            List<VectorOfDMatch> good_point = new List<VectorOfDMatch>();
            for (int i = 0; i < matches.Size; i++)
            {
                VectorOfDMatch m = matches[i];
                if (m.Size > 1)
                {
                    double dist = m[1].Distance;
                    if (dist < 0.6 * max_dist)
                        good_point.Add(m);
                }
            }

            // 
            List<PointF> obj;
            List<PointF> scene;
            for(int i=0; i<good_point.Count; i++)
            {

            }

        }
        
        static void test()
        {
            Image<Bgr, byte> img1 = new Image<Bgr, byte>(@"C:\Users\qa\Desktop\picture\scroll_lect.jpg");
            Image<Gray, byte> g = img1.Convert<Gray, byte>();
            //Mat c = new Mat();
            //CvInvoke.CornerHarris(g, c, 2);
            //Image<Gray, float> g1 = c.ToImage<Gray, float>();
            //g1.Dilate(1);
            ////CvInvoke.Dilate(c,g1,IntPtr.Zero, new Point(-1,-1), 1, BorderType.Default, )
            //double maxv =0.0;
            //double minv =0.0;
            //Point maxP = new Point();
            //Point minP = new Point();
            //CvInvoke.MinMaxLoc(g1, ref minv, ref maxv, ref maxP, ref minP);

            //CvInvoke.Threshold(g1, g, 0.01* maxv, 255.0, ThresholdType.BinaryInv);
            //g.Save("temp_1.jpg");
            //CvInvoke.Imshow("a", g);
            //CvInvoke.WaitKey(0);
            //CvInvoke.DestroyAllWindows();
            g = g.ThresholdBinary(new Gray(127), new Gray(255));
            //CvInvoke.Imshow("a", g);
            //CvInvoke.WaitKey(0);
            //CvInvoke.DestroyAllWindows();
            MCvMoments m = g.GetMoments(true);

            using (VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint())
            {
                Mat hier = new Mat();
                CvInvoke.FindContours(g, contours, hier, RetrType.Tree, ChainApproxMethod.ChainApproxSimple);
                //Image<Gray, Emgu.CV.CvEnum.DepthType.Cv32F> h = hier.ToImage<Gray, Emgu.CV.CvEnum.DepthType.Cv32S>();
                //Image<Bgra, byte> h = hier.ToImage<Bgra, byte>();
                Matrix<Int32> mh = new Matrix<Int32>(hier.Rows,hier.Cols,hier.NumberOfChannels);
                hier.CopyTo(mh);

                int count = contours.Size;
                int pos = 0;
                int next = -1;
                int pre = -1;
                int first_child = -1;
                int parent = -1;
                for (int i = 0; i < count; i++)
                {
                    next = mh.Data[0, pos++];
                    pre = mh.Data[0, pos++];
                    first_child = mh.Data[0, pos++];
                    parent = mh.Data[0, pos++];
                    VectorOfPoint contour = contours[i];
                    CvInvoke.DrawContours(img1, contours, i, new MCvScalar(0, 0, 255));
                    CvInvoke.Imshow("a", img1);
                    CvInvoke.WaitKey(0);
                    CvInvoke.DestroyAllWindows();
                }
            }
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
        static void test_2()
        {
            Image<Bgr, byte> b1 = new Image<Bgr, byte>(@"C:\Users\qa\Desktop\picture\menu_1.jpg");
            Image<Bgr, byte> b2 = new Image<Bgr, byte>(@"C:\Users\qa\Desktop\picture\iphone_icon\scroll_left_icon.jpg");

            double hessianThresh = 300;
            //SURF surfCPU = new SURF(hessianThresh);
            SIFT s = new SIFT();
            VectorOfKeyPoint modelKeyPoints = new VectorOfKeyPoint();
            VectorOfKeyPoint observedKeyPoints = new VectorOfKeyPoint();
            UMat modelDescriptors = new UMat();
            UMat observedDescriptors = new UMat();
            //surfCPU.Compute(b2, modelKeyPoints, modelDescriptors);
            //surfCPU.Compute(b1, observedKeyPoints, observedDescriptors);
            s.DetectAndCompute(b2, null, modelKeyPoints, modelDescriptors, false);
            s.DetectAndCompute(b1, null, observedKeyPoints, observedDescriptors, false);

            VectorOfVectorOfDMatch matches = new VectorOfVectorOfDMatch();
            BFMatcher matcher = new BFMatcher(DistanceType.L2);
            matcher.Add(modelDescriptors);
            matcher.KnnMatch(observedDescriptors, matches, 2, null);

            Mat homography = null;
            Mat mask = new Mat(matches.Size, 1, DepthType.Cv8U, 1);
            mask.SetTo(new MCvScalar(255));
            Features2DToolbox.VoteForUniqueness(matches, 0.8, mask);
            int Count = CvInvoke.CountNonZero(mask);      //用于寻找模板在图中的位置
            if (Count >= 4)
            {
                Count = Features2DToolbox.VoteForSizeAndOrientation(modelKeyPoints, observedKeyPoints, matches, mask, 1.5, 20);
                if (Count >= 4)
                    homography = Features2DToolbox.GetHomographyMatrixFromMatchedFeatures(modelKeyPoints, observedKeyPoints, matches, mask, 2);
            }

            Mat result = new Mat();
            Features2DToolbox.DrawMatches(b2, modelKeyPoints, b1, observedKeyPoints, matches, result, new MCvScalar(255, 0, 255), new MCvScalar(0, 255, 255), mask);

            if (homography != null)
            {
                Rectangle rect = new Rectangle(Point.Empty, b2.Size);
                PointF[] points = new PointF[]
                                {
                  new PointF(rect.Left, rect.Bottom),
                  new PointF(rect.Right, rect.Bottom),
                  new PointF(rect.Right, rect.Top),
                  new PointF(rect.Left, rect.Top)
                                };
                points = CvInvoke.PerspectiveTransform(points, homography);
                Point[] points2 = Array.ConvertAll<PointF, Point>(points, Point.Round);
                VectorOfPoint vp = new VectorOfPoint(points2);
                CvInvoke.Polylines(result, vp, true, new MCvScalar(255, 0, 0, 255), 2);
            }

            CvInvoke.Imshow("a", result);
            CvInvoke.WaitKey(0);
            CvInvoke.DestroyAllWindows();
        }
        static void test_1()
        {
            Mat img = CvInvoke.Imread(@"C:\Users\qa\Desktop\picture\save_00.jpg");

            CvInvoke.Imshow("a", img);
            CvInvoke.WaitKey(0);
            CvInvoke.DestroyAllWindows();
        }
        static float Classify(Image<Bgr, Byte> testImg, string folder)
        {
            float ret = 0.0f;
            int class_num = 3;  //number of clusters/classes
            int input_num = 0;  //number of train images
            int j = 0;

            SURF surf = new SURF(300);

            return ret;
        }
    }
}

