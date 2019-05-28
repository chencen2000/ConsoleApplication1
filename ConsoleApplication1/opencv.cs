using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Features2D;
using Emgu.CV.Flann;
using Emgu.CV.ML;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using Emgu.CV.XFeatures2D;
using System;
using System.Collections.Generic;
using System.Drawing;
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
            //test_haar();
            //test_cornet();
            //test_match_shape();
            //test_bg();
        }
        static void test_bg()
        {
            VideoCapture vc = new VideoCapture(0);
            for (int i = 0; i < 41; i++)
            {
                double d = vc.GetCaptureProperty((CapProp)i);
            }
            BackgroundSubtractorMOG2 bg = new BackgroundSubtractorMOG2();
            //BackgroundSubtractorKNN bg = new BackgroundSubtractorKNN(10, 16, true);
            VideoWriter vw = new VideoWriter("temp_v.avi", 5, new Size(640, 480), true);
            int kb = 0;
            Mat f = new Mat();
            Mat fgMask = new Mat();
            while (kb != 27)
            {
                vc.Read(f);
                bg.Apply(f, fgMask);
                vw.Write(f);
                CvInvoke.Imshow("a", f);
                CvInvoke.Imshow("b", fgMask);
                kb = CvInvoke.WaitKey(10);
            }

            /*
            bg.History = 3;
            Mat fgMask = new Mat();
            for (int i = 0; i < 5; i++)
            {
                Mat f = CvInvoke.Imread(@"C:\Users\qa\Desktop\picture\save_00.jpg");
                bg.Apply(f, fgMask);
                CvInvoke.Imshow("a", f);
                CvInvoke.Imshow("b", fgMask);
                CvInvoke.WaitKey(0);
                f = CvInvoke.Imread(@"C:\Users\qa\Desktop\picture\save_01.jpg");
                bg.Apply(f, fgMask);
                CvInvoke.Imshow("a", f);
                CvInvoke.Imshow("b", fgMask);
                CvInvoke.WaitKey(0);
                f = CvInvoke.Imread(@"C:\Users\qa\Desktop\picture\save_02.jpg");
                bg.Apply(f, fgMask);
                CvInvoke.Imshow("a", f);
                CvInvoke.Imshow("b", fgMask);
                CvInvoke.WaitKey(0);
                f = CvInvoke.Imread(@"C:\Users\qa\Desktop\picture\save_03.jpg");
                bg.Apply(f, fgMask);
                CvInvoke.Imshow("a", f);
                CvInvoke.Imshow("b", fgMask);
                CvInvoke.WaitKey(0);
            }
            */
            CvInvoke.DestroyAllWindows();
        }
        static void test_cornet()
        {
            Mat mi = CvInvoke.Imread(@"C:\Users\qa\Desktop\picture\iphone_icon\scroll_left_icon.jpg", Emgu.CV.CvEnum.ImreadModes.Grayscale);
            Mat corner = new Mat();
            CvInvoke.CornerHarris(mi, corner, 3);
            Mat corner_img = new Mat();
            CvInvoke.Normalize(corner, corner_img, 0, 255, Emgu.CV.CvEnum.NormType.MinMax);
            Mat m1 = new Mat();
            CvInvoke.ConvertScaleAbs(corner_img, m1, 1, 0);
            m1.Save("temp_1.jpg");

            double mind = 0;
            double maxd = 0;
            System.Drawing.Point minp = new Point();
            System.Drawing.Point maxp = new Point();
            int[] mini = new int[4];
            int[] maxi = new int[4];
            CvInvoke.MinMaxLoc(m1, ref mind, ref maxd, ref minp, ref maxp);
            CvInvoke.MinMaxIdx(m1, out mind, out maxd, mini, maxi);

            Matrix<float> mx = new Matrix<float>(corner.Rows, corner.Cols, corner.NumberOfChannels);
            corner.CopyTo(mx);
            Matrix<float> mx1 = new Matrix<float>(corner.Rows, corner.Cols, corner.NumberOfChannels);
            corner_img.CopyTo(mx1);
            Matrix<Byte> mx2 = new Matrix<Byte>(corner.Rows, corner.Cols, corner.NumberOfChannels);
            m1.CopyTo(mx2);

            List<System.Drawing.Point> points = new List<System.Drawing.Point>();

            for (int i = 0; i < mx.Rows; i++)
            {
                for (int j = 0; j < mx.Cols; j++)
                {
                    float v1 = mx[i, j];
                    float v2 = mx1[i, j];
                    if (v2 > 200)
                    {
                        points.Add(new System.Drawing.Point(i, j));

                    }
                }
            }

            if (points.Count > 0)
            {
                foreach (var p in points)
                    Program.logIt(string.Format("{0}", p));

                RotatedRect rr = CvInvoke.MinAreaRect(new VectorOfPoint(points.ToArray()));
                PointF[] pf = rr.GetVertices();
                foreach (var p in pf)
                    Program.logIt(string.Format("{0}", p));

                //Rectangle r = rr.MinAreaRect();
                //Program.logIt(string.Format("rect={0}", r));
            }

            return;
        }

        static void test_match_shape()
        {
            //Mat mi = CvInvoke.Imread(@"C:\Users\qa\Desktop\picture\iphone_icon\scroll_left_icon.jpg", Emgu.CV.CvEnum.ImreadModes.Grayscale);
            Mat mi = CvInvoke.Imread(@"C:\Users\qa\Desktop\picture\iphone_icon\scroll_left_icon.jpg", Emgu.CV.CvEnum.ImreadModes.Grayscale);
            Mat mi2 = new Mat();
            //CvInvoke.BitwiseNot(mi, mi2);
            //mi = mi2;

            //BoostDesc d = new BoostDesc();
            //VGG d = new VGG();
            //LATCH d = new LATCH();
            //Freak d = new Freak();
            //DAISY d = new DAISY();
            //GFTTDetector d = new GFTTDetector();
            //SIFT d = new SIFT();
            //Brisk d = new Brisk();
            KAZE d = new KAZE();
            //AKAZE d = new AKAZE();
            //ORBDetector d = new ORBDetector();
            //SURF d = new SURF(400);
            MKeyPoint[] kp = d.Detect(mi);
            Mat desc = new Mat();
            d.Compute(mi, new VectorOfKeyPoint(kp), desc);

            mi2 = CvInvoke.Imread(@"C:\Users\qa\Desktop\picture\menu_1.jpg", Emgu.CV.CvEnum.ImreadModes.Grayscale);
            MKeyPoint[] kp2 = d.Detect(mi2);
            Mat desc2 = new Mat();
            d.Compute(mi2, new VectorOfKeyPoint(kp2), desc2);

            VectorOfVectorOfDMatch matches = new VectorOfVectorOfDMatch();
            //BFMatcher m = new BFMatcher(DistanceType.L2Sqr);
            //m.Add(desc);
            //m.KnnMatch(desc2, matches, 2, null);

            FlannBasedMatcher fbm = new FlannBasedMatcher(new KdTreeIndexParams(), new SearchParams());
            fbm.Add(desc);
            fbm.KnnMatch(desc2, matches, 2, null);

            Mat mask = new Mat(matches.Size, 1, DepthType.Cv8U, 1);
            mask.SetTo(new MCvScalar(255));
            Features2DToolbox.VoteForUniqueness(matches, 0.8, mask);
            int nonZeroCount = CvInvoke.CountNonZero(mask);
            PointF[] pts = null;
            {
                nonZeroCount = Features2DToolbox.VoteForSizeAndOrientation(new VectorOfKeyPoint(kp), new VectorOfKeyPoint(kp2), matches, mask, 1.5, 1);
                if (nonZeroCount > 4)
                {
                    Mat homography = Features2DToolbox.GetHomographyMatrixFromMatchedFeatures(new VectorOfKeyPoint(kp),
                               new VectorOfKeyPoint(kp2), matches, mask, 2);
                    if (homography != null)
                    {
                        Rectangle rect = new Rectangle(Point.Empty, mi.Size);
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
            Features2DToolbox.DrawMatches(mi, new VectorOfKeyPoint(kp), mi2, new VectorOfKeyPoint(kp2),
               matches, result, new MCvScalar(255, 255, 255), new MCvScalar(255, 255, 255), mask);
            Matrix<Byte> mx = new Matrix<Byte>(mask.Rows, mask.Cols, mask.NumberOfChannels);
            mask.CopyTo(mx);
            List<PointF> pf = new List<PointF>();
            for (int i=0; i<mx.Rows; i++)
            {
                for (int j = 0; j < mx.Cols; j++)
                {
                    Byte b = mx[i, j];
                    if (b != 0)
                    {
                        VectorOfDMatch dm = matches[i];
                        MDMatch dm1 = dm[0].Distance < dm[1].Distance ? dm[0] : dm[1];
                        MKeyPoint p1 = kp[dm1.TrainIdx];
                        MKeyPoint p2 = kp2[dm1.QueryIdx];
                        Program.logIt(string.Format("{0}: {1}-{2}", dm1.Distance, p2.Point, p1.Point));
                        pf.Add(p2.Point);
                    }
                }
            }
            Rectangle r = PointCollection.BoundingRectangle(pf.ToArray());
            CvInvoke.Rectangle(result, r, new MCvScalar(0, 0, 255, 255));
            if (pts != null)
            {
                System.Drawing.Point[] pps = Array.ConvertAll<PointF, Point>(pts, Point.Round);
                using (VectorOfPoint vp = new VectorOfPoint(pps))
                {
                    CvInvoke.Polylines(result, vp, true, new MCvScalar(255, 0, 0, 255), 5);
                }
            }
            result.Save("temp_1.jpg");
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
