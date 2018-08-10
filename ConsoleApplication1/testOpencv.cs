using Accord.Imaging;
using Emgu.CV;
using Emgu.CV.Cuda;
using Emgu.CV.CvEnum;
using Emgu.CV.Features2D;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using Emgu.CV.XFeatures2D;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
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
                    SURF surfCPU = new SURF(hessianThresh);
                    //extract features from the object image
                    UMat modelDescriptors = new UMat();
                    //surfCPU.DetectAndCompute(uModelImage, null, modelKeyPoints, modelDescriptors, false);
                    surfCPU.DetectRaw(modelImage, modelKeyPoints);
                    surfCPU.Compute(modelImage, modelKeyPoints, modelDescriptors);

                    watch = Stopwatch.StartNew();

                    // extract features from the observed image
                    UMat observedDescriptors = new UMat();
                    surfCPU.DetectAndCompute(uObservedImage, null, observedKeyPoints, observedDescriptors, false);
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
            test();
        }
        static void test()
        {
            Mat img1 = CvInvoke.Imread(@"C:\Users\qa\Desktop\picture\save_10.jpg", Emgu.CV.CvEnum.ImreadModes.AnyColor);
            Mat img2 = CvInvoke.Imread(@"C:\Users\qa\Desktop\picture\save_12.jpg", Emgu.CV.CvEnum.ImreadModes.AnyColor);
            int Threshold = 60;
            Image<Bgr, Byte> bmp = img1.ToImage<Bgr, Byte>().AbsDiff(img2.ToImage<Bgr, Byte>());
            bmp = bmp.ThresholdBinary(new Bgr(Threshold, Threshold, Threshold), new Bgr(255, 255, 255));
            bmp.Save("temp_1.jpg");
            //Bitmap b = bmp.Bitmap;
            BlobCounter bc = new BlobCounter();
            bc.BlobsFilter = null;
            bc.FilterBlobs = true;
            bc.MinHeight = 32;
            bc.MinWidth = 32;
            bc.CoupledSizeFiltering = true;
            bc.ObjectsOrder = ObjectsOrder.Area;
            bc.ProcessImage(bmp.Bitmap);
            foreach (Rectangle rect in bc.GetObjectsRectangles())
            {
                Program.logIt(string.Format("r={0}", rect));
            }
            CvInvoke.Imshow("a", bmp);
            CvInvoke.WaitKey(0);
            CvInvoke.DestroyAllWindows();

        }
        static void test_1()
        {
            Mat img1 = CvInvoke.Imread(@"C:\Users\qa\Desktop\picture\save_06.jpg", Emgu.CV.CvEnum.ImreadModes.AnyColor);
            Mat img2 = CvInvoke.Imread(@"C:\Users\qa\Desktop\picture\iphone_icon\app_switch_close.jpg", Emgu.CV.CvEnum.ImreadModes.AnyColor);
            //SIFT sift = new SIFT();
            //long time;
            //Mat i = DrawMatches.Draw(img2, img1, out time);
            //CvInvoke.Imshow("a", i);
            //CvInvoke.WaitKey(0);
            //CvInvoke.DestroyAllWindows();
            DrawMatches.test(img1, img2);
        }
    }
}
