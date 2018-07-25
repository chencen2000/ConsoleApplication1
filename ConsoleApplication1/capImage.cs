using AForge;
using AForge.Imaging;
using AForge.Imaging.Filters;
using AForge.Imaging.Formats;
using AForge.Math.Geometry;
using AForge.Video;
using AForge.Video.DirectShow;
using AForge.Vision.Motion;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.UI;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tesseract;

namespace ConsoleApplication1
{
    class capImage
    {
        public static void main(System.Collections.Specialized.StringDictionary args)
        {
            Bitmap b1 = ImageDecoder.DecodeFromFile(@"C:\Users\qa\Desktop\picture\iphone_icon\scoll_down_selected_icon.jpg");
            if (true)
            {
                Grayscale g_filter = new Grayscale(0.2125, 0.7154, 0.0721);
                Bitmap b1_g = g_filter.Apply(b1);
                Threshold t_filter = new Threshold(180);
                t_filter.ApplyInPlace(b1_g);
                Invert filter = new Invert();
                filter.ApplyInPlace(b1_g);
                b1_g.Save("temp_1.jpg");
                
                BlobCounter blobCounter = new BlobCounter();
                blobCounter.BlobsFilter = null;
                blobCounter.FilterBlobs = true;
                blobCounter.MaxWidth = b1_g.Width - 10;
                blobCounter.MaxHeight = b1_g.Height - 10;
                blobCounter.MinHeight = 2;
                blobCounter.MinWidth = 2;
                blobCounter.ObjectsOrder = ObjectsOrder.Area;
                blobCounter.ProcessImage(b1_g);
                Blob[] blobs = blobCounter.GetObjectsInformation();
                // blobs shoule be 2. one is a rectangle, another is an arrow.
                if (blobs.Length == 2)
                {
                    Blob r_blob = blobs[0];
                    List<IntPoint> r_edgePoints = blobCounter.GetBlobsEdgePoints(blobs[0]);
                    Blob a_blob = blobs[1];
                    List<IntPoint> a_edgePoints = blobCounter.GetBlobsEdgePoints(blobs[1]);
                    SimpleShapeChecker shapeChecker = new SimpleShapeChecker();
                    List<IntPoint> r_corners;
                    if (shapeChecker.IsQuadrilateral(r_edgePoints, out r_corners))
                    {
                        var v = shapeChecker.CheckShapeType(r_edgePoints);
                        var v1 = shapeChecker.CheckPolygonSubType(r_corners);
                        var v2 = shapeChecker.CheckShapeType(a_edgePoints);
                        int x = (int)(0.05 * r_blob.Rectangle.Width);
                        int y = (int)(0.5 * r_blob.Rectangle.Height);
                        Rectangle r = new Rectangle(x + r_blob.Rectangle.X, y + r_blob.Rectangle.Y, r_blob.Rectangle.Width - x - x, r_blob.Rectangle.Height - y);
                        if (r.Contains(a_blob.Rectangle))
                        {
                            Program.logIt(string.Format("{0} in {1}", a_blob.Rectangle, r));
                        }
                    }
                }
            }
            else
            {
                Size ns = new Size(46, 46);
                int x = (b1.Width - ns.Width) / 2;
                int y = (b1.Height - ns.Height) / 2;
                Crop c1 = new Crop(new Rectangle(x, y, ns.Width, ns.Height));
                Bitmap b11 = c1.Apply(b1);
                b11.Save("temp_1.jpg");
                Grayscale g_filter = new Grayscale(0.2125, 0.7154, 0.0721);
                Bitmap b12 = g_filter.Apply(b11);
                b12.Save("temp_2.jpg");
                Threshold t_filter = new Threshold(180);
                t_filter.ApplyInPlace(b12);
                Rectangle r_middle_low = new Rectangle(10, b12.Height / 2, b12.Width - 20, b12.Height / 2);
                Invert filter = new Invert();
                filter.ApplyInPlace(b12);
                b12.Save("temp_2.jpg");
                BlobCounter blobCounter = new BlobCounter();
                blobCounter.BlobsFilter = null;
                blobCounter.FilterBlobs = true;
                blobCounter.MinHeight = 2;
                blobCounter.MinWidth = 2;
                blobCounter.ObjectsOrder = ObjectsOrder.Area;
                blobCounter.ProcessImage(b12);
                Blob[] blobs = blobCounter.GetObjectsInformation();
                SimpleShapeChecker shapeChecker = new SimpleShapeChecker();
                for (int i = 0; i < blobs.Length; i++)
                {
                    List<IntPoint> edgePoints = blobCounter.GetBlobsEdgePoints(blobs[i]);
                    if (edgePoints.Count > 4)
                    {
                        bool isq = shapeChecker.IsQuadrilateral(edgePoints);
                        bool ita = shapeChecker.IsTriangle(edgePoints);
                        bool in_middle_low = r_middle_low.Contains(blobs[i].Rectangle);
                        List<IntPoint> cs = PointsCloud.FindQuadrilateralCorners(edgePoints);
                        if (!isq)
                        {
                            using (Graphics g = Graphics.FromImage(b11))
                            {
                                g.DrawRectangle(new Pen(Color.Blue, 1), r_middle_low);
                                g.DrawRectangle(new Pen(Color.Red, 1), blobs[i].Rectangle);
                            }
                        }
                    }
                }
                b11.Save("temp_1.jpg");
            }
            /*
            SusanCornersDetector scd = new SusanCornersDetector();
            scd.DifferenceThreshold = 33;
            scd.GeometricalThreshold = 10;
            List<IntPoint> corners = scd.ProcessImage(b12);
            SolidBrush aBrush = new SolidBrush(Color.Red);
            foreach (IntPoint p in corners)
            {
                using (Graphics g = Graphics.FromImage(b11))
                {
                    g.FillRectangle(aBrush, p.X, p.Y, 1, 1);
                }
            }
            b11.Save("temp.jpg");
            */
        }
        public static void main_6(System.Collections.Specialized.StringDictionary args)
        {
            Bitmap f1 = ImageDecoder.DecodeFromFile(@"C:\Users\qa\Desktop\picture\save_10.jpg");
            Bitmap f2 = ImageDecoder.DecodeFromFile(@"C:\Users\qa\Desktop\picture\save_12.jpg");
            //Bitmap f1 = smart_rotate_cop(@"C:\Users\qa\Desktop\picture\save_10.jpg");
            //Bitmap f2 = smart_rotate_cop(@"C:\Users\qa\Desktop\picture\save_11.jpg");
            //f1.Save("temp_1.jpg");
            //f2.Save("temp_2.jpg");
            Subtract filter = new Subtract(f1);
            Bitmap resultImage = filter.Apply(f2);
            //Image<Bgr, Byte> imageCV = new Image<Bgr, byte>(resultImage);
            //ImageViewer.Show(imageCV, "b");
            resultImage.Save("temp.jpg");
            ThresholdedDifference td_filter = new ThresholdedDifference(60);
            td_filter.OverlayImage = f1;
            Bitmap r = filter.Apply(f2);
            r.Save("temp_r.jpg");
        }
        public static void main_4(System.Collections.Specialized.StringDictionary args)
        {
            Bitmap b = new Bitmap(@"temp_4.jpg");
            var engine = new TesseractEngine(@"./tessdata", "eng", EngineMode.Default);
            engine.SetVariable("tessedit_char_whitelist", "0123456789");
            var p = engine.Process(b);

        }
        public static void main_1(System.Collections.Specialized.StringDictionary args)
        {
            Bitmap image = ImageDecoder.DecodeFromFile(@"temp.jpg");
            Grayscale g_filter = new Grayscale(0.2125, 0.7154, 0.0721);
            Bitmap grayImage = g_filter.Apply(image);
            Blur filter = new Blur();
            filter.ApplyInPlace(grayImage);
            OtsuThreshold o_filter = new OtsuThreshold();
            o_filter.ApplyInPlace(grayImage);
            /*
            ExtractBiggestBlob e_filter = new ExtractBiggestBlob();
            Bitmap b = e_filter.Apply(grayImage);
            int x = e_filter.BlobPosition.X;
            int y = e_filter.BlobPosition.Y;
            int w = b.Size.Width;
            int h = b.Size.Height;
            //biggestBlobsImage.Save("temp.jpg");
            Crop c_filter = new Crop(new Rectangle(x, y, w, h));
            Bitmap nb = c_filter.Apply(rotated);
            nb.Save("temp.jpg");
            */
            BlobCounter blobCounter = new BlobCounter();
            blobCounter.MinHeight = 20;
            blobCounter.MinWidth = 20;
            blobCounter.FilterBlobs = false;
            blobCounter.BlobsFilter = null;
            blobCounter.ObjectsOrder = ObjectsOrder.YX;
            blobCounter.ProcessImage(grayImage);
            Blob[] blobs = blobCounter.GetObjectsInformation();
            Program.logIt(string.Format("blobs={0}", blobCounter.ObjectsCount));
            Rectangle r = Rectangle.Empty;
            for(int i=1; i < blobs.Length; i++)
            {
                Blob b = blobs[i];
                Program.logIt(string.Format("{0}: {1}", b.ID, b.Rectangle));
                if (r == Rectangle.Empty) r = b.Rectangle;
                else r = Rectangle.Union(r, b.Rectangle);
            }
            //Blob toppest_blob = null;
            //foreach (Blob b in blobs)
            //{
            //    Program.logIt(string.Format("{0}: {1}", b.ID, b.Rectangle));
            //    if (r == Rectangle.Empty) r = b.Rectangle;
            //    else r = Rectangle.Union(r, b.Rectangle);
            //    if (toppest_blob == null) toppest_blob = b;
            //    else
            //    {
            //        if (b.Rectangle.Y < toppest_blob.Rectangle.Y)
            //            toppest_blob = b;
            //    }
            //}
            Program.logIt(string.Format("rect: {0}", r));
            Crop c_filter = new Crop(r);
            Bitmap nb = c_filter.Apply(image);
            nb.Save("temp_1.jpg");

            //Program.logIt(string.Format("toppest Blob: id={0}, rect={1}", toppest_blob.ID, toppest_blob.Rectangle));
        }
        public static Bitmap smart_rotate_cop(string orgfile)
        {
            Bitmap rotated = null;
            //Bitmap image = ImageDecoder.DecodeFromFile(@"C:\Users\qa\Desktop\picture\save_001.jpg");
            Bitmap image = ImageDecoder.DecodeFromFile(orgfile);
            //AForge.Range angles = new AForge.Range();
            {
                // anti-clockwise 90
                RotateBicubic filter = new RotateBicubic(90);
                Bitmap newImage = filter.Apply(image);
                newImage.Save("temp_1.jpg");
                Bitmap grayImage = Grayscale.CommonAlgorithms.BT709.Apply(newImage);
                //grayImage.Save("temp_2.jpg");
                DifferenceEdgeDetector edgeDetector = new DifferenceEdgeDetector();
                Bitmap edgesImage = edgeDetector.Apply(grayImage);
                edgesImage.Save("temp_3.jpg");
                OtsuThreshold o_filter = new OtsuThreshold();
                o_filter.ApplyInPlace(edgesImage);
                edgesImage.Save("temp_4.jpg");
                HoughLineTransformation lineTransform = new HoughLineTransformation();
                lineTransform.ProcessImage(edgesImage);
                Program.logIt(string.Format("lines={0}", lineTransform.LinesCount));
                AForge.Range angles = new AForge.Range();
                HoughLine[] lines = lineTransform.GetLinesByRelativeIntensity(1.0);
                double angle = 0;
                foreach (HoughLine l in lines)
                {
                    Program.logIt(string.Format("Intensity={0}, Radius={1}, Theta={2}", l.Intensity, l.Radius, l.Theta));
                    if (l.Radius < 0)
                    {
                        if (l.Theta < 90) angle = -l.Theta;
                        else angle = 180.0 - l.Theta;
                    }
                    else
                    {
                        if (l.Theta < 90) angle = -l.Theta;
                        else angle = 180.0 - l.Theta;
                    }
                }
                Program.logIt(string.Format("angle={0}", angle));
                RotateBicubic r_filter = new RotateBicubic(angle);
                Bitmap nb = r_filter.Apply(newImage);
                nb.Save("temp_2.jpg");
                rotated = nb;
            }
            /*
            HoughLine[] lines = lineTransform.GetLinesByRelativeIntensity(0.6);
            List<AForge.Math.Geometry.Line> alines = new List<Line>();
            foreach(HoughLine l in lines)
            {
                Program.logIt(string.Format("Intensity={0}, Radius={1}, Theta={2}", l.Intensity, l.Radius, l.Theta));
                float r, t;
                if (l.Radius < 0)
                {
                    r = -l.Radius;
                    t = (float)(180.0 + l.Theta);
                }
                else
                {
                    r = l.Radius;
                    t = (float)l.Theta;
                }

                AForge.Math.Geometry.Line al = AForge.Math.Geometry.Line.FromRTheta(r,t);
                Program.logIt(string.Format("slope={0}", al.Slope));
                alines.Add(al);
            }
            AForge.Math.Geometry.Line vl = AForge.Math.Geometry.Line.FromRTheta(1, 0);
            foreach(AForge.Math.Geometry.Line al in alines)
            {
                float angle = al.GetAngleBetweenLines(vl);
                Program.logIt(string.Format("angle={0}", angle));
            }
            */
            // extra
            // input rotated
            // output copped;
            Bitmap copped = null;
            {
                Grayscale g_filter = new Grayscale(0.2125, 0.7154, 0.0721);
                Bitmap grayImage = g_filter.Apply(rotated);
                Blur filter = new Blur();
                filter.ApplyInPlace(grayImage);
                OtsuThreshold o_filter = new OtsuThreshold();
                o_filter.ApplyInPlace(grayImage);
                /*
                ExtractBiggestBlob e_filter = new ExtractBiggestBlob();
                Bitmap b = e_filter.Apply(grayImage);
                int x = e_filter.BlobPosition.X;
                int y = e_filter.BlobPosition.Y;
                int w = b.Size.Width;
                int h = b.Size.Height;
                //biggestBlobsImage.Save("temp.jpg");
                Crop c_filter = new Crop(new Rectangle(x, y, w, h));
                Bitmap nb = c_filter.Apply(rotated);
                nb.Save("temp.jpg");
                */
                BlobCounter blobCounter = new BlobCounter();
                blobCounter.MinHeight = 20;
                blobCounter.MinWidth = 20;
                blobCounter.FilterBlobs = false;
                blobCounter.BlobsFilter = null;
                blobCounter.ObjectsOrder = ObjectsOrder.YX; 
                blobCounter.ProcessImage(grayImage);
                Blob[] blobs = blobCounter.GetObjectsInformation();
                Program.logIt(string.Format("blobs={0}", blobCounter.ObjectsCount));
                Rectangle r = Rectangle.Empty;
                /*
                foreach (Blob b in blobs)
                {
                    Program.logIt(string.Format("{0}: {1}", b.ID, b.Rectangle));
                    if (r == Rectangle.Empty) r = b.Rectangle;
                    else r = Rectangle.Union(r, b.Rectangle);
                }
                */
                for (int i = 1; i < blobs.Length; i++)
                {
                    Blob b = blobs[i];
                    Program.logIt(string.Format("{0}: {1}", b.ID, b.Rectangle));
                    if (r == Rectangle.Empty) r = b.Rectangle;
                    else r = Rectangle.Union(r, b.Rectangle);
                }
                Program.logIt(string.Format("rect: {0}", r));
                Crop c_filter = new Crop(r);
                Bitmap nb = c_filter.Apply(rotated);
                nb.Save("temp.jpg");
                copped = nb;
            }
            // double roate
            // input copped
            if(false)
            {
                Crop c_filter = new Crop(new Rectangle(0, copped.Height - 15, copped.Width, 10));
                Bitmap img = c_filter.Apply(copped);
                img.Save("temp_1.jpg");
                Bitmap grayImage = Grayscale.CommonAlgorithms.BT709.Apply(img);
                DifferenceEdgeDetector edgeDetector = new DifferenceEdgeDetector();
                Bitmap edgesImage = edgeDetector.Apply(grayImage);
                edgesImage.Save("temp_2.jpg");
                OtsuThreshold o_filter = new OtsuThreshold();
                o_filter.ApplyInPlace(edgesImage);
                edgesImage.Save("temp_3.jpg");
                HoughLineTransformation lineTransform = new HoughLineTransformation();
                lineTransform.ProcessImage(edgesImage);
                Program.logIt(string.Format("lines={0}", lineTransform.LinesCount));
                HoughLine[] lines = lineTransform.GetLinesByRelativeIntensity(1.0);
                double angle = 0;
                foreach (HoughLine l in lines)
                {
                    Program.logIt(string.Format("Intensity={0}, Radius={1}, Theta={2}", l.Intensity, l.Radius, l.Theta));
                    if (l.Radius < 0)
                    {
                        if (l.Theta < 90) angle = 90.0 - l.Theta;
                        else angle = l.Theta - 90;
                    }
                    else
                    {
                        if (l.Theta < 90) angle = l.Theta;
                        else angle = 90.0 - l.Theta;
                    }
                }
                Program.logIt(string.Format("angle={0}", angle));
                RotateBicubic r_filter = new RotateBicubic(angle);
                Bitmap nb = r_filter.Apply(copped);
                nb.Save("temp_4.jpg");
                rotated = nb;
            }
            if(false)
            {
                Grayscale g_filter = new Grayscale(0.2125, 0.7154, 0.0721);
                Bitmap grayImage = g_filter.Apply(rotated);
                Blur filter = new Blur();
                filter.ApplyInPlace(grayImage);
                OtsuThreshold o_filter = new OtsuThreshold();
                o_filter.ApplyInPlace(grayImage);
                /*
                ExtractBiggestBlob e_filter = new ExtractBiggestBlob();
                Bitmap b = e_filter.Apply(grayImage);
                int x = e_filter.BlobPosition.X;
                int y = e_filter.BlobPosition.Y;
                int w = b.Size.Width;
                int h = b.Size.Height;
                //biggestBlobsImage.Save("temp.jpg");
                Crop c_filter = new Crop(new Rectangle(x, y, w, h));
                Bitmap nb = c_filter.Apply(rotated);
                nb.Save("temp.jpg");
                */
                BlobCounter blobCounter = new BlobCounter();
                blobCounter.MinHeight = 32;
                blobCounter.MinWidth = 32;
                blobCounter.FilterBlobs = true;
                blobCounter.BlobsFilter = null;
                blobCounter.ObjectsOrder = ObjectsOrder.Size;
                blobCounter.ProcessImage(grayImage);
                Blob[] blobs = blobCounter.GetObjectsInformation();
                Program.logIt(string.Format("blobs={0}", blobCounter.ObjectsCount));
                Rectangle r = Rectangle.Empty;
                foreach (Blob b in blobs)
                {
                    Program.logIt(string.Format("{0}: {1}", b.ID, b.Rectangle));
                    if (r == Rectangle.Empty) r = b.Rectangle;
                    else r = Rectangle.Union(r, b.Rectangle);
                }
                Program.logIt(string.Format("rect: {0}", r));
                Crop c_filter = new Crop(r);
                Bitmap nb = c_filter.Apply(rotated);
                nb.Save("temp.jpg");
                copped = nb;
            }
            return copped;
        }
        public static void main_7(System.Collections.Specialized.StringDictionary args)
        {
            Bitmap icon = ImageDecoder.DecodeFromFile(@"C:\Users\qa\Desktop\picture\iphone_icon\tap_1_icon.jpg");
            Bitmap image = ImageDecoder.DecodeFromFile(@"C:\Users\qa\Desktop\picture\save_07.jpg");
            ColorFiltering filter = new ColorFiltering();
            filter.Red = new IntRange(180, 255);
            filter.Green = new IntRange(180, 255);
            filter.Blue = new IntRange(180, 255);
            Bitmap img = filter.Apply(image);
            img.Save("temp_1.jpg");
            // color filt
            Bitmap grayImage = Grayscale.CommonAlgorithms.BT709.Apply(img);
            // 2 - Edge detection
            DifferenceEdgeDetector edgeDetector = new DifferenceEdgeDetector();
            Bitmap edgesImage = edgeDetector.Apply(grayImage);
            // 3 - Threshold edges
            Threshold thresholdFilter = new Threshold(40);
            thresholdFilter.ApplyInPlace(edgesImage);
            edgesImage.Save("temp_2.jpg");
            // create and configure blob counter
            BlobCounter blobCounter = new BlobCounter();

            blobCounter.MinHeight = 32;
            blobCounter.MinWidth = 32;
            blobCounter.FilterBlobs = true;
            blobCounter.ObjectsOrder = ObjectsOrder.Size;

            // 4 - find all stand alone blobs
            blobCounter.ProcessImage(edgesImage);
            Blob[] blobs = blobCounter.GetObjectsInformation();

            // 5 - check each blob
            for (int i = 0, n = blobs.Length; i < n; i++)
            {
                // ...
                System.Diagnostics.Trace.WriteLine(string.Format("blob: {0}", i));
                System.Diagnostics.Trace.WriteLine(string.Format("ID: {0}", blobs[i].ID));
                System.Diagnostics.Trace.WriteLine(string.Format("rectangle: {0}", blobs[i].Rectangle));
                System.Diagnostics.Trace.WriteLine(string.Format("ColorMean: {0}", blobs[i].ColorMean));
                System.Diagnostics.Trace.WriteLine(string.Format("ColorStdDev: {0}", blobs[i].ColorStdDev));
                System.Diagnostics.Trace.WriteLine(string.Format("Fullness: {0}", blobs[i].Fullness));
                Crop c_filter = new Crop(blobs[i].Rectangle);
                Bitmap nb = c_filter.Apply(image);
                nb.Save(string.Format("blob_{0}.jpg", i));
            }
        }
        public static void main_2(System.Collections.Specialized.StringDictionary args)
        {
            /*
            using (Mat img = new Mat(200, 400, DepthType.Cv8U, 3))
            {
                img.SetTo(new Bgr(255, 0, 0).MCvScalar);
                CvInvoke.PutText(
                                  img,
                                  "Hello, world",
                                  new System.Drawing.Point(10, 80),
                                  FontFace.HersheyComplex,
                                  1.0,
                                  new Bgr(0, 255, 0).MCvScalar);
                //Image<Bgr, Byte> imgeOrigenal = img.ToImage<Bgr, Byte>();
                //ImageViewer.Show(imgeOrigenal, "Test Window");
                ImageViewer.Show(img, "Test Window");
            }
            */
            Image<Bgr, Byte> img = new Image<Bgr, byte>(@"C:\Users\qa\Desktop\picture\WIN_20180717_23_02_14_Pro.jpg");
            UMat uimage = new UMat();
            CvInvoke.CvtColor(img, uimage, ColorConversion.Bgr2Gray);
            UMat pyrDown = new UMat();
            CvInvoke.PyrDown(uimage, pyrDown);
            CvInvoke.PyrUp(pyrDown, uimage);
            //ImageViewer.Show(uimage, "b");
            double cannyThreshold = 180.0;
            double cannyThresholdLinking = 120.0;
            UMat cannyEdges = new UMat();
            CvInvoke.Canny(uimage, cannyEdges, cannyThreshold, cannyThresholdLinking);
            
            LineSegment2D[] lines = CvInvoke.HoughLinesP(
                               cannyEdges,
                               1, //Distance resolution in pixel-related units
                               Math.PI / 180.0, //Angle resolution measured in radians.
                               200, //threshold
                               30, //min Line width
                               10); //gap between lines
            AForge.Math.Geometry.Line vline = AForge.Math.Geometry.Line.FromPoints(new AForge.Point(0, 0), new AForge.Point(0, 400));
            foreach (LineSegment2D line in lines)
            {
                AForge.Point p1 = new AForge.Point(line.P1.X, line.P1.Y);
                AForge.Point p2 = new AForge.Point(line.P2.X, line.P2.Y);
                AForge.Math.Geometry.Line l1 = AForge.Math.Geometry.Line.FromPoints(p1, p2);
                AForge.Math.Geometry.LineSegment segment = new AForge.Math.Geometry.LineSegment(p1, p2);
                float angle = l1.GetAngleBetweenLines(vline);
                float length = segment.Length;
                System.Diagnostics.Trace.WriteLine(string.Format("angle={0:f}, len={1:f}", angle, length));
                img.Draw(line, new Bgr(Color.Green), 3);
            }
            ImageViewer.Show(img, "b");
            //LineSegment2D[] lines = CvInvoke.HoughLines(cannyEdges, 100, Math.PI / 180, 200, 0, 0);

        }
        public static void main_ocr(System.Collections.Specialized.StringDictionary args)
        {
            //Bitmap b = new Bitmap(@"C:\Users\qa\Desktop\picture\upgrade-galaxy-s2-i9100-to-xxls8-android-4-1-2-official-firmware-jelly-bean1.jpg");
            //var engine = new TesseractEngine(@"./tessdata", "eng", EngineMode.Default);
            //var p = engine.Process(b);

            Bitmap bmp = null;
            System.Threading.AutoResetEvent take_picture = new System.Threading.AutoResetEvent(false);
            System.Threading.AutoResetEvent take_picture_done = new System.Threading.AutoResetEvent(false);
            MJPEGStream videoSource = new MJPEGStream("http://192.168.1.77:8080/video");
            videoSource.NewFrame += (s, e) =>
            {
                if (take_picture.WaitOne(0))
                {
                    bmp = new Bitmap(e.Frame);
                    take_picture_done.Set();
                }
            };
            videoSource.Start();
            System.Threading.Thread.Sleep(3000);
            take_picture.Set();
            take_picture_done.WaitOne(3000);
            videoSource.SignalToStop();
            videoSource.WaitForStop();
            if (bmp != null)
            {
                bmp.Save("captured_01.png");
            }
        }

        public static void main_3(System.Collections.Specialized.StringDictionary args)
        {
            //D:\projects\bitbucket\pytest\images
            string dir = @"D:\projects\bitbucket\pytest\images";
            foreach(string s in System.IO.Directory.GetFiles(dir))
            {
                Bitmap src = ImageDecoder.DecodeFromFile(s);
                Bitmap temp = ImageDecoder.DecodeFromFile(@"C:\Users\qa\Desktop\picture\setting_icon.jpg");
                Grayscale g_filter = new Grayscale(0.2125, 0.7154, 0.0721);
                Bitmap src_gray = g_filter.Apply(src);
                Bitmap temp_gray = g_filter.Apply(temp);
                CannyEdgeDetector c1_filter = new CannyEdgeDetector();
                CannyEdgeDetector c2_filter = new CannyEdgeDetector();
                c1_filter.ApplyInPlace(src_gray);
                c2_filter.ApplyInPlace(temp_gray);
                ExhaustiveTemplateMatching tm = new ExhaustiveTemplateMatching(0.9f);
                TemplateMatch[] matchings = tm.ProcessImage(src_gray, temp_gray);
                src_gray.Save("temp.jpg");
                temp_gray.Save("temp1.jpg");
                foreach (TemplateMatch m in matchings)
                {
                    //
                }
            }
        }
        public static void extrac_blue_block(System.Collections.Specialized.StringDictionary args)
        {
            Bitmap src = ImageDecoder.DecodeFromFile(@"test.jpg");
            ColorFiltering filter = new ColorFiltering();
            filter.Red = new IntRange(0, 100);
            filter.Green = new IntRange(127, 200);
            filter.Blue = new IntRange(250, 255);
            Bitmap img = filter.Apply(src);
            img.Save("temp.jpg");
            Grayscale g_filter = new Grayscale(0.2125, 0.7154, 0.0721);
            Bitmap grayImage = g_filter.Apply(img);
            OtsuThreshold o_filter = new OtsuThreshold();
            o_filter.ApplyInPlace(grayImage);
            ExtractBiggestBlob e_filter = new ExtractBiggestBlob();
            Bitmap b = e_filter.Apply(grayImage);
            int x = e_filter.BlobPosition.X;
            int y = e_filter.BlobPosition.Y;
            int w = b.Size.Width;
            int h = b.Size.Height;
            Crop c_filter = new Crop(new Rectangle(x, y, w, h));
            Bitmap nb = c_filter.Apply(src);
            nb.Save("temp1.jpg");
        }
        public static void extrac_image(System.Collections.Specialized.StringDictionary args)
        {
            Bitmap src = ImageDecoder.DecodeFromFile(@"test.jpg");
            Grayscale g_filter = new Grayscale(0.2125, 0.7154, 0.0721);
            Bitmap grayImage = g_filter.Apply(src);
            OtsuThreshold o_filter = new OtsuThreshold();
            o_filter.ApplyInPlace(grayImage);
            ExtractBiggestBlob e_filter = new ExtractBiggestBlob();
            Bitmap b = e_filter.Apply(grayImage);
            int x = e_filter.BlobPosition.X;
            int y = e_filter.BlobPosition.Y;
            int w = b.Size.Width;
            int h = b.Size.Height;
            //biggestBlobsImage.Save("temp.jpg");
            Crop c_filter = new Crop(new Rectangle(x, y, w, h));
            Bitmap nb = c_filter.Apply(src);
            nb.Save("temp.jpg");
        }
        public static void smart_rotate(System.Collections.Specialized.StringDictionary args)
        {
            Bitmap b = ImageDecoder.DecodeFromFile(@"D:\projects\bitbucket\pytest\images\test_01.jpg");
            Grayscale g_filter = new Grayscale(0.2125, 0.7154, 0.0721);
            Bitmap grayImage = g_filter.Apply(b);
            GaussianBlur gb_filter = new GaussianBlur(7, 7);
            gb_filter.ApplyInPlace(grayImage);
            CannyEdgeDetector c_filter = new CannyEdgeDetector(20, 150);
            c_filter.ApplyInPlace(grayImage);
            HoughLineTransformation lineTransform = new HoughLineTransformation();
            lineTransform.ProcessImage(grayImage);
            Bitmap houghLineImage = lineTransform.ToBitmap();
            //HoughLine[] lines = lineTransform.GetLinesByRelativeIntensity(1);
            HoughLine[] lines = lineTransform.GetMostIntensiveLines(1);
            double angle = 0.0;
            foreach (HoughLine line in lines)
            {
                System.Diagnostics.Trace.WriteLine(line.Theta);
                // ...
                angle = line.Theta;
            }
            angle = 90.0 - angle;
            RotateBicubic r_filter = new RotateBicubic(angle);
            Bitmap nb = r_filter.Apply(b);
            // flip 180
            RotateBicubic flip_filter = new RotateBicubic(180);
            Bitmap nb1 = flip_filter.Apply(nb);
            nb1.Save("temp.jpg");
        }
        public static void main_5(System.Collections.Specialized.StringDictionary args)
        {
            Bitmap bmp = null;
            System.Threading.AutoResetEvent take_picture = new System.Threading.AutoResetEvent(false);
            System.Threading.AutoResetEvent take_picture_done = new System.Threading.AutoResetEvent(false);
            var videoDevices = new FilterInfoCollection(FilterCategory.VideoInputDevice);
            VideoCaptureDevice videoSource = new VideoCaptureDevice(videoDevices[0].MonikerString);
            videoSource.NewFrame += (s, e) => 
            {
                if (take_picture.WaitOne(0))
                {
                    bmp = new Bitmap(e.Frame);
                    take_picture_done.Set();
                }
            };
            videoSource.Start();
            System.Threading.Thread.Sleep(3000);
            take_picture.Set();
            take_picture_done.WaitOne(3000);
            videoSource.SignalToStop();
            videoSource.WaitForStop();
            if(bmp!=null)
            {
                bmp.Save("test.png");
            }

        }
    }
}
