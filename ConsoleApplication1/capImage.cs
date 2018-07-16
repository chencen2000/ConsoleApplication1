using AForge;
using AForge.Imaging;
using AForge.Imaging.Filters;
using AForge.Imaging.Formats;
using AForge.Video;
using AForge.Video.DirectShow;
using System;
using System.Collections.Generic;
using System.Drawing;
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

        public static void main_1(System.Collections.Specialized.StringDictionary args)
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
        public static void main_2(System.Collections.Specialized.StringDictionary args)
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
