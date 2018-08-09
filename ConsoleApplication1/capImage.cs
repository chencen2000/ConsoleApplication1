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

        static void Main(string[] args)
        {
            Bitmap img1 = new Bitmap(@"C:\Users\qa\Desktop\picture\save_10.jpg");
            Bitmap img2 = new Bitmap(@"C:\Users\qa\Desktop\picture\iphone_icon\email_icon.jpg");
            HarrisCornersDetector harris = new HarrisCornersDetector(0.04f, 1000f);
            IntPoint[] harrisPoints1 = harris.ProcessImage(img1).ToArray();
            IntPoint[] harrisPoints2 = harris.ProcessImage(img2).ToArray();

            CorrelationMatching matcher = new CorrelationMatching(9, img1, img2);
            IntPoint[][] matches = matcher.Match(harrisPoints1, harrisPoints2);

            Concatenate concat = new Concatenate(img1);
            Bitmap img3 = concat.Apply(img2);

            PairsMarker pairs = new PairsMarker(
                    matches[0], // Add image1's width to the X points
                                // to show the markings correctly
                    matches[1].Apply(p => new IntPoint(p.X + img1.Width, p.Y)));

            pairs.ApplyInPlace(img3);
            img3.Save("temp_1.jpg");

            List<double[]> data = new List<double[]>();
            foreach (var v in matches[0])
            {
                data.Add(new double[] { v.X, v.Y });
            }

            Accord.Math.Random.Generator.Seed = 0;
            BalancedKMeans kmeans = new BalancedKMeans(3)
            {
                // Note: in balanced k-means the chances of the algorithm oscillating
                // between two solutions increases considerably. For this reason, we 
                // set a max-iterations limit to avoid iterating indefinitely.
                MaxIterations = 100
            };
            KMeansClusterCollection clusters = kmeans.Learn(data.ToArray());
            int[] labels = clusters.Decide(data.ToArray());
        }

        static void Main_2(string[] args)
        {
            Bitmap b1 = new Bitmap(@"C:\Users\qa\Desktop\picture\save_10.jpg");
            Bitmap b2 = new Bitmap(@"C:\Users\qa\Desktop\picture\iphone_icon\email_icon_t.jpg");
            var surf = new SpeededUpRobustFeaturesDetector(threshold: 0.0002f, octaves: 5, initial: 2);
            var f1 = surf.Transform(b1);
            var f2 = surf.Transform(b2);

            KNearestNeighborMatching k = new KNearestNeighborMatching(4);
            //k.Threshold = 1.0;
            IntPoint[][] ps = k.Match(f1, f2);

            //Bitmap img1mark = new PointsMarker(ps[0]).Apply(b1);
            //Bitmap img2mark = new PointsMarker(ps[1]).Apply(b2);
            //img1mark.Save("temp_1.jpg");
            //img2mark.Save("temp_2.jpg");
            Concatenate concat = new Concatenate(b1);
            Bitmap img3 = concat.Apply(b2);
            PairsMarker pairs = new PairsMarker(
                    ps[0], // Add image1's width to the X points
                                // to show the markings correctly
                    ps[1].Apply(pp => new IntPoint(pp.X + b1.Width, pp.Y)));
            pairs.ApplyInPlace(img3);
            img3.Save("temp_1.jpg");


            Accord.Point p = Accord.Math.Geometry.PointsCloud.GetCenterOfGravity(ps[0]);
            IntPoint minP, maxP;
            Accord.Math.Geometry.PointsCloud.GetBoundingRectangle(ps[0], out minP, out maxP);
            List<double[]> data = new List<double[]>();
            foreach (var v in ps[0])
            {
                data.Add(new double[] { v.X, v.Y });
            }

            //MeanShift meanShift = new MeanShift()
            //{
            //    // Use a uniform kernel density
            //    Kernel = new UniformKernel(),
            //    Bandwidth = 2.0
            //};
            //MeanShiftClusterCollection clustering = meanShift.Learn(data.ToArray());
            //int[] labels = clustering.Decide(data.ToArray());


            KMeans kmeans = new KMeans(k: 4);
            //KMeans kmeans = new KMeans(k: 2)
            //{
            //    Distance = new Dice(),

            //    // We will compute the K-Means algorithm until cluster centroids
            //    // change less than 0.5 between two iterations of the algorithm
            //    // Tolerance = 0.05
            //};
            var clusters = kmeans.Learn(data.ToArray());
            int[] labels = clusters.Decide(data.ToArray());
            System.Collections.Generic.Dictionary<int, List<Accord.IntPoint>> d = new Dictionary<int, List<Accord.IntPoint>>();
            foreach(var v in ps[0])
            {
                int i = clusters.Decide(new double[] { v.X, v.Y });
                if (!d.ContainsKey(i))
                    d.Add(i, new List<Accord.IntPoint>());
                d[i].Add(v);
            }
            KeyValuePair<int, List<Accord.IntPoint>> max = new KeyValuePair<int, List<Accord.IntPoint>>(0, new List<Accord.IntPoint>());
            foreach(var v in d)
            {
                if (v.Value.Count > max.Value.Count)
                    max = v;
            }
            Accord.IntPoint[] poi = max.Value.ToArray();
            Accord.Math.Geometry.PointsCloud.GetBoundingRectangle(poi, out minP, out maxP);

            // 2nd round
            KMeans kmeans_2 = new KMeans(k: 4);
            data.Clear();
            foreach (var v in poi)
            {
                data.Add(new double[] { v.X, v.Y });
            }
            var clusters_2 = kmeans_2.Learn(data.ToArray());
            labels = clusters_2.Decide(data.ToArray());
        }
        static void Main_1(string[] args)
        {
            Bitmap b1 = new Bitmap(@"C:\Users\qa\Desktop\picture\save_10.jpg");
            Bitmap b2 = new Bitmap(@"C:\Users\qa\Desktop\picture\iphone_icon\phone_icon.jpg");
            HarrisCornersDetector harris = new HarrisCornersDetector();
            IntPoint[] harrisPoints1 = harris.ProcessImage(b1).ToArray();
            IntPoint[] harrisPoints2 = harris.ProcessImage(b2).ToArray();
            //Bitmap img1mark = new PointsMarker(harrisPoints1).Apply(b1);
            Bitmap img1mark = new PointsMarker(harrisPoints1).Apply(b1);
            Bitmap img2mark = new PointsMarker(harrisPoints2).Apply(b2);

            Concatenate concatenate = new Concatenate(img1mark);
            Bitmap res = concatenate.Apply(img2mark);
            res.Save("temp_1.jpg");

            CorrelationMatching matcher = new CorrelationMatching(9, b1, b2);
            //IntPoint[][] matches = matcher.Match(b1, b2, harrisPoints1, harrisPoints2);
            IntPoint[][] matches = matcher.Match(harrisPoints1, harrisPoints2);

            IntPoint[] correlationPoints1 = matches[0];
            IntPoint[] correlationPoints2 = matches[1];

            img1mark = new PointsMarker(correlationPoints1).Apply(b1);
            img2mark = new PointsMarker(correlationPoints2).Apply(b2);

            img1mark.Save("temp_1.jpg");
            img2mark.Save("temp_2.jpg");
            //Concatenate concat = new Concatenate(b1);
            //Bitmap img3 = concat.Apply(b2);
            //PairsMarker pairs = new PairsMarker(
            //    correlationPoints1, // Add image1's width to the X points
            //                // to show the markings correctly
            //    correlationPoints2.Apply(p => new IntPoint(p.X + b1.Width, p.Y)));
            //img3.Save("temp_2.jpg");
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
