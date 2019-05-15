using Emgu.CV;
using Emgu.CV.ML;
using System;
using System.Collections.Generic;
using System.IO.Ports;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace ConsoleApplication1
{
    class ColorSensor
    {
        static SerialPort _port = null;
        static System.Collections.Concurrent.ConcurrentQueue<string> _incoming_data = new System.Collections.Concurrent.ConcurrentQueue<string>();
        static System.Threading.ManualResetEvent _need_data = new System.Threading.ManualResetEvent(false);
        static void Main(string[] args)
        {
            System.Configuration.Install.InstallContext _args = new System.Configuration.Install.InstallContext(null, args);
            if (_args.IsParameterTrue("debug"))
            {
                System.Console.WriteLine("Wait for debugger, press any key to continue");
                System.Console.ReadKey();
            }
            if (_args.IsParameterTrue("train"))
            {
                // train process
                train(_args.Parameters);
            }
            else if (_args.IsParameterTrue("sample"))
            {
                sample_device(_args.Parameters);
            }
            else
            {
                //test();
                //test_svm();
                sample_device(_args.Parameters);
            }
        }
        static void train(System.Collections.Specialized.StringDictionary args)
        {
            try
            {
                _port = new System.IO.Ports.SerialPort(args["port"]);
                //_port = new SerialPort(args["port"], 9600);
                _port.BaudRate = 9600;
                _port.Parity = Parity.None;
                _port.StopBits = StopBits.One;
                _port.DataBits = 8;
                _port.Handshake = Handshake.None;
                _port.RtsEnable = true;
                _port.DtrEnable = true;
                _port.ReadTimeout = 1000;
                _port.WriteTimeout = 1000;
                _port.DataReceived += _port_DataReceived;
                _port.Open();
                System.Threading.Thread.Sleep(3000);
                _port.Write(new byte[] { 0xff }, 0, 1);
            }
            catch (Exception)
            {
                _port = null;
            }
            string data = "";
            if (_port != null)
            {
                System.Console.WriteLine("Please remove the device. Press any key to continue.");
                System.Console.ReadKey();
                data = get_data();
                if (!string.IsNullOrEmpty(data))
                {
                    Program.logIt($"get: {data}");
                }
                //
                System.Console.WriteLine("Please put the device. Press any key to continue.");
                System.Console.ReadKey();
                data = get_data();
                if (!string.IsNullOrEmpty(data))
                {
                    Program.logIt($"get: {data}");
                }
            }
            if (_port != null)
            {
                if (_port.IsOpen)
                {
                    _port.Write(new byte[] { 0x00 }, 0, 1);
                    _port.Close();
                }
            }
        }

        private static void _port_DataReceived(object sender, SerialDataReceivedEventArgs e)
        {
            SerialPort sp = (SerialPort)sender;
            if (sp.IsOpen)
            {
                try
                {
                    //string indata = sp.ReadExisting();
                    string indata = sp.ReadLine();
                    //Program.logIt(indata);
                    if (_need_data.WaitOne(0))
                        _incoming_data.Enqueue(indata);
                }
                catch (Exception) { }
            }
        }
        static void clear_queue()
        {
            string s;
            while (!_incoming_data.IsEmpty)
            {
                _incoming_data.TryDequeue(out s);
            }
        }
        static string get_data()
        {
            string ret = "";
            StringBuilder sb = new StringBuilder();
            _need_data.Set();
            // get data
            while (_incoming_data.IsEmpty)
                System.Threading.Thread.Sleep(100);
            _need_data.Reset();
            // get data
            while (!_incoming_data.IsEmpty)
            {
                if (_incoming_data.TryDequeue(out ret))
                    sb.Append(ret);
            }
            return sb.ToString();
        }

        static void sample_device(System.Collections.Specialized.StringDictionary args)
        {
            System.Console.WriteLine($"Open COM port: {args["port"]}.");
            //1.open serial port
            try
            {
                _port = new System.IO.Ports.SerialPort(args["port"]);
                //_port = new SerialPort(args["port"], 9600);
                _port.BaudRate = 9600;
                _port.Parity = Parity.None;
                _port.StopBits = StopBits.One;
                _port.DataBits = 8;
                _port.Handshake = Handshake.None;
                _port.RtsEnable = true;
                _port.DtrEnable = true;
                _port.ReadTimeout = 1000;
                _port.WriteTimeout = 1000;
                _port.DataReceived += _port_DataReceived;
                _port.Open();
            }
            catch (Exception)
            {
                _port = null;
                System.Console.WriteLine($"Fail to open COM port: {args["port"]}.");
                goto exit;
            }

            DateTime _start = DateTime.Now;
            bool done = false;
            System.Console.WriteLine($"Waiting for sensor ready.");
            //2.wait for sensor ready
            while (!done)
            {
                string s = get_data();
                Match m = Regex.Match(s, "Found sensor", RegexOptions.None, Regex.InfiniteMatchTimeout);
                if(m.Success)
                {
                    System.Console.WriteLine($"Sensor is ready.");
                    done = true;
                }
                if ((DateTime.Now - _start).TotalSeconds > 10)
                    break;
            }
            if (!done)
            {
                System.Console.WriteLine($"Sensor is not ready.");
                goto exit;
            }

            Regex r = new Regex(@"^Color Temp: (\d+) K - Lux: (\d+) - R: (\d+) G: (\d+) B: (\d+) C: (\d+)\s*$");
            //3.turn off led
            System.Console.WriteLine($"Trun off LED.");
            _port.Write(new byte[] { 0x00 }, 0, 1);

            System.Console.WriteLine($"Read data for white noise.");
            System.Console.WriteLine($"Please remove device from sensor, and press any key to continue and q to quit.");
            ConsoleKeyInfo k = System.Console.ReadKey();
            if(k.KeyChar=='q' || k.KeyChar == 'q')
            {
                goto exit;
            }
            //4.read data for white noise
            int samples = 10;
            int[,] white_noise = new int[samples,6];
            System.Console.WriteLine($"Read {samples} sample data for white noise.");
            done = false;
            int i = 0;
            int[] white_noise_lux = new int[samples];
            int[] white_noise_c = new int[samples];
            while (!done && i<samples)
            {
                System.Threading.Thread.Sleep(1000);
                string s = get_data();
                System.Console.WriteLine($"White noise data: {s}");
                Match m = r.Match(s);
                if(m.Success && m.Groups.Count > 6)
                {
                    white_noise[i, 0] = Int32.Parse(m.Groups[1].Value);
                    white_noise[i, 1] = Int32.Parse(m.Groups[2].Value);
                    white_noise[i, 2] = Int32.Parse(m.Groups[3].Value);
                    white_noise[i, 3] = Int32.Parse(m.Groups[4].Value);
                    white_noise[i, 4] = Int32.Parse(m.Groups[5].Value);
                    white_noise[i, 5] = Int32.Parse(m.Groups[6].Value);
                    white_noise_lux[i] = white_noise[i, 1];
                    white_noise_c[i] = white_noise[i, 5];
                    i++;
                }
            }
            System.Console.WriteLine($"Complete to sample data for white noise.");
            // MeanStandardDeviation 
            Tuple<double, double> wn_lux = MathNet.Numerics.Statistics.ArrayStatistics.MeanStandardDeviation(white_noise_lux);
            Tuple<double, double> wn_c = MathNet.Numerics.Statistics.ArrayStatistics.MeanStandardDeviation(white_noise_c);
            System.Console.WriteLine($"White noise. mean of lux={wn_lux.Item1}, stddev={wn_lux.Item2}");
            System.Console.WriteLine($"White noise. mean of C={wn_c.Item1}, stddev={wn_c.Item2}");
            done = false;
            int color_index = -1;
            string data = "";
            while (!done)
            {
                System.Console.WriteLine($"Read device color. please enter the color index (1,2,14...) and enter: ");
                data = System.Console.ReadLine();
                if(Int32.TryParse(data, out color_index))
                {
                    done = true;
                }
            }
            done = false;
            System.Console.WriteLine($"Read device color. please place devices, press q to quit.");
            List<int[]> color_data = new List<int[]>();
            int device_stage = 0;
            while (!done)
            {
                System.Threading.Thread.Sleep(1000);
                if (System.Console.KeyAvailable)
                {
                    k = System.Console.ReadKey();
                    if (k.KeyChar == 'q' || k.KeyChar == 'q')
                    {
                        done = true;
                        continue;
                    }
                }
                //string data;
                data = get_data();
                Match m = r.Match(data);
                if (m.Success)
                {
                    System.Console.WriteLine($"Data: {data}");
                    if (m.Groups.Count > 6)
                    {
                        if (device_stage == 0)
                        {
                            // wiat for device in place,
                            // get lux and c
                            int lux = Int32.Parse(m.Groups[2].Value);
                            int c = Int32.Parse(m.Groups[6].Value);
                            double r1 = (wn_lux.Item1 - lux) / wn_lux.Item1;
                            double r2 = (wn_c.Item1 - c) / wn_c.Item1;
                            if (r1 > 0.5 && r2 > 0.5)
                            {
                                // device in place
                                System.Console.WriteLine($"Device In-Place.");
                                device_stage = 1;
                            }
                        }
                        else if (device_stage == 1)
                        {
                            // device in-place
                            // led on.
                            System.Console.WriteLine($"Turn On LED .");
                            _port.Write(new byte[] { 0xff }, 0, 1);
                            device_stage = 2;
                            System.Threading.Thread.Sleep(2000);
                        }
                        else if (device_stage == 2)
                        {
                            System.Console.WriteLine($"Color Data: {data}");
                            // save color data
                            int []c = new int[6];
                            c[0] = Int32.Parse(m.Groups[1].Value);
                            c[1] = Int32.Parse(m.Groups[2].Value);
                            c[2] = Int32.Parse(m.Groups[3].Value);
                            c[3] = Int32.Parse(m.Groups[4].Value);
                            c[4] = Int32.Parse(m.Groups[5].Value);
                            c[5] = Int32.Parse(m.Groups[6].Value);
                            color_data.Add(c);
                            device_stage = 3;
                        }
                        else if (device_stage == 3)
                        {
                            // device in-place
                            // led on.
                            System.Console.WriteLine($"Turn Off LED .");
                            _port.Write(new byte[] { 0x00 }, 0, 1);
                            device_stage = 4;
                            System.Threading.Thread.Sleep(1000);
                        }
                        else if (device_stage == 4)
                        {
                            System.Console.WriteLine($"Please remove device and place another one.");
                            // wiat for device remove,
                            // get lux and c
                            int lux = Int32.Parse(m.Groups[2].Value);
                            int c = Int32.Parse(m.Groups[6].Value);
                            double r1 = (wn_lux.Item1 - lux) / wn_lux.Item1;
                            double r2 = (wn_c.Item1 - c) / wn_c.Item1;
                            if (r1 < 0.2 && r2 < 0.2)
                            {
                                // device in place
                                System.Console.WriteLine($"Device removed.");
                                device_stage = 0;
                            }
                        }
                        else
                        {

                        }
                    }
                }
            }
            //5.press any key to continue to read device color
            //6.place device
            //7.wait for device in-place
            //8.read data for device color
            //9.wait for device removal
            //10.press 'q' to quit or go to 7.
            //11.done.
            // dump color data;
            System.Console.WriteLine($"Color Index: {color_index}");
            foreach(int[] a in color_data)
            {
                StringBuilder sb = new StringBuilder();
                sb.Append($"{color_index},");
                int R = a[2];
                int G = a[3];
                int B = a[4];
                int C = a[5];
                foreach (int b in a)
                {
                    sb.Append($"{b},");
                }
                double d = 1.0 * R / C * 255;
                sb.Append($"{(int)d},");
                d = 1.0 * G / C * 255;
                sb.Append($"{(int)d},");
                d = 1.0 * B / C * 255;
                sb.Append($"{(int)d},");
                System.Console.WriteLine(sb.ToString());
            }
            exit:
            if (_port != null)
            {
                if (_port.IsOpen)
                {
                    _port.Write(new byte[] { 0x00 }, 0, 1);
                    _port.Close();
                }
            }
        }
        static void test_knn()
        {
            //Regex r = new Regex(@"^Color Temp: (\d+) K - Lux: (\d+) - R: (\d+) G: (\d+) B: (\d+) Rr: (\d+) Gr: (\d+) Br: (\d+) C: (\d+)\s*$");
            //string[] lines = System.IO.File.ReadAllLines(@"data\test.txt");
            //foreach(string s in lines)
            //{
            //    Match m = r.Match(s);
            //    if (m.Success)
            //    {
            //        if (m.Groups.Count > 9)
            //        {

            //        }
            //    }
            //}
            string s = "knn.xml";
            if (System.IO.File.Exists(s))
            {
                using (KNearest knn = new KNearest())
                {
                    knn.Load(s);
                    //bool ok = knn.Train(data, Emgu.CV.ML.MlEnum.DataLayoutType.RowSample, response);
                    Matrix<float> sample;
                    test_data(out sample);
                    float r = knn.Predict(sample);

                }
            }
            else
            {
                Matrix<float> data;
                Matrix<float> response;
                ReadMushroomData(out data, out response);

                // 
                using (KNearest knn = new KNearest())
                {
                    knn.DefaultK = 3;
                    knn.IsClassifier = true;
                    bool ok = knn.Train(data, Emgu.CV.ML.MlEnum.DataLayoutType.RowSample, response);
                    if (ok)
                    {
                        knn.Save("knn.xml");
                        //int cols = data.Cols;
                        //Matrix<float> sample = new Matrix<float>(1, cols);
                        Matrix<float> sample;
                        test_data(out sample);
                        float r = knn.Predict(sample);
                    }
                }
            }
        }
        static private void test_data(out Matrix<float> data)
        {
            string raw_data = "x,y,w,t,p,f,c,n,w,e,e,s,y,w,w,p,w,o,p,k,s,g";
            int varCount = raw_data.Split(',').Length ;
            data = new Matrix<float>(1, varCount);
            string[] values = raw_data.Split(',');
            for (int i = 0; i < values.Length; i++)
            {
                data[0, i] = System.Convert.ToByte(System.Convert.ToChar(values[i]));
            }
        }
        static private void ReadMushroomData(out Matrix<float> data, out Matrix<float> response)
        {
            string[] rows = System.IO.File.ReadAllLines(@"data\agaricus-lepiota.txt");

            int varCount = rows[0].Split(',').Length - 1;
            data = new Matrix<float>(rows.Length, varCount);
            response = new Matrix<float>(rows.Length, 1);
            int count = 0;
            foreach (string row in rows)
            {
                string[] values = row.Split(',');
                Char c = System.Convert.ToChar(values[0]);
                response[count, 0] = System.Convert.ToInt32(c);
                for (int i = 1; i < values.Length; i++)
                    data[count, i - 1] = System.Convert.ToByte(System.Convert.ToChar(values[i]));
                count++;
            }
        }
        static private void ReadMushroomData_for_SVM(out Matrix<float> data, out Matrix<int> response)
        {
            string[] rows = System.IO.File.ReadAllLines(@"data\agaricus-lepiota.txt");

            int varCount = rows[0].Split(',').Length - 1;
            data = new Matrix<float>(rows.Length, varCount);
            response = new Matrix<int>(rows.Length, 1);
            int count = 0;
            foreach (string row in rows)
            {
                string[] values = row.Split(',');
                Char c = System.Convert.ToChar(values[0]);
                response[count, 0] = System.Convert.ToInt32(c);
                for (int i = 1; i < values.Length; i++)
                    data[count, i - 1] = System.Convert.ToByte(System.Convert.ToChar(values[i]));
                count++;
            }
        }
        static void test_svm()
        {
            Matrix<float> data;
            Matrix<int> response;
            ReadMushroomData_for_SVM(out data, out response);

            using (SVM model = new SVM())
            {
                //model.KernelType = SVM.SvmKernelType.Linear;
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
                test_data(out sample);
                float r = model.Predict(sample);

            }
        }
    }
}
