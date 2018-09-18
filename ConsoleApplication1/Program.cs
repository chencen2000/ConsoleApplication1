using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Text;
using System.Threading.Tasks;
using System.Xml;

namespace ConsoleApplication1
{
    class Program
    {
        public static void logIt(String msg)
        {
            System.Diagnostics.Trace.WriteLine(msg);
        }
        static void Main(string[] args)
        {
            // test
            /*
            try
            {
                WebRequest request = WebRequest.Create("https://raw.githubusercontent.com/chencen2000/powershellscript/master/test.xml");
                WebResponse response = request.GetResponse();
                Stream dataStream = response.GetResponseStream();
                StreamReader reader = new StreamReader(dataStream);
                String s = reader.ReadToEnd();
                System.Console.WriteLine(s);
                reader.Close();
                response.Close();
            }
            catch (Exception) { }
            */
            //try
            //{
            //    System.Threading.EventWaitHandle e = System.Threading.EventWaitHandle.OpenExisting("Global\\captureimage");
            //    //System.Threading.EventWaitHandle e = new System.Threading.EventWaitHandle(false, System.Threading.EventResetMode.AutoReset, @"Global\\captureimage");
            //    //System.Threading.EventWaitHandle e = new System.Threading.EventWaitHandle(false, System.Threading.EventResetMode.AutoReset, @"captureimage");
            //    e.Set();
            //}
            //catch (Exception ex) { }
            test_xml();
        }
        static void test_xml()
        {
            string s = @"test.xml";
            bool b = false;
            if (b)
            {
                var settings = new XmlWriterSettings();
                settings.OmitXmlDeclaration = true;
                settings.Indent = true;
                using (XmlWriter writer = XmlWriter.Create(s,settings))
                {
                    writer.WriteStartDocument();
                    writer.WriteStartElement("test");
                    writer.WriteStartElement("img");
                    writer.WriteAttributeString("id", "img1");
                    byte[] d = System.IO.File.ReadAllBytes(@"C:\Users\qa\Desktop\picture\save_01.jpg");
                    writer.WriteBase64(d, 0, d.Length);
                    writer.WriteEndElement();
                    writer.WriteEndElement();
                    writer.WriteEndDocument();
                    writer.Flush();
                }
            }
            else
            {
                byte[] ret = null;
                XmlDocument doc = new XmlDocument();
                doc.Load(s);
                XmlNode n = doc.DocumentElement.SelectSingleNode("//img[@id='img1']");
                using(XmlNodeReader r = new XmlNodeReader(n))
                {
                    r.MoveToContent();
                    using (MemoryStream memStream = new MemoryStream())
                    {
                        int read = 0;
                        do
                        {
                            byte[] data = new byte[1024 * 64];
                            read = r.ReadElementContentAsBase64(data, 0, data.Length);
                            if (read > 0)
                                memStream.Write(data, 0, data.Length);
                        } while (read > 0);
                        ret = memStream.ToArray();
                    }
                    r.Close();
                }
                /*
                using (XmlReader xr = XmlReader.Create(s))
                {
                    XmlDocument doc = new XmlDocument();
                    doc.Load(xr);
                    XmlNode n = doc.DocumentElement["img2"];
                    XmlNodeReader r = new XmlNodeReader(n);
                    r.MoveToContent();
                    //while (r.Read())
                    {
                        //if (r.NodeType == XmlNodeType.Text)
                        {
                            using (MemoryStream memStream = new MemoryStream())
                            {
                                //r.Read();
                                byte[] data = new byte[10240];
                                int read = 0;
                                do
                                {
                                    read = r.ReadElementContentAsBase64(data, 0, data.Length);
                                    memStream.Write(data, 0, data.Length);
                                } while (read > 0);
                                FileStream outStream = File.OpenWrite("temp_1.jpg");
                                memStream.WriteTo(outStream);
                                outStream.Flush();
                                outStream.Close();
                            }
                        }
                    }
                    xr.Close();
                }
                */
            }
        }
        static void Main2(string[] args)
        {
            double d = 2.5f;
            byte[] b = BitConverter.GetBytes(d);
            string s = Convert.ToBase64String(b);

            double[][] inputs =
{
    // The first two are from class 0
    new double[] { -5, -2, -1 },
    new double[] { -5, -5, -6 },

    // The next four are from class 1
    new double[] {  2,  1,  1 },
    new double[] {  1,  1,  2 },
    new double[] {  1,  2,  2 },
    new double[] {  3,  1,  2 },

    // The last three are from class 2
    new double[] { 11,  5,  4 },
    new double[] { 15,  5,  6 },
    new double[] { 10,  5,  6 },
};


            System.Collections.Generic.Dictionary<string, object> data = new Dictionary<string, object>();
            //foreach(var a in inputs)
            //{

            //}
            data.Add("good", inputs);
            System.Web.Script.Serialization.JavaScriptSerializer jss = new System.Web.Script.Serialization.JavaScriptSerializer();
            s = jss.Serialize(data);

            System.Collections.Generic.Dictionary<string, object> d1 = jss.Deserialize<System.Collections.Generic.Dictionary<string, object>>(s);
            //capImage.main(null);
        }
        static void Main1(string[] args)
        {
            System.Configuration.Install.InstallContext _args = new System.Configuration.Install.InstallContext(null, args);
            if (_args.IsParameterTrue("debug"))
            {
                System.Console.WriteLine("Wait for debugger, press any key to continue...");
                System.Console.ReadKey();
            }

            System.Net.HttpListener listener = new System.Net.HttpListener();
            listener.Prefixes.Add("http://+:80/");
            listener.Start();
            listener.BeginGetContext(new AsyncCallback(httpListenerCallback), listener);
            bool quit = false;
            System.Console.WriteLine("Http starts, press x to terminate...");
            while (!quit)
            {
                if (System.Console.KeyAvailable)
                {
                    System.ConsoleKeyInfo k = System.Console.ReadKey();
                    if (k.KeyChar == 'x')
                        quit = true;
                }
                if (!quit)
                    System.Threading.Thread.Sleep(1000);
            }
            if(listener.IsListening)
                listener.Stop();
            listener.Close();
        }

        static void httpListenerCallback(IAsyncResult result)
        {
            System.Net.HttpListener listener = (System.Net.HttpListener)result.AsyncState;
            try
            {
                if (listener.IsListening)
                {
                    // continue to listen
                    listener.BeginGetContext(new AsyncCallback(httpListenerCallback), listener);

                    // handle the incoming request
                    System.Net.HttpListenerContext context = listener.EndGetContext(result);
                    System.Net.HttpListenerRequest request = context.Request;
                    string responseString;
                    if (string.Compare("/appletv/us/js/application.js", request.Url.LocalPath, true) == 0)
                    {
                        responseString = System.IO.File.ReadAllText(@"D:\projects\local\atv\com.apple.trailers\application.js");
                    }
                    else if (string.Compare("/appletv/us/nav.xml", request.Url.LocalPath, true) == 0)
                    {
                        responseString = System.IO.File.ReadAllText(@"D:\projects\local\atv\com.apple.trailers\index.xml");
                    }
                    else if (string.Compare("/appletv/studios/marvel/ironman3/index-hd.xml", request.Url.LocalPath, true) == 0)
                    {
                        responseString = System.IO.File.ReadAllText(@"D:\projects\local\atv\com.apple.trailers\ironman3.index-hd.xml");
                    }
                    else if (string.Compare("/appletv/studios/marvel/ironman3/videos/trailer1-hd.xml", request.Url.LocalPath, true) == 0)
                    {
                        responseString = System.IO.File.ReadAllText(@"D:\projects\local\atv\com.apple.trailers\ironman3.videos.trailer1-hd.xml");
                    }
                    else
                    {
                        responseString = System.IO.File.ReadAllText(@"D:\projects\local\atv\atv\index.xml");
                    }
                    System.Net.HttpListenerResponse response = context.Response;
                    //string responseString = System.IO.File.ReadAllText(@"D:\projects\local\atv\atv\index.xml");
                    byte[] buffer = System.Text.Encoding.UTF8.GetBytes(responseString);
                    response.ContentLength64 = buffer.Length;
                    System.IO.Stream output = response.OutputStream;
                    output.Write(buffer, 0, buffer.Length);
                    output.Close();
                }
            }
            catch (Exception ex)
            {
            }
        }
    }
}
