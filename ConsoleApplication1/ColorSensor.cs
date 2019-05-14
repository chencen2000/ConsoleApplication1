﻿using System;
using System.Collections.Generic;
using System.IO.Ports;
using System.Linq;
using System.Text;
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
                System.Threading.Thread.Sleep(1000);
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
                _port.Write(new byte[] { 0xff }, 0, 1);
                data = get_data();
                if (!string.IsNullOrEmpty(data))
                {
                    Program.logIt($"get: {data}");
                }
            }
            if (_port != null)
            {
                if (_port.IsOpen)
                    _port.Close();
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
            //while (_incoming_data.IsEmpty)
            System.Threading.Thread.Sleep(1000);
            _need_data.Reset();
            // get data
            while (!_incoming_data.IsEmpty)
            {
                if (_incoming_data.TryDequeue(out ret))
                    sb.Append(ret);
            }
            return sb.ToString();
        }
    }
}