using ARSoft.Tools.Net;
using ARSoft.Tools.Net.Dns;
using ARSoft.Tools.Net.Dns.DynamicUpdate;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Text;
using System.Threading.Tasks;

namespace ConsoleApplication1
{
    class dnsServer
    {
        static void Main(string[] args)
        {
            // test dnsServer.
            using (DnsServer server = new DnsServer(System.Net.IPAddress.Loopback, 10, 10))
            {
                server.QueryReceived += Server_QueryReceived;
                server.Start();

                // test
                DnsClient dc = new DnsClient(IPAddress.Parse("127.0.0.1"), 5000);
                //DnsMessage msg = dc.Resolve(DomainName.Parse("www.google.com"), RecordType.A);
                DnsMessage msg = dc.Resolve(DomainName.Parse("www.google.com"));

                Console.WriteLine("Press any key to stop server");
                Console.ReadLine();

                server.Stop();
            }

        }

        private static async Task Server_QueryReceived(object sender, QueryReceivedEventArgs eventArgs)
        {
            DnsMessage query = eventArgs.Query as DnsMessage;
            if (query != null)
            {
                DnsMessage response = query.CreateResponseInstance();
                if ((query.Questions.Count == 1))
                {
                    DnsQuestion question = query.Questions[0];
                    System.Diagnostics.Trace.WriteLine(question.Name.ToString());
                    //DnsMessage upstreamResponse = await DnsClient.Default.ResolveAsync(question.Name, question.RecordType, question.RecordClass);
                    DnsClient dc = new DnsClient(IPAddress.Parse("192.168.1.254"), 5000);
                    DnsMessage upstreamResponse = await dc.ResolveAsync(question.Name, question.RecordType, question.RecordClass);
                    if (upstreamResponse != null)
                    {
                        foreach (DnsRecordBase record in (upstreamResponse.AnswerRecords))
                        {
                            response.AnswerRecords.Add(record);
                        }
                        foreach (DnsRecordBase record in (upstreamResponse.AdditionalRecords))
                        {
                            response.AdditionalRecords.Add(record);
                        }

                        response.ReturnCode = ReturnCode.NoError;

                        // set the response
                        eventArgs.Response = response;

                    }
                }
            }
            
            //await Task.Delay(1000);

            
        }
    }
}
