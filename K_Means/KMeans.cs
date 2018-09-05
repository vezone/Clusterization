//#define Debug 

using System;
using System.Collections.Generic;

namespace K_Means.K_Means
{

    public struct Point
    {
        private double m_X, m_Y;
        public Point(double x, double y)
        {
            m_X = x;
            m_Y = y;
        }
        public Point(Point data)
            : this(data.m_X, data.m_Y)
        {
        }

        public double X
        {
            get { return m_X; }
            set { m_X = value; }
        }
        public double Y
        {
            get { return m_Y; }
            set { m_Y = value; }
        }

        public int Dimension => 2;

        public double Distance(Point other)
        {
            Point result =
                (this - other) * (this - other);
            return Math.Sqrt(result.X + result.Y);
        }

        public static Point operator +(
            Point right,
            Point left)
        {
            return new Point(right.X + left.X,
                             right.Y + left.Y);
        }

        public static Point operator -(
            Point right,
            Point left)
        {
            return new Point(right.X - left.X,
                             right.Y - left.Y);
        }

        public static Point operator *(
            Point right,
            Point left)
        {
            return new Point(right.X * right.X,
                             right.Y * right.Y);
        }

        public static Point operator /(
            Point right,
            int val)
        {
            return new Point(right.X / val,
                             right.Y / val);
        }

        public static Point operator /(
            Point right,
            Point left)
        {
            return new Point(right.X / left.X,
                             right.Y / left.Y);
        }

        public override string ToString()
        {
            return $"{m_X}, {m_Y}";
        }
    }

    public class KMeans
    {

        static Point[] rawData;

        public static void Log(object obj)
        {
#if Debug
            Console.WriteLine(obj);
#endif
        }

        public static void Logn(object obj)
        {
#if Debug
            Console.WriteLine(obj);
#endif
        }

        public static string Read()
        {
#if Debug
            return Console.ReadLine();
#else
            return "";
#endif
        }

        public static Point[] GetRawData()
        {
            return rawData;
        }

        public static void SetRawData()
        {
            rawData = new Point[]
            {
                new Point(65.0, 220.0),
                new Point(73.0, 160.0),
                new Point(59.0, 110.0),
                new Point(61.0, 120.0),
                new Point(75.0, 150.0),
                new Point(67.0, 240.0),
                new Point(68.0, 230.0),
                new Point(70.0, 220.0),
                new Point(62.0, 130.0),
                new Point(66.0, 210.0),
                new Point(77.0, 190.0),
                new Point(75.0, 180.0),
                new Point(74.0, 170.0),
                new Point(70.0, 210.0),
                new Point(61.0, 110.0),
                new Point(58.0, 100.0),
                new Point(66.0, 230.0),
                new Point(59.0, 120.0),
                new Point(68.0, 210.0),
                new Point(61.0, 130.0)
            };
        }

        public static List<List<Point>> Run()
        {
            SetRawData();
            
            Logn("Input RawData:");
            ShowData(rawData, true, true);

            int numClusters = 3;
            Logn($"Setting numClusters to {numClusters}");

            int[] clustering = Cluster(rawData, numClusters);
            Logn("K-means clustering complete!");

            Logn("Final clustering in internal form:");
            ShowVector(clustering);

            Logn("Raw data by cluster:");
            //ShowClustered(rawData, clustering, numClusters, 1);
            List<List<Point>> list = GetClusters(rawData, clustering);

            int clusterNum = 0;
            foreach (var cluster in list)
            {
                Logn($"cluster: {clusterNum}");
                Logn("===========");
                foreach (var Point in cluster)
                {
                    Logn(Point);
                }
                ++clusterNum;
                Logn("===========");
            }

            Read();
            return list;
        }

        //K-means
        public static int[] Cluster(
            Point[] rawData,
            int numClusters)
        {
            Point[] data = Normalized(rawData);
            bool changed = true, success = true;
            int[] clustering = InitClustering(data.Length, numClusters);
            Point[] means = Allocate(numClusters);

            int maxCount = 10 * data.Length;
            int iterations = 0;

            while (changed &&
                   success &&
                   iterations < maxCount)
            {
                ++iterations;
                success = UpdateMeans(data, clustering, ref means);
                changed = UpdateClustering(data, ref clustering, means);
            }

            return clustering;
        }


        public static Point[] Normalized(Point[] rawData)
        {
            Point[] result = new Point[rawData.Length];
            for (int i = 0; i < rawData.Length; i++)
                result[i] = new Point(rawData[i]);

            Point Sum = new Point(0.0, 0.0);
            for (int i = 0; i < result.Length; i++)
            {
                Sum += result[i];
            }

            Point mean = Sum / result.Length;

            Sum = new Point(0.0, 0.0);
            for (int i = 0; i < result.Length; i++)
            {
                Sum += (result[i] - mean) * (result[i] - mean);
            }

            Point standartDeviation = Sum / result.Length;
            for (int i = 0; i < result.Length; i++)
                result[i] = (result[i] - mean) / standartDeviation;

            return result;
        }

        private static int[] InitClustering(
            int numTuples,
            int numClusters)
        {
            Random random = new Random(0);
            int[] clustering = new int[numTuples];

            for (int i = 0; i < numClusters / 3; i++)
                clustering[i] = 0;

            for (int i = numClusters / 3; i < 2 * (numClusters / 3); i++)
                clustering[i] = 1;

            for (int i = 2 * (numClusters / 3); i < numClusters; i++)
                clustering[i] = 2;

            return clustering;
        }

        private static Point[] Allocate(
            int numClusters)
        {
            Point[] result = new Point[numClusters];
            for (int k = 0; k < numClusters; ++k)
                result[k] = new Point();
            return result;
        }

        private static bool UpdateMeans(
            Point[] data,
            int[] clustering,
            ref Point[] means)
        {
            int numClusters = means.Length;
            int[] clusterCounts = new int[numClusters];

            for (int i = 0; i < data.Length; i++)
                ++clusterCounts[clustering[i]]; //cluster = clustering[i]

            for (int k = 0; k < numClusters; ++k)
                if (clusterCounts[k] == 0)
                    return false;

            for (int k = 0; k < means.Length; ++k)
                means[k] = new Point(0.0, 0.0);

            for (int i = 0; i < data.Length; i++)
                means[clustering[i]] += data[i];

            for (int k = 0; k < means.Length; ++k)
                means[k] /= clusterCounts[k];

            return true;
        }

        private static bool UpdateClustering(
            Point[] data,
            ref int[] clustering,
            Point[] means)
        {
            int numClusters = means.Length;
            bool changed = false;

            int[] newClustering = new int[clustering.Length];
            Array.Copy(clustering,
                       newClustering,
                       clustering.Length);

            double[] distances =
                new double[numClusters];

            for (int i = 0; i < data.Length; ++i)
            {
                for (int k = 0; k < numClusters; ++k)
                    distances[k] =
                        data[i].Distance(means[k]);

                int newClusterID = MinIndex(distances);
                if (newClusterID != newClustering[i])
                {
                    changed = true;
                    newClustering[i] = newClusterID;
                }
            }

            if (!changed)
                return false;

            int[] clusterCounts = new int[numClusters];
            for (int i = 0; i < data.Length; ++i)
                ++clusterCounts[newClustering[i]];

            for (int k = 0; k < numClusters; ++k)
                if (clusterCounts[k] == 0)
                    return false;

            Array.Copy(newClustering, clustering,
                newClustering.Length);

            return true;
        }

        private static int MinIndex(
            double[] distance)
        {
            int indexOfMin = 0;
            double smallDist = distance[0];

            for (int k = 1;
                 k < distance.Length;
                 ++k)
            {
                if (distance[k] < smallDist)
                {
                    smallDist = distance[k];
                    indexOfMin = k;
                }
            }

            return indexOfMin;
        }

        //additional
        private static void ShowData(Point[] data,
            bool showIndices, bool newLine)
        {
            string output = "";
            for (int i = 0; i < data.Length; i++)
            {
                output =
                    showIndices == true
                    ? $"{i}: {data[i].ToString()}"
                    : $"{data[i].ToString()}";
                if (newLine)
                    Logn(output);
                else
                    Log(output + " ");
            }
        }

        private static void ShowVector(int[] vec)
        {
            string output = "";
            for (int i = 0; i < vec.Length; i++)
                output += $"{vec[i]} ";
            Log($"{output}\n");
        }

        static void ShowClustered(
            Point[] data, int[] clustering,
            int numClusters, int decimals)
        {
            for (int k = 0; k < numClusters; ++k)
            {
                Logn("===================");
                for (int i = 0; i < data.Length; ++i)
                {
                    int clusterID = clustering[i];
                    if (clusterID != k) continue;
                    Log(i.ToString().PadLeft(3) + " ");

                    if (data[i].X >= 0.0 &&
                        data[i].Y >= 0.0)
                    {
                        Log(" ");
                    }
                    Log(data[i].X.ToString("F" + decimals) + " " +
                                  data[i].Y.ToString("F" + decimals));
                    Logn("");
                }
                Logn("===================");
            } // k


        }

        private static List<List<Point>> GetClusters(
            Point[] data, int[] clustering)
        {
            List<List<Point>> list = new List<List<Point>>();

            List<Point> cluster0 = new List<Point>();
            List<Point> cluster1 = new List<Point>();
            List<Point> cluster2 = new List<Point>();

            for (int i = 0; i < data.Length; i++)
            {
                if (clustering[i] == 0)
                    cluster0.Add(data[i]);
                else if (clustering[i] == 1)
                    cluster1.Add(data[i]);
                else
                    cluster2.Add(data[i]);
            }

            list.Add(cluster0);
            list.Add(cluster1);
            list.Add(cluster2);

            return list;
        }

    }
}
