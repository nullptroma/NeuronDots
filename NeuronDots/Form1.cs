using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using Neural_Network;

namespace NeuronDots
{
    public partial class Form1 : Form
    {
        private List<ColorPoint> Points = new List<ColorPoint>();
        private Bitmap Field = new Bitmap(1280 / 8, 720/8);
        private NeuralNetwork Net;
        static double Sigmoid(double x)
        {
            return 1 / (1 + Math.Exp(-x));
        }
        static double DerivativeSigmoid(double x) => (1 - Sigmoid(x)) * Sigmoid(x);
        private int Count = 0;
        private int W = 1280;
        private int H = 720;

        public Form1()
        {
            InitializeComponent();
            Net = new NeuralNetwork(Sigmoid, DerivativeSigmoid, 
                                    new NeuronLayer() { NumOfNeurons=2, Bias= true },
                                    new NeuronLayer() { NumOfNeurons = 60, Bias = true },
                                    new NeuronLayer() { NumOfNeurons = 3});
            Net.LearningRatio = 0.05;
            this.Width = W;
            this.Height = H;
            Task.Factory.StartNew(UpdateNN, TaskCreationOptions.LongRunning);
        }

        Random rnd = new Random();
        private void UpdateNN()
        {
            while (true)
            {
                double err = 0;
                if (Points.Count > 0)
                {
                    int k;
                    for (k = 0; k < Points.Count*100; k++)
                    {
                        var p = Points[rnd.Next(Points.Count)];
                        double nx = (double)p.X / W - 0.5;
                        double ny = (double)p.Y / H - 0.5;
                        Net.ForwardPassData(new double[] { nx, ny });
                        double[] targets = new double[2];
                        if (p.Type == 1)
                            Net.SetExpectedOutput(new double[] { 0, 1, 0 });
                        else if (p.Type == 2)
                            Net.SetExpectedOutput(new double[] { 0, 0, 1 });
                        else
                            Net.SetExpectedOutput(new double[] { 1, 0, 0 });
                        err += Net.Error;
                        Net.AdjustWeights();
                    }
                    err /= k;
                }


                for (int i = 0; i < W / 8; i++)
                {
                    for (int j = 0; j < H / 8; j++)
                    {
                        double nx = (double)i / W * 8 - 0.5;
                        double ny = (double)j / H * 8 - 0.5;
                        double[] outputs = Net.ForwardPassData(new double[] { nx, ny });
                        double red = Math.Max(0, Math.Min(1, outputs[0] - outputs[1] - outputs[2] + 0.5));
                        double green = Math.Max(0, Math.Min(1, outputs[1] - outputs[0] - outputs[2] + 0.5));
                        double blue = Math.Max(0, Math.Min(1, outputs[2] - outputs[1] - outputs[0] + 0.5));
                        red = 0.3 + red * 0.5;
                        green = 0.3 + green * 0.5;
                        blue = 0.3 + blue * 0.5;
                        Color c = Color.FromArgb((int)(red*255), (int)(green*255), (int)(blue*255));
                        lock(Field)
                            Field.SetPixel(i, j, c);
                    }
                }

                Invoke(new Action(() => {
                    this.Text = string.Format("{0:F8} Count:{1}", err, Count);
                    this.Refresh();
                }));
                Count++;
            }
        }

        private void Form1_Paint(object sender, PaintEventArgs e)
        {
            var g = e.Graphics;
            lock (Field)
                g.DrawImage(Field, 0,0, 1280, 720);
            foreach (var p in Points)
                g.FillEllipse(p.Type == 1 ? Brushes.Green : (p.Type == 2? Brushes.Blue : Brushes.Red), new Rectangle(p.X, p.Y, 10, 10));
        }

        private void Form1_MouseDown(object sender, MouseEventArgs e)
        {
            if (e.Button == MouseButtons.Left)
                Points.Add(new ColorPoint() { X = e.X, Y = e.Y, Type = 1 });
            else if(e.Button == MouseButtons.Right)
                Points.Add(new ColorPoint() { X = e.X, Y = e.Y, Type = 2 });
            else
                Points.Add(new ColorPoint() { X = e.X, Y = e.Y, Type = 3 });
            this.Refresh();
        }

        private void Form1_FormClosed(object sender, FormClosedEventArgs e)
        {
            //Net.SaveToXMLFile(Environment.GetFolderPath(Environment.SpecialFolder.Desktop) + "/save.xml");
        }
    }

    struct ColorPoint
    {
        public int X { get; set; }
        public int Y { get; set; }
        public int Type { get; set; }
    }
}
