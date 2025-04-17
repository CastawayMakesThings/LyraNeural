package lyra;

import java.util.Arrays;

public class ExampleUsage {
    public static void main(String[] args) {

        LyraNetwork LN = new LyraNetwork();
        int[] architecture = {7,6,5,6,5,6,3};
        LN.setNeurons(architecture);
        LN.setLearningRate(0.01);
        LN.setActivationFunction("relu");
        LN.setEpochThreshold(3000000);
        LN.shouldPrintTrainingStatus(true, 100000);
        LN.init();

        double[] out = LN.feed(new double[]{1,1,1,1,1,1,1}); // or 0.0 / 2.0
        System.out.println("Initial output: " + Arrays.toString(out));

        double[][] input = {
                {1,0,0,0,0,0,0},
                {0,1,0,0,0,0,0},
                {0,0,1,0,0,0,0},
                {0,0,0,1,0,0,0},
                {0,0,0,0,1,0,0},
                {0,0,0,0,0,1,0},
                {0,0,0,0,0,0,1}
        };

        double[][] wantedResults = {
                {1,0,0},
                {0,1,0},
                {1,1,0},
                {0,0,1},
                {1,0,1},
                {0,1,1},
                {1,1,1}
        };

        LN.train(input, wantedResults);

        LyraNetwork.printResults(LN.feed(new double[] {0,0,0,0,0,1,0}));

        LN.saveModel("l.lyra");

    }
}
