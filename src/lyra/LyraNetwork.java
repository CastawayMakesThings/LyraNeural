package lyra;

import java.io.*;
import java.util.Random;
import java.util.Scanner;
import java.util.*;



//DISCLAIMER, PLEASE DON'T JUDGE MY CODE. FOR SUGGESTIONS AND REPORTING ISSUES, CONTACT MY DISCORD IN THE README FILE.



public class LyraNetwork {
    Neuron[][] neuronNetwork;
    private int[] hiddenLayerSize;
    private int frontLayerSize;
    private int outputLayerSize;
    private final Random random = new Random();
    private double[][] hiddenValues;
    private double learningRate = 0.5;
    private int epochSensitivity;
    private boolean showStatus = false;
    private long time = 0;
    private double goal = 0;
    public final double lyraFileVersion = 1.0;
    public final double lyraAPIVersion = 1.01;
    private boolean isInitialized = false;
    private int activationFunction = 0;
    private String modelMetadata;


    //Sets the size of the hidden layers.
    public void hiddenLayerSize(int[] size) {
        hiddenLayerSize = size;
    }

    //Changes the activation function
    public void setActivationFunction(String function) {
        function = function.toLowerCase();
        switch (function) {
            case "sigmoid":
                activationFunction = 0;
                break;
            case "relu":
                activationFunction = 1;
                break;
            case "tanh":
                activationFunction = 2;
                break;
            default:
                throw new IllegalArgumentException("ACTIVATION FUNCTION \""+function+"\" NOT RECOGNIZED \nThe options are sigmoid, relu, and tanh");
        }
    }

    //Sets the activation function, but an int input
    public void setActivationFunction(int function) {
        if(function > 2) {
            throw new IllegalArgumentException("UNKNOWN ACTIVATION FUNCTION SELECTED. \n USE 0 for Sigmoid,\n 1 for ReLU,\n and 2 for tanh");
        }
        activationFunction = function;
    }

    //Prints the intro message
    public LyraNetwork() {
        System.out.println("LyraNetwork version "+lyraAPIVersion+" loaded!");
    }

    //Sets any model metadata
    public void setModelMetadata(String i) {
        if(i.contains("#") || i.contains("@@")) {
            throw new modelFileLoaderException("ERROR: METADATA CONTAINS ILLEGAL CHARACTERS");
        } else {
            modelMetadata = i;
        }
    }

    //Makes it easier to print feed results
    public static void printResults(double[] input) {
        for (double v : input) {
            System.out.println(v);
        }
    }

    //Sets all the neurons
    public void setNeurons(int[] allNeurons) {
        if(!(allNeurons.length > 2)) {
            throw new IllegalArgumentException("ERROR: NETWORK MUST HAVE MORE THAN TWO LAYERS.");
        }

        frontLayerSize = allNeurons[0];
        hiddenLayerSize = new int[allNeurons.length - 2];
        System.arraycopy(allNeurons, 1, hiddenLayerSize, 0, allNeurons.length - 1 - 1);
        outputLayerSize = allNeurons[allNeurons.length-1];
    }

    //Sets an error goal for training
    public void precisionGoal(double goal) {
        this.goal = goal;
    }

    //Sets the front layer size
    public void frontLayerSize(int size) {
        frontLayerSize = size;
    }

    //Returns any model metadata
    public String getModelMetadata() {
        if(! modelMetadata.isEmpty() || modelMetadata.isBlank()) {
            return modelMetadata;
        } else {
            System.out.println("No Metadata found!");
            return null;
        }
    }

    //Sets the output layer size
    public void outputLayerSize(int size) {
        outputLayerSize = size;
    }

    //Sets the learning rate
    public void setLearningRate(double rate) {
        learningRate = rate;
    }

    //Sets how many epochs the model has to train
    public void setEpochThreshold(int epochs) {
        epochSensitivity = epochs;
    }

    //If it should print the status of training
    public void shouldShowStatus(boolean x) {
        showStatus = x;
    }

    //Sets a time limit for training, in seconds
    public void timeLimit(long seconds) {
        this.time = seconds * 1000000000;
    }


    //All of the activation functions
    private double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }
    private double sigmoidDerivative(double x) {
        return x * (1 - x); // NOTE: x should be output of sigmoid, not raw input!
    }
    private double relu(double x) {
        return Math.max(0, x);
    }
    private double reluDerivative(double x) {
        return x > 0 ? 1 : 0;
    }
    private double tanh(double x) {
        return Math.tanh(x);
    }
    private double tanhDerivative(double x) {
        double tanhVal = Math.tanh(x);
        return 1 - tanhVal * tanhVal;
    }
    private double leakyRelu(double x) {
        return x > 0 ? x : 0.01 * x;
    }
    private double leakyReluDerivative(double x) {
        return x > 0 ? 1 : 0.01;
    }
    private double calculate(double x, boolean isDerivative) {
        switch (activationFunction) {
            case 0:
                return isDerivative ? sigmoidDerivative(x) : sigmoid(x);
            case 1:
                return isDerivative ? reluDerivative(x) : relu(x);
            case 2:
                return isDerivative ? tanhDerivative(x) : tanh(x);
            case 3:
                return isDerivative ? leakyReluDerivative(x) : leakyRelu(x);
            default:
                throw new IllegalStateException("Unknown activation function selected");
        }
    }

    //A method that extracts all digits from a string
    public String extractNumbers(String input) {
        StringBuilder result = new StringBuilder();
        boolean hasDecimal = false;
        boolean hasMinus = false;

        for (int i = 0; i < input.length(); i++) {
            char c = input.charAt(i);

            if (Character.isDigit(c)) {
                result.append(c);
            } else if (c == '-' && !hasMinus && result.isEmpty()) {
                result.append(c);
                hasMinus = true;
            } else if (c == '.' && !hasDecimal) {
                result.append(c);
                hasDecimal = true;
            }
        }

        return result.toString();
    }

    //Initializes the network
    public void init() {
        if (isInitialized) return;

        neuronNetwork = new Neuron[hiddenLayerSize.length + 2][];
        hiddenValues = new double[neuronNetwork.length][];

        for (int i = 0; i < neuronNetwork.length; i++) {
            if (i == 0) {
                neuronNetwork[i] = new Neuron[frontLayerSize];
                hiddenValues[i] = new double[frontLayerSize];
            } else if (i == neuronNetwork.length - 1) {
                neuronNetwork[i] = new Neuron[outputLayerSize];
                hiddenValues[i] = new double[outputLayerSize];
            } else {
                neuronNetwork[i] = new Neuron[hiddenLayerSize[i - 1]];
                hiddenValues[i] = new double[hiddenLayerSize[i - 1]];
            }

            for (int j = 0; j < neuronNetwork[i].length; j++) {
                neuronNetwork[i][j] = new Neuron();
                neuronNetwork[i][j].bias = random.nextDouble() * 2 - 1; // [-1, 1]

                if (i > 0) {
                    neuronNetwork[i][j].weights = new double[neuronNetwork[i - 1].length];
                    for (int k = 0; k < neuronNetwork[i][j].weights.length; k++) {
                        neuronNetwork[i][j].weights[k] = random.nextDouble() * 2 - 1; // [-1, 1]
                    }
                }
            }
        }

        isInitialized = true;
    }

    //Checks the network for it's error
    public double evaluate(double[][] inputs, double[][] wantedResults) {
        double[] error = new double[inputs.length];
        double[] result;

        if (inputs.length != wantedResults.length) {
            throw new IllegalArgumentException("ERROR: INPUT SET SIZE DOES NOT MATCH OUTPUT SET SIZE");
        }

        for (int i = 0; i < inputs.length; i++) {

            result = feed(inputs[i]);

            // Calculate output layer error
            double[] outputError = new double[result.length];
            double[] outputDelta = new double[result.length];
            error[i] = 0;

            for (int x = 0; x < result.length; x++) {
                outputError[x] = wantedResults[i][x] - result[x];
                error[i] += Math.pow(outputError[x], 2);
                outputDelta[x] = outputError[x] * calculate(hiddenValues[neuronNetwork.length - 1][x], true);
            }

            // MSE component: 0.5 * sum of squared errors for this sample
            error[i] *= 0.5;
        }

        // Calculate average error (Mean Squared Error)
        double totalError = 0;
        for (double v : error) {
            totalError += v;
        }
        totalError = totalError / error.length;

        return totalError;
    }

    //Feeds the network
    public double[] feed(double[] inputs) {
        double[] outputs = new double[outputLayerSize];

        if(!isInitialized) {
            throw new IllegalStateException ("ERROR: MODEL NOT YET INITIALIZED");
        }

        //Prechecks
        if(inputs.length != frontLayerSize) {
            throw new IllegalStateException("ERROR: INPUT COUNT DOES NOT MATCH FIRST LAYER NEURON COUNT");
        }

        //Enter the inputs in the network
        for (int i = 0; i < inputs.length; i++) {
            neuronNetwork[0][i].value = inputs[i];
        }

        //The actual process
        //For every layer
        for (int i = 1; i <= neuronNetwork.length - 1; i++) {
            //For every neuron
            for (int j = 0; j < neuronNetwork[i].length; j++) {
                //Sets the bias
                hiddenValues[i][j] = neuronNetwork[i][j].bias;
                //For every weight
                for (int k = 0; k < neuronNetwork[i][j].weights.length; k++) {
                    //Adds up the previous neuron values times the weights
                    hiddenValues[i][j] += neuronNetwork[i][j].weights[k] * neuronNetwork[i-1][k].value;
                }
                //Does a function on the value.
                neuronNetwork[i][j].value = calculate(hiddenValues[i][j], false);
            }
        }

        //Returns the values of the last layer.
        for (int i = 0; i <= outputLayerSize - 1; i++) {
            outputs[i] = neuronNetwork[neuronNetwork.length - 1][i].value;
        }

        return outputs;
    }


    //Trains the network
    public void train(double[][] inputs, double[][] wantedResults) {
        System.out.println("Training...");

        if(!isInitialized) {
            System.out.println("NOTE: MODEL NOT YET INITIALIZED, AUTO INITIALIZING...");
            init();
        }

        if(hiddenLayerSize.length > 2 && !(activationFunction == 0)) {
            throw new IllegalStateException("ERROR: SHALLOW NETWORKS (Networks with only 1 hidden layer) ARE ONLY COMPATIBLE WITH SIGMOID");
        }

        if (inputs == null || wantedResults == null || inputs.length != wantedResults.length) {
            throw new IllegalArgumentException("Invalid training data");
        }

        int dataPair = 0;
        int iterations = 0;
        double totalError = 999;
        boolean isIterating = true;
        long iterationTime = 0;

        long startTime = System.nanoTime();

        while(isIterating) {

            if (iterations > epochSensitivity) {
                isIterating = false;
            }

            if (iterationTime > time && time != 0) {
                isIterating = false;
            }

            if(showStatus) {
                System.out.println("EPOCH " + iterations);
            }


            if(dataPair >= inputs.length) {
                dataPair = 0;
            }

            double[] result = feed(inputs[dataPair]);
            double[] wantedResult = wantedResults[dataPair];

            // Calculate output layer error
            double[] outputError = new double[result.length];
            double[] outputDelta = new double[result.length];
            totalError = 0;

            for (int i = 0; i < result.length; i++) {
                outputError[i] = wantedResult[i] - result[i];
                totalError += Math.pow(outputError[i], 2);
                outputDelta[i] = outputError[i] * calculate(hiddenValues[neuronNetwork.length - 1][i], true);
            }
            totalError *= 0.5;

            if(totalError < goal) {
                isIterating = false;
            }

            // Calculate hidden layer errors
            double[][] hiddenError = new double[neuronNetwork.length][];
            double[][] hiddenDelta = new double[neuronNetwork.length][];

            // Initialize arrays
            for (int h = 0; h < neuronNetwork.length; h++) {
                hiddenError[h] = new double[neuronNetwork[h].length];
                hiddenDelta[h] = new double[neuronNetwork[h].length];
            }

            for (int h = neuronNetwork.length - 2; h > 0; h--) {
                for (int i = 0; i < neuronNetwork[h].length; i++) {
                    hiddenError[h][i] = 0;
                    for (int j = 0; j < neuronNetwork[h + 1].length; j++) {
                        hiddenError[h][i] += neuronNetwork[h + 1][j].weights[i] *
                                (h == neuronNetwork.length - 2 ? outputDelta[j] : hiddenDelta[h + 1][j]);
                    }
                    hiddenDelta[h][i] = hiddenError[h][i] * calculate(hiddenValues[h][i], true);
                }
            }

            // Update output layer weights
            for (int i = 0; i < outputLayerSize; i++) {
                for (int j = 0; j < neuronNetwork[neuronNetwork.length - 2].length; j++) {
                    neuronNetwork[neuronNetwork.length - 1][i].weights[j] +=
                            learningRate * outputDelta[i] * neuronNetwork[neuronNetwork.length - 2][j].value;
                }
                neuronNetwork[neuronNetwork.length - 1][i].bias += learningRate * outputDelta[i];
            }

            // Update hidden layer weights
            for (int h = neuronNetwork.length - 2; h > 0; h--) {
                for (int i = 0; i < neuronNetwork[h].length; i++) {
                    for (int j = 0; j < neuronNetwork[h - 1].length; j++) {
                        neuronNetwork[h][i].weights[j] +=
                                learningRate * hiddenDelta[h][i] * neuronNetwork[h - 1][j].value;
                    }
                    neuronNetwork[h][i].bias += learningRate * hiddenDelta[h][i];
                }
            }

            dataPair++;
            iterations++;

            iterationTime = System.nanoTime() -startTime;

        }
        long endTime = System.nanoTime();
        long totalTime = endTime - startTime;
        System.out.println("PROCESS COMPLETE! \nTRAINING TOOK "+ totalTime / 1000000000+" SECONDS.");
        System.out.println("TOTAL ERROR: "+totalError);
    }

    //Saves the model
    public void saveModel(String filePath) {
        if(!isInitialized) {
            throw new IllegalStateException("ERROR: CAN NOT SAVE UNINITIALIZED MODEL");
        }
        File file = new File(filePath);
        FileWriter writer;
        try {
            writer = new FileWriter(file);
            writer.write("LYRAMODEL v"+ lyraFileVersion +"#");
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        String s = "";

        s = s+ frontLayerSize + "#";

        for (int value : hiddenLayerSize) {
            s = s + value + "h#";
        }

        s=s+outputLayerSize+"#";
        s=s+activationFunction+"#";

        for (int i = 1; i < neuronNetwork.length; i++) {
            for (int j = 0; j < neuronNetwork[i].length; j++) {
                s = s+neuronNetwork[i][j].bias + "b#";
                for (int k = 0; k < neuronNetwork[i][j].weights.length; k++) {
                    s = s+ neuronNetwork[i][j].weights[k] + "#";
                }
                s = s+"#";
            }
        }

        s = s+"@@"+modelMetadata;

        try {
            writer.write(s);
            writer.close();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        System.out.println("Successfully wrote model to "+filePath);

    }

    //Loads the model
    public void loadModel(String filePath) {
        File file = new File(filePath);
        if(!file.exists()) {
            throw new versionMismatchException("ERROR: CAN NOT FIND SPECIFIED FILE");
        }
        StringBuilder s = new StringBuilder();
        try (Scanner scanner = new Scanner(file)) {
            while (scanner.hasNextLine()) {
                s.append(scanner.nextLine());
            }
        } catch (FileNotFoundException e) {
            throw new RuntimeException(e);
        }

        int index = s.indexOf("@@");
        String metaData = s.substring(index + 2);
        StringBuilder newS = new StringBuilder(s.substring(0, index));
        s = newS;


        if (!s.toString().startsWith("LYRAMODEL")) {
            throw new versionMismatchException("ERROR: FILE IS NOT A LYRA MODEL");
        }

        String[] a = s.toString().split("#");

        if(a[0].contains("v")){
            String fyi = String.valueOf(lyraFileVersion);
            if(! a[0].contains(fyi)) {
                throw new versionMismatchException("ERROR: CAN NOT LOAD "+filePath+" AS IT IS OUTDATED");
            }

        } else {
            throw new modelFileLoaderException("ERROR LOADING FILE, FILE NOT COMPATIBLE");
        }


        int i = 0;
        frontLayerSize = Integer.parseInt(a[i+1]);

        // Extract hidden layer sizes
        i = 1;
        List<Integer> hiddenSizesList = new ArrayList<>();
        while (a[i].contains("h")) {
            hiddenSizesList.add(Integer.parseInt(a[i].replace("h", "")));
            i++;
        }

        hiddenLayerSize = new int[hiddenSizesList.size()];
        for (int j = 0; j < hiddenSizesList.size(); j++) {
            hiddenLayerSize[j] = hiddenSizesList.get(j);
        }

        outputLayerSize = Integer.parseInt(a[i++]);
        setActivationFunction(Integer.parseInt(a[i+2]));
        i ++;

        // Initialize the neuron network
        neuronNetwork = new Neuron[hiddenLayerSize.length + 2][];
        neuronNetwork[0] = new Neuron[frontLayerSize];
        for (int j = 1; j < neuronNetwork.length - 1; j++) {
            neuronNetwork[j] = new Neuron[hiddenLayerSize[j - 1]];
        }
        neuronNetwork[neuronNetwork.length - 1] = new Neuron[outputLayerSize];

        for (int j = 0; j < neuronNetwork.length; j++) {
            for (int k = 0; k < neuronNetwork[j].length; k++) {
                neuronNetwork[j][k] = new Neuron();
                if (j != 0) {
                    neuronNetwork[j][k].weights = new double[neuronNetwork[j - 1].length];
                }
            }
        }

        hiddenValues = new double[neuronNetwork.length][];
        for (int g = 0; g < neuronNetwork.length; g++) {
            if(g == 0) {
                hiddenValues[g] = new double[frontLayerSize];
            }
            else if(g < neuronNetwork.length - 1) {
                hiddenValues[g] = new double[hiddenLayerSize[g - 1]];
            } else {
                hiddenValues[g] = new double[outputLayerSize];
            }
        }

        // Load bias and weights
        for (int j = 1; j < neuronNetwork.length; j++) {
            for (int k = 0; k < neuronNetwork[j].length; k++) {
                neuronNetwork[j][k].bias = Double.parseDouble(extractNumbers(a[i++]));
                for (int w = 0; w < neuronNetwork[j][k].weights.length; w++) {
                    if (i < a.length && !a[i].isEmpty() && !a[i].contains("b") && !a[i].contains("h")) {
                        neuronNetwork[j][k].weights[w] = Double.parseDouble(extractNumbers(a[i++]));
                    }
                }
                if (i < a.length && a[i].isEmpty()) {
                    i++; // skip empty string between weights and next bias
                }
            }
        }
        modelMetadata = metaData;

        if(modelMetadata.contains("~")) {
            System.out.println("Message from model: "+modelMetadata.substring(modelMetadata.indexOf("~") + 1));
        }
        isInitialized = true;
    }
}
