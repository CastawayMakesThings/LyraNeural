package lyra;

import java.io.*;
import java.util.Random;
import java.util.Scanner;
import java.util.*;



//DISCLAIMER, PLEASE DON'T JUDGE MY CODE. FOR SUGGESTIONS AND REPORTING ISSUES, CONTACT MY DISCORD IN




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
    public double calculate(double x, boolean derivative) {
        switch (activationFunction) {
            case 0: // sigmoid
                if (derivative) {
                    // assuming x is already sigmoid(x)
                    return x * (1 - x);
                } else {
                    return 1.0 / (1.0 + Math.exp(-x));
                }

            case 1: // ReLU
                if (derivative) {
                    return x > 0 ? 1.0 : 0.0;
                } else {
                    return x > 0 ? x : 0.0;
                }

            case 2: // tanh
                if (derivative) {
                    double tanhVal = Math.tanh(x);
                    return 1.0 - tanhVal * tanhVal;
                } else {
                    return Math.tanh(x);
                }

            case 3: // leaky ReLU
                if (derivative) {
                    return x > 0 ? 1.0 : 0.01;
                } else {
                    return x > 0 ? x : 0.01 * x;
                }

            default:
                throw new IllegalStateException("Unknown activation function");
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
            int layerSize;
            if (i == 0) {
                layerSize = frontLayerSize;
            } else if (i == neuronNetwork.length - 1) {
                layerSize = outputLayerSize;
            } else {
                layerSize = hiddenLayerSize[i - 1];
            }

            neuronNetwork[i] = new Neuron[layerSize];
            hiddenValues[i] = new double[layerSize];

            for (int j = 0; j < layerSize; j++) {
                Neuron neuron = new Neuron();

                // Bias: small value to push ReLU neurons into action
                neuron.bias = random.nextDouble() * 0.2 - 0.1; // [-0.1, 0.1]

                if (i > 0) {
                    int inputSize = neuronNetwork[i - 1].length;
                    neuron.weights = new double[inputSize];

                    // Choose initialization strategy based on activation function
                    double scale;
                    switch (activationFunction) {
                        case 1:
                        case 3:
                            scale = Math.sqrt(2.0 / inputSize); // He initialization
                            break;
                        case 2:
                        case 0:
                        default:
                            scale = Math.sqrt(1.0 / inputSize); // Xavier initialization
                            break;
                    }

                    for (int k = 0; k < inputSize; k++) {
                        neuron.weights[k] = random.nextGaussian() * scale;
                    }
                }

                neuronNetwork[i][j] = neuron;
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

    private double sumDeltas(Neuron[] nextLayer, double[] nextDeltas, int neuronIndex) {
        double sum = 0.0;
        for (int i = 0; i < nextLayer.length; i++) {
            sum += nextDeltas[i] * nextLayer[i].weights[neuronIndex];
        }
        return sum;
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


    public void train(double[][] inputs, double[][] expectedOutputs) {
        if (!isInitialized) {
            init();
        }

        int samples = inputs.length;
        long startTime = System.nanoTime();
        int epochs = 0;

        while (true) {
            double totalError = 0.0;

            for (int sampleIndex = 0; sampleIndex < samples; sampleIndex++) {
                double[] input = inputs[sampleIndex];
                double[] expected = expectedOutputs[sampleIndex];

                // Forward pass
                hiddenValues[0] = input;
                for (int layer = 1; layer < neuronNetwork.length; layer++) {
                    hiddenValues[layer] = new double[neuronNetwork[layer].length];
                    for (int j = 0; j < neuronNetwork[layer].length; j++) {
                        double sum = 0.0;
                        for (int k = 0; k < neuronNetwork[layer - 1].length; k++) {
                            sum += hiddenValues[layer - 1][k] * neuronNetwork[layer][j].weights[k];
                        }
                        sum += neuronNetwork[layer][j].bias;
                        hiddenValues[layer][j] = calculate(sum, false);
                    }
                }

                // Calculate error (output layer)
                double[] output = hiddenValues[hiddenValues.length - 1];
                double[] outputError = new double[output.length];
                for (int i = 0; i < output.length; i++) {
                    outputError[i] = expected[i] - output[i];
                    totalError += outputError[i] * outputError[i];
                }

                // Backward pass
                double[][] deltas = new double[neuronNetwork.length][];
                for (int i = neuronNetwork.length - 1; i > 0; i--) {
                    deltas[i] = new double[neuronNetwork[i].length];
                    for (int j = 0; j < neuronNetwork[i].length; j++) {
                        double outputVal = hiddenValues[i][j];
                        double error = (i == neuronNetwork.length - 1) ?
                                outputError[j] :
                                sumDeltas(neuronNetwork[i + 1], deltas[i + 1], j);
                        deltas[i][j] = error * calculate(outputVal, true);
                    }
                }

                // Weight & bias update
                for (int i = 1; i < neuronNetwork.length; i++) {
                    for (int j = 0; j < neuronNetwork[i].length; j++) {
                        for (int k = 0; k < neuronNetwork[i - 1].length; k++) {
                            neuronNetwork[i][j].weights[k] += learningRate * deltas[i][j] * hiddenValues[i - 1][k];
                        }
                        neuronNetwork[i][j].bias += learningRate * deltas[i][j];
                    }
                }
            }

            totalError /= samples;
            epochs++;

            if (showStatus) {
                System.out.println("Epoch " + epochs + ", Error: " + totalError);
            }

            if ((goal > 0 && totalError < goal) ||
                    (epochSensitivity > 0 && epochs >= epochSensitivity) ||
                    (time > 0 && System.nanoTime() - startTime > time)) {
                break;
            }
        }

        System.out.println("Training complete after " + epochs + " epochs.");
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
