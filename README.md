
![Logo.png](<https://media-hosting.imagekit.io/073275a5510f4116/Logo.png?Expires=1838970184&Key-Pair-Id=K2ZIVPTIP2VGHC&Signature=UZp2S-sRp2LydlRq32Zm0cI4M2TO~riZIASCxjHM2cxK0KnVof-lI4oHFHbO3O3b6vSTWepYFKdgZoDN0DmtKjoG58arsbEf4UzHaysFWz2D~0gihKhJ-ufMqF7KWeKg1s2caE1oqZnyMGYFnakXquVc6X73dyuHHiQlYiq~t7bbEZvt03V~qWwNhwObtLuS620aP3kHF2PSJtXAgTCJoviWi5tzcL4ga4cfSZwA5zQa~hninzZGglfSyBR1HejoTCmeZDQ7Zhjf55zYBq0rGuPc1CnrRSJQvJJlZLvkRx6y~xfJX7qQPGHnS9-gxI2fcmTR~vxEan6DqipeK6fj5w__>)


# Lyra Neural v1.0.2

A simple, lightweight, open-source, and easy API for integrating simple ai into your projects.
NOTE: PLEASE DON'T JUDGE THE CODE.

## WARNING, SIGMOID IS CURRENTLY BROKEN, USE ReLU.


## Authors

- [AnotherCastaway (@CastawayMakesThings)](https://www.github.com/CastawayMakesThings)


## Description

Lyra Neural is a simple ML API for Java, that works in a very basic way. Although it may require some previous knowledge of how Neural Networks work, the API itself is quite easy to learn, with only a handful of methods you need to know.

![Static Badge](https://img.shields.io/badge/License-APACHE2.0-orange)

## Downloads

[Latest Release | v 1.0.2](https://github.com/CastawayMakesThings/LyraNeural/releases/tag/v1.0.2)

## Changelog

April 17, 2025:
Fixed ReLU and tanh.
Uploaded Lyra v1.0.3.

April 13, 2025:
Fixed training process.
Uploaded Lyra v1.0.2.

April 11, 2025:
First version released! 🎉
Licensed the project!

## Documentation

## -Installation-

To install it is simple. Download the official release from the from [here](https://github.com/CastawayMakesThings/LyraNeural/releases/tag/Release). Make sure you have the JAR in the location you want.
This API was made for IntelliJ, but should work for others.

###  IntelliJ IDEA

1. Create or open your project.
2. In the **Project** panel, locate your module (usually a blue folder labeled `src`).
3. Right-click the module and select **"Open Module Settings"**.
4. Go to the **"Libraries"** tab.
5. Click the **`+`** icon near the top-left and select **Java**.
6. Find and select the JAR file you downloaded.
7. Click **OK** or **Apply**.

You can now use classes from the JAR using `import lyra.LyraNetwork;`, etc.

---

###  Eclipse

1. Open your project.
2. Right-click the project folder in the **Project Explorer**.
3. Select **Properties**.
4. Go to **Java Build Path** > **Libraries** tab.
5. Click **"Add External JARs..."**.
6. Select the JAR file you downloaded.
7. Click **Apply and Close**.

You can now import and use any class from the JAR in your code.

---

###  VS Code (Java Extension Pack)

1. Open your Java project in VS Code.
2. Create a folder called `lib` (or any name) in your project root.
3. Move the JAR file into the `lib` folder.
4. Open (or create) `.vscode/settings.json` in your project, and add:

   ```json
   {
     "java.project.referencedLibraries": [
       "lib/*.jar"
     ]
   }

## -Getting started-

LYRA is very easy, with just a handful of methods you need to know.

### Step 1: Create the LyraNetwork object.

Creating the object is simple, just initialize the LyraNetwork like any other object.

    LyraNetwork myNetwork = new LyraNetwork;
No parameters required!

### Step 2: Define neurons.

To define the neurons and layers, use three methods:

    frontLayerSize(count), hiddenLayerSize(array), outputLayerSize(count).
Or you can use this one method:

   ```java 
    setNeurons(); //Use this to set all the whole architecture.
```

Here is an example of using it:
    
    myNetwork.frontLayerSize(2); //Two neurons in the first layer
    myNetwork.outputLayerSize(3); //Three neurons in the last layer
    
    int[] neuronsHidden = {5, 6, 7}; 

    //The number of ints in this array is equal to how many layers are in the network
    //Each value represents how many neurons are in that layer
    //As showed, we have 5 in the first hidden layer, 6 in the second, and 7 in the last.

    myNetwork.hiddenLayerSize(neuronsHidden); //Inputting the array we made.

Or we can use the setNeurons method
   ```java
    myNetwork.setNeurons(new double[] {2,5,6,7,3});
```

### Step 3: Other parameters.

There are some important configurations that can be made. NOTE: ReLU and Tanh only work for deep networks (Networks with more than 1 hidden layer.)

    myNetwork.setEpochThreshold(); //How many epochs (iterations) the network should train
    myNetwork.timeLimit(); //How much time in seconds it has to train
    myNetwork.precisionGoal(); //When a model reaches this accuracy, it will stop training
    myNetwork.setActivationFunction(); //Sets the activation functions. Options are sigmoid, relu, and tanh.
    myNetwork.setLearningRate(); //How course the adjustments are.
    //The higher this is, the quicker the training, but it also causes lower accuracy
### Step 4: Initialization:

After parameters are configured to your desire, you then need to "create"
the network based off of you previous configurations.

    myNetwork.init; //Simple, innit?
### Step 5: Prepare training data

This is the most complicated step. You need two arrays of arrays of doubles (names don't matter).

    double[][] inputData;
    double[][] desiredResults;
The first represents the data that will be inputted, the second represents the results you want.
Let's dive a little deeper on how it works. 

    inputData = {{0.5, 1}, {0,0}, {1, 0.5} ... };
These numbers must be between -1 and 1. The amount of numbers you put in these arrays
should equal the number of neurons in the first layer of your network.
In this case, I made 2 neurons in the front, so I am putting 2 values in each sub-array.
Same with desiredResults, except the number of values should match the number of neurons in the 
last layer. In this case three.

    desiredResults = {{0.8, 1, 0.2}, {1, 1, 0.5}, {1, 1, 1} ... }
The number of arrays in the main array should have the same amount on both inputData and desiredResults.
The first array in desiredResults are the results that you would want if you fed
the first array of inputData to the model. For example, let's say I want a model that has two
neurons on both sides of the network, and I want it to invert the values of the input.
Say, if I entered 0,1, I would want it to output 1,0 Here is how I would do that:

    inputData = {{0,0},{0,1},{1,0},{1,1}};
    desiredResults = {{1,1},{1,0},{0,1},{0,0}};
The model will loop through the data if there is fewer data than epochs (Which is probably the case.)

### Step 6: Train Model.

To train the model, use the train() method. In our case:
    
    myNetwork.train(inputData, desiredResults);
If all goes well, and you gave it enough time, the model is now ready to try using.

### Step 7: Run it.

The feed() method takes in an array of doubles as an input (Once again, the amount of doubles in the array must match
the amount of neurons in the first layer) and returns an array of doubles. For example:

    double[] input = {0,1};
    double[] result = myNetwork.feed(input);
Congrats! You have just trained and ran your own AI! If you want to see your networks total error, use the evaluate() method.

    myNetwork.evaluate(input, output);
The way these parameters work is the same the train() method's works. What it does is it takes an array 
of arrays, and the first representing inputs, and the second representing wanted values.
It goes over each array and calculates the error, and returns the average.

Here is the full code to make a network, train it, feed it, and evaluate it's error.

    public class myLyraNetwork {
    public static void main(String[] args) {

        LyraNetwork myAI = new LyraNetwork();

        int[] neurons = {2, 2};

        myAI.hiddenLayerSize(neurons);
        myAI.frontLayerSize(2);
        myAI.outputLayerSize(2);
        myAI.setLearningRate(0.2);
        myAI.setEpochThreshold(50000000);
        myAI.setActivationFunction("relu");

        myAI.init();

        double[][] input = {{0,0},{0,1},{1,0},{1,1}};
        double[][] wantedResults = {{1,1},{1,0},{0,1},{0,0}};

        myAI.train(input, wantedResults);

        double[] test = {0,1};
        double[] result = myAI.feed(test);

        LyraNetwork.printResults(result);
        System.out.println(myAI.evaluate(input, wantedResults));
    }
If you want to save your network, use the saveModel() method, and specify the filepath.

    myAI.saveModel("C://file/where/you/will/put/your/model.lyra")
It will save it to a .lyra file. Note: It only saves neuron counts, weights, and biases.


If you want to load a model, create a new LyraNetwork object.

    LyraNetwork ai = new LyraNetwork();
Next, use the loadModel method and specify the path of the .lyra file.

    ai.loadModel("C://filepath/to/the/lyra/file.lyra")
And voilà! You have loaded up a model you can now feed.

## -METHODS-

| Method Name            | Inputs                                           | Returns   | Description                                                                                                                                                         |
|------------------------|--------------------------------------------------|-----------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `hiddenLayerSize`      | int[] size                                       | void      | Sets the hidden layer's neuron count.                                                                                                                               |
| `setActivationFunction`| String function OR int func                      | void      | Sets the activation function. Options: `relu`, `sigmoid`, `tanh`, or use 0, 1, or 2.                                                                                |
| `setModelMetadata`     | String data                                      | void      | Sets metadata that will be embedded when `saveModel` is called. If `~` is typed in metadata, anything after it will be printed upon loading.                       |
| `setNeurons`           | int[] allNeurons                                 | void      | Sets all the neurons in a network. E.g., {2,4,4,2} means layers with 2, 4, 4, and 2 neurons respectively.                                                           |
| `precisionGoal`        | double goal                                      | void      | If the model reaches the given accuracy, training will automatically end.                                                                                           |
| `frontLayerSize`       | int size                                         | void      | Sets the size of the front layer.                                                                                                                                   |
| `getModelMetadata`     | none                                             | String    | Returns any metadata in the model.                                                                                                                                  |
| `outputLayerSize`      | int size                                         | void      | Sets the size of the last layer.                                                                                                                                    |
| `setLearningRate`      | double rate                                      | void      | Sets the learning rate — how coarse the adjustments are.                                                                                                            |
| `setEpochThreshold`    | int epochs                                       | void      | Sets how many epochs (iterations) the network has to train.                                                                                                         |
| `timeLimit`            | long seconds                                     | void      | Sets the time limit for training in seconds. **MAY BE BROKEN**                                                                                                      |
| `init`                 | none                                             | void      | Initializes the network with random weights and biases. Required to train custom networks.                                                                          |
| `evaluate`             | double[][] inputs, double[][] wantedResults      | double    | Feeds all the data into the network and returns the average error.                                                                                                  |
| `feed`                 | double[] inputs                                  | double[]  | Feeds the network the values and returns the output. Input array size must match neuron count in the first layer.                                                  |
| `train`                | double[][] inputs, double[][] wantedResults      | void      | Trains the network on the data given.                                                                                                                               |
| `saveModel`            | String filepath                                  | void      | Saves the neurons, weights, biases, activation functions, and metadata to the selected filepath.                                                                   |
| `loadModel`            | String filePath                                  | void      | Loads the model from the selected `.lyra` file.                                                                                                                     |
| `printResults`         | double[] input                                   | void      | A tool to easily print the results from feeding a network.                                                                                                          |



## -OTHER INFO-

### Metadata

The network does support saving various metadata to a network file by using the `setModelMetadata` method. At the moment, 
metadata is pretty useless, except for doing a custom model loading message. To add one, add any metadata you like, and 
add a '~'. Anything after the `~` will be printed when it is loaded up.

    myAI.setModelMetadata("date:10/04/2025, name:myCoolModel /*You can type whatever you want here.*/ ~Hello, World!")
As you can see in the example above, the text Hello, World! is after the '~', so that will be printed when the model is loaded.

### Training

The training process is rather complicated, so here I will explain it in more detail. There are two arrays of arrays. Each array
in the first array of arrays, the data that will be fed to each neuron in the front layer. Let's say your network has 3 neurons in the front layer.
One array could look like this: `{0,1,0.5}`. In the second array of arrays, each value in each array will be the hoped-for output of each
neuron in the last layer. Let's say our network has 4 neurons in the last layer. One array could look like this: `{0,0.25,0.75,1}`. That is 
what you _hope_ the output will be if you inputted `{0,1,0.5}`. In each array of arrays, the number of arrays in each array must match.
It will loop through all of your arrays and train on it. Let's make an example of a network that has 4 neurons in the first layer and two
in the last. Let's say we want it so that the first neuron of the output layer (we will call o1) will light up when most of the neurons in
the first layer have high values, and we want o2 to have a higher value of all the neurons in the first layer have a lower value.
We would prepare the something like this:

```java
double[][] inputs = {{0,0,0,0}, {1,1,1,1}, {0.2,0.2,0.2,0.2}, {0.7,0.7,0.7,0.7}, {1,0.75,0.2,0.89}, {0,0.25,0.6,0.3}};
double[][] wantedResults = {{0,1}, {1,0}, {0,1}, {1,0}, {1,0}, {0,1}};
myAI.train(inputs, wantedResults);
```
What this is saying is that:

If we input the FIRST array of inputs[][] (0,0,0,0) into the model, we want it to output the FIRST array of outputs[][] (0,1), 

and if we input the SECOND array of inputs[][] (1,1,1,1) into the model, we want it to output the SECOND array of outputs[][] (1,0),

and on and on. For a model like this, you would probably want more data.

## Possible Future Additions

Some features that I would hope to add are:

Model Presets

More Advanced Metadata

Example models

Most important: Maven/Gradle support.

## For support, contact my Discord @castawaymakesthings

