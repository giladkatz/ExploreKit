package explorekit.data;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Random;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

/**
 * Created by giladkatz on 16/02/2016.
 */
public class Fold {
    private List<Integer> indices;
    private int[] numInstancesPerClass;
    private List<Integer>[] indicesByClass;
    private int numOfInstancesInFold = 0;
    private boolean isTestFold;
    private HashMap<List<String>, List<Integer>> distinctValMappings = new HashMap<>();

    public Fold(int numOfClasses, boolean isTestFold) {
        this.indices = new ArrayList<>();
        this.numInstancesPerClass = new int[numOfClasses];
        this.indicesByClass = new List[numOfClasses];
        for (int i=0; i<numOfClasses; i++) {
            this.numInstancesPerClass[i] = 0;
            this.indicesByClass[i] = new ArrayList<>();
        }
        this.isTestFold = isTestFold;
    }


    /**
     * Used to gnerate a sub-fold by randomly sampling a predefined number of samples from the fold. In the case
     * of distinct values, the the number of samples referred to the distinct values and all the associated indices
     * will be added.
     * @param numOfSamples
     * @param randomSeed
     * @return
     */
    public Fold generateSubFold(int numOfSamples, int randomSeed) {
        Fold subFold = new Fold(this.numInstancesPerClass.length, this.isTestFold);

        //determine how many instances of each class needs to be added
        double[] requiredNumOfSamplesPerClass = getRequiredNumberOfInstancesPerClass(numOfSamples);

        //now we need to randomly select the samples
        Random random = new Random(randomSeed);
        for (int i=0; i<numInstancesPerClass.length;i++) {
            if (this.distinctValMappings.size() == 0) {
                List<Integer> selectedIndicesPerClass = new ArrayList<>();
                while (selectedIndicesPerClass.size() < requiredNumOfSamplesPerClass[i]) {
                    int instanceIndex = indicesByClass[i].get(random.nextInt(indicesByClass[i].size()));
                    if (!selectedIndicesPerClass.contains(instanceIndex)) {
                        selectedIndicesPerClass.add(instanceIndex);
                        subFold.addInstance(instanceIndex, i);
                    }
                }
            } else {
                List<String>[] keySetValues = new List[distinctValMappings.keySet().size()];
                int counter = 0;
                for (List<String> key : distinctValMappings.keySet()) {
                    keySetValues[counter] = key;
                    counter++;
                }
                List<List<String>> selectedIndicesPerClass = new ArrayList<>();
                while (selectedIndicesPerClass.size() < requiredNumOfSamplesPerClass[i]) {
                    List<String> distictValKey = keySetValues[random.nextInt(keySetValues.length)];
                    if (!selectedIndicesPerClass.contains(distictValKey) && indicesByClass[i].contains(distinctValMappings.get(distictValKey).get(0))) {
                        selectedIndicesPerClass.add(distictValKey);
                        subFold.addDistinctValuesBatch(distictValKey, distinctValMappings.get(distictValKey),i);
                    }
                }
            }
        }

        return subFold;
    }

    private double[] getRequiredNumberOfInstancesPerClass(int numOfSamples) {
        double[] numOfInstancesPerClass = new double[numInstancesPerClass.length];

        //If there are no distinct values, the problem is simple
        if (distinctValMappings.size() == 0) {
            for (int i=0; i<numOfInstancesPerClass.length; i++) {
                numOfInstancesPerClass[i] = ((double)numInstancesPerClass[i]/ (double)IntStream.of(numInstancesPerClass).sum()) * numOfSamples;
            }
        }
        else {
            //We need to find the number of DISTINCT VALUES per class
            for (List<String> item : distinctValMappings.keySet()) {
                int index = distinctValMappings.get(item).get(0);
                for (int i=0; i<indicesByClass.length; i++) {
                    if (indicesByClass[i].contains(index)) {
                        numOfInstancesPerClass[i]++;
                        break;
                    }
                }
            }

            double sum = DoubleStream.of(numOfInstancesPerClass).sum();
            for (int i=0; i<numOfInstancesPerClass.length; i++) {
                numOfInstancesPerClass[i] = (numOfInstancesPerClass[i]/sum)* numOfSamples;
            }
        }

        return numOfInstancesPerClass;
    }

    /**
     * Receives a list of indices which all have the same distinct value. The function adds all the indices
     * to the fold and updates the map that keeps track of the relations among them
     * @param indices
     * @param instancesClass
     */
    public void addDistinctValuesBatch(List<String> key, List<Integer> indices, int instancesClass) {
        distinctValMappings.put(key, indices);
        for (int i : indices) {
            addInstance(i, instancesClass);
        }
    }

    /**
     * Sets the distinct values and indices of this fold
     * @param distinctValMappings
     */
    public void setDistinctValMappings(HashMap<List<String>, List<Integer>> distinctValMappings) {
        this.distinctValMappings = distinctValMappings;
    }

    /**
     * Adds an instance to the fold and updates the counter and indices list
     * @param index
     * @param classIdx
     */
    public void addInstance(int index, int classIdx){
        this.indices.add(index);
        this.indicesByClass[classIdx].add(index);
        this.numInstancesPerClass[classIdx] += 1;
        this.numOfInstancesInFold += 1;
    }

    /**
     * Returns the distinct val mappings required for validaion and the generation of the Weka-compatible
     * dataset used in the acutal classification.
     * @return
     */
    public HashMap<List<String>, List<Integer>> getDistinctValMappings() {
        return distinctValMappings;
    }

    /**
     * Returns the indices of the instances that belong to a specific class
     * @param classIdx
     * @return
     */
    public List<Integer> getIndicesPerClass(int classIdx) {
        return this.indicesByClass[classIdx];
    }

    public List<Integer>[] getIndicesPerClass() {
        return this.indicesByClass;
    }

    public void setIndicesPerClass(List<Integer>[] indicesByClass) {
        this.indicesByClass = indicesByClass;
    }

    /**
     * Returns the overall number of instances in the fold, regardless of class
     * @return
     */
    public int getNumOfInstancesInFold() {
        return this.numOfInstancesInFold;
    }

    /**
     * Sets the overall number of instnaces in the fold, regardless of class
     * @param numOfInstancesInFold
     */
    public void setNumOfInstancesInFold(int numOfInstancesInFold) { this.numOfInstancesInFold = numOfInstancesInFold; }

    /**
     * Gets the number of instances of a certain class in the fold
     * @param classIdx
     * @return
     */
    public int getNumOfInstancesPerClass(int classIdx) {
        return this.numInstancesPerClass[classIdx];
    }

    public int[] getInstancesClassDistribution() {
        return numInstancesPerClass;
    }

    public void setInstancesClassDistribution(int[] numInstancesPerClass) {this.numInstancesPerClass = numInstancesPerClass; }

    /**
     * Returns all the indices in the fold
     * @return
     */
    public List<Integer> getIndices() {
        return indices;
    }

    /**
     * Sets the indices of the fold
     * @param indices
     */
    public void setIndices(List<Integer> indices) {
        this.indices = indices;
    }

    /**
     * Returns true if this fold needs to be used as the test fold
     * @return
     */
    public boolean isTestFold() {
        return isTestFold;
    }

    /**
     * Used to define a fold as test of train
     * @param isTest
     */
    public void setIsTestFold(boolean isTest) {
        this.isTestFold = isTest;
    }

}
