package explorekit.data;

import com.google.common.base.Throwables;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.io.InputStream;
import java.io.Reader;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Created by giladkatz on 12/02/2016.
 */
public class Loader {

    /**
     * Used to load and process an ARFF file. The Dataset object and created and onstances are assigned to different
     * folds (proportionally by class)
     * @param reader
     * @param randomSeed
     * @param distinctValIndices
     * @return
     */
    public Dataset readArff(Reader reader, int randomSeed, List<Integer> distinctValIndices, int classAttIndex, double trainingSetPercentageOfDataset) throws Exception{
        ArffLoader.ArffReader arffReader;

        Properties properties = new Properties();
        InputStream input = this.getClass().getClassLoader().getResourceAsStream("config.properties");
        properties.load(input);

        try {
            arffReader = new ArffLoader.ArffReader(reader);
            Instances structure = arffReader.getStructure();
            Instances data = arffReader.getData();

            System.out.println("num of attributes:  " + structure.numAttributes());
            System.out.println("num of instances:  " + data.numInstances());

            //Begin by iterating over the columns and generating the corresponding objects
            List<ColumnInfo> columns = processHeader(structure, data.numInstances(),classAttIndex);

            //now we process the data itself and populate the columns
            processData(data, columns);

            //Next, we generate the folds
            //We only need the target attribute column to determine the folds (in case we use stratified sampling)
            int targetClassColumnIndex = classAttIndex;
            if (classAttIndex == -1) {
                targetClassColumnIndex = columns.size()-1;
            }
            List<Fold> folds;
            if (distinctValIndices == null) {
                folds = GenerateFolds(columns.get(targetClassColumnIndex), randomSeed, trainingSetPercentageOfDataset);
            }
            else {
                folds = GenerateFoldsWithDistinctValues(columns.get(targetClassColumnIndex), randomSeed, trainingSetPercentageOfDataset,distinctValIndices,columns);
            }

            List<ColumnInfo> distinctValColumnInfos = new ArrayList<>();
            if (distinctValIndices != null) {
                for (int distinctColumnIndex : distinctValIndices) {
                    distinctValColumnInfos.add(columns.get(distinctColumnIndex));
                }
            }

            //Fially, we can create the Dataset object
            Dataset dataset = new Dataset(columns, folds, targetClassColumnIndex, structure.relationName() + "_" + Integer.toString(randomSeed),
                    data.numInstances(), distinctValColumnInfos, randomSeed, Integer.parseInt(properties.getProperty("maxNumberOfDiscreteValuesForInclusionInSet")));



            return dataset;
        } catch (Exception e) {
            Throwables.propagate(e);
        }


        return  null;
    }

    /**
     * Identical to the readArff function, but receives two arff files. The additional file is used as the test set
     * @param trainSetReader
     * @param testSetReader
     * @param randomSeed
     * @param distinctValIndices
     * @return
     */
    public Dataset readArffWithFixedTrainTestFolds(Reader trainSetReader, Reader testSetReader, int randomSeed, List<Integer> distinctValIndices, int classAttIndex, double trainingSetPercentage) {
        ArffLoader.ArffReader arffReader;
        try {
            Dataset trainDataset = readArff(trainSetReader, randomSeed, distinctValIndices, classAttIndex, trainingSetPercentage);
            Dataset testDataset = readArff(testSetReader, randomSeed, distinctValIndices, classAttIndex, trainingSetPercentage);

        }
        catch (Exception e) {
            Throwables.propagate(e);
        }
        return  null;
    }


    /**
     * Used to generate folds when there are distinct values columns that are used to group instances together
     * in the final dataset
     * @param targetColumnInfo
     * @param randomSeed
     * @param trainingSetPercentage
     * @param distinctValIndices
     * @param columns
     * @return
     * @throws Exception
     */
    private List<Fold> GenerateFoldsWithDistinctValues(ColumnInfo targetColumnInfo, int randomSeed, double trainingSetPercentage,
                                     List<Integer> distinctValIndices, List<ColumnInfo> columns) throws Exception {
        Properties properties = new Properties();
        InputStream input = this.getClass().getClassLoader().getResourceAsStream("config.properties");
        properties.load(input);

        //Next, we need to get the number of classes (we assume the target class is discrete)
        int numOfClasses = ((DiscreteColumn)targetColumnInfo.getColumn()).getNumOfPossibleValues();

        //Store the indices of the instances, partitioned by their class
        ArrayList<ArrayList<Integer>> itemIndicesByClass = new ArrayList<ArrayList<Integer>>();
        for (int i = 0; i < numOfClasses; i++) {
            itemIndicesByClass.add(new ArrayList<Integer>());
        }

        //one item index refers to all the indices of the other items of the same value
        HashMap<List<String>,List<Integer>> distinctValMappings = new HashMap<>();
        List<ColumnInfo> distinctValColumns = new ArrayList<>();
        for (int colIdx : distinctValIndices) {
            distinctValColumns.add(columns.get(colIdx));
        }
        for (int i =0; i<targetColumnInfo.getColumn().getNumOfInstances(); i++) {
            final int j = i;
            List<String> sourceValues = distinctValColumns.stream().map(c -> c.getColumn().getValue(j).toString()).collect(Collectors.toList());
            if (!distinctValMappings.containsKey(sourceValues)) {
                distinctValMappings.put(sourceValues, new ArrayList<>());
            }
            distinctValMappings.get(sourceValues).add(i);
        }

        //Now we have the lines grouped by their distinct values. We now need to select a single represetative

        for (List<String> val : distinctValMappings.keySet()) {
            int firstItemIndex = distinctValMappings.get(val).get(0);
            int instanceClass = (Integer) targetColumnInfo.getColumn().getValue(firstItemIndex);
            itemIndicesByClass.get(instanceClass).add(firstItemIndex);
        }

        //Now we calculate the number of instances from each class we want to assign to fold
        int numOfFolds = Integer.parseInt(properties.getProperty("numOfFolds"));
        double[] maxNumOfInstancesPerTrainingClassPerFold = new double[numOfClasses];
        double[] maxNumOfInstancesPerTestClassPerFold = new double[numOfClasses];
        for (int i=0; i< itemIndicesByClass.size(); i++)
        {
            //If the training set overall size (in percentages) is predefined, use it. Otherwise, just create equal folds
            if (trainingSetPercentage == -1) {
                maxNumOfInstancesPerTrainingClassPerFold[i] = itemIndicesByClass.get(i).size()/numOfFolds;
                maxNumOfInstancesPerTestClassPerFold[i] = itemIndicesByClass.get(i).size()/numOfFolds;
            }
            else {
                //The total number of instances, multipllied by the training percentage and then divided by the number of the TRAINING folds
                maxNumOfInstancesPerTrainingClassPerFold[i] = itemIndicesByClass.get(i).size() * trainingSetPercentage /(numOfFolds-1);
                maxNumOfInstancesPerTestClassPerFold[i] = itemIndicesByClass.get(i).size() - maxNumOfInstancesPerTrainingClassPerFold[i];
            }

        }

        //We're using a fixed seed so we can reproduce our results
        //int randomSeed = Integer.parseInt(properties.getProperty("randomSeed"));
        Random rnd = new Random(randomSeed);

        //Now create the Fold objects and start filling them
        ArrayList<Fold> folds = new ArrayList<>(numOfClasses);
        for (int i=0; i<numOfFolds; i++) {
            boolean isTestFold = designateFoldAsTestSet(numOfFolds, i, properties.getProperty("testFoldDesignation"));
            Fold fold = new Fold(numOfClasses, isTestFold);
            folds.add(fold);
        }

        //for (int i=0; i < targetColumnInfo.getColumn().getNumOfInstances(); i++) {
        for (List<String> key: distinctValMappings.keySet()) {
            int i = distinctValMappings.get(key).get(0);
            int instanceClass = (Integer)targetColumnInfo.getColumn().getValue(i);

            boolean foundAssignment = false;
            List<String> exploredIndices = new ArrayList<>();
            while (!foundAssignment) {
                //We now randomly sample a fold and see whether the instance can be assigned to it. If not, sample again
                int foldIdx = rnd.nextInt(numOfFolds);
                if (!exploredIndices.contains(Integer.toString(foldIdx))) {
                    exploredIndices.add(Integer.toString(foldIdx));
                }

                //Now see if the instance can be assigned to the fold
                Fold fold = folds.get(foldIdx);
                if (!fold.isTestFold()) {
                    if (fold.getNumOfInstancesPerClass(instanceClass) < maxNumOfInstancesPerTrainingClassPerFold[instanceClass] || exploredIndices.size() == numOfFolds) {
                        //now that we found a match, instead of inserting one element, insert all of them
                        fold.addDistinctValuesBatch(key, distinctValMappings.get(key), instanceClass);
                        foundAssignment = true;
                    }
                }
                else {
                    if (fold.getNumOfInstancesPerClass(instanceClass) < maxNumOfInstancesPerTestClassPerFold[instanceClass] || exploredIndices.size() == numOfFolds) {
                        //now that we found a match, instead of inserting one element, insert all of them
                        fold.addDistinctValuesBatch(key, distinctValMappings.get(key), instanceClass);
                        foundAssignment = true;
                    }
                }
            }
        }
        return folds;
    }


    public InputStream getProperties() {
        return this.getClass().getClassLoader().getResourceAsStream("config.properties");
    }

    /**
     * Used to generate the folds. We don't use Weka's fold generation code because it does not provide us with
     * the indices of the instances which are assigned to each fold.
     * @param targetColumnInfo
     * @param randomSeed
     * @param trainingSetPercentage
     * @return
     * @throws Exception
     */
    private List<Fold> GenerateFolds(ColumnInfo targetColumnInfo, int randomSeed, double trainingSetPercentage) throws Exception {
        Properties properties = new Properties();
        InputStream input = this.getClass().getClassLoader().getResourceAsStream("config.properties");
        properties.load(input);

        //Next, we need to get the number of classes (we assume the target class is discrete)
        int numOfClasses = ((DiscreteColumn)targetColumnInfo.getColumn()).getNumOfPossibleValues();

        //Store the indices of the instances, partitioned by their class
        ArrayList<ArrayList<Integer>> itemIndicesByClass = new ArrayList<ArrayList<Integer>>();
        for (int i = 0; i < numOfClasses; i++) {
            itemIndicesByClass.add(new ArrayList<Integer>());
        }

        for (int i = 0; i < targetColumnInfo.getColumn().getNumOfInstances(); i++) {
            int instanceClass = (Integer) targetColumnInfo.getColumn().getValue(i);
            itemIndicesByClass.get(instanceClass).add(i);
        }

        //Now we calculate the number of instances from each class we want to assign to fold
        int numOfFolds = Integer.parseInt(properties.getProperty("numOfFolds"));
        double[] maxNumOfInstancesPerTrainingClassPerFold = new double[numOfClasses];
        double[] maxNumOfInstancesPerTestClassPerFold = new double[numOfClasses];
        for (int i=0; i< itemIndicesByClass.size(); i++)
        {
            //If the training set overall size (in percentages) is predefined, use it. Otherwise, just create equal folds
            if (trainingSetPercentage == -1) {
                maxNumOfInstancesPerTrainingClassPerFold[i] = itemIndicesByClass.get(i).size()/numOfFolds;
                maxNumOfInstancesPerTestClassPerFold[i] = itemIndicesByClass.get(i).size()/numOfFolds;
            }
            else {
                //The total number of instances, multipllied by the training percentage and then divided by the number of the TRAINING folds
                maxNumOfInstancesPerTrainingClassPerFold[i] = itemIndicesByClass.get(i).size() * trainingSetPercentage /(numOfFolds-1);
                maxNumOfInstancesPerTestClassPerFold[i] = itemIndicesByClass.get(i).size() - maxNumOfInstancesPerTrainingClassPerFold[i];
            }

        }

        //We're using a fixed seed so we can reproduce our results
        //int randomSeed = Integer.parseInt(properties.getProperty("randomSeed"));
        Random rnd = new Random(randomSeed);

        //Now create the Fold objects and start filling them
        ArrayList<Fold> folds = new ArrayList<>(numOfClasses);
        for (int i=0; i<numOfFolds; i++) {
            boolean isTestFold = designateFoldAsTestSet(numOfFolds, i, properties.getProperty("testFoldDesignation"));
            Fold fold = new Fold(numOfClasses, isTestFold);
            folds.add(fold);
        }

        for (int i=0; i < targetColumnInfo.getColumn().getNumOfInstances(); i++) {
            int instanceClass = (Integer)targetColumnInfo.getColumn().getValue(i);

            boolean foundAssignment = false;
            List<String> exploredIndices = new ArrayList<>();
            while (!foundAssignment) {
                //We now randomly sample a fold and see whether the instance can be assigned to it. If not, sample again
                int foldIdx = rnd.nextInt(numOfFolds);
                if (!exploredIndices.contains(Integer.toString(foldIdx))) {
                    exploredIndices.add(Integer.toString(foldIdx));
                }

                //Now see if the instance can be assigned to the fold
                Fold fold = folds.get(foldIdx);
                if (!fold.isTestFold()) {
                    if (fold.getNumOfInstancesPerClass(instanceClass) < maxNumOfInstancesPerTrainingClassPerFold[instanceClass] || exploredIndices.size() == numOfFolds) {
                        fold.addInstance(i, instanceClass);
                        foundAssignment = true;
                    }
                }
                else {
                    if (fold.getNumOfInstancesPerClass(instanceClass) < maxNumOfInstancesPerTestClassPerFold[instanceClass] || exploredIndices.size() == numOfFolds) {
                        fold.addInstance(i, instanceClass);
                        foundAssignment = true;
                    }
                }
            }
        }
        return folds;
    }

    private boolean designateFoldAsTestSet(int numOfFolds, int currentFoldIdx, String designationMethod) throws  Exception{
        switch(designationMethod){
            case("last"):
                if (currentFoldIdx == (numOfFolds-1)) {
                    return true;
                }
                else {
                    return false;
                }
            default:
                throw new Exception("unknown test fold selection method");
        }
    }

    /**
     * Processes the weka Instances object and populates the columns of our representation
     * @param structure
     * @param columns
     * @throws Exception
     */
    private void processData(Instances structure, List<ColumnInfo> columns) throws Exception {
        Enumeration<Instance> instances = structure.enumerateInstances();
        int elementsCounter = 0;
        while (instances.hasMoreElements()) {
            Instance instance = instances.nextElement();
            for (int i=0; i<structure.numAttributes(); i++) {
                ColumnInfo currentColumn = columns.get(i);
                switch(currentColumn.getColumn().getType())
                {
                    case Numeric:
                        double numericValue = instance.value(i);
                        currentColumn.getColumn().setValue(elementsCounter, numericValue);
                        break;
                    case Discrete:
                        int discreteValIndex = (int)instance.value(i);
                        currentColumn.getColumn().setValue(elementsCounter, discreteValIndex);
                        break;
                    case Date:
                        DateFormat dateFormat = new SimpleDateFormat(((DateColumn)currentColumn.getColumn()).getDateFomat());
                        Date dateVal = dateFormat.parse(instance.toString(i).replace("'",""));
                        currentColumn.getColumn().setValue(elementsCounter, dateVal);
                        break;
                    case String:
                        String stringVal = instance.toString(i);
                        currentColumn.getColumn().setValue(elementsCounter, stringVal);
                        break;
                    default:
                        throw new Exception("unsupported column type");
                }
            }
            elementsCounter++;
        }
    }

    /**
     * Iterates over the Attribute objects of the loaded datasets and creates the column
     * objects used by the Dataset object.
     * @param structure The Weka object containing all the data
     * @return A list of columns, initiated with their relevant types
     * @throws Exception
     */
    private List<ColumnInfo> processHeader(Instances structure, int numOfInstances, int classAttributeIndex) throws Exception {
        List<ColumnInfo> columns = new ArrayList<ColumnInfo>();
        Enumeration attributes = structure.enumerateAttributes();
        while (attributes.hasMoreElements()) {
            Attribute attribute = (Attribute) attributes.nextElement();
            switch (attribute.type()) {
                //numeric
                case 0:
                    Column numericColumn = new NumericColumn(numOfInstances);
                    ColumnInfo numericColumnInfo = new ColumnInfo(numericColumn, null, null, null, attribute.name());
                    columns.add(numericColumnInfo);
                    break;
                //discrete
                case 1:
                    Column discreteColumn = new DiscreteColumn(numOfInstances, attribute.numValues());
                    ColumnInfo discreteColumnInfo = new ColumnInfo(discreteColumn, null, null, null, attribute.name());
                    columns.add(discreteColumnInfo);
                    break;
                //String
                case 2:
                    Column stringColumn = new StringColumn(numOfInstances);
                    ColumnInfo stringColumnInfo = new ColumnInfo(stringColumn, null, null, null, attribute.name());
                    columns.add(stringColumnInfo);
                    break;
                //Date
                case 3:
                    String[] splitLine = attribute.toString().split(" ");
                    String dateFormat = "";
                    for (int i=3; i<splitLine.length; i++) {
                        dateFormat += splitLine[i] + " ";
                    }
                    dateFormat = dateFormat.substring(1,dateFormat.length()-2);
                    Column dateColumn = new DateColumn(numOfInstances,dateFormat);
                    ColumnInfo dateColumnInfo = new ColumnInfo(dateColumn, null, null, null, attribute.name());
                    columns.add(dateColumnInfo);
                    break;
                default:
                    System.out.println("unsupported column type");
                    throw new Exception("unsupported column type");
            }
        }

        //finally, we set the last column to be the target class
        ColumnInfo classColumn;
        if (classAttributeIndex == -1) {
            classColumn = columns.get(columns.size() - 1);
        }
        else {
            classColumn = columns.get(classAttributeIndex);
        }
        classColumn.SetTargetClassValue(true);
        return columns;
    }

}
