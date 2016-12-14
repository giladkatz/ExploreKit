package explorekit.data;


import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Created by giladkatz on 11/02/2016.
 */
public class Dataset {
    private List<ColumnInfo> columns;
    private int numOfInstancesPerColumn;

    private List<Fold> folds;
    private List<Integer> indices;
    private List<Integer> indicesOfTrainingFolds;
    private List<Integer> indicesOfTestFolds;
    private List<Integer>[] trainingIndicesByClass;

    private int[] numOfTrainingInstancesPerClass;
    private int[] numOfTestInstancesPerClass;

    private int numOfTrainingRows = 0;
    private int numOfTestRows = 0;

    private int targetColumnIndex;
    private String name;

    private List<ColumnInfo> distinctValColumns = new ArrayList<>();
    private List<ColumnInfo> distinctValueCompliantColumns = new ArrayList<>();
    private HashMap<List<String>, List<Integer>> trainFoldDistinctValMappings;
    private HashMap<List<String>, List<Integer>> testFoldDistinctValMappings;
    private List<Integer> trainFoldsDistinctValRepresentatives;
    private List<Integer> testFoldsDistinctValRepresentatives;

    //Defines the maximal number of distinct values a discrete attribute can have in order to be included in the ARFF file
    private int maxNumOFDiscreteValuesForInstancesObject;

    /**
     * Used in all the operations which require a random variable. A fixed seed enables us to recreate experiments.
     */
    private int randomSeed;

    Dataset(List<ColumnInfo> columns, List<Fold> folds, int targetClassIdx, String name, int numOfInstancesPerColumn, List<ColumnInfo> distinctValColumns, int randomSeed, int maxNumOfValsPerDiscreteAttribtue) throws Exception {
        this.randomSeed = randomSeed;
        this.columns = columns;
        this.numOfInstancesPerColumn = numOfInstancesPerColumn;
        this.folds = folds;
        this.targetColumnIndex = targetClassIdx;
        this.name = name;
        this.maxNumOFDiscreteValuesForInstancesObject = maxNumOfValsPerDiscreteAttribtue;
        if (distinctValColumns != null) {
            this.distinctValColumns = distinctValColumns;
            trainFoldDistinctValMappings = new HashMap<>();
            testFoldDistinctValMappings = new HashMap<>();
        }


        this.indices = new ArrayList<>();
        this.indicesOfTrainingFolds = new ArrayList<>();
        this.indicesOfTestFolds = new ArrayList<>();
        this.trainingIndicesByClass = new List[folds.get(0).getInstancesClassDistribution().length];

        this.numOfTrainingInstancesPerClass = new int[folds.get(0).getInstancesClassDistribution().length];
        this.numOfTestInstancesPerClass = new int[folds.get(0).getInstancesClassDistribution().length];
        for (Fold fold: folds) {
            this.indices.addAll(fold.getIndices());

            if (!fold.isTestFold()) {
                this.indicesOfTrainingFolds.addAll(fold.getIndices());
                for (int classIdx = 0; classIdx < fold.getInstancesClassDistribution().length; classIdx++) {
                    int numOfInstance = fold.getNumOfInstancesPerClass(classIdx);
                    this.numOfTrainingInstancesPerClass[classIdx] += numOfInstance;
                    this.numOfTrainingRows += numOfInstance;
                }

                for (int i=0; i<folds.get(0).getInstancesClassDistribution().length; i++) {
                    if (this.trainingIndicesByClass[i] == null) {
                        this.trainingIndicesByClass[i] = new ArrayList<>();
                    }
                    this.trainingIndicesByClass[i].addAll(fold.getIndicesPerClass(i));
                }

                //Add all the distint values of the fold to the dataset object
                trainFoldDistinctValMappings.putAll(fold.getDistinctValMappings());
            }
            else {
                this.indicesOfTestFolds.addAll(fold.getIndices());
                for (int classIdx = 0; classIdx < fold.getInstancesClassDistribution().length; classIdx++) {
                    int numOfInstance = fold.getNumOfInstancesPerClass(classIdx);
                    this.numOfTestInstancesPerClass[classIdx] += numOfInstance;
                    this.numOfTestRows += numOfInstance;
                }
                //Add all the distint values of the fold to the dataset object
                testFoldDistinctValMappings.putAll(fold.getDistinctValMappings());
            }
        }

        //Now that we are done processing the indices, we select one "representative" for each distinct value
        trainFoldsDistinctValRepresentatives = new ArrayList<>();
        for (List<String> key : trainFoldDistinctValMappings.keySet()) {
            int index = trainFoldDistinctValMappings.get(key).get(0);
            trainFoldsDistinctValRepresentatives.add(index);
        }
        testFoldsDistinctValRepresentatives = new ArrayList<>();
        for (List<String> key : testFoldDistinctValMappings.keySet()) {
            int index = testFoldDistinctValMappings.get(key).get(0);
            testFoldsDistinctValRepresentatives.add(index);
        }

        //finally, we sort the indices so that they will correspond with the order of the values in the columns
        Collections.sort(this.indices);
        Collections.sort(this.indicesOfTrainingFolds);
        Collections.sort(this.indicesOfTestFolds);
        Collections.sort(trainFoldsDistinctValRepresentatives);
        Collections.sort(testFoldsDistinctValRepresentatives);
        for (int i=0; i<this.trainingIndicesByClass.length; i++) {
            Collections.sort(this.trainingIndicesByClass[i]);
        }

        for (ColumnInfo ci : columns) {
            if (isColumnDistinctValuesCompatibe(ci)) {
                distinctValueCompliantColumns.add(ci);
            }
        }
    }

    /**
     * Recieved another dataset that needs to be added to the current dataset as a test set
     * @param testSet
     */
    public void AttachExternalTestFold(Dataset testSet) throws Exception {

        int numOfRowsInBaseDataset = this.numOfTrainingRows + this.numOfTestRows;

        //If an existing fold is defined as test, change it to train
        for (Fold fold : folds) {
            if (fold.isTestFold()) {
                fold.setIsTestFold(false);
            }
        }

        Fold newTestFold = new Fold(numOfTrainingInstancesPerClass.length, true);
        if (testSet.getDistinctValueColumns() == null || testSet.getDistinctValueColumns().size() == 0) {
            for (int i = 0; i < testSet.numOfTrainingRows + testSet.numOfTestRows; i++) {
                newTestFold.addInstance(this.numOfTrainingRows + this.numOfTestRows + i, (Integer) testSet.getTargetClassColumn().getColumn().getValue(i));
            }
        }
        else {
            for (Fold testSetFold : testSet.folds) {
                for (List<String> sources : testSetFold.getDistinctValMappings().keySet()) {
                    int firstItemIndexInBatch = testSetFold.getDistinctValMappings().get(sources).get(0);
                    int groupClass = (Integer) testSet.getTargetClassColumn().getColumn().getValue(firstItemIndexInBatch);
                    List<Integer> newIndices = new ArrayList<>();
                    for (int val : testSetFold.getDistinctValMappings().get(sources)) {
                        newIndices.add(val + numOfRowsInBaseDataset);
                    }
                    newTestFold.addDistinctValuesBatch(sources, newIndices, groupClass);
                }
            }
        }

        this.folds.add(newTestFold);

        //change the folds indices
        indices.addAll(testSet.getIndices());
        indicesOfTrainingFolds.addAll(indicesOfTestFolds);

        indicesOfTestFolds = testSet.getIndicesOfTestInstances();
        indicesOfTestFolds.addAll(testSet.getIndicesOfTrainingInstances());

        //update the total size of the joined datast
        numOfTrainingRows = indicesOfTrainingFolds.size();
        numOfTestRows = indicesOfTestFolds.size();

        //now we need to update every aspect of the current Dataset object
        numOfInstancesPerColumn += testSet.getNumOfInstancesPerColumn();

        //the current division of train/test needs to be discarded. All items need to be transferred to the training
        for (int i=0; i<numOfTrainingInstancesPerClass.length; i++) {
            numOfTrainingInstancesPerClass[i] += numOfTestInstancesPerClass[i];
            numOfTestInstancesPerClass[i] = testSet.getNumOfRowsPerClassInTrainingSet()[i] + testSet.getNumOfRowsPerClassInTestSet()[i];
        }

        //If the dataset has distinct values then we need to modify additional parameters
        //move all the distinct values in the test (of the training dataset) to the train
        if (distinctValColumns != null) {
            AttachExternalDatasetDistinctValues(testSet);
        }


        //Finally, the task pf updating the column objects
        for (int i=0; i<columns.size(); i++) {
            Column currentColumn = columns.get(i).getColumn();
            List<Integer> tempList = new ArrayList<>(); tempList.add(i);
            Column testSetColumn = testSet.getColumns(tempList).get(0).getColumn();

            switch (currentColumn.getType()) {
                case Discrete:
                    DiscreteColumn discreteReplacementColumn = new DiscreteColumn(numOfInstancesPerColumn, ((DiscreteColumn)currentColumn).getNumOfPossibleValues());
                    populateJoinedColumnValues(discreteReplacementColumn, currentColumn, testSetColumn);
                    columns.get(i).setColumn(discreteReplacementColumn);
                    break;
                case Numeric:
                    NumericColumn numericReplacementColumn = new NumericColumn(numOfInstancesPerColumn);
                    populateJoinedColumnValues(numericReplacementColumn, currentColumn, testSetColumn);
                    columns.get(i).setColumn(numericReplacementColumn);
                    break;
                case Date:
                    DateColumn dateReplacementColumn = new DateColumn(numOfInstancesPerColumn, ((DateColumn)currentColumn).getDateFomat());
                    populateJoinedColumnValues(dateReplacementColumn, currentColumn, testSetColumn);
                    columns.get(i).setColumn(dateReplacementColumn);
                    break;
                case String:
                    StringColumn stringReplacementColumn = new StringColumn(numOfInstancesPerColumn);
                    populateJoinedColumnValues(stringReplacementColumn, currentColumn, testSetColumn);
                    columns.get(i).setColumn(stringReplacementColumn);
                    break;
                default:
                    throw new Exception("unidentified column type");
            }
        }
    }

    private void AttachExternalDatasetDistinctValues(Dataset testSet) {
        trainFoldDistinctValMappings.clear();
        testFoldDistinctValMappings.clear();
        trainFoldsDistinctValRepresentatives.clear();
        testFoldsDistinctValRepresentatives.clear();

        for (Fold fold: folds) {
            //Add all the distint values of the fold to the dataset object
            if (!fold.isTestFold()) {
                trainFoldDistinctValMappings.putAll(fold.getDistinctValMappings());
            }
            else {
                testFoldDistinctValMappings.putAll(fold.getDistinctValMappings());
            }
        }

        //Now that we are done processing the indices, we select one "representative" for each distinct value
        trainFoldsDistinctValRepresentatives.clear();
        testFoldsDistinctValRepresentatives.clear();
        trainFoldsDistinctValRepresentatives = new ArrayList<>();
        for (List<String> key : trainFoldDistinctValMappings.keySet()) {
            int index = trainFoldDistinctValMappings.get(key).get(0);
            trainFoldsDistinctValRepresentatives.add(index);
        }
        testFoldsDistinctValRepresentatives = new ArrayList<>();
        for (List<String> key : testFoldDistinctValMappings.keySet()) {
            int index = testFoldDistinctValMappings.get(key).get(0);
            testFoldsDistinctValRepresentatives.add(index);
        }


    }

    private void populateJoinedColumnValues(Column newColumn, Column currentColumn, Column testSetColumn) {
        for (int j=0; j<numOfTrainingRows; j++) {
            newColumn.setValue(j,currentColumn.getValue(j));
        }
        for (int j=numOfTrainingRows; j<numOfTrainingRows+numOfTestRows; j++) {
            newColumn.setValue(j,testSetColumn.getValue(j-numOfTrainingRows));
        }
    }

    /**
     * Returns the required size of each
     * @return
     */
    public int getNumOfInstancesPerColumn() {
        return this.numOfInstancesPerColumn;
    }

    /**
     * Internal constructor, used for replication
     */
    private Dataset() {}

    /**
     * Gets the indeices of the instances assigned to the training folds
     * @return
     */
    public List<Integer> getIndicesOfTrainingInstances() {
        return indicesOfTrainingFolds;
    }

    /**
     * Gets the idices of the instances assigned to the test folds
     * @return
     */
    public List<Integer> getIndicesOfTestInstances() {
        return indicesOfTestFolds;
    }

    /**
     * Returns the indices of the samples allocated to this dataset
     * @return
     */
    public List<Integer> getIndices() {
        return indices;
    }

    /**
     * Returns the number of samples in this dataset (both training and test)
     * @return
     */
    public int getNumberOfRows() {
        return this.numOfTrainingRows + this.numOfTestRows;
    }

    /**
     * Returns the name of the dataset
     * @return
     */
    public String getName() {
        return this.name;
    }

    /**
     * Returns the total number of lines in the training dataset
     * @return
     */
    public int getNumOfTrainingDatasetRows() {
        return this.numOfTrainingRows;
    }

    /**
     * Returns the total number of lines in the test dataset
     * @return
     */
    public int getNumOfTestDatasetRows() {
        return this.numOfTestRows;
    }

    /**
     * Returns the number of classes in the dataset
     * @return
     */
    public int getNumOfClasses() {
        return numOfTrainingInstancesPerClass.length;
    }

    /**
     * Returns the number of samples that belong to each class in the training set
     * @return
     */
    public int[] getNumOfRowsPerClassInTrainingSet() {
        return numOfTrainingInstancesPerClass;
    }

    /**
     * Returns the number of samples that belong to each class in the test set
     * @return
     */
    public int[] getNumOfRowsPerClassInTestSet() {
        return numOfTestInstancesPerClass;
    }

    public List<Integer>[] getTrainingIndicesByClass() {
        return this.trainingIndicesByClass;
    }

    public List<Fold> getFolds() {
        return this.folds;
    }

    /**
     * Returns the index of the class with the least number of instances
     * @return
     */
    public int getMinorityClassIndex() {
        int currentIdx = -1;
        int numOfInstances = Integer.MAX_VALUE;
        for (int i=0; i<numOfTrainingInstancesPerClass.length; i++) {
            if (numOfTrainingInstancesPerClass[i]< numOfInstances) {
                numOfInstances = numOfTrainingInstancesPerClass[i];
                currentIdx = i;
            }
        }
        return currentIdx;
    }

    /**
     * Returns speofic columns from the dataset
     * @param columnIndices
     * @return
     */
    public List<ColumnInfo> getColumns(List<Integer> columnIndices) {
        List<ColumnInfo> columnsList = new ArrayList<>();
        for (int columnIndex: columnIndices) {
            columnsList.add(columns.get(columnIndex));
        }
        return columnsList;
    }

    public void addColumn(ColumnInfo column) {
        this.columns.add(column);
    }

    /**
     * Returns all the colums of the dataset object
     * @param includeTargetColumn whether the target column should also be returned
     * @return
     */
    public List<ColumnInfo> getAllColumns(boolean includeTargetColumn) {
        List<ColumnInfo> columnsList = new ArrayList<>();
        for (ColumnInfo column: columns) {
            if ((!distinctValColumns.contains(column)) && (!column.isTargetClass() || includeTargetColumn)) {
                columnsList.add(column);
            }
        }
        return columnsList;
    }

    /**
     * Returns all the columns of a specified type
     * @param columnType
     * @param includeTargetColumn whether the target column should also be returned if it meets the criterion
     * @return
     */
    public List<ColumnInfo> getAllColumnsOfType (Column.columnType columnType, boolean includeTargetColumn) {
        List<ColumnInfo> columnsToReturn = new ArrayList<>();
        for (ColumnInfo ci: columns) {
            if (ci.getColumn().getType() == columnType) {
                if (ci == getTargetClassColumn()) {
                    if (includeTargetColumn) {
                        columnsToReturn.add(ci);
                    }
                }
                else {
                    columnsToReturn.add(ci);
                }
            }
        }
        return columnsToReturn;
    }

    /**
     * Returns the target class column
     * @return
     */
    public ColumnInfo getTargetClassColumn() {
        return columns.get(targetColumnIndex);
    }

    /**
     * Returns the columns used to create the distinct value of the instances
     * @return
     */
    public List<ColumnInfo> getDistinctValueColumns() {
        return this.distinctValColumns;
    }


    /**
     * Samples a predefined number of samples from the dataset (while maintaining the ratio)
     * and generates a Weka Instances object.
     * IMPORTANT:this function is written so that it can only be applied on the training set, because
     * the classification model is trained on it. The test set is meant to be used as a whole.
     * @param numOfSamples
     * @param randomSeed
     * @return
     * @throws Exception
     */
    public Instances generateSetWithSampling(int numOfSamples, int randomSeed) throws Exception {
        double[] numOfRequiredIntancesPerClass = new double[numOfTrainingInstancesPerClass.length];

        //Start by getting the number of items we need from each class
        for (int i=0; i<numOfRequiredIntancesPerClass.length; i++) {
            numOfRequiredIntancesPerClass[i] = numOfSamples * (((double)numOfTrainingInstancesPerClass[i])/((double)numOfTrainingRows));
        }

        //Now we extract the subset for each class
        Random rnd = new Random(randomSeed);
        List<Integer> subsetIndicesList = new ArrayList<>();
        for (int i=0; i<numOfRequiredIntancesPerClass.length; i++) {
            int assignedItemsFromClass = 0;
            while (assignedItemsFromClass < numOfRequiredIntancesPerClass[i]) {
                int pos = rnd.nextInt(this.trainingIndicesByClass[i].size());
                int index = trainingIndicesByClass[i].get(pos);
                if (!subsetIndicesList.contains(index)) {
                    subsetIndicesList.add(index);
                    assignedItemsFromClass++;
                }
            }
        }

        //get all the attributes that need to be included in the set
        ArrayList<Attribute> attributes = new ArrayList<>();
        getAttributesListForClassifier(attributes);
        Instances finalSet = new Instances("trainingSet", attributes, 0);
        double[][] dataMatrix = getDataMatrixByIndices(subsetIndicesList);
        for (int i=0; i<dataMatrix[0].length; i++) {
            double[] arr = new double[dataMatrix.length];
            for (int j=0; j<dataMatrix.length; j++) {
                arr[j] = dataMatrix[j][i];
            }
            DenseInstance di = new DenseInstance(1.0, arr);
            finalSet.add(i, di);
        }
        finalSet.setClassIndex(targetColumnIndex-getNumberOfDateStringAndDistinctColumns());

        return finalSet;
    }

    /**
     * Used to obtain either the training or test set of the dataset
     * @param getTrainingSet
     * @return
     * @throws Exception
     */
    public Instances generateSet(boolean getTrainingSet) throws Exception {
        ArrayList<Attribute> attributes = new ArrayList<>();

        //get all the attributes that need to be included in the set
        getAttributesListForClassifier(attributes);

        //Create an empty set of instances and populate with the instances
        double[][] dataMatrix;
        Instances finalSet;
        if (getTrainingSet) {
            finalSet = new Instances("trainingSet", attributes, 0);
            dataMatrix = getTrainingDataMatrix();
        }
        else {
            finalSet = new Instances("testSet", attributes, 0);
            dataMatrix = getTestDataMatrix();
        }

        for (int i=0; i<dataMatrix[0].length; i++) {
            double[] arr = new double[dataMatrix.length];
            for (int j=0; j<dataMatrix.length; j++) {
                arr[j] = dataMatrix[j][i];
            }
            DenseInstance di = new DenseInstance(1.0, arr);
            finalSet.add(i, di);
        }
        finalSet.setClassIndex(targetColumnIndex-getNumberOfDateStringAndDistinctColumnsBeforeTargetClass());

        return finalSet;
    }

    /**
     * Iterates over all the columns of the dataset object and returns those that can be
     * included in the Instances object that will be fed to Weka (i.e. excluding the Date
     * and String columns)
     * @param attributes
     * @throws Exception
     */
    private void getAttributesListForClassifier(ArrayList<Attribute> attributes) throws Exception {
        for (int i =0; i< columns.size(); i++) {
            ColumnInfo currentColumn = columns.get(i);

            //The dataset will not include the distinct value columns, if they exist
            if (this.distinctValColumns.contains(currentColumn)) {
                continue;
            }

            Attribute att = null;
            switch(currentColumn.getColumn().getType())
            {
                case Numeric:
                    att = new Attribute(Integer.toString(i),i);
                    break;
                case Discrete:
                    List<String> values = new ArrayList<>();
                    int numOfDiscreteValues = ((DiscreteColumn)currentColumn.getColumn()).getNumOfPossibleValues();
                    //if the number of distinct values exceeds the maximal amount, skip it
                    if (numOfDiscreteValues > this.maxNumOFDiscreteValuesForInstancesObject) {
                        break;
                    }
                    for (int j=0; j<numOfDiscreteValues; j++) { values.add(Integer.toString(j)); }
                    att = new Attribute(Integer.toString(i), values, i);
                    break;
                case String:
                    //Most classifiers can't handle Strings. Currently we don't include them in the dataset
                    break;
                case Date:
                    //Currently we don't include them in the dataset. We don't have a way of handling "raw" dates
                    break;
                default:
                    throw new Exception("unsupported column type");
            }
            if (att != null) {
                attributes.add(att);
            }
        }
    }

    /**
     * Returns the number of columns in the dataset which are either String or Date
     * @return
     */
    private int getNumberOfDateStringAndDistinctColumns() {
        int numOfColumns = 0;
        for (ColumnInfo ci : columns) {
            if (ci.getColumn().getType().equals(Column.columnType.Date) || ci.getColumn().getType().equals(Column.columnType.String) ||
                    distinctValColumns.contains(ci)) {
                numOfColumns++;
            }
            if (ci.getColumn().getType() == Column.columnType.Discrete && ((DiscreteColumn)ci.getColumn()).getNumOfPossibleValues() > this.maxNumOFDiscreteValuesForInstancesObject) {
                numOfColumns++;
            }
        }
        return numOfColumns;
    }

    /**
     * In cases where the target class is not the last attribute, we need to determine its new index.
     * @return
     */
    private int getNumberOfDateStringAndDistinctColumnsBeforeTargetClass() {
        int numOfColumns = 0;
        for (ColumnInfo ci : columns) {
            if (ci == columns.get(targetColumnIndex)) {
                return numOfColumns;
            }
            if (ci.getColumn().getType().equals(Column.columnType.Date) || ci.getColumn().getType().equals(Column.columnType.String) ||
                    distinctValColumns.contains(ci)) {
                numOfColumns++;
            }
            if (ci.getColumn().getType() == Column.columnType.Discrete && ((DiscreteColumn)ci.getColumn()).getNumOfPossibleValues() > this.maxNumOFDiscreteValuesForInstancesObject) {
                numOfColumns++;
            }
        }
        return numOfColumns;
    }


    /**
     * Returns a two-dimensional, Weka-friendly array. The array contains only the lines whose indices
     * were provided
     * @param indicesList
     * @return
     */
    public double[][] getDataMatrixByIndices(List<Integer> indicesList) {
        //we distinct val column(s) is not included in the matrix
        double[][] data = new double[columns.size() - (this.distinctValColumns.size() + getNumberOfDateStringAndDistinctColumns())][indicesList.size()];
        int skippedColumnsCounter = 0;
        for (int col = 0; col < columns.size(); col++) {
            //if this is a distinct val column or if the column is a raw string
            if ( shouldColumnBeExncludedInDataMatrix(col)) {
                skippedColumnsCounter++;
                continue;
            }
            int rowCounter = 0;
            boolean isNumericColumn = columns.get(col).getColumn().getType().equals(Column.columnType.Numeric);

            for (int row : indicesList) {
                if (isNumericColumn) {
                    data[col-skippedColumnsCounter][rowCounter] = (Double) columns.get(col).getColumn().getValue(row); }
                else {
                    data[col-skippedColumnsCounter][rowCounter] = (Integer) columns.get(col).getColumn().getValue(row);
                }
                rowCounter++;
            }
        }
        return data;
    }

    /**
     * Returns the training set instances in a Weka-friendly, two-dimentsional array format
     * @return
     */
    public double[][] getTrainingDataMatrix() {
        if (trainFoldDistinctValMappings != null && trainFoldDistinctValMappings.size() > 0)
            return getTrainingDataMatrixWithDistinctVals();

        double[][] data = new double[columns.size() - (getNumberOfDateStringAndDistinctColumns())][numOfTrainingRows];
        int skippedColumnsCounter = 0;
        for (int col = 0; col < columns.size(); col++) {
            //if this is a distinct val column or if the column is a raw string
            if ( shouldColumnBeExncludedInDataMatrix(col)) {
                skippedColumnsCounter++;
                continue;
            }
            int rowCounter = 0;
            boolean isNumericColumn = columns.get(col).getColumn().getType().equals(Column.columnType.Numeric);

            for (int row : indicesOfTrainingFolds) {
                if (isNumericColumn) {
                    data[col-skippedColumnsCounter][rowCounter] = (Double) columns.get(col).getColumn().getValue(row); }
                else {
                    data[col-skippedColumnsCounter][rowCounter] = (Integer) columns.get(col).getColumn().getValue(row);
                }
                rowCounter++;
            }
        }
        return data;
    }

    /**
     * Returns the test set instances in a Weka-friendly, two-dimentsional array format
     * @return
     */
    public double[][] getTestDataMatrix() {
        if (testFoldDistinctValMappings != null && testFoldDistinctValMappings.size() > 0)
            return getTestDataMatrixWithDistinctVals();

        double[][] data = new double[columns.size() - (getNumberOfDateStringAndDistinctColumns())][numOfTestRows];
        int skippedColumnsCounter = 0;
        for (int col = 0; col < columns.size(); col++) {
            if ( shouldColumnBeExncludedInDataMatrix(col)) {
                skippedColumnsCounter++;
                continue;
            }
            int rowCounter = 0;
            boolean isNumericColumn = columns.get(col).getColumn().getType().equals(Column.columnType.Numeric);
            for (int row : indicesOfTestFolds) {
                if (isNumericColumn) {
                    data[col-skippedColumnsCounter][rowCounter] = (Double) columns.get(col).getColumn().getValue(row);
                }
                else {
                    data[col-skippedColumnsCounter][rowCounter] = (Integer) columns.get(col).getColumn().getValue(row);
                }
                rowCounter++;
            }
        }
        return data;
    }

    /**
     * Identical to the getTrainingDataMatrix function, but returns a matrix containing a single index for
     * each distinct value combination
     * @return
     */
    public double[][] getTrainingDataMatrixWithDistinctVals() {
        double[][] data = new double[columns.size() - (getNumberOfDateStringAndDistinctColumns())][trainFoldDistinctValMappings.size()];
        int skippedColumnsCounter = 0;
        for (int col = 0; col < columns.size(); col++) {
            //if this is a distinct val column or if the column is a raw string
            if ( shouldColumnBeExncludedInDataMatrix(col)) {
                skippedColumnsCounter++;
                continue;
            }
            int rowCounter = 0;
            boolean isNumericColumn = columns.get(col).getColumn().getType().equals(Column.columnType.Numeric);

            for (List<String> key : trainFoldDistinctValMappings.keySet()) {
                //now we take a single representative from this group
                int index = trainFoldDistinctValMappings.get(key).get(0);
                if (isNumericColumn) {
                    data[col-skippedColumnsCounter][rowCounter] = (Double) columns.get(col).getColumn().getValue(index); }
                else {
                    data[col-skippedColumnsCounter][rowCounter] = (Integer) columns.get(col).getColumn().getValue(index);
                }
                rowCounter++;
            }
        }
        return data;
    }

    /**
     * Returns the distinct value mappings of each instance in the training set (for each instance, we get
     * a list of all the instances with the same distinct value)
     * @return
     */
    public HashMap<List<String>, List<Integer>> getTrainFoldDistinctValMappings() { return trainFoldDistinctValMappings; }

    /**
     * Returns the distinct value mappings of each instance in the test set (for each instance, we get
     * a list of all the instances with the same distinct value)
     * @return
     */
    public HashMap<List<String>, List<Integer>> getTestFoldDistinctValMappings() {
        return testFoldDistinctValMappings;
    }



    private boolean shouldColumnBeExncludedInDataMatrix(int columnIndex) {

        if (this.distinctValColumns.contains(columns.get(columnIndex)))
            return true;
        if (columns.get(columnIndex).getColumn().getType().equals(Column.columnType.String))
            return true;
        if (columns.get(columnIndex).getColumn().getType().equals(Column.columnType.Date))
            return true;
        if ((columns.get(columnIndex).getColumn().getType().equals(Column.columnType.Discrete) &&
                        ((DiscreteColumn)columns.get(columnIndex).getColumn()).getNumOfPossibleValues() > this.maxNumOFDiscreteValuesForInstancesObject))
            return true;
        return false;
    }

    /**
     * * Identical to the getTrainingDataMatrix function, but returns a matrix containing a single index for
     * each distinct value combination
     * @return
     */
    public double[][] getTestDataMatrixWithDistinctVals() {
        double[][] data = new double[columns.size() - (getNumberOfDateStringAndDistinctColumns())][testFoldDistinctValMappings.size()];
        int skippedColumnsCounter = 0;
        for (int col = 0; col < columns.size(); col++) {
            if ( shouldColumnBeExncludedInDataMatrix(col)) {
                skippedColumnsCounter++;
                continue;
            }
            int rowCounter = 0;
            boolean isNumericColumn = columns.get(col).getColumn().getType().equals(Column.columnType.Numeric);
            for (List<String> key : testFoldDistinctValMappings.keySet()) {
                //now we take a single representative from this group
                int index = testFoldDistinctValMappings.get(key).get(0);
                if (isNumericColumn) {
                    data[col-skippedColumnsCounter][rowCounter] = (Double) columns.get(col).getColumn().getValue(index); }
                else {
                    data[col-skippedColumnsCounter][rowCounter] = (Integer) columns.get(col).getColumn().getValue(index);
                }
                rowCounter++;
            }
        }
        return data;
    }

    /**
     * Partitions the training folds into a set of LOO folds. One of the training folds is designated as "test",
     * while the remaining folds are used for training. All possible combinations are returned.
     * @return
     */
    public List<Dataset> GenerateTrainingSetSubFolds() throws Exception {
        //first, get all the training folds in the current dataset

        List<Fold> trainingFolds = new ArrayList<>();
        for (Fold fold: folds) {
            if (!fold.isTestFold()) {
                trainingFolds.add(fold);
            }
        }
        List<Dataset> trainingDatasets = new ArrayList<>();
        for (int i=0; i<trainingFolds.size(); i++) {
            List<Fold> newFoldsList = new ArrayList<>();
            for (int j=0; j<trainingFolds.size(); j++) {
                Fold currentFold = trainingFolds.get(j);
                //if i==j, then this is the test fold
                Fold newFold = new Fold(getNumOfClasses(),(i==j));
                newFold.setIndices(currentFold.getIndices());
                newFold.setNumOfInstancesInFold(currentFold.getNumOfInstancesInFold());
                newFold.setInstancesClassDistribution(currentFold.getInstancesClassDistribution());
                newFold.setIndicesPerClass(currentFold.getIndicesPerClass());
                newFold.setDistinctValMappings(currentFold.getDistinctValMappings());
                newFoldsList.add(newFold);
            }
            //now that we have the folds, we can generate the Dataset object
            Dataset subDataset = new Dataset(this.columns, newFoldsList,this.targetColumnIndex, this.name, this.numOfInstancesPerColumn, this.distinctValColumns, this.randomSeed, this.maxNumOFDiscreteValuesForInstancesObject);
            trainingDatasets.add(subDataset);
        }

        return trainingDatasets;
    }

    /**
     * Determines whether the values of the a column adhere to the distinct value requirements.
     * These columns will be used fot the initial candiate features generation
     * @param ci
     * @return
     */
    private boolean isColumnDistinctValuesCompatibe(ColumnInfo ci) throws Exception {
        if (ci.isTargetClass() || distinctValColumns.contains(ci) ||
                ci.getColumn().getType().equals(Column.columnType.Date)  || ci.getColumn().getType().equals(Column.columnType.String)) {
            return false;
        }

        try {
            HashMap<Object, Object> distinctValsDict = new HashMap<>();
            HashMap<Object, Object> valuesMap = new HashMap<>();
            for (int i = 0; i < indices.size(); i++) {
                int j = indices.get(i);
                List<Object> sourceValues = distinctValColumns.stream().map(c -> c.getColumn().getValue(j)).collect(Collectors.toList());
                if (!distinctValsDict.containsKey(sourceValues)) {
                    distinctValsDict.put(sourceValues, ci.getColumn().getValue(j));
                } else {
                    if (!distinctValsDict.get(sourceValues).equals(ci.getColumn().getValue(j))) {
                        return false;
                    }
                }
            }
        }
        catch (Exception ex) {
            throw new Exception("Error in isColumnDistinctValuesCompatibe");
        }
        return true;
    }

    /**
     * Generates a new Dataset object which points to a subset of the columns in the original dataset. The
     * target class attribute is always added (if only the target class is returned then this function becomes
     * the equivalent of emptyReplica()
     * @param indices
     * @return
     */
    public Dataset replicateDatasetByColumnIndices(List<Integer> indices) {
        Dataset dataset = new Dataset();

        //We need to create a new columns object and just reference the same objects
        dataset.columns = new ArrayList<>();
        for (int index : indices) {
            dataset.columns.add(columns.get(index));
        }
        if (!dataset.columns.contains(getTargetClassColumn())) {
            dataset.addColumn(getTargetClassColumn());
        }
        dataset.targetColumnIndex = dataset.columns.size()-1;

        dataset.numOfInstancesPerColumn = this.numOfInstancesPerColumn;
        dataset.indices = this.indices;
        dataset.folds = this.folds;
        dataset.indicesOfTrainingFolds = this.indicesOfTrainingFolds;
        dataset.indicesOfTestFolds = this.indicesOfTestFolds;
        dataset.numOfTrainingInstancesPerClass = this.numOfTrainingInstancesPerClass;
        dataset.numOfTestInstancesPerClass = this.numOfTestInstancesPerClass;
        dataset.numOfTrainingRows = this.numOfTrainingRows;
        dataset.numOfTestRows = this.numOfTestRows;
        dataset.name = this.name;
        dataset.distinctValColumns = this.distinctValColumns;
        dataset.trainingIndicesByClass = this.trainingIndicesByClass;
        dataset.trainFoldDistinctValMappings = this.trainFoldDistinctValMappings;
        dataset.testFoldDistinctValMappings = this.testFoldDistinctValMappings;
        dataset.trainFoldsDistinctValRepresentatives = this.trainFoldsDistinctValRepresentatives;
        dataset.testFoldsDistinctValRepresentatives = this.testFoldsDistinctValRepresentatives;
        dataset.distinctValueCompliantColumns = this.distinctValueCompliantColumns;
        dataset.maxNumOFDiscreteValuesForInstancesObject = this.maxNumOFDiscreteValuesForInstancesObject;

        return dataset;
    }

    /**
     * Creates an exact replica of the dataset, except for the fact that it creates a new List of columns
     * instead of referencing to the existing list. This enables the addition of columns to this object without
     * adding them to the original.
     * @return
     */
    public Dataset replicateDataset() {
        Dataset dataset = new Dataset();

        //We need to create a new columns object and just reference the same objects
        dataset.columns = new ArrayList<>();
        for (ColumnInfo ci: this.columns) {
            dataset.columns.add(ci);
        }

        dataset.numOfInstancesPerColumn = this.numOfInstancesPerColumn;
        dataset.indices = this.indices;
        dataset.folds = this.folds;
        dataset.indicesOfTrainingFolds = this.indicesOfTrainingFolds;
        dataset.indicesOfTestFolds = this.indicesOfTestFolds;
        dataset.numOfTrainingInstancesPerClass = this.numOfTrainingInstancesPerClass;
        dataset.numOfTestInstancesPerClass = this.numOfTestInstancesPerClass;
        dataset.numOfTrainingRows = this.numOfTrainingRows;
        dataset.numOfTestRows = this.numOfTestRows;
        dataset.targetColumnIndex = this.targetColumnIndex;
        dataset.name = this.name;
        dataset.distinctValColumns = this.distinctValColumns;
        dataset.trainingIndicesByClass = this.trainingIndicesByClass;
        dataset.trainFoldDistinctValMappings = this.trainFoldDistinctValMappings;
        dataset.testFoldDistinctValMappings = this.testFoldDistinctValMappings;
        dataset.trainFoldsDistinctValRepresentatives = this.trainFoldsDistinctValRepresentatives;
        dataset.testFoldsDistinctValRepresentatives = this.testFoldsDistinctValRepresentatives;
        dataset.distinctValueCompliantColumns = this.distinctValueCompliantColumns;
        dataset.maxNumOFDiscreteValuesForInstancesObject = this.maxNumOFDiscreteValuesForInstancesObject;

        return dataset;
    }

    public Dataset generateRandomSubDataSet(int numOfInstancesPerFold, int randomSeed) throws Exception {

        List<Fold> newFoldsList = new ArrayList<>();
        //operatorAssignments.parallelStream().forEach(oa -> {
        folds.parallelStream().forEach(fold -> {
        //for (Fold fold: this.folds) {
            Fold subFold = fold.generateSubFold(numOfInstancesPerFold, randomSeed);
            newFoldsList.add(subFold);
        });

        Dataset dataset = new Dataset(this.columns,newFoldsList,this.targetColumnIndex,this.name+"_subfold",this.numOfInstancesPerColumn,this.distinctValColumns,randomSeed,this.maxNumOFDiscreteValuesForInstancesObject);
        return dataset;
    }


    /**
     * Creates a replica of the given Dataset object, but without any columns except for the target column and the
     * distinct value columns (if they exist)
     * @return
     */
    public Dataset emptyReplica() {
        Dataset dataset = new Dataset();

        //We need to create a new columns object and just reference the same objects
        dataset.columns = new ArrayList<>();
        dataset.numOfInstancesPerColumn = this.numOfInstancesPerColumn;
        dataset.indices = this.indices;
        dataset.indicesOfTrainingFolds = this.indicesOfTrainingFolds;
        dataset.indicesOfTestFolds = this.indicesOfTestFolds;
        dataset.numOfTrainingInstancesPerClass = this.numOfTrainingInstancesPerClass;
        dataset.numOfTestInstancesPerClass = this.numOfTestInstancesPerClass;
        dataset.numOfTrainingRows = this.numOfTrainingRows;
        dataset.numOfTestRows = this.numOfTestRows;
        dataset.name = this.name;
        dataset.distinctValColumns = this.distinctValColumns;
        dataset.trainingIndicesByClass = this.trainingIndicesByClass;
        dataset.trainFoldDistinctValMappings = this.trainFoldDistinctValMappings;
        dataset.testFoldDistinctValMappings = this.testFoldDistinctValMappings;
        dataset.trainFoldsDistinctValRepresentatives = this.trainFoldsDistinctValRepresentatives;
        dataset.testFoldsDistinctValRepresentatives = this.testFoldsDistinctValRepresentatives;
        dataset.distinctValueCompliantColumns = this.distinctValueCompliantColumns;
        dataset.maxNumOFDiscreteValuesForInstancesObject = this.maxNumOFDiscreteValuesForInstancesObject;

        //since we only add the target column, it's index in 0
        dataset.targetColumnIndex = 0;
        dataset.columns.add(this.columns.get(this.targetColumnIndex));

        //add the distinct value columns to the empty dataset
        for (ColumnInfo ci: distinctValColumns) {
            dataset.columns.add(ci);
        }

        return dataset;
    }

    /**
     * Returns a single indice for each distinct values combination in the training folds
     * @return
     */
    public List<Integer> getTrainFoldsDistinctValRepresentatives() {
        return trainFoldsDistinctValRepresentatives;
    }

    /**
     * Returns a single indice for each distinct values combination in the test folds
     * @return
     */
    public List<Integer> getTestFoldsDistinctValRepresentatives() {
        return testFoldsDistinctValRepresentatives;
    }

    /**
     * Returns the list of columns whose values satisfy the constraint of the distinct value
     * @return
     */
    public List<ColumnInfo> getDistinctValueCompliantColumns() {
        return distinctValueCompliantColumns;
    }
}
