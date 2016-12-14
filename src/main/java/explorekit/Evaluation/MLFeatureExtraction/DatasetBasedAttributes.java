package explorekit.Evaluation.MLFeatureExtraction;

import explorekit.Evaluation.ClassificationResults;
import explorekit.Evaluation.FilterEvaluators.InformationGainFilterEvaluator;
import explorekit.Evaluation.WrapperEvaluation.WrapperEvaluator;
import explorekit.data.*;
import explorekit.operators.UnaryOperators.EqualRangeDiscretizerUnaryOperator;
import explorekit.search.FilterWrapperHeuristicSearch;
import explorekit.search.Search;
import org.apache.commons.math3.stat.inference.ChiSquareTest;
import org.apache.commons.math3.stat.inference.TTest;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.Attribute;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.InputStream;
import java.util.*;

import static com.sun.org.apache.xalan.internal.xsltc.compiler.util.Type.Attribute;

/**
 * Created by giladkatz on 26/04/2016.
 */
public class DatasetBasedAttributes {

    //Basic information on the dataset
    private double numOfInstances;
    private double numOfClasses;
    private double numOfFeatures;
    private double numOfNumericAtributes;
    private double numOfDiscreteAttributes;
    private double ratioOfNumericAttributes;
    private double ratioOfDiscreteAttributes;

    //discrete features-specific attributes (must not include the target class)
    private double maxNumberOfDiscreteValuesPerAttribute;
    private double minNumberOfDiscreteValuesPerAttribtue;
    private double avgNumOfDiscreteValuesPerAttribute;
    private double stdevNumOfDiscreteValuesPerAttribute;

    //Statistics on the initial performance of the dataset
    private double numOfFoldsInEvaluation;
    private double maxAUC;
    private double minAUC;
    private double avgAUC;
    private double stdevAUC;

    private double maxLogLoss;
    private double minLogLoss;
    private double avgLogLoss;
    private double stdevLogLoss;

    private TreeMap<Double,Double> maxPrecisionAtFixedRecallValues = new TreeMap<>();
    private TreeMap<Double,Double> minPrecisionAtFixedRecallValues = new TreeMap<>();
    private TreeMap<Double,Double> avgPrecisionAtFixedRecallValues = new TreeMap<>();
    private TreeMap<Double,Double> stdevPrecisionAtFixedRecallValues = new TreeMap<>();


    //Statistics on the initial attributes' entropy with regards to the target class and their interactions
    private double maxIGVal;
    private double minIGVal;
    private double avgIGVal;
    private double stdevIGVal;

    private double discreteAttsMaxIGVal;
    private double discreteAttsMinIGVal;
    private double discreteAttsAvgIGVal;
    private double discreteAttsStdevIGVal;

    private double numericAttsMaxIGVal;
    private double numericAttsMinIGVal;
    private double numericAttsAvgIGVal;
    private double numericAttsStdevIGVal;

    //Statistics on the correlation of the dataset's features
    private double maxPairedTTestValueForNumericAttributes;
    private double minPairedTTestValueForNumericAttributes;
    private double avgPairedTTestValueForNumericAttributes;
    private double stdevPairedTTestValueForNumericAttributes;

    private double maxChiSquareValueforDiscreteAttributes;
    private double minChiSquareValueforDiscreteAttributes;
    private double avgChiSquareValueforDiscreteAttributes;
    private double stdevChiSquareValueforDiscreteAttributes;

    private double maxChiSquareValueforDiscreteAndDiscretizedAttributes;
    private double minChiSquareValueforDiscreteAndDiscretizedAttributes;
    private double avgChiSquareValueforDiscreteAndDiscretizedAttributes;
    private double stdevChiSquareValueforDiscreteAndDiscretizedAttributes;

    //support parameters - not to be included in the output of the class
    List<ColumnInfo> discreteAttributesList = new ArrayList<>();
    List<ColumnInfo> numericAttributesList = new ArrayList<>();
    Properties properties;

    public HashMap<Integer,AttributeInfo> getDatasetBasedFeatures(Dataset dataset, String classifier, Properties properties) throws Exception {
        try {
            this.properties = properties;

            processGeneralDatasetInfo(dataset);

            processInitialEvaluationInformation(dataset, classifier);

            processEntropyBasedMeasures(dataset, properties);

            processAttributesStatisticalTests(dataset);

            HashMap<Integer,AttributeInfo> attributes = generateDatasetAttributesMap();

            return attributes;
        }
        catch (Exception ex) {
            int x=5;
        }
        return null;
    }

    /**
     * Returns a HashMap with all the attributes. For each attribute, in addition to the value we also record the
     * type and name of each attribute.
     * @return
     */
    public HashMap<Integer,AttributeInfo> generateDatasetAttributesMap() {
        HashMap<Integer, AttributeInfo> attributes = new HashMap<>();

        AttributeInfo att1 = new AttributeInfo("numOfInstances", Column.columnType.Numeric, numOfInstances,-1);
        AttributeInfo att2 = new AttributeInfo("numOfClasses", Column.columnType.Numeric, numOfClasses,-1);
        AttributeInfo att3 = new AttributeInfo("numOfFeatures", Column.columnType.Numeric, numOfFeatures,-1);
        AttributeInfo att4 = new AttributeInfo("numOfNumericAtributes", Column.columnType.Numeric, numOfNumericAtributes,-1);
        AttributeInfo att5 = new AttributeInfo("numOfDiscreteAttributes", Column.columnType.Numeric, numOfDiscreteAttributes,-1);
        AttributeInfo att6 = new AttributeInfo("ratioOfNumericAttributes", Column.columnType.Numeric, ratioOfNumericAttributes,-1);
        AttributeInfo att7 = new AttributeInfo("ratioOfDiscreteAttributes", Column.columnType.Numeric, ratioOfDiscreteAttributes,-1);
        AttributeInfo att8 = new AttributeInfo("maxNumberOfDiscreteValuesPerAttribute", Column.columnType.Numeric, maxNumberOfDiscreteValuesPerAttribute,-1);
        AttributeInfo att9 = new AttributeInfo("minNumberOfDiscreteValuesPerAttribtue", Column.columnType.Numeric, minNumberOfDiscreteValuesPerAttribtue,-1);
        AttributeInfo att10 = new AttributeInfo("avgNumOfDiscreteValuesPerAttribute", Column.columnType.Numeric, avgNumOfDiscreteValuesPerAttribute,-1);
        AttributeInfo att11 = new AttributeInfo("stdevNumOfDiscreteValuesPerAttribute", Column.columnType.Numeric, stdevNumOfDiscreteValuesPerAttribute,-1);
        AttributeInfo att12 = new AttributeInfo("numOfFoldsInEvaluation", Column.columnType.Numeric, numOfFoldsInEvaluation,-1);
        AttributeInfo att13 = new AttributeInfo("maxAUC", Column.columnType.Numeric, maxAUC,-1);
        AttributeInfo att14 = new AttributeInfo("minAUC", Column.columnType.Numeric, minAUC,-1);
        AttributeInfo att15 = new AttributeInfo("avgAUC", Column.columnType.Numeric, avgAUC,-1);
        AttributeInfo att16 = new AttributeInfo("stdevAUC", Column.columnType.Numeric, stdevAUC,-1);
        AttributeInfo att17 = new AttributeInfo("maxLogLoss", Column.columnType.Numeric, maxLogLoss,-1);
        AttributeInfo att18 = new AttributeInfo("minLogLoss", Column.columnType.Numeric, minLogLoss,-1);
        AttributeInfo att19 = new AttributeInfo("avgLogLoss", Column.columnType.Numeric, avgLogLoss,-1);
        AttributeInfo att20 = new AttributeInfo("stdevLogLoss", Column.columnType.Numeric, stdevLogLoss,-1);
        AttributeInfo att21 = new AttributeInfo("maxIGVal", Column.columnType.Numeric, maxIGVal,-1);
        AttributeInfo att22 = new AttributeInfo("minIGVal", Column.columnType.Numeric, minIGVal,-1);
        AttributeInfo att23 = new AttributeInfo("avgIGVal", Column.columnType.Numeric, avgIGVal,-1);
        AttributeInfo att24 = new AttributeInfo("stdevIGVal", Column.columnType.Numeric, stdevIGVal,-1);
        AttributeInfo att25 = new AttributeInfo("discreteAttsMaxIGVal", Column.columnType.Numeric, discreteAttsMaxIGVal,-1);
        AttributeInfo att26 = new AttributeInfo("discreteAttsMinIGVal", Column.columnType.Numeric, discreteAttsMinIGVal,-1);
        AttributeInfo att27 = new AttributeInfo("discreteAttsAvgIGVal", Column.columnType.Numeric, discreteAttsAvgIGVal,-1);
        AttributeInfo att28 = new AttributeInfo("discreteAttsStdevIGVal", Column.columnType.Numeric, discreteAttsStdevIGVal,-1);
        AttributeInfo att29 = new AttributeInfo("numericAttsMaxIGVal", Column.columnType.Numeric, numericAttsMaxIGVal,-1);
        AttributeInfo att30 = new AttributeInfo("numericAttsMinIGVal", Column.columnType.Numeric, numericAttsMinIGVal,-1);
        AttributeInfo att31 = new AttributeInfo("numericAttsAvgIGVal", Column.columnType.Numeric, numericAttsAvgIGVal,-1);
        AttributeInfo att32 = new AttributeInfo("numericAttsStdevIGVal", Column.columnType.Numeric, numericAttsStdevIGVal,-1);
        AttributeInfo att33 = new AttributeInfo("maxPairedTTestValueForNumericAttributes", Column.columnType.Numeric, maxPairedTTestValueForNumericAttributes,-1);
        AttributeInfo att34 = new AttributeInfo("minPairedTTestValueForNumericAttributes", Column.columnType.Numeric, minPairedTTestValueForNumericAttributes,-1);
        AttributeInfo att35 = new AttributeInfo("avgPairedTTestValueForNumericAttributes", Column.columnType.Numeric, avgPairedTTestValueForNumericAttributes,-1);
        AttributeInfo att36 = new AttributeInfo("stdevPairedTTestValueForNumericAttributes", Column.columnType.Numeric, stdevPairedTTestValueForNumericAttributes,-1);
        AttributeInfo att37 = new AttributeInfo("maxChiSquareValueforDiscreteAttributes", Column.columnType.Numeric, maxChiSquareValueforDiscreteAttributes,-1);
        AttributeInfo att38 = new AttributeInfo("minChiSquareValueforDiscreteAttributes", Column.columnType.Numeric, minChiSquareValueforDiscreteAttributes,-1);
        AttributeInfo att39 = new AttributeInfo("avgChiSquareValueforDiscreteAttributes", Column.columnType.Numeric, avgChiSquareValueforDiscreteAttributes,-1);
        AttributeInfo att40 = new AttributeInfo("stdevChiSquareValueforDiscreteAttributes", Column.columnType.Numeric, stdevChiSquareValueforDiscreteAttributes,-1);
        AttributeInfo att41 = new AttributeInfo("maxChiSquareValueforDiscreteAndDiscretizedAttributes", Column.columnType.Numeric, maxChiSquareValueforDiscreteAndDiscretizedAttributes,-1);
        AttributeInfo att42 = new AttributeInfo("minChiSquareValueforDiscreteAndDiscretizedAttributes", Column.columnType.Numeric, minChiSquareValueforDiscreteAndDiscretizedAttributes,-1);
        AttributeInfo att43 = new AttributeInfo("avgChiSquareValueforDiscreteAndDiscretizedAttributes", Column.columnType.Numeric, avgChiSquareValueforDiscreteAndDiscretizedAttributes,-1);
        AttributeInfo att44 = new AttributeInfo("stdevChiSquareValueforDiscreteAndDiscretizedAttributes", Column.columnType.Numeric, stdevChiSquareValueforDiscreteAndDiscretizedAttributes,-1);

        attributes.put(attributes.size(), att1);
        attributes.put(attributes.size(), att2);
        attributes.put(attributes.size(), att3);
        attributes.put(attributes.size(), att4);
        attributes.put(attributes.size(), att5);
        attributes.put(attributes.size(), att6);
        attributes.put(attributes.size(), att7);
        attributes.put(attributes.size(), att8);
        attributes.put(attributes.size(), att9);
        attributes.put(attributes.size(), att10);
        attributes.put(attributes.size(), att11);
        attributes.put(attributes.size(), att12);
        attributes.put(attributes.size(), att13);
        attributes.put(attributes.size(), att14);
        attributes.put(attributes.size(), att15);
        attributes.put(attributes.size(), att16);
        attributes.put(attributes.size(), att17);
        attributes.put(attributes.size(), att18);
        attributes.put(attributes.size(), att19);
        attributes.put(attributes.size(), att20);
        attributes.put(attributes.size(), att21);
        attributes.put(attributes.size(), att22);
        attributes.put(attributes.size(), att23);
        attributes.put(attributes.size(), att24);
        attributes.put(attributes.size(), att25);
        attributes.put(attributes.size(), att26);
        attributes.put(attributes.size(), att27);
        attributes.put(attributes.size(), att28);
        attributes.put(attributes.size(), att29);
        attributes.put(attributes.size(), att30);
        attributes.put(attributes.size(), att31);
        attributes.put(attributes.size(), att32);
        attributes.put(attributes.size(), att33);
        attributes.put(attributes.size(), att34);
        attributes.put(attributes.size(), att35);
        attributes.put(attributes.size(), att36);
        attributes.put(attributes.size(), att37);
        attributes.put(attributes.size(), att38);
        attributes.put(attributes.size(), att39);
        attributes.put(attributes.size(), att40);
        attributes.put(attributes.size(), att41);
        attributes.put(attributes.size(), att42);
        attributes.put(attributes.size(), att43);
        attributes.put(attributes.size(), att44);

        //now we need to process the multiple values of the precision/recall analysis.
        for (double key : maxPrecisionAtFixedRecallValues.keySet()) {
            AttributeInfo maxPrecisionAtt = new AttributeInfo("maxPrecisionAtFixedRecallValues_" + key, Column.columnType.Numeric, maxPrecisionAtFixedRecallValues.get(key),-1);
            AttributeInfo minPrecisionAtt = new AttributeInfo("minPrecisionAtFixedRecallValues_" + key, Column.columnType.Numeric, minPrecisionAtFixedRecallValues.get(key),-1);
            AttributeInfo avgPrecisionAtt = new AttributeInfo("avgPrecisionAtFixedRecallValues_" + key, Column.columnType.Numeric, avgPrecisionAtFixedRecallValues.get(key),-1);
            AttributeInfo stdevPrecisionAtt = new AttributeInfo("stdevPrecisionAtFixedRecallValues_" + key, Column.columnType.Numeric, stdevPrecisionAtFixedRecallValues.get(key),-1);
            attributes.put(attributes.size(), maxPrecisionAtt);
            attributes.put(attributes.size(), minPrecisionAtt);
            attributes.put(attributes.size(), avgPrecisionAtt);
            attributes.put(attributes.size(), stdevPrecisionAtt);
        }

        return attributes;
    }

    /**
     * Used to calcualte the dependency of the different attributes in the dataset. For the numeric attributes we conduct a paired T-Test
     * between every pair. For the discrete attributes we conduct a Chi-Square test. Finally, we discretize the numeric attributes and
     * conduct an additional Chi-Suqare test on all attributes.
     * @param dataset
     * @throws Exception
     */
    public void processAttributesStatisticalTests(Dataset dataset) throws Exception {
        //We start by calculating the Paired T-Test on every pair of the numeric attributes
        TTest tTest = new TTest();
        List<Double> pairedTTestValuesList = new ArrayList<>();
        for (int i=0; i<numericAttributesList.size()-1; i++) {
            for (int j=i+1; j<numericAttributesList.size(); j++) {
                if (i != j) {
                    double tTestVal = Math.abs(tTest.pairedT((double[]) numericAttributesList.get(i).getColumn().getValues(), (double[]) numericAttributesList.get(j).getColumn().getValues()));
                    if (!Double.isNaN(tTestVal) && !Double.isInfinite(tTestVal)) {
                        pairedTTestValuesList.add(tTestVal);
                    }
                }
            }
        }

        if (pairedTTestValuesList.size() > 0) {
            this.maxPairedTTestValueForNumericAttributes = pairedTTestValuesList.stream().mapToDouble(a -> a).max().getAsDouble();
            this.minPairedTTestValueForNumericAttributes = pairedTTestValuesList.stream().mapToDouble(a -> a).min().getAsDouble();
            this.avgPairedTTestValueForNumericAttributes = pairedTTestValuesList.stream().mapToDouble(a -> a).average().getAsDouble();
            double tempStdev = pairedTTestValuesList.stream().mapToDouble(a -> Math.pow(a - this.avgPairedTTestValueForNumericAttributes, 2)).sum();
            this.stdevPairedTTestValueForNumericAttributes = Math.sqrt(tempStdev / pairedTTestValuesList.size());
        }
        else {
            this.maxPairedTTestValueForNumericAttributes = 0;
            this.minPairedTTestValueForNumericAttributes = 0;
            this.avgPairedTTestValueForNumericAttributes = 0;
            this.stdevPairedTTestValueForNumericAttributes = 0;
        }

        //Next we calculate the Chi-Square TEST OF INDEPENDENCE for the discrete attributes
        ChiSquareTest chiSquareTest = new ChiSquareTest();
        List<Double> chiSquaredTestValuesList = new ArrayList<>();
        for (int i=0; i<discreteAttributesList.size()-1; i++) {
            for (int j = i+1; j < discreteAttributesList.size(); j++) {
                if (i!=j) {
                    long[][] counts = generateDiscreteAttributesCategoryIntersection((DiscreteColumn) discreteAttributesList.get(i).getColumn(), (DiscreteColumn) discreteAttributesList.get(j).getColumn());
                    double testVal = chiSquareTest.chiSquare(counts);
                    if (!Double.isNaN(testVal) && !Double.isInfinite(testVal)) {
                        chiSquaredTestValuesList.add(testVal);
                    }
                }
            }
        }

        if (chiSquaredTestValuesList.size() > 0) {
            this.maxChiSquareValueforDiscreteAttributes = chiSquaredTestValuesList.stream().mapToDouble(a -> a).max().getAsDouble();
            this.minChiSquareValueforDiscreteAttributes = chiSquaredTestValuesList.stream().mapToDouble(a -> a).min().getAsDouble();
            this.avgChiSquareValueforDiscreteAttributes = chiSquaredTestValuesList.stream().mapToDouble(a -> a).average().getAsDouble();
            double tempStdev = chiSquaredTestValuesList.stream().mapToDouble(a -> Math.pow(a - this.avgChiSquareValueforDiscreteAttributes, 2)).sum();
            this.stdevChiSquareValueforDiscreteAttributes = Math.sqrt(tempStdev / chiSquaredTestValuesList.size());
        }
        else {
            this.maxChiSquareValueforDiscreteAttributes = 0;
            this.minChiSquareValueforDiscreteAttributes = 0;
            this.avgChiSquareValueforDiscreteAttributes = 0;
            this.stdevChiSquareValueforDiscreteAttributes = 0;
        }

        //finally, we discretize the numberic features and conduct an additional Chi-Square evaluation
        double[] bins = new double[Integer.parseInt(properties.getProperty("equalRangeDiscretizerBinsNumber"))];
        EqualRangeDiscretizerUnaryOperator erduo;
        List<ColumnInfo> discretizedColumns = new ArrayList<>();
        for (ColumnInfo ci : numericAttributesList) {
            erduo = new EqualRangeDiscretizerUnaryOperator(bins);
            List<ColumnInfo> tempColumnsList = new ArrayList<>();
            tempColumnsList.add(ci);
            erduo.processTrainingSet(dataset,tempColumnsList,null);
            ColumnInfo discretizedAttribute = erduo.generate(dataset,tempColumnsList,null,false);
            discretizedColumns.add(discretizedAttribute);
        }
        //now we add all the original discrete attributes to this list and run the Chi-Square test again
        discretizedColumns.addAll(discreteAttributesList);
        chiSquaredTestValuesList = new ArrayList<>();
        for (int i=0; i<discretizedColumns.size()-1; i++) {
            for (int j = i+1; j < discretizedColumns.size(); j++) {
                if (i!=j) {
                    long[][] counts = generateDiscreteAttributesCategoryIntersection((DiscreteColumn) discretizedColumns.get(i).getColumn(), (DiscreteColumn) discretizedColumns.get(j).getColumn());
                    double testVal = chiSquareTest.chiSquare(counts);
                    if (!Double.isNaN(testVal) &&  !Double.isInfinite(testVal)) {
                        chiSquaredTestValuesList.add(testVal);
                    }
                }
            }
        }

        if (chiSquaredTestValuesList.size() > 0) {
            this.maxChiSquareValueforDiscreteAndDiscretizedAttributes = chiSquaredTestValuesList.stream().mapToDouble(a -> a).max().getAsDouble();
            this.minChiSquareValueforDiscreteAndDiscretizedAttributes = chiSquaredTestValuesList.stream().mapToDouble(a -> a).min().getAsDouble();
            this.avgChiSquareValueforDiscreteAndDiscretizedAttributes = chiSquaredTestValuesList.stream().mapToDouble(a -> a).average().getAsDouble();
            double tempStdev = chiSquaredTestValuesList.stream().mapToDouble(a -> Math.pow(a - this.avgChiSquareValueforDiscreteAndDiscretizedAttributes, 2)).sum();
            this.stdevChiSquareValueforDiscreteAndDiscretizedAttributes = Math.sqrt(tempStdev / chiSquaredTestValuesList.size());
        }
        else {
            this.maxChiSquareValueforDiscreteAndDiscretizedAttributes = 0;
            this.minChiSquareValueforDiscreteAndDiscretizedAttributes = 0;
            this.avgChiSquareValueforDiscreteAndDiscretizedAttributes = 0;
            this.stdevChiSquareValueforDiscreteAndDiscretizedAttributes = 0;
        }

    }

    private long[][] generateDiscreteAttributesCategoryIntersection(DiscreteColumn col1, DiscreteColumn col2) throws Exception {
        long[][] intersectionsMatrix = new long[col1.getNumOfPossibleValues()][col2.getNumOfPossibleValues()];
        int[] col1Values = (int[])col1.getValues();
        int[] col2Values = (int[])col2.getValues();

        if (col1Values.length != col2Values.length) {
            throw new Exception("Columns do not have the same number of instances");
        }

        for (int i=0; i<col1Values.length; i++) {
            intersectionsMatrix[col1Values[i]][col2Values[i]]++;
        }

        return intersectionsMatrix;
    }

    private void processEntropyBasedMeasures(Dataset dataset, Properties properties) throws Exception {

        List<Double> IGScoresPerColumnIndex = new ArrayList<>();
        List<Double> IGScoresPerDiscreteColumnIndex = new ArrayList<>();
        List<Double> IGScoresPerNumericColumnIndex = new ArrayList<>();

        //start by getting the IG scores of all the attributes
        InformationGainFilterEvaluator ige = new InformationGainFilterEvaluator();
        for (int i=0; i<dataset.getAllColumns(false).size(); i++) {
            ColumnInfo ci = dataset.getAllColumns(false).get(i);
            if (dataset.getTargetClassColumn() == ci) {
                continue;
            }

            //if the attribute is string or date, not much we can do about that
            if (ci.getColumn().getType() != Column.columnType.Discrete && ci.getColumn().getType() != Column.columnType.Numeric) {
                continue;
            }
            List<Integer> indicedList = new ArrayList<>();
            indicedList.add(i);
            Dataset replicatedDataset = dataset.replicateDatasetByColumnIndices(indicedList);
            List<ColumnInfo> tempList = new ArrayList<>();
            tempList.add(ci);
            ige.initFilterEvaluator(tempList);
            double score = ige.produceScore(replicatedDataset, null, dataset, null, null, properties);
            IGScoresPerColumnIndex.add(score);

            if (ci.getColumn().getType() == Column.columnType.Discrete) {
                IGScoresPerDiscreteColumnIndex.add(score);
            }
            else {
                IGScoresPerNumericColumnIndex.add(score);
            }
        }

        this.maxIGVal = IGScoresPerColumnIndex.stream().mapToDouble(a -> a).max().getAsDouble();
        this.minIGVal = IGScoresPerColumnIndex.stream().mapToDouble(a -> a).min().getAsDouble();
        this.avgIGVal = IGScoresPerColumnIndex.stream().mapToDouble(a -> a).average().getAsDouble();
        double tempStdev = IGScoresPerColumnIndex.stream().mapToDouble(a -> Math.pow(a - this.avgIGVal, 2)).sum();
        this.stdevIGVal = Math.sqrt(tempStdev / IGScoresPerColumnIndex.size());

        if (IGScoresPerDiscreteColumnIndex.size() > 0) {
            this.discreteAttsMaxIGVal = IGScoresPerDiscreteColumnIndex.stream().mapToDouble(a -> a).max().getAsDouble();
            this.discreteAttsMinIGVal = IGScoresPerDiscreteColumnIndex.stream().mapToDouble(a -> a).min().getAsDouble();
            this.discreteAttsAvgIGVal = IGScoresPerDiscreteColumnIndex.stream().mapToDouble(a -> a).average().getAsDouble();
            tempStdev = IGScoresPerDiscreteColumnIndex.stream().mapToDouble(a -> Math.pow(a - this.avgIGVal, 2)).sum();
            this.discreteAttsStdevIGVal = Math.sqrt(tempStdev / IGScoresPerDiscreteColumnIndex.size());
        }
        else {
            this.discreteAttsMaxIGVal = 0;
            this.discreteAttsMinIGVal = 0;
            this.discreteAttsAvgIGVal = 0;
            this.discreteAttsStdevIGVal = 0;
        }

        if (IGScoresPerNumericColumnIndex.size() > 0) {
            this.numericAttsMaxIGVal = IGScoresPerNumericColumnIndex.stream().mapToDouble(a -> a).max().getAsDouble();
            this.numericAttsMinIGVal = IGScoresPerNumericColumnIndex.stream().mapToDouble(a -> a).min().getAsDouble();
            this.numericAttsAvgIGVal = IGScoresPerNumericColumnIndex.stream().mapToDouble(a -> a).average().getAsDouble();
            tempStdev = IGScoresPerNumericColumnIndex.stream().mapToDouble(a -> Math.pow(a - this.avgIGVal, 2)).sum();
            this.numericAttsStdevIGVal = Math.sqrt(tempStdev / IGScoresPerNumericColumnIndex.size());
        }
        else {
            this.numericAttsMaxIGVal = 0;
            this.numericAttsMinIGVal = 0;
            this.numericAttsAvgIGVal = 0;
            this.numericAttsStdevIGVal = 0;
        }
    }

    /**
     * Used to obtain information about the performance of the classifier on the initial dataset. For training
     * datasets the entire dataset needs to be provided. For test datasets - only the training folds.
     * @param dataset
     * @param classifier
     * @throws Exception
     */
    private void processInitialEvaluationInformation(Dataset dataset, String classifier) throws Exception {
        //We now need to test all folds combinations (the original train/test allocation is disregarded, which is
        //not a problem for the offline training. The test set dataset MUST submit a new dataset object containing
        //only the training folds
        for (Fold fold : dataset.getFolds()) {
            fold.setIsTestFold(false);
        }
        FilterWrapperHeuristicSearch fwhs = new FilterWrapperHeuristicSearch(10);
        WrapperEvaluator wrapperEvaluator = fwhs.getWrapper("AucWrapperEvaluator");
        List<Dataset> leaveOneFoldOutDatasets = dataset.GenerateTrainingSetSubFolds();
        List<ClassificationResults> classificationResults = wrapperEvaluator.produceClassificationResults(leaveOneFoldOutDatasets, properties);

        List<Double> aucVals = new ArrayList<>();
        List<Double> logLossVals = new ArrayList<>();
        List<TreeMap<Double,Double>> recallPrecisionValues = new ArrayList<>();
        for (ClassificationResults classificationResult : classificationResults) {
            aucVals.add(classificationResult.getAuc());
            logLossVals.add(classificationResult.getLogLoss());
            recallPrecisionValues.add(classificationResult.getRecallPrecisionValues());
        }

        this.numOfFoldsInEvaluation = dataset.getFolds().size();

        this.maxAUC = aucVals.stream().mapToDouble(a -> a).max().getAsDouble();
        this.minAUC = aucVals.stream().mapToDouble(a -> a).min().getAsDouble();
        this.avgAUC = aucVals.stream().mapToDouble(a -> a).average().getAsDouble();
        double tempStdev = aucVals.stream().mapToDouble(a -> Math.pow(a - this.avgAUC, 2)).sum();
        this.stdevAUC = Math.sqrt(tempStdev / aucVals.size());

        this.maxLogLoss = logLossVals.stream().mapToDouble(a -> a).max().getAsDouble();
        this.minLogLoss = logLossVals.stream().mapToDouble(a -> a).min().getAsDouble();
        this.avgLogLoss = logLossVals.stream().mapToDouble(a -> a).average().getAsDouble();
        tempStdev = logLossVals.stream().mapToDouble(a -> Math.pow(a - this.avgLogLoss, 2)).sum();
        this.stdevLogLoss = Math.sqrt(tempStdev / logLossVals.size());

        for (double recallVal : recallPrecisionValues.get(0).keySet()) {
            double max = -1;
            double min = 2;
            List<Double> valuesList = new ArrayList<>();
            for (TreeMap<Double,Double> precisionRecallVals : recallPrecisionValues) {
                max = Math.max(precisionRecallVals.get(recallVal), max);
                min = Math.min(precisionRecallVals.get(recallVal), min);
                valuesList.add(precisionRecallVals.get(recallVal));
            }

            //now the assignments
            maxPrecisionAtFixedRecallValues.put(recallVal, max);
            minPrecisionAtFixedRecallValues.put(recallVal, min);
            avgPrecisionAtFixedRecallValues.put(recallVal, valuesList.stream().mapToDouble(a -> a).average().getAsDouble());
            tempStdev = valuesList.stream().mapToDouble(a -> Math.pow(a - avgPrecisionAtFixedRecallValues.get(recallVal), 2)).sum();
            stdevPrecisionAtFixedRecallValues.put(recallVal, Math.sqrt(tempStdev / valuesList.size()));
        }
    }

    /**
     * Extracts general information regarding the analyzed dataset
     * @param dataset
     */
    private void processGeneralDatasetInfo(Dataset dataset) throws Exception {
        this.numOfInstances = dataset.getNumOfInstancesPerColumn();

        //If an index to the target class was not provided, it's the last attirbute.
        this.numOfClasses = dataset.getNumOfClasses();
        this.numOfFeatures = dataset.getAllColumns(false).size(); //the target class is not included
        this.numOfNumericAtributes = 0;
        this.numOfDiscreteAttributes = 0;

        for (ColumnInfo columnInfo : dataset.getAllColumns(false)) {
            if (columnInfo.getColumn().getType() == Column.columnType.Numeric) {
                this.numOfNumericAtributes++;
                this.numericAttributesList.add(columnInfo);
            }
            if (columnInfo.getColumn().getType() == Column.columnType.Discrete) {
                this.numOfDiscreteAttributes++;
                this.discreteAttributesList.add(columnInfo);
            }
        }

        this.ratioOfNumericAttributes = this.numOfNumericAtributes / (this.numOfNumericAtributes + this.numOfDiscreteAttributes);
        this.ratioOfDiscreteAttributes = this.ratioOfDiscreteAttributes / (this.numOfNumericAtributes + this.numOfDiscreteAttributes);

        List<Double> numOfValuesperDiscreteAttribute = new ArrayList<>();
        for (ColumnInfo columnInfo : discreteAttributesList) {
            numOfValuesperDiscreteAttribute.add((double)((DiscreteColumn)columnInfo.getColumn()).getNumOfPossibleValues());
        }

        if (numOfValuesperDiscreteAttribute.size() > 0) {
            this.maxNumberOfDiscreteValuesPerAttribute = numOfValuesperDiscreteAttribute.stream().mapToDouble(a -> a).max().getAsDouble();
            this.minNumberOfDiscreteValuesPerAttribtue = numOfValuesperDiscreteAttribute.stream().mapToDouble(a -> a).min().getAsDouble();
            this.avgNumOfDiscreteValuesPerAttribute = numOfValuesperDiscreteAttribute.stream().mapToDouble(a -> a).average().getAsDouble();
            //the stdev requires an interim step
            double tempStdev = numOfValuesperDiscreteAttribute.stream().mapToDouble(a -> Math.pow(a - avgNumOfDiscreteValuesPerAttribute, 2)).sum();
            this.stdevNumOfDiscreteValuesPerAttribute = Math.sqrt(tempStdev / numOfValuesperDiscreteAttribute.size());
        }
        else {
            this.maxNumberOfDiscreteValuesPerAttribute = 0;
            this.minNumberOfDiscreteValuesPerAttribtue = 0;
            this.avgNumOfDiscreteValuesPerAttribute = 0;
            this.stdevNumOfDiscreteValuesPerAttribute = 0;
        }
    }
}
