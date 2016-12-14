package explorekit.Evaluation.WrapperEvaluation;

import com.sun.applet2.AppletParameters;
import explorekit.Evaluation.ClassificationItem;
import explorekit.Evaluation.ClassificationResults;
import explorekit.Evaluation.Evaluator;
import explorekit.data.Column;
import explorekit.data.ColumnInfo;
import explorekit.data.Dataset;
import explorekit.operators.OperatorAssignment;
import explorekit.operators.OperatorsAssignmentsManager;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.evaluation.IntervalBasedEvaluationMetric;
import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.evaluation.Prediction;
import weka.classifiers.functions.SMO;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;

import java.io.FileWriter;
import java.io.InputStream;
import java.io.PrintWriter;
import java.util.*;

/**
 * Created by giladkatz on 20/02/2016.
 */
public abstract class WrapperEvaluator implements Evaluator {

    public class ClassificationItemsComparator implements Comparator<ClassificationItem> {
        public int minorityClassIndex;

        public ClassificationItemsComparator(int minorityClassIndex) {
            this.minorityClassIndex = minorityClassIndex;
        }

        @Override
        public int compare(ClassificationItem o1, ClassificationItem o2) {
            if (o1.getProbabilitiesOfClass(minorityClassIndex) == (o2.getProbabilitiesOfClass(minorityClassIndex))) {
                return 0;
            }
            if (o1.getProbabilitiesOfClass(minorityClassIndex) - (o2.getProbabilitiesOfClass(minorityClassIndex)) > 0) {
                return 1;
            }
            else {
                return -1;
            }
        }
    }

    public abstract WrapperEvaluator getCopy();

    public abstract double produceScoreWithSampling(Dataset dataset, ClassificationResults currentScore,
                                                    int numOfSamplesPerFold, int numOfTimesToSample, int randomSeed, Properties properties) throws Exception;

    /**
     * Gets the ClassificationResults items for each of the analyzed datasets (contains the class probabilites and true
     * class for each instance)
     * @param datasets
     * @return
     * @throws Exception
     */
    public List<ClassificationResults> produceClassificationResults(List<Dataset> datasets, Properties properties) throws Exception {
        List<ClassificationResults> classificationResultsPerFold = new ArrayList<>();
        for (Dataset dataset : datasets) {
            Date date = new Date();
            System.out.println("Starting to run classifier " + date.toString());
            Evaluation evaluationResults = runClassifier(properties.getProperty("classifier"), dataset.generateSet(true), dataset.generateSet(false), properties);
            date = new Date();
            System.out.println("Starting to process classification results " + date.toString());
            ClassificationResults classificationResults = getClassificationResults(evaluationResults, dataset, properties);
            date = new Date();
            System.out.println("Done " + date.toString());
            classificationResultsPerFold.add(classificationResults);
        }
        return classificationResultsPerFold;
    }

    /**
     * Similar to produceClassificationResults, but instead of using the entire training set for each dataset we
     * sample a subset several times. Results for each dataset are averaged and then averaged again across datasets
     * @param datasets
     * @param numOfSamplesPerFold
     * @param numOfTimesToSample
     * @param randomSeed
     * @return
     * @throws Exception
     */
    public List<ClassificationResults> produceClassificationResultsWithSampling(List<Dataset> datasets,
                                                                                int numOfTimesToSample, int numOfSamplesPerFold, int randomSeed, Properties properties) throws Exception {

        List<ClassificationResults> classificationResultsPerFold = new ArrayList<>();
        for (Dataset dataset : datasets) {
            List<ClassificationResults> resultsPerFold = new ArrayList<>();
            for (int i=0; i<numOfTimesToSample; i++) {
                //it's important to note that the training set is generated using the sampling. The random seed is (original seed+i) so that
                // we have different indices each time but also the same indices across training iterations
                Evaluation evaluationResults = runClassifier(properties.getProperty("classifier"), dataset.generateSetWithSampling(numOfSamplesPerFold,randomSeed+i), dataset.generateSet(false), properties);
                ClassificationResults classificationResults = getClassificationResults(evaluationResults, dataset, properties);
                resultsPerFold.add(classificationResults);
            }
            //here we average all the performance metrics and create one object to represent the fold
            classificationResultsPerFold.add(averageClassificationResults(resultsPerFold));
        }
        return classificationResultsPerFold;
    }

    /**
     * Used to calculcate the AUC across all wrapper evaluators. Currently does an UNWEIGHTED average of AUCs. May
     * need to change this to a configurable value
     * @param evaluation
     * @param dataset
     * @return
     */
    public double CalculateAUC(Evaluation evaluation, Dataset dataset) {
        double auc = 0;
        for (int i=0; i<dataset.getNumOfClasses(); i++) {
            auc += evaluation.areaUnderROC(i);
        }
        auc = auc/dataset.getNumOfClasses();
        return auc;
    }

    /**
     * Used to calculcate the Logarithmic Loss scoring function across all wrapper evaluators. LogLoss is
     * defined only for NominalPredictions, otherwise returns zero.
     * @param evaluation
     * @param dataset
     * @return
     */
    public double CalculateLogLoss(Evaluation evaluation, Dataset dataset) {
        ArrayList<Prediction> predictions = evaluation.predictions();
        // LogLoss is defined for NominalPredictions.
        if (predictions.get(0) instanceof NominalPrediction) {
            double logloss = 0;
            for (Prediction p : predictions) {
                double pred_actual = ((NominalPrediction) p).distribution()[(int) p.actual()];
                // Fix value so that we avoid the logarithms of of 0 and 1.
                pred_actual = Math.max(Math.min(pred_actual, 1 - 1E-15), 1E-15);
                logloss += Math.log(pred_actual);
            }
            logloss = logloss * ((-1.0) / predictions.size());
            // Return negative LogLoss so that bigger score means better results (as in most other scoring functions).
            return -logloss;
        }
        return 0;
    }

    private ClassificationResults averageClassificationResults(List<ClassificationResults> classificationResultses) {
        List<ClassificationItem> averageClassifications = new ArrayList<>();
        double auc = 0;
        double logloss = 0;
        TreeMap<Double,Double> tprFprValues = new TreeMap<>();
        TreeMap<Double,Double> recallPrecisionValues = new TreeMap<>();
        HashMap<Double,Double> fMeasureValuesPerRecall = new HashMap<>();

        for (ClassificationResults classificationResult : classificationResultses) {
            //deal with the case this is a new object
            if (averageClassifications.size() == 0) {
                //add the classification items
                for (int i=0; i<classificationResult.getItemClassifications().size(); i++) {
                    averageClassifications.add(new ClassificationItem(classificationResult.getItemClassifications().get(i).getTrueClass(),
                            classificationResult.getItemClassifications().get(i).getProbabilities()));
                }
                //add the auc
                auc += classificationResult.getAuc();

                //add the logloss
                logloss += classificationResult.getLogLoss();

                //tpr-fpr values
                for (double key : classificationResult.getTprFprValues().keySet()) {
                    tprFprValues.put(key, classificationResult.getTprFprValues().get(key));
                }

                //recall precision values
                for (double key : classificationResult.getRecallPrecisionValues().keySet()) {
                    recallPrecisionValues.put(key, classificationResult.getRecallPrecisionValues().get(key));
                }

                //fMeasureValuesPerRecall values
                for (double key : classificationResult.getFMeasureValuesPerRecall().keySet()) {
                    fMeasureValuesPerRecall.put(key, classificationResult.getFMeasureValuesPerRecall().get(key));
                }
            }
            else {
                //Classification items
                for (int i=0; i<classificationResult.getItemClassifications().size(); i++) {
                    double[] currentProbs = averageClassifications.get(i).getProbabilities();
                    double[] probsToAdd = classificationResult.getItemClassifications().get(i).getProbabilities();
                    for (int j=0; j<currentProbs.length; j++) {
                        currentProbs[j] += probsToAdd[j];
                    }
                    averageClassifications.get(i).setProbabilities(currentProbs);
                }

                //add the auc
                auc += classificationResult.getAuc();

                //tpr-fpr values
                //for (double key : classificationResult.getTprFprValues().keySet()) {
                //    tprFprValues.put(key, tprFprValues.get(key) + classificationResult.getTprFprValues().get(key));
                //}

                //recall precision values
                for (double key : classificationResult.getRecallPrecisionValues().keySet()) {
                    recallPrecisionValues.put(key, recallPrecisionValues.get(key) + classificationResult.getRecallPrecisionValues().get(key));
                }

                //fMeasureValuesPerRecall values
                for (double key : classificationResult.getFMeasureValuesPerRecall().keySet()) {
                    fMeasureValuesPerRecall.put(key, fMeasureValuesPerRecall.get(key) + classificationResult.getFMeasureValuesPerRecall().get(key));
                }
            }
        }

        //now we need to average all the values by dividing them by the number of the ClassificationResults object that were involved
        for (int i=0; i<averageClassifications.size(); i++) {
            ClassificationItem tempItem = averageClassifications.get(i);
            for (int j=0; j<tempItem.getProbabilities().length; j++) {
                tempItem.setProbabilityOfClass(j, tempItem.getProbabilitiesOfClass(j)/classificationResultses.size());
            }
        }
        auc = auc/classificationResultses.size();

        //for (double key : tprFprValues.keySet()) {
        //    tprFprValues.put(key, tprFprValues.get(key)/classificationResultses.size());
        //}

        //recall precision values
        for (double key : recallPrecisionValues.keySet()) {
            recallPrecisionValues.put(key, recallPrecisionValues.get(key)/classificationResultses.size());
        }

        //fMeasureValuesPerRecall values
        for (double key : fMeasureValuesPerRecall.keySet()) {
            fMeasureValuesPerRecall.put(key, fMeasureValuesPerRecall.get(key)/classificationResultses.size());
        }

        return new ClassificationResults(averageClassifications,auc, logloss, tprFprValues,recallPrecisionValues,fMeasureValuesPerRecall);
    }

    /**
     * Obtains the classification probabilities assigned to each instance and returns them as a ClassificationResults object
     * @param evaluation
     * @return
     */
    public ClassificationResults getClassificationResults(Evaluation evaluation, Dataset dataset, Properties properties) throws Exception {
        Date date = new Date();

        //used for validation - by making sure that that the true classes of the instances match we avoid "mix ups"
        Column actualTargetColumn = dataset.getTargetClassColumn().getColumn();

        List<ClassificationItem> classificationItems = new ArrayList<>();
        int counter = 0;
        for (Prediction prediction: evaluation.predictions()) {
            if ((counter%10000) == 0) {
                if ((int) prediction.actual() != (Integer) actualTargetColumn.getValue(dataset.getIndicesOfTestInstances().get(counter))) {
                    if (dataset.getTestDataMatrixWithDistinctVals() == null || dataset.getTestDataMatrixWithDistinctVals().length == 0) {
                        throw new Exception("the target class values do not match");
                    }
                }
            }
            counter++;
            ClassificationItem ci = new ClassificationItem((int)prediction.actual(),((NominalPrediction)prediction).distribution());
            classificationItems.add(ci);
        }
        //Now generate all the statistics we may want to use
        double auc = CalculateAUC(evaluation, dataset);

        double logloss = CalculateLogLoss(evaluation, dataset);

        //We calcualte the TPR/FPR rate. We do it ourselves because we want all the values
        TreeMap<Double,Double> tprFprValues = calculateTprFprRate(evaluation, dataset);

        //The TRR/FPR values enable us to calculate the precision/recall values.
        TreeMap<Double,Double> recallPrecisionValues = calculateRecallPrecisionValues(dataset, tprFprValues,
                Double.parseDouble(properties.getProperty("precisionRecallIntervals")));

        //Next, we calculate the F-Measure at the selected points
        HashMap<Double,Double> fMeasureValuesPerRecall = new HashMap<>();
        String[] fMeasurePrecisionValues = properties.getProperty("FMeausrePoints").split(",");
        for (String recallVal: fMeasurePrecisionValues) {
            double recall = Double.parseDouble(recallVal);
            double precision = recallPrecisionValues.get(recall);
            double F1Measure = (2*precision*recall)/(precision+recall);
            fMeasureValuesPerRecall.put(recall,F1Measure);
        }

        ClassificationResults classificationResults = new ClassificationResults(classificationItems, auc, logloss, tprFprValues, recallPrecisionValues, fMeasureValuesPerRecall);

        return classificationResults;
    }

    /**
     * Calculates the score for each of the datasets in the list (subfolds) and returns the average score
     * @param analyzedDatasets
     * @param classificationResults
     * @return
     * @throws Exception
     */
    public double produceAverageScore(List<Dataset> analyzedDatasets, List<ClassificationResults> classificationResults, Dataset completeDataset, OperatorAssignment oa, ColumnInfo candidateAttribute, Properties properties) throws Exception {

        double score = 0;
        for (int i = 0; i< analyzedDatasets.size(); i++) {
            Dataset dataset = analyzedDatasets.get(i);

            //the ClassificationResult can be null for the initial run.
            ClassificationResults classificationResult = null;
            if (classificationResults != null) {
                classificationResult = classificationResults.get(i);
            }
            score += produceScore(dataset, classificationResult, completeDataset, oa, candidateAttribute, properties);
        }
        return  score/(analyzedDatasets.size());
    }

    public double produceAverageScoreWithSampling(List<Dataset> datasets, List<ClassificationResults> classificationResults,
                                                  int numOfSamplesPerFold, int numOfInstancesInSample, int randomSeed, Properties properties) throws Exception {

        double score = 0;
        for (int i=0; i<datasets.size(); i++) {
            Dataset dataset = datasets.get(i);

            //the ClassificationResult can be null for the initial run.
            ClassificationResults classificationResult = null;
            if (classificationResults != null) {
                classificationResult = classificationResults.get(i);
            }
            score += produceScoreWithSampling(dataset, classificationResult,numOfSamplesPerFold,numOfInstancesInSample,randomSeed, properties);
        }
        return  score/(datasets.size());
    }

    public Evaluator.evaluatorType getType() {
        return evaluatorType.Wrapper;
    }

    public Evaluation runClassifier(String classifierName, Instances trainingSet, Instances testSet, Properties properties) throws Exception {
        try {
            OperatorsAssignmentsManager oam = new OperatorsAssignmentsManager(properties);
            Classifier classifier = oam.getClassifier(classifierName);
            classifier.buildClassifier(trainingSet);
            Evaluation evaluation;

            evaluation = new Evaluation(trainingSet);
            evaluation.evaluateModel(classifier, testSet);

            return evaluation;
        }
        catch (Exception ex) {
            System.out.println("problem running classifier");
        }

        return null;
    }

    /**
     * This procedure should only be called when an attribute (or attributes) have been selected. It will
     * several statistics into the output file, statistics whose calculation requires additional time to the
     * one spent by Weka itself.
     * @param dataset
     * @param newFile used to determine whether the text needs to be appended or override any existing text
     * @throws Exception
     */
    public void EvaluationAndWriteResultsToFile(Dataset dataset, String addedAttribute, int iteration, String runInfo,
                    boolean newFile, int evaluatedAttsCounter, double filterEvaluatorScore, double wrapperEvaluationScore, Properties properties) throws Exception {
        Evaluation evaluation = runClassifier(properties.getProperty("classifier"),dataset.generateSet(true), dataset.generateSet(false), properties);

        //We calcualte the TPR/FPR rate. We do it ourselves because we want all the values
        TreeMap<Double,Double> tprFprValues = calculateTprFprRate(evaluation, dataset);

        //The TRR/FPR values enable us to calculate the precision/recall values.
        TreeMap<Double,Double> recallPrecisionValues = calculateRecallPrecisionValues(dataset, tprFprValues,
                Double.parseDouble(properties.getProperty("precisionRecallIntervals")));


        //Next, we calculate the F-Measure at the selected points
        TreeMap<Double,Double> fMeasureValuesPerRecall = new TreeMap<>();
        String[] fMeasurePrecisionValues = properties.getProperty("FMeausrePoints").split(",");
        for (String recallVal: fMeasurePrecisionValues) {
            double recall = Double.parseDouble(recallVal);
            double precision = recallPrecisionValues.get(recall);
            double F1Measure = (2*precision*recall)/(precision+recall);
            fMeasureValuesPerRecall.put(recall,F1Measure);
        }

        //now we can write everything to file
        StringBuilder sb = new StringBuilder();

        //If it's a new file, we need to create a header for the file
        if (newFile) {
            sb.append("Iteration,Added_Attribute,LogLoss,AUC,");
            for (double recallVal : fMeasureValuesPerRecall.keySet() ) {
                sb.append("F1_Measure_At_Recall_" + Double.toString(recallVal) + ",");
            }
            for (double recallVal: recallPrecisionValues.keySet()) {
                sb.append("Precision_At_Recall_Val_" + Double.toString(recallVal)+",");
            }
            sb.append("Chosen_Attribute_Filter_Score,Chosen_Attribute_Wrapper_Score,Num_Of_Evaluated_Attributes_In_Iteration");
            sb.append("Iteration_Completion_time");
            sb.append("\n");
        }
        sb.append(Integer.toString(iteration) + ",");
        sb.append("\"" + addedAttribute + "\""  +",");

        // The LogLoss
        sb.append(Double.toString(CalculateLogLoss(evaluation, dataset)).concat(","));

        //The AUC
        sb.append(Double.toString(evaluation.areaUnderROC(dataset.getMinorityClassIndex())).concat(","));

        //The F1 measure
        for (double recallVal : fMeasureValuesPerRecall.keySet() ) {
            sb.append(Double.toString(fMeasureValuesPerRecall.get(recallVal)).concat(","));
        }

        //Recall/Precision values
        for (double recallVal: recallPrecisionValues.keySet()) {
            sb.append(Double.toString(recallPrecisionValues.get(recallVal)).concat(","));
        }

        sb.append(Double.toString(filterEvaluatorScore) + ",");
        sb.append(Double.toString(wrapperEvaluationScore) + ",");
        sb.append(Integer.toString(evaluatedAttsCounter).concat(","));

        Date date = new Date();
        sb.append(date.toString());

        try
        {
            String filename= properties.getProperty("resultsFilePath") + dataset.getName() + runInfo + ".csv";
            FileWriter fw = new FileWriter(filename,!newFile);
            fw.write(sb.toString() + "\n");
            fw.close();
        }
        catch(Exception ioe)
        {
            System.err.println("IOException: " + ioe.getMessage());
        }
    }

    /**
     * Used to calculate the recall/precision values from the TPR/FPR values. We use the recall values as the basis for
     * our calculation because they are monotonic and becuase it enables the averaging of different fold values
     * @param dataset
     * @param tprFprValues
     * @param recallInterval
     * @return
     */
    private TreeMap<Double,Double> calculateRecallPrecisionValues(Dataset dataset, TreeMap<Double,Double> tprFprValues, double recallInterval) {
        //start by getting the number of samples in the minority class and in other classes
        int minorityClassIndex = dataset.getMinorityClassIndex();
        double numOfMinorityClassItems = dataset.getNumOfRowsPerClassInTestSet()[minorityClassIndex];
        double numOfNonMinorityClassItems = 0; //all non-minority class samples are counted together (multi-class cases)
        for (int i=0; i< dataset.getNumOfRowsPerClassInTestSet().length; i++) {
            if (i != minorityClassIndex) {
                numOfNonMinorityClassItems += dataset.getNumOfRowsPerClassInTestSet()[i];
            }
        }

        TreeMap<Double,Double> recallPrecisionValues = new TreeMap<>();
        for (double i=0; i<=1; i+=recallInterval) {
            double recallKey = getClosestRecallValue(tprFprValues, i); //the recall is the TPR
            double precision = (recallKey*numOfMinorityClassItems)/((recallKey*numOfMinorityClassItems) + (tprFprValues.get(recallKey)*numOfNonMinorityClassItems));
            if (Double.isNaN(precision)) {
                precision = 0;
            }
            recallPrecisionValues.put(round(i,2),precision);
        }
        return recallPrecisionValues;
    }

    private double round(double value, int places) {
        if (places < 0) throw new IllegalArgumentException();

        long factor = (long) Math.pow(10, places);
        value = value * factor;
        long tmp = Math.round(value);
        return (double) tmp / factor;
    }

    /**
     * Returns the ACTUAL recall value that is closest to the requested value. It is important to note that there are
     * no limitations in this function, so in end-cases the function may return strange results.
     * @param tprFprValues
     * @param recallVal
     * @return
     */
    private double getClosestRecallValue(TreeMap<Double, Double> tprFprValues, double recallVal) {
        for (double key: tprFprValues.keySet()) {
            if (key >= recallVal) {
                return key;
            }
        }
        return 0;
    }

    /**
     * Used to calculate all the TPR-FPR values of the provided evaluation
     * @param evaluation
     * @param dataset
     * @return
     */
    private TreeMap<Double,Double> calculateTprFprRate(Evaluation evaluation, Dataset dataset) {
        Date date = new Date();
        System.out.println("Starting TPR/FPR calculations : " + date.toString());

        HashMap<Double,Double> trpFprRates = new HashMap<>();

        //we convert the results into a format that's more comfortable to work with
        List<ClassificationItem> classificationItems = new ArrayList<>();
        for (Prediction prediction: evaluation.predictions()) {
            ClassificationItem ci = new ClassificationItem((int)prediction.actual(),((NominalPrediction)prediction).distribution());
            classificationItems.add(ci);
        }

        //now we need to know what is the minority class and the number of samples for each class
        int minorityClassIndex = dataset.getMinorityClassIndex();
        double numOfNonMinorityClassItems = 0; //all non-minority class samples are counted together (multi-class cases)
        for (int i=0; i< dataset.getNumOfRowsPerClassInTestSet().length; i++) {
            if (i != minorityClassIndex) {
                numOfNonMinorityClassItems += dataset.getNumOfRowsPerClassInTestSet()[i];
            }
        }

        //sort all samples by their probability of belonging to the minority class
        Collections.sort(classificationItems, new ClassificationItemsComparator(minorityClassIndex));
        Collections.reverse(classificationItems);

        TreeMap<Double,Double> tprFprValues = new TreeMap<>();
        tprFprValues.put(0.0,0.0);
        double minoritySamplesCounter = 0;
        double majoritySamplesCounter = 0;
        double currentProb = 2;
        for (ClassificationItem ci : classificationItems) {
            double currentSampleProb = ci.getProbabilitiesOfClass(minorityClassIndex);
            //if the probability is different, time to update the TPR/FPR statistics
            if (currentSampleProb != currentProb) {
                double tpr =  minoritySamplesCounter/dataset.getNumOfRowsPerClassInTestSet()[minorityClassIndex];
                double fpr = majoritySamplesCounter/numOfNonMinorityClassItems;
                tprFprValues.put(tpr,fpr);
                currentProb = currentSampleProb;
            }
            if (ci.getTrueClass() == minorityClassIndex) {
                minoritySamplesCounter++;
            }
            else {
                majoritySamplesCounter++;
            }
        }
        tprFprValues.put(1.0,1.0);
        tprFprValues.put(1.0001,1.0);
        date = new Date();
        System.out.println("Done : " + date.toString());
        return tprFprValues;
    }


}