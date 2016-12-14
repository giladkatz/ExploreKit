package explorekit.Evaluation.FilterEvaluators;

import explorekit.Evaluation.ClassificationItem;
import explorekit.Evaluation.ClassificationResults;
import explorekit.Evaluation.Evaluator;
import explorekit.Evaluation.MLFeatureExtraction.AttributeInfo;
import explorekit.Evaluation.MLFeatureExtraction.DatasetBasedAttributes;
import explorekit.Evaluation.MLFeatureExtraction.MLAttributesManager;
import explorekit.Evaluation.MLFeatureExtraction.OperatorAssignmentBasedAttributes;
import explorekit.data.Column;
import explorekit.data.ColumnInfo;
import explorekit.data.Dataset;
import explorekit.operators.OperatorAssignment;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.evaluation.Prediction;
import weka.core.Instance;
import weka.core.Instances;

import java.io.File;
import java.io.InputStream;
import java.util.HashMap;
import java.util.Properties;

import static explorekit.Evaluation.Evaluator.evaluatorScoringMethod.ClassifierProbability;

/**
 * Created by giladkatz on 08/05/2016.
 */
public class MLFilterEvaluator extends FilterEvaluator {


    private Classifier classifier;

    private Evaluation evaluation;

    HashMap<Integer, AttributeInfo> datasetAttributes;

    public MLFilterEvaluator() {

    }

    /**
     * Constructor
     * @param dataset
     * @param properties
     * @throws Exception
     */
    public MLFilterEvaluator(Dataset dataset, Properties properties) throws Exception {
        initializeBackgroundModel(dataset, properties);
    }

    @Override
    public double produceScoreWithDistinctValues(Dataset dataset, ClassificationResults currentScore, OperatorAssignment oa, ColumnInfo candidateAttribute) {
        throw new NotImplementedException();
    }

    @Override
    public FilterEvaluator getCopy() {
        MLFilterEvaluator mlf = new MLFilterEvaluator();
        mlf.setClassifier(this.classifier);
        mlf.setDatasetAttributes(this.datasetAttributes);
        mlf.setEvaluation(this.evaluation);
        return mlf;
    }

    @Override
    public boolean needToRecalculateScoreAtEachIteration() {
        return true;
    }

    @Override
    public evaluatorScoringMethod getEvaluatorScoreingMethod() {
        return ClassifierProbability;
    }

    @Override
    public double produceScore(Dataset analyzedDatasets, ClassificationResults currentScore, Dataset completeDataset, OperatorAssignment oa, ColumnInfo candidateAttribute, Properties properties) throws Exception {

        try {
            MLAttributesManager mlam = new MLAttributesManager();
            if (classifier == null) {
                System.out.println("Classifier is not initialized");
                throw new Exception("Classifier is not initialized");
            }

            //we need to generate the features for this candidate attribute and then run the (previously) calculated classification model
            OperatorAssignmentBasedAttributes oaba = new OperatorAssignmentBasedAttributes();
            HashMap<Integer, AttributeInfo> candidateAttributes = oaba.getOperatorAssignmentBasedAttributes(analyzedDatasets, oa, candidateAttribute, properties);

            //now add the dataset attributes to the candidate attribute's attribute
            for (AttributeInfo datasetAttInfo : datasetAttributes.values()) {
                candidateAttributes.put(candidateAttributes.size(), datasetAttInfo);
            }

            //We need to add the type of the classifier we're using
            AttributeInfo classifierAttribute = new AttributeInfo("Classifier", Column.columnType.Discrete, mlam.getClassifierIndex(properties.getProperty("classifier")), properties.getProperty("classifiersForMLAttributesGeneration").split(",").length);
            candidateAttributes.put(candidateAttributes.size(), classifierAttribute);

            //In order to have attributes of the same set size, we need to add the class attribute. We don't know the true value, so we set it to negative
            AttributeInfo classAttrubute = new AttributeInfo("classAttribute", Column.columnType.Discrete, 0, 2);
            candidateAttributes.put(candidateAttributes.size(), classAttrubute);

            //finally, we need to set the index of the target class
            Instances testInstances = mlam.generateValuesMatrix(candidateAttributes);
            testInstances.setClassIndex(testInstances.numAttributes() - 1);


            evaluation = new Evaluation(testInstances);
            evaluation.evaluateModel(classifier, testInstances);

            //we have a single prediction, so it's easy to process
            Prediction prediction = evaluation.predictions().get(0);
            ClassificationItem ci = new ClassificationItem((int) prediction.actual(), ((NominalPrediction) prediction).distribution());
            return ci.getProbabilities()[analyzedDatasets.getMinorityClassIndex()];
        }
        catch (Exception ex) {
            System.out.println("Error in ML score generation : " + ex.getMessage());
            return -1;
        }
    }

    /**
     * Used to create or load the data used by the background model - all the datasets that are evaluated "offline" to create
     * the meta-features classifier.
     * @param analyzedDatasets
     * @param properties
     * @throws Exception
     */
    public void initializeBackgroundModel(Dataset analyzedDatasets, Properties properties) throws Exception {
        MLAttributesManager mlam = new MLAttributesManager();
        classifier = mlam.generateBackgroundClassificationModel(analyzedDatasets, properties);

        DatasetBasedAttributes dba = new DatasetBasedAttributes();
        datasetAttributes = dba.getDatasetBasedFeatures(analyzedDatasets, properties.getProperty("classifier"), properties);
    }

    public void recalculateDatasetBasedFeatures(Dataset analyzedDatasets, Properties properties) throws Exception {
        DatasetBasedAttributes dba = new DatasetBasedAttributes();
        datasetAttributes = dba.getDatasetBasedFeatures(analyzedDatasets, properties.getProperty("classifier"), properties);
    }

    public Classifier getClassifier() {
        return classifier;
    }

    public void setClassifier(Classifier classifier) {
        this.classifier = classifier;
    }

    public HashMap<Integer, AttributeInfo> getDatasetAttributes() {
        return this.datasetAttributes;
    }

    public void setDatasetAttributes(HashMap<Integer, AttributeInfo> datasetAttributes) {
        this.datasetAttributes = datasetAttributes;
    }

    public void deleteBackgroundClassificationModel(Dataset dataset, Properties properties) throws Exception {
        String backgroundFilePath = properties.getProperty("backgroundClassifierLocation") + "_background_" + dataset.getName() + "_classifier_obj";
        File file = new File(backgroundFilePath);
        file.delete();
    }

    public Evaluation getEvaluation() {
        return evaluation;
    }

    public void setEvaluation(Evaluation evaluation) {
        this.evaluation = evaluation;
    }
}
