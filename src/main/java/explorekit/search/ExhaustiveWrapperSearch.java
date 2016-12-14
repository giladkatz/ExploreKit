package explorekit.search;

import explorekit.Evaluation.ClassificationResults;
import explorekit.Evaluation.FilterEvaluators.FilterEvaluator;
import explorekit.Evaluation.WrapperEvaluation.WrapperEvaluator;
import explorekit.data.ColumnInfo;
import explorekit.data.Dataset;
import explorekit.operators.Operator;
import explorekit.operators.OperatorAssignment;
import explorekit.operators.OperatorsAssignmentsManager;
import explorekit.search.AttributeRankersFilters.AttributeRankerFilter;

import java.io.InputStream;
import java.util.*;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Created by giladkatz on 24/03/2016.
 */
public class ExhaustiveWrapperSearch extends Search {
    private final int maxIterations;
    Properties properties;

    public ExhaustiveWrapperSearch(int maxIterations) throws Exception {
        this.maxIterations = maxIterations;
        Properties properties = new Properties();
        InputStream input = this.getClass().getClassLoader().getResourceAsStream("config.properties");
        properties.load(input);
    }

    public void run(Dataset originalDataset, String runInfo) throws Exception {
        WrapperEvaluator wrapperEvaluator = super.getWrapper(properties.getProperty("wrapperApproach"));

        //The first step is to evaluate the initial attributes, so we get a reference point to how well we did
        wrapperEvaluator.EvaluationAndWriteResultsToFile(originalDataset, "", 0, runInfo, true,0, -1, -1, properties);

        //now we create the replica of the original dataset, to which we can add columns
        Dataset dataset = originalDataset.replicateDataset();

        //Get the training set sub-folds, used to evaluate the various candidate attributes
        List<Dataset> originalDatasetTrainingFolds = originalDataset.GenerateTrainingSetSubFolds();
        List<Dataset> subFoldTrainingDatasets = dataset.GenerateTrainingSetSubFolds();

        //We now apply the wrapper on the training subfolds in order to get the baseline score. This is the score a candidate attribute needs to "beat"
        double currentScore = wrapperEvaluator.produceAverageScore(subFoldTrainingDatasets, null, null, null, null, properties);
        System.out.println("Initial score: " + Double.toString(currentScore));

        //The probabilities assigned to each instance using the ORIGINAL dataset (training folds only)
        List<ClassificationResults> currentClassificationProbs = wrapperEvaluator.produceClassificationResults(originalDatasetTrainingFolds, properties);

        //Apply the unary operators (discretizers, normalizers) on all the original features. The attributes generated
        //here are different than the ones generated at later stages because they are included in the dataset that is
        //used to generate attributes in the iterative search phase
        OperatorsAssignmentsManager oam = new OperatorsAssignmentsManager(properties);
        List<OperatorAssignment> candidateAttributes = oam.applyUnaryOperators(dataset,null, null, subFoldTrainingDatasets, currentClassificationProbs);

        //Now we add the new attributes to the dataset (they are added even though they may not be included in the
        //final dataset beacuse they are essential to the full generation of additional features
        oam.GenerateAndAddColumnToDataset(dataset, candidateAttributes);

        //The initial dataset has been populated with the discretized/normalized features. Time to begin the search
        int iterationsCounter = 1;
        List<ColumnInfo> columnsAddedInthePreviousIteration = null;
        performExhaustiveWrapperSearch(originalDataset, runInfo, wrapperEvaluator, dataset, originalDatasetTrainingFolds, subFoldTrainingDatasets, currentClassificationProbs, oam, candidateAttributes, iterationsCounter, columnsAddedInthePreviousIteration);
    }

    private void performExhaustiveWrapperSearch(Dataset originalDataset, String runInfo, WrapperEvaluator wrapperEvaluator,
                                        Dataset dataset, List<Dataset> originalDatasetTrainingFolds, List<Dataset> subFoldTrainingDatasets, List<ClassificationResults> currentClassificationProbs,
                                        OperatorsAssignmentsManager oam, List<OperatorAssignment> candidateAttributes, int iterationsCounter, List<ColumnInfo> columnsAddedInthePreviousIteration) throws Exception {
        Properties properties = new Properties();
        InputStream input = OperatorsAssignmentsManager.class.getClassLoader().getResourceAsStream("config.properties");
        properties.load(input);
        AttributeRankerFilter rankerFilter = getRankerFilter(properties.getProperty("rankerApproach"));

        while (iterationsCounter <= this.maxIterations) {
            Date date = new Date();
            System.out.println("Starting search iteration " + Integer.toString(iterationsCounter) + " : " + date.toString());
            OperatorAssignment topRankingAssignment = null;
            OperatorAssignment chosenOperatorAssignment = null;

            //now we get all the possible operator assignments
            List<Operator> operators = oam.getNonUnaryOperatorsList();
            candidateAttributes.addAll(oam.getOperatorAssignments(dataset, columnsAddedInthePreviousIteration,
                    operators, Integer.parseInt(properties.getProperty("maxNumOfAttsInOperatorSource"))));

            //now we run the wrapper evaluation on all of the operator assignments and get the scores
            ReentrantLock attributeGenerationLock = new ReentrantLock();
            ReentrantLock wrapperEvaluationLock = new ReentrantLock();
            List<ClassificationResults> tempCurrentClassificationProbs = currentClassificationProbs;
            candidateAttributes.parallelStream().forEach(oa -> {
                try {
                    attributeGenerationLock.lock();
                    Dataset replicatedDataset = dataset.replicateDataset();

                    attributeGenerationLock.unlock();

                    ColumnInfo ci = oam.generateColumn(replicatedDataset, oa, true);
                    wrapperEvaluationLock.lock();
                    WrapperEvaluator cloneEvaluator = wrapperEvaluator.getCopy();
                    List<Dataset> replicatedSubFoldsList = new ArrayList<>();
                    for (Dataset subFoldDataset: subFoldTrainingDatasets) {
                        replicatedSubFoldsList.add(subFoldDataset.replicateDataset());
                    }
                    wrapperEvaluationLock.unlock();
                    //run the wrapper evaluator and get the score
                    double score = oam.applyOperatorAndPerformWrapperEvaluation(originalDatasetTrainingFolds,oa,cloneEvaluator, tempCurrentClassificationProbs, null);
                    oa.setWrapperEvaluatorScore(score);
                }
                catch (Exception ex) {
                    System.out.println("error when generating and evaluating attribute: " + oa.getName());
                    System.out.println("The error: " + ex.getMessage());
                }
            });

            //once all candidates have been evaluated, sort them and choose the top ranking attribute
            candidateAttributes = rankerFilter.rankAndFilter(candidateAttributes,null,subFoldTrainingDatasets,currentClassificationProbs);
            chosenOperatorAssignment = candidateAttributes.get(0);
            System.out.print("Chosen att score: " + chosenOperatorAssignment.getWrapperEvaluatorScore());

            //remove the chosen attribute from the list of "candidates"
            candidateAttributes.remove(chosenOperatorAssignment);

            //The final step - add the new attribute to the datasets
            //start with the dataset used in the following search iterations
            columnsAddedInthePreviousIteration = oam.addAddtibuteToDataset(dataset, chosenOperatorAssignment, true, currentClassificationProbs);

            //continue with the final dataset
            oam.addAddtibuteToDataset(originalDataset, chosenOperatorAssignment, false, currentClassificationProbs);

            //finally, we need to recalculate the baseline score used for the attribute selection (using the updated final dataset)
            currentClassificationProbs = wrapperEvaluator.produceClassificationResults(originalDatasetTrainingFolds, properties);

            StringBuilder expDescription = new StringBuilder();
            expDescription.append("Evaluation results for iteration " + Integer.toString(iterationsCounter) + "\n");
            expDescription.append("Added attribute: " + chosenOperatorAssignment.getName() + "\n");
            ////(Dataset dataset, String addedAttribute, int iteration, String runInfo, boolean newFile) throws Exception {
            wrapperEvaluator.EvaluationAndWriteResultsToFile(originalDataset, chosenOperatorAssignment.getName(),
                    iterationsCounter, runInfo, false, candidateAttributes.size()+1, chosenOperatorAssignment.getFilterEvaluatorScore(),chosenOperatorAssignment.getWrapperEvaluatorScore(), properties);
            iterationsCounter++;
        }
    }
}
