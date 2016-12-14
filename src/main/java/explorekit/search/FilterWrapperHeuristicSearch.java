package explorekit.search;

import explorekit.Evaluation.ClassificationResults;
import explorekit.Evaluation.FilterEvaluators.FilterEvaluator;
import explorekit.Evaluation.WrapperEvaluation.WrapperEvaluator;
import explorekit.data.ColumnInfo;
import explorekit.data.Dataset;
import explorekit.operators.OperatorAssignment;
import explorekit.operators.OperatorsAssignmentsManager;
import explorekit.search.AttributeRankersFilters.AttributeRankerFilter;

import java.io.InputStream;
import java.util.*;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Created by giladkatz on 11/02/2016.
 * The search approach presented in the paper. The candidate features are first ranked using a filter (i.e. conputationally efficient) approach.
 * The ranked features are then evaluated sequentally until one that improves the performance on the validation set is found.
 */
public class FilterWrapperHeuristicSearch extends Search {
    private Date experimentStartDate;
    private final int maxIterations;
    private Properties properties;

    public FilterWrapperHeuristicSearch(int maxIterations) throws Exception {
        this.maxIterations = maxIterations;
        properties = new Properties();
        InputStream input = this.getClass().getClassLoader().getResourceAsStream("config.properties");
        properties.load(input);
    }

    public void run(Dataset originalDataset, String runInfo) throws Exception{
        //the initialization of the oevaluators (and the generation of background models, if needed) is not counted in the experiment time
        FilterEvaluator filterEvaluator = super.getFilter(properties.getProperty("filterApproach"), originalDataset, properties);
        WrapperEvaluator wrapperEvaluator = super.getWrapper(properties.getProperty("wrapperApproach"));

        experimentStartDate = new Date();
        System.out.println("Experiment Start Date/Time: " + experimentStartDate.toString());

        //The first step is to evaluate the initial attributes, so we get a reference point to how well we did
        wrapperEvaluator.EvaluationAndWriteResultsToFile(originalDataset, "", 0, runInfo, true,0, -1, -1, properties);

        //now we create the replica of the original dataset, to which we can add columns
        Dataset dataset = originalDataset.replicateDataset();

        //Get the training set sub-folds, used to evaluate the various candidate attributes
        List<Dataset> originalDatasetTrainingFolds = originalDataset.GenerateTrainingSetSubFolds();
        List<Dataset> subFoldTrainingDatasets = dataset.GenerateTrainingSetSubFolds();

        Date date = new Date();

        //We now apply the wrapper on the training subfolds in order to get the baseline score. This is the score a candidate attribute needs to "beat"
        double currentScore = wrapperEvaluator.produceAverageScore(subFoldTrainingDatasets, null, null, null, null, properties);
        System.out.println("Initial score: " + Double.toString(currentScore)  + " : " + date.toString());

        //The probabilities assigned to each instance using the ORIGINAL dataset (training folds only)
        System.out.println("Producing initial classification results"  + " : " + date.toString());
        List<ClassificationResults> currentClassificationProbs = wrapperEvaluator.produceClassificationResults(originalDatasetTrainingFolds, properties);
        date = new Date();
        System.out.println("  .....done " + date.toString());

        //Apply the unary operators (discretizers, normalizers) on all the original features. The attributes generated
        //here are different than the ones generated at later stages because they are included in the dataset that is
        //used to generate attributes in the iterative search phase
        System.out.println("Starting to apply unary operators:   "  + " : " + date.toString());
        OperatorsAssignmentsManager oam = new OperatorsAssignmentsManager(properties);
        List<OperatorAssignment> candidateAttributes = oam.applyUnaryOperators(dataset,null, filterEvaluator, subFoldTrainingDatasets, currentClassificationProbs);
        date = new Date();
        System.out.println("  .....done " + date.toString());

        //Now we add the new attributes to the dataset (they are added even though they may not be included in the
        //final dataset beacuse they are essential to the full generation of additional features
        System.out.println("Starting to generate and add columns to dataset:   "  + " : " + date.toString());
        oam.GenerateAndAddColumnToDataset(dataset, candidateAttributes);
        date = new Date();
        System.out.println("  .....done " + date.toString());

        //The initial dataset has been populated with the discretized/normalized features. Time to begin the search
        int iterationsCounter = 1;
        List<ColumnInfo> columnsAddedInthePreviousIteration = null;

        performIterativeSearch(originalDataset, runInfo, filterEvaluator, wrapperEvaluator, dataset, originalDatasetTrainingFolds, subFoldTrainingDatasets, currentClassificationProbs, oam, candidateAttributes, iterationsCounter, columnsAddedInthePreviousIteration);
    }


    OperatorAssignment chosenOperatorAssignment = null;
    OperatorAssignment topRankingAssignment = null;
    int evaluatedAttsCounter = 0;
    boolean terminateSearch = false;


    /**
     * Performs the iterative search - the selection of the candidate features and the generation of the additional candidates that are added to the pool
     * in the next round.
     * @param originalDataset The dataset with the original attributes set
     * @param runInfo
     * @param filterEvaluator The type of FilterEvaluator chosen for the expriments
     * @param wrapperEvaluator The type of wrapper evaluator chosen for the experiments
     * @param dataset The dataset with the "augmented" attributes set (to this object we add the selected attributes)
     * @param originalDatasetTrainingFolds The training folds and the test fold (the original partitioning of the data)
     * @param subFoldTrainingDatasets Only the training folds (a subset of the previous parameter)
     * @param currentClassificationProbs The probabilities assigned to each instance by the classifier of belonging to each of the classes
     * @param oam Manages the applying of the various operators on the attributes
     * @param candidateAttributes The attributes that are being ocnsidered for adding to the dataset
     * @param iterationsCounter
     * @param columnsAddedInthePreviousIteration The attriubtes that were already added to the dataset
     * @throws Exception
     */
    private void performIterativeSearch(Dataset originalDataset, String runInfo, FilterEvaluator filterEvaluator, WrapperEvaluator wrapperEvaluator,
                                        Dataset dataset, List<Dataset> originalDatasetTrainingFolds, List<Dataset> subFoldTrainingDatasets, List<ClassificationResults> currentClassificationProbs,
                                        OperatorsAssignmentsManager oam, List<OperatorAssignment> candidateAttributes, int iterationsCounter, List<ColumnInfo> columnsAddedInthePreviousIteration) throws Exception {
        int totalNumOfWrapperEvaluations = 0;

        AttributeRankerFilter rankerFilter = getRankerFilter(properties.getProperty("rankerApproach"));

        while (iterationsCounter <= this.maxIterations) {
            filterEvaluator.recalculateDatasetBasedFeatures(originalDataset, properties);
            Date date = new Date();
            System.out.println("Starting search iteration " + Integer.toString(iterationsCounter) + " : " + date.toString());


            //recalculte the filter evaluator score of the existing attributes
            oam.recalculateFilterEvaluatorScores(dataset,candidateAttributes,subFoldTrainingDatasets,filterEvaluator,currentClassificationProbs);

            //now we generate all the candidate features
            date = new Date(); System.out.println("            Starting feature generation : " + date.toString());
            candidateAttributes.addAll(oam.applyNonUnaryOperators(dataset, columnsAddedInthePreviousIteration, filterEvaluator, subFoldTrainingDatasets, currentClassificationProbs));
            date = new Date(); System.out.println("            Finished feature generation : " + date.toString());

            //Sort the candidates by their initial (filter) score and test them using the wrapper evaluator
            candidateAttributes = rankerFilter.rankAndFilter(candidateAttributes,columnsAddedInthePreviousIteration,subFoldTrainingDatasets,currentClassificationProbs);

            System.out.println("            Starting wrapper evaluation : " + date.toString());
            evaluatedAttsCounter = 0;
            chosenOperatorAssignment = null;
            topRankingAssignment = null;

            terminateSearch = false;
            ReentrantLock wrapperResultsLock = new ReentrantLock();
            int numOfThreads = Integer.parseInt(properties.getProperty("numOfThreads"));

            final List<ClassificationResults> localCurrentClassificationProbs = currentClassificationProbs;
            for (int i=0; i< candidateAttributes.size(); i+=numOfThreads) {
                if (chosenOperatorAssignment != null) {
                    break;
                }
                List<OperatorAssignment> oaList = candidateAttributes.subList(i,i+Math.min(numOfThreads, candidateAttributes.size()-i));
                oaList.parallelStream().forEach(oa -> {
                    try {
                        if (oa.getFilterEvaluatorScore() != Double.MIN_VALUE && evaluatedAttsCounter <= Double.parseDouble(properties.getProperty("maxNumOfWrapperEvaluationsPerIteration"))
                                && oa.getFilterEvaluatorScore() > 0.001) {
                            double score = oam.applyOperatorAndPerformWrapperEvaluation(originalDatasetTrainingFolds, oa, wrapperEvaluator, localCurrentClassificationProbs, null);
                            oa.setWrapperEvaluatorScore(score);

                            wrapperResultsLock.lock();
                            evaluatedAttsCounter++;

                            //we want to keep tabs on the OA with the best observed wrapper performance
                            if (topRankingAssignment == null || topRankingAssignment.getWrapperEvaluatorScore() < score) {
                                topRankingAssignment = oa;
                            }

                            if (isStoppingCriteriaMet(filterEvaluator, wrapperEvaluator, oa, score, topRankingAssignment)) {
                                chosenOperatorAssignment = oa;
                            }
                            if ((evaluatedAttsCounter % 100) == 0) {
                                Date currentDate = new Date();
                                System.out.println("                     Evaluated : " + evaluatedAttsCounter + "attributes:   " + currentDate.toString());
                            }
                            wrapperResultsLock.unlock();
                        }
                    }
                    catch (Exception ex) {

                    }
                });
            }
            System.out.println("            Finished wrapper evaluation : " + date.toString());

            //Sum the number of evaluated attributes into the global counter
            totalNumOfWrapperEvaluations += evaluatedAttsCounter;

            //check if the chosenOperatorAssignment parameter contains a value.
            if (chosenOperatorAssignment == null) {
                if (topRankingAssignment != null) {
                    chosenOperatorAssignment = topRankingAssignment;
                }
                else {
                    System.out.println("No attributes available. Terminating search.");
                    break;
                }
            }

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
            wrapperEvaluator.EvaluationAndWriteResultsToFile(originalDataset, chosenOperatorAssignment.getName(), iterationsCounter, runInfo, false, evaluatedAttsCounter, chosenOperatorAssignment.getFilterEvaluatorScore() ,chosenOperatorAssignment.getWrapperEvaluatorScore(), properties);
            iterationsCounter++;
        }

        //some cleanup, if required
        filterEvaluator.deleteBackgroundClassificationModel(originalDataset,properties);

        //After the search process is over, write the total amount of time spent and the number of wrapper evaluations that were conducted
        writeFinalStatisticsToResultsFile(dataset.getName(), runInfo,experimentStartDate,totalNumOfWrapperEvaluations);

    }

    /**
     * Determines whether to terminate the wrapper evaluation of the candidates. If returns "true", it also
     * sets the value of the chosenOperatorAssignment parameter that contains the attribute that will be added
     * to the dataset
     * @param filterEvaluator
     * @param wrapperEvaluator
     * @param currentAssignment
     * @param score
     * @param topRankingAssignment
     * @return
     */
    private boolean isStoppingCriteriaMet(FilterEvaluator filterEvaluator, WrapperEvaluator wrapperEvaluator,
                   OperatorAssignment currentAssignment, double score, OperatorAssignment topRankingAssignment) {
        if (score > 0.01) {
            return true;
        }

        return false;
    }
}
