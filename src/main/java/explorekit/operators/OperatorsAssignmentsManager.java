package explorekit.operators;

import explorekit.Evaluation.ClassificationResults;
import explorekit.Evaluation.FilterEvaluators.FilterEvaluator;
import explorekit.Evaluation.FilterEvaluators.InformationGainFilterEvaluator;
import explorekit.Evaluation.WrapperEvaluation.WrapperEvaluator;
import explorekit.data.Column;
import explorekit.data.ColumnInfo;
import explorekit.data.Dataset;
import explorekit.operators.BinaryOperators.AddBinaryOperator;
import explorekit.operators.BinaryOperators.DivisionBinaryOperator;
import explorekit.operators.BinaryOperators.MultiplyBinaryOperator;
import explorekit.operators.BinaryOperators.SubtractBinaryOperator;
import explorekit.operators.GroupByThenOperators.*;
import explorekit.operators.TimeBasedGroupByThenOperators.*;
import explorekit.operators.UnaryOperators.*;
import weka.classifiers.Classifier;
import weka.classifiers.functions.SMO;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;

import java.io.*;
import java.security.MessageDigest;
import java.util.*;


/**
 * Created by giladkatz on 29/02/2016.
 */
public class OperatorsAssignmentsManager {

    public static Properties properties;

    public OperatorsAssignmentsManager(Properties properties) throws Exception {
        this.properties = properties;
    }

    /**
     * Activates the applyOperatorsAndPerformInitialEvaluation function, but only for Unary Operators
     * @param dataset
     * @param mustIncluseAttributes Attributes which must be in either the source or the target of every generated feature
     */
    public static List<OperatorAssignment> applyUnaryOperators(Dataset dataset, List<ColumnInfo> mustIncluseAttributes,
                         FilterEvaluator filterEvaluator, List<Dataset> subFoldTrainingDatasets, List<ClassificationResults> currentScores) throws Exception {
        List<Operator> unaryOperatorsList = getUnaryOperatorsList();
        return applyOperatorsAndPerformInitialEvaluation(dataset, unaryOperatorsList,mustIncluseAttributes, 1, filterEvaluator, subFoldTrainingDatasets, currentScores, false);
    }

    /**
     * Activates the applyOperatorsAndPerformInitialEvaluation function, for all operator types by Unary
     * @param dataset
     * @param mustIncluseAttributes Attributes which must be in either the source or the target of every generated feature
     */
    public static List<OperatorAssignment> applyNonUnaryOperators(Dataset dataset, List<ColumnInfo> mustIncluseAttributes,
                    FilterEvaluator filterEvaluator, List<Dataset> subFoldTrainingDatasets, List<ClassificationResults> currentScores) throws Exception {
        List<Operator> nonUnaryOperatorsList = getNonUnaryOperatorsList();
        return applyOperatorsAndPerformInitialEvaluation(dataset, nonUnaryOperatorsList,mustIncluseAttributes,
                Integer.parseInt(properties.getProperty("maxNumOfAttsInOperatorSource")), filterEvaluator, subFoldTrainingDatasets, currentScores, true);
    }

    /**
     * Adds the attribute to the dataset. If the operator that was used to generate the attribute is not unary
     * then we apply all relevant Unary operators to generate additional attirbutes that are added to the dataset
     * @param dataset
     * @param oa
     * @param applyAdditionalUnaryOperators Whether to apply unary operators (discretizers/normalizers) on the chosen
     *                                      feature and add these values to the dataset object (desirable for the dataset
     *                                      that is used for search, not for the dataset containing the final set of
     *                                      attributes)
     */
    public static List<ColumnInfo> addAddtibuteToDataset(Dataset dataset, OperatorAssignment oa, boolean applyAdditionalUnaryOperators, List<ClassificationResults> currentScores) throws Exception {
        List<ColumnInfo> newlyGeneratedColumns = new ArrayList<>();

        //start by generating the new attribute and adding it to the dataset
        ColumnInfo newColumn = generateColumn(dataset, oa, true);
        dataset.addColumn(newColumn);
        newlyGeneratedColumns.add(newColumn);

        //if the operator that was used to generate the attribute was not unary, apply all relevant unary operators and add them to the dataset as well
        if (applyAdditionalUnaryOperators && oa.getOperator().getType() != Operator.operatorType.Unary) {
            List<OperatorAssignment> additionalAttributes = applyUnaryOperators(dataset, newlyGeneratedColumns, null, new ArrayList<>(), currentScores);

            for (OperatorAssignment operatorAssignment : additionalAttributes) {
                ColumnInfo ci = generateColumn(dataset, operatorAssignment, true);
                dataset.addColumn(ci);
                newlyGeneratedColumns.add(ci);
            }
        }
        return newlyGeneratedColumns;
    }

    /**
     * Ranks all current DISCRETE columns in the dataset using a filter evaluator and returns the top X ranking columns (ties are broken randomly)
     * @param dataset
     * @param filterEvaluator
     * @param numOfAttributesToReturn
     * @return
     * @throws Exception
     */
    private static List<ColumnInfo> getTopRankingDiscreteAttributesByFilterScore(Dataset dataset, FilterEvaluator filterEvaluator, int numOfAttributesToReturn) throws Exception {

        TreeMap<Double,List<Integer>> IGScoresPerColumnIndex = new TreeMap<>(Collections.reverseOrder());
        for (int i=0; i<dataset.getAllColumns(false).size(); i++) {
            ColumnInfo ci = dataset.getAllColumns(false).get(i);
            if (dataset.getTargetClassColumn() == ci) {
                continue;
            }

            //if the attribute is string or date, not much we can do about that
            if (ci.getColumn().getType() != Column.columnType.Discrete) {
                continue;
            }

            List<Integer> indicedList = new ArrayList<>();
            indicedList.add(i);
            Dataset replicatedDataset = dataset.emptyReplica();

            List<ColumnInfo> columnsToAnalyze = new ArrayList<>();
            columnsToAnalyze.add(ci);
            filterEvaluator.initFilterEvaluator(columnsToAnalyze);
            double score = filterEvaluator.produceScore(replicatedDataset, null, dataset, null, ci, properties);
            if (!IGScoresPerColumnIndex.containsKey(score)) {
                IGScoresPerColumnIndex.put(score, new ArrayList<>());
            }
            IGScoresPerColumnIndex.get(score).add(i);
        }

        List<ColumnInfo> columnsToReturn = new ArrayList<>();

        for (double score : IGScoresPerColumnIndex.keySet()) {
            for (int index : IGScoresPerColumnIndex.get(score)) {
                List<Integer> tempList = new ArrayList<>();
                tempList.add(index);
                columnsToReturn.add(dataset.getColumns(tempList).get(0));
                if (columnsToReturn.size() >= numOfAttributesToReturn) {
                    return columnsToReturn;
                }
            }
        }
        return columnsToReturn;
    }

    /**
     * Recieves a a dataset and a list of operators, finds all possible combinations, generates and writes the attributes to file
     * and returs the assignments list
     * @param dataset The full dataset. The new attribute generated for it is the one to be saved to file
     * @param operators The operators for which assignments will be generated
     * @param mustIncluseAttributes The attributes that must be present in EITHER the source or the target. Empty lists or null mean there's no restriction
     * @param maxNumOfSourceAttributes The maximal number of attributes that can be in the source (if the operator permits). Smaller number down to 1 (including) will also be generated
     * @param filterEvaluator The filter evaluator that will be used to compute the initial ranking of the attriubte. The calculation is carried out on the sibfolds
     * @param subFoldTrainingDatasets The training set sub-folds. Used in order to calculate the score, as the test set cannot be used for this purpose here.
     * @return
     * @throws Exception
     */
    public static  List<OperatorAssignment> applyOperatorsAndPerformInitialEvaluation(Dataset dataset, List<Operator> operators,
                List<ColumnInfo> mustIncluseAttributes, int maxNumOfSourceAttributes, FilterEvaluator filterEvaluator,
                List<Dataset> subFoldTrainingDatasets, List<ClassificationResults> currentScores, boolean reduceNumberOfAttributes) throws Exception {

        //in case the number of initial attributes is very high, we need narrow the search space
        if (reduceNumberOfAttributes && (mustIncluseAttributes == null || mustIncluseAttributes.size() == 0)) {
            //It is important to break the condition in two, because in advanced interations we always have a "must include" attribute
            if (dataset.getAllColumns(false).size() > 60) {
                InformationGainFilterEvaluator initialSelectionAttEvaluator = new InformationGainFilterEvaluator();
                mustIncluseAttributes = getTopRankingDiscreteAttributesByFilterScore(dataset, initialSelectionAttEvaluator, 10);
            }
        }

        List<OperatorAssignment> operatorAssignments = getOperatorAssignments(dataset, mustIncluseAttributes, operators, maxNumOfSourceAttributes);

        //Create all the new features, save them to file and evaluate them using the filter evaluator
        generateAttributeAndCalculateFilterEvaluatorScore(dataset, filterEvaluator, subFoldTrainingDatasets, currentScores, operatorAssignments);

        /*
        // The single thread version
        for (OperatorAssignment os: operatorAssignments) {
            ColumnInfo ci = generateColumn(dataset, os, true);
            //if the filter evaluator is not null, we'll conduct the initial evaluation of the new attribute
            if (filterEvaluator != null) {
                os.setFilterEvaluatorScore(EvaluateAttributeUsingTrainingSubFolds(subFoldTrainingDatasets, filterEvaluator, os));
            }
        }*/
        return operatorAssignments;
    }

    static int counter = 0;
    private static void generateAttributeAndCalculateFilterEvaluatorScore(Dataset dataset, FilterEvaluator filterEvaluator,
                 List<Dataset> subFoldTrainingDatasets, List<ClassificationResults> currentScores,
                 List<OperatorAssignment> operatorAssignments) throws Exception {
        //System.setProperty("java.util.concurrent.ForkJoinPool.common.parallelism", "1");
        System.out.println("num of attributes to evaluate: " + operatorAssignments.size());
        counter = 0;
        int numOfThread = Integer.parseInt(properties.getProperty("numOfThreads"));

        if (numOfThread > 1) {
            //ReentrantLock attributeGenerationLock = new ReentrantLock();
            //ReentrantLock filterEvaluationLock = new ReentrantLock();
            operatorAssignments.parallelStream().forEach(oa -> {
                try {
                    //attributeGenerationLock.lock();
                    Dataset replicatedDataset = dataset.replicateDataset();
                    counter++;
                    if ((counter % 1000) == 0) {
                        Date date = new Date();
                        System.out.println("analyzed " + counter + " attributes : " + date.toString());
                    }
                    //attributeGenerationLock.unlock();

                    ColumnInfo ci = generateColumn(replicatedDataset, oa, true);
                    //if the filter evaluator is not null, we'll conduct the initial evaluation of the new attribute
                    if ((ci != null) && (filterEvaluator != null)) {
                        //filterEvaluationLock.lock();
                        FilterEvaluator cloneEvaluator = filterEvaluator.getCopy();
                        List<Dataset> replicatedSubFoldsList = new ArrayList<Dataset>();
                        for (Dataset subFoldDataset : subFoldTrainingDatasets) {
                            replicatedSubFoldsList.add(subFoldDataset.replicateDataset());
                        }
                        //filterEvaluationLock.unlock();
                        double filterEvaluatorScore = EvaluateAttributeUsingTrainingSubFolds(replicatedSubFoldsList, cloneEvaluator, oa, currentScores);
                        oa.setFilterEvaluatorScore(filterEvaluatorScore);
                    }
                } catch (Exception ex) {
                    System.out.println("error when generating and evaluating attribute: " + oa.getName());
                    System.out.println("The error: " + ex.getMessage());
                }
            });
        }
        else {
            for (OperatorAssignment oa : operatorAssignments) {
                try {
                    Dataset replicatedDataset = dataset.replicateDataset();
                    counter++;
                    if ((counter % 1000) == 0) {
                        System.out.println("analyzed " + counter + " attributes");
                    }
                    ColumnInfo ci = generateColumn(replicatedDataset, oa, true);
                    //if the filter evaluator is not null, we'll conduct the initial evaluation of the new attribute
                    if ((ci != null) && (filterEvaluator != null)) {
                        FilterEvaluator cloneEvaluator = filterEvaluator.getCopy();
                        List<Dataset> replicatedSubFoldsList = new ArrayList<Dataset>();
                        for (Dataset subFoldDataset : subFoldTrainingDatasets) {
                            replicatedSubFoldsList.add(subFoldDataset.replicateDataset());
                        }
                        double filterEvaluatorScore = -1;
                        try {
                            filterEvaluatorScore = EvaluateAttributeUsingTrainingSubFolds(replicatedSubFoldsList, cloneEvaluator, oa, currentScores);
                        }
                        catch (Exception ex){
                            int x=5;
                        }
                        oa.setFilterEvaluatorScore(filterEvaluatorScore);
                    }
                } catch (Exception ex) {
                    System.out.println("error when generating and evaluating attribute: " + oa.getName());
                    System.out.println("The error: " + ex.getMessage());
                }
            }
        }
    }

    /**
     * Used to recalculate the scores of existing attributes when a new search iteration begins.
     * @param dataset
     * @param candidateAttributes
     * @param subFoldTrainingDatasets
     * @param filterEvaluator
     * @param currentScores
     * @throws Exception
     */
    public static void recalculateFilterEvaluatorScores(Dataset dataset, List<OperatorAssignment> candidateAttributes, List<Dataset> subFoldTrainingDatasets, FilterEvaluator filterEvaluator,
                                                        List<ClassificationResults> currentScores) throws Exception {
        //If the filter is not of a type that requires recalculcation (like IG) then terminate
        if (!filterEvaluator.needToRecalculateScoreAtEachIteration()) {
            return;
        }
        generateAttributeAndCalculateFilterEvaluatorScore(dataset, filterEvaluator, subFoldTrainingDatasets, currentScores, candidateAttributes);
    }

    /**
     * Evaluates a set of datasets using a leave-one-out evaluation
     * @param datasets
     * @param filterEvaluator
     * @param operatorAssignment
     * @param currentScores
     * @return
     * @throws Exception
     */
    private static double EvaluateAttributeUsingTrainingSubFolds(List<Dataset> datasets, FilterEvaluator filterEvaluator,
                                                                 OperatorAssignment operatorAssignment, List<ClassificationResults> currentScores) throws Exception {
        double finalScore = 0;

        for (int i=0; i<datasets.size(); i++) {
            Dataset dataset = datasets.get(i);
            ClassificationResults currentScore = null;
            if (currentScores != null) {
                currentScore = currentScores.get(i);
            }
            ColumnInfo ci = generateColumn(dataset, operatorAssignment, false);
            if (ci == null) {
                return Double.MIN_VALUE;
            }
            List<ColumnInfo> tempList = new ArrayList<>();
            tempList.add(ci);
            filterEvaluator.initFilterEvaluator(tempList);
            Dataset datasetEmptyReplica = dataset.emptyReplica();

            try {
                finalScore += filterEvaluator.produceScore(datasetEmptyReplica, currentScore, dataset, operatorAssignment, ci, properties);
            }
            catch (Exception ex) {
                finalScore += filterEvaluator.produceScore(datasetEmptyReplica, currentScore, dataset, operatorAssignment, ci, properties);
            }
        }
        return (finalScore/datasets.size());
    }

    /**
     * Generates/retrieves the attribute specified, adds it to a replica of the dataset and calculates the score
     * based on the provided wrapper method
     * @param datasets
     * @param operatorAssignment
     * @param wrapperEvaluator
     * @return
     * @throws Exception
     */
    public static double applyOperatorAndPerformWrapperEvaluation(List<Dataset> datasets, OperatorAssignment operatorAssignment,
                                                                   WrapperEvaluator wrapperEvaluator, List<ClassificationResults> currentScores, Dataset completeDataset) throws Exception{
        double score = 0;
        for (int i=0; i<datasets.size(); i++) {
            Dataset dataset = datasets.get(i);
            ClassificationResults currentScore = null;
            if (currentScores != null) {
                currentScore = currentScores.get(i);
            }
            Dataset datasetReplica = dataset.replicateDataset();
            ColumnInfo ci = generateColumn(datasetReplica, operatorAssignment, true);
            //datasetReplica.addColumn(ci);
            double iterationScore = wrapperEvaluator.produceScore(datasetReplica, currentScore, completeDataset, operatorAssignment, ci, properties);
            score += iterationScore;
        }
        return score/(datasets.size());
    }

    public static double applyOperatorAndPerformWrapperEvaluationWithSampling(List<Dataset> datasets, OperatorAssignment operatorAssignment,
                                                                   WrapperEvaluator wrapperEvaluator, List<ClassificationResults> currentScores,
                                                                              int numOfTimesToSamplePerFold, int numOfInstancesPerSampling, int randomSeed) throws Exception{
        double score = 0;
        for (int i=0; i<datasets.size(); i++) {
            Dataset dataset = datasets.get(i);
            ClassificationResults currentScore = null;
            if (currentScores != null) {
                currentScore = currentScores.get(i);
            }
            Dataset datasetReplica = dataset.replicateDataset();
            ColumnInfo ci = generateColumn(datasetReplica, operatorAssignment, true);
            datasetReplica.addColumn(ci);
            double iterationScore = wrapperEvaluator.produceScoreWithSampling(datasetReplica, currentScore,numOfTimesToSamplePerFold, numOfInstancesPerSampling, randomSeed, properties);
            score += iterationScore;
        }
        return score/(datasets.size());
    }

    /**
     * Receives a dataset and a list of OperatorAssignment objects, generates/gets them from file and
     * adds them to the dataset
     * @param dataste
     * @param oaList
     * @throws Exception
     */
    public static  void GenerateAndAddColumnToDataset(Dataset dataste, List<OperatorAssignment> oaList) throws  Exception {
        for (OperatorAssignment oa : oaList) {
            ColumnInfo ci = generateColumn(dataste, oa, true);
            dataste.addColumn(ci);
        }
    }

    /**
     * Creates the new attribute. Also writes it to a file.
     * @param dataset
     * @param finalAttribute indicates if this is the version that is generated from the COMPLETE training set. This
     *                       is the only version that needs to be written or read from the file system
     * @param os
     */
    public static ColumnInfo generateColumn(Dataset dataset, OperatorAssignment os, boolean finalAttribute) throws Exception {
        boolean writeToFile = false;
        try {
            ColumnInfo ci = null;
            if (finalAttribute && writeToFile) {
                ci = readColumnInfoFromFile(dataset.getName(), os.getName());
            }
            if (ci == null) {
                Operator operator = null;
                try {
                    operator = getOperator(os.getOperator());
                }
                catch (Exception ex) {
                    System.out.println("Sleeping");
                    Thread.sleep(100);
                    operator = getOperator(os.getOperator());
                }

                operator.processTrainingSet(dataset, os.getSources(), os.getTragets());

                try {
                    ci = operator.generate(dataset, os.getSources(), os.getTragets(), true);
                }
                catch (Exception ex) {
                    int x=5;
                }

                if (ci != null && os != null && os.getSecondaryOperator() != null) {
                    Dataset replica = dataset.emptyReplica();
                    replica.addColumn(ci);
                    UnaryOperator uOperator = os.getSecondaryOperator();
                    List<ColumnInfo> tempList = new ArrayList<>();
                    tempList.add(ci);
                    try {
                        uOperator.processTrainingSet(replica, tempList, null);
                        ColumnInfo ci2 = uOperator.generate(replica, tempList, null, true);
                        ci = ci2;
                    }
                    catch (Exception ex) {

                    }

                }
                if (finalAttribute && writeToFile) {
                    //write the column to file, so we don't have to calculate it again
                    writeColumnInfoToFile(dataset.getName(), os.getName(), ci);
                }
            }
            if (ci == null) {

            }
            return ci;
        }
        catch (Exception ex) {
            Operator operator = getOperator(os.getOperator());
            operator.processTrainingSet(dataset, os.getSources(), os.getTragets());
            System.out.println("Error while generating column: " + ex.getMessage());
            throw new Exception("Failure to generate column");
        }
    }

    /**
     * Read a ColumnInfo object from file
     * @param datasetName
     * @param operatorAssignmentName
     * @return
     * @throws Exception
     */
    public static  ColumnInfo readColumnInfoFromFile(String datasetName, String operatorAssignmentName) throws Exception {
        String fileName = getHashedName(datasetName + operatorAssignmentName) + ".ser";
        String filePath = properties.get("operatorAssignmentFilesLocation") + fileName;

        File file = new File(filePath);
        if (!file.exists()) {
            return null;
        }
        FileInputStream streamIn = new FileInputStream(filePath);
        ObjectInputStream objectinputstream = new ObjectInputStream(streamIn);
        try {
            ColumnInfo ci = (ColumnInfo) objectinputstream.readObject();
            return ci;
        }
        catch (Exception ex) {
            System.out.println("Error reading ColumnInfo from file");
        }
        return null;
    }

    /**
     * Writes a ColumnInfo object to file
     * @param ci
     */
    public static  void writeColumnInfoToFile(String datasetName, String operatorAssignmentName, ColumnInfo ci) throws Exception {
        String fileName = getHashedName(datasetName + operatorAssignmentName) + ".ser";
        FileOutputStream fout = new FileOutputStream(properties.getProperty("operatorAssignmentFilesLocation") + fileName, true);
        ObjectOutputStream oos = new ObjectOutputStream(fout);
        oos.writeObject(ci);
    }

    /**
     * Used to get hashes for the written ColumnInfo objects
     * @param name
     * @return
     */
    private static  String getHashedName(String name) throws Exception {
        MessageDigest md = MessageDigest.getInstance("MD5");
        md.update(name.getBytes());
        byte[] digest = md.digest();
        StringBuffer sb = new StringBuffer();
        for (byte b : digest) {
            sb.append(String.format("%02x", b & 0xff));
        }
        return sb.toString();
    }

    /**
     * Receives a dataset with a set of attributes and a list of operators and generates all possible source/target/operator/secondary operator assignments
     * @param dataset The dataset with the attributes that need to be analyzed
     * @param attributesToInclude A list of attributes that must be included in either the source or target of every generated assignment. If left empty, there are no restrictions
     * @param operators A list of all the operators whose assignment will be considered
     * @param maxCombinationSize the maximal number of attributes that can be a in the source of each operator. Smaller number (down to 1) are also considered
     * @return
     */
    public static  List<OperatorAssignment> getOperatorAssignments(Dataset dataset, List<ColumnInfo> attributesToInclude, List<Operator> operators,
                                                                   int maxCombinationSize) throws Exception {
        boolean areNonUniaryOperatorsBeingUsed = false;
        if (operators.size() > 0 && !operators.get(0).getType().equals(Operator.operatorType.Unary)) {
            areNonUniaryOperatorsBeingUsed = true;
        }

        if (attributesToInclude == null) {attributesToInclude = new ArrayList<>();}
        List<OperatorAssignment> operatorsAssignments = new ArrayList();
        for (int i=maxCombinationSize; i>0; i--) {
            List<List<ColumnInfo>> sourceAttributeCombinations = getAttributeCombinations(dataset.getAllColumns(false), i);

            //for each of the candidate source attributes combinations
            for (List<ColumnInfo> sources: sourceAttributeCombinations) {
                //if a distinct dolumn(s) exists, we need to make sure that at least one column (or one of its ancestors) satisfies the constraint
                if (dataset.getDistinctValueColumns() != null && dataset.getDistinctValueColumns().size() > 0) {
                    if (areNonUniaryOperatorsBeingUsed && !isDistinctValueCompliantAttributeExists(dataset.getDistinctValueCompliantColumns(), sources)) {
                        continue;
                    }
                }

                //first check if any of the required atts (if there are any) are included
                if (attributesToInclude.size() > 0) {
                    ArrayList<ColumnInfo> tempList = new ArrayList<>(sources);
                    tempList.retainAll(attributesToInclude);
                    if (tempList.size() == 0) { continue; }
                }

                //Now we check all the operators on the source attributes alone.
                for (Operator operator: operators) {
                    if (operator.isApplicable(dataset, sources, new ArrayList<ColumnInfo>())) {
                        OperatorAssignment os = new OperatorAssignment(sources, null, getOperator(operator), null);
                        operatorsAssignments.add(os);
                    }

                    //now we pair the source attributes with a target attribute and check again
                    for (ColumnInfo targetColumn : dataset.getAllColumns(false)) {
                        //if (sources.contains(targetColumn)) { continue; }
                        if (overlapExistsBetweenSourceAndTargetAttributes(sources,targetColumn)) { continue; }
                        List<ColumnInfo> tempList = new ArrayList<>();
                        tempList.add(targetColumn);
                        if (operator.isApplicable(dataset, sources, tempList)) {
                            OperatorAssignment os = new OperatorAssignment(sources, tempList, getOperator(operator), null);
                            operatorsAssignments.add(os);
                        }
                    }
                }
            }
        }

        //Finally, we go over all the operator assignments. For every assignment that is not performed on
        //an unary operator, we check if any of the unary operators can be applied on it.
        List<OperatorAssignment> additionalAssignments = new ArrayList<>();
        for (OperatorAssignment os : operatorsAssignments) {
            if (os.getOperator().getType() != Operator.operatorType.Unary) {
                for (Operator operator : getUnaryOperatorsList()) {
                    if (operator.getType().equals(Operator.operatorType.Unary)) {
                        UnaryOperator tempOperator = (UnaryOperator)operator;
                        if (tempOperator.requiredInputType().equals(os.getOperator().getOutputType())) {
                            OperatorAssignment additionalAssignment = new OperatorAssignment(os.getSources(), os.getTragets(), os.getOperator(), tempOperator);
                            additionalAssignments.add(additionalAssignment);
                        }
                    }
                }
            }
        }
        operatorsAssignments.addAll(additionalAssignments);
        return operatorsAssignments;
    }

    /**
     * For a given set of attributes, the function determines whether at least on attribute (or one of its ancestors) satisfies the distinct value constraint
     * @param distinctValueCompliantColumns The original coulumns that satisfy the distinct value constraint
     * @param columns the list of columns that make up the source attributes of the currently analyzed attribute
     * @return
     */
    private static boolean isDistinctValueCompliantAttributeExists(List<ColumnInfo> distinctValueCompliantColumns, List<ColumnInfo> columns) {
        for (ColumnInfo ci : columns) {
            if (distinctValueCompliantColumns.contains(ci) ||
                    ((ci.getSourceColumns() != null) && isDistinctValueCompliantAttributeExists(distinctValueCompliantColumns, ci.getSourceColumns()))) {
                return true;
            }
        }
        return false;
    }

    private static boolean overlapExistsBetweenSourceAndTargetAttributes(List<ColumnInfo> sourceAtts, ColumnInfo targetAtt) {
        //the simplest case - the same attribute appears both in the source and the target
        if (sourceAtts.contains(targetAtt)) {
            return true;
        }

        //Now we need to check that the source atts and the target att has no shared columns (including after the application of an operator)
        List<ColumnInfo> sourceAttsAndAncestors = new ArrayList<>();
        for (ColumnInfo sourceAtt: sourceAtts) {
            sourceAttsAndAncestors.add(sourceAtt);
            if (sourceAtt.getSourceColumns() != null) {
                for (ColumnInfo ancestorAtt : sourceAtt.getSourceColumns()) {
                    if (!sourceAttsAndAncestors.contains(ancestorAtt)) {
                        sourceAttsAndAncestors.add(ancestorAtt);
                    }
                }
            }
            if (sourceAtt.getTargetColumns() != null) {
                for (ColumnInfo ancestorAtt : sourceAtt.getTargetColumns()) {
                    if (!sourceAttsAndAncestors.contains(ancestorAtt)) {
                        sourceAttsAndAncestors.add(ancestorAtt);
                    }
                }
            }
        }

        //do the same for the target att (because we only have one we don't need the external loop)
        List<ColumnInfo> targetAttsAndAncestors = new ArrayList<>();
        targetAttsAndAncestors.add(targetAtt);
        if (targetAtt.getSourceColumns() != null) {
            for (ColumnInfo ancestorAtt : targetAtt.getSourceColumns()) {
                if (!targetAttsAndAncestors.contains(ancestorAtt)) {
                    targetAttsAndAncestors.add(ancestorAtt);
                }
            }
        }
        if (targetAtt.getTargetColumns() != null) {
            for (ColumnInfo ancestorAtt : targetAtt.getTargetColumns()) {
                if (!targetAttsAndAncestors.contains(ancestorAtt)) {
                    targetAttsAndAncestors.add(ancestorAtt);
                }
            }
        }

        boolean overlap =  !Collections.disjoint(sourceAttsAndAncestors, targetAttsAndAncestors);

        if (overlap &&  targetAttsAndAncestors.size() > 1) {
            int x=5;
        }

        return overlap;
    }

    /**
     * Returns lists of column-combinations
     * @param attributes
     * @param numOfAttributesInCombination
     * @return
     */
    private static  List<List<ColumnInfo>> getAttributeCombinations(List<ColumnInfo> attributes, int numOfAttributesInCombination) {
        List<List<ColumnInfo>> attributeCombinations = new ArrayList<>();
        CombinationGenerator gen = new CombinationGenerator (attributes.size(), numOfAttributesInCombination);
        while (gen.hasMore ()) {
            int[] indices = gen.getNext();
            List<ColumnInfo> tempColumns = new ArrayList<>();
            for (int index: indices) {
                tempColumns.add(attributes.get(index));
            }
            attributeCombinations.add(tempColumns);
        }
        return attributeCombinations;
    }

    /**
     * Returns a list of unary operators from the configuration file
     * @return
     */
    public static  List<Operator> getUnaryOperatorsList() throws Exception {
        String[] operatorNames = properties.getProperty("unaryOperators").split(",");
        List<Operator> unaryOperatorsList = new ArrayList();
        for (String unaryOperator: operatorNames) {
            UnaryOperator uo = getUnaryOperator(unaryOperator);
            unaryOperatorsList.add(uo);
        }
        return unaryOperatorsList;
    }

    /**
     * Returns a list of nonUnary operators from the configuration file (i.e. all other operator types)
     * @return
     * @throws Exception
     */
    public static  List<Operator> getNonUnaryOperatorsList() throws Exception {
        String[] operatorNames = properties.getProperty("nonUnaryOperators").split(",");
        List<Operator> operatorsList = new ArrayList();
        for (String unaryOperator: operatorNames) {
            Operator operator = getNonUnaryOperator(unaryOperator);
            operatorsList.add(operator);
        }
        return operatorsList;
    }

    /**
     * Gets a new copy of the provided operator
     * @param operator
     * @return
     * @throws Exception
     */
    private static Operator getOperator(Operator operator) throws Exception {
        if (operator.getType().equals(Operator.operatorType.Unary)) {
            return getUnaryOperator(operator.getName());
        }
        return getNonUnaryOperator(operator.getName());
    }

    /**
     * Returns an unary operator by name
     * @param operatorName
     * @return
     * @throws Exception
     */
    private static  UnaryOperator getUnaryOperator(String operatorName) throws Exception {
        switch (operatorName) {
            case "EqualRangeDiscretizerUnaryOperator":
                double[] bins = new double[Integer.parseInt(properties.getProperty("equalRangeDiscretizerBinsNumber"))];
                EqualRangeDiscretizerUnaryOperator erd = new EqualRangeDiscretizerUnaryOperator(bins);
                return erd;
            case "StandardScoreUnaryOperator":
                StandardScoreUnaryOperator ssuo = new StandardScoreUnaryOperator();
                return ssuo;
            case "DayOfWeekUnaryOperator":
                DayOfWeekUnaryOperator dowuo = new DayOfWeekUnaryOperator();
                return dowuo;
            case "HourOfDayUnaryOperator":
                HourOfDayUnaryOperator hoduo = new HourOfDayUnaryOperator();
                return hoduo;
            case "IsWeekendUnaryOperator":
                IsWeekendUnaryOperator iwuo = new IsWeekendUnaryOperator();
                return iwuo;
            default:
                throw new Exception("unindentified unary operator: " + operatorName);
        }
    }

    /**
     * Returns a non-unary operator by name
     * @param operatorName
     * @return
     * @throws Exception
     */
    private static  Operator getNonUnaryOperator(String operatorName) throws Exception {

        double timeSpan = 0;
        if (operatorName.startsWith("TimeBasedGroupByThen")) {
            timeSpan = Double.parseDouble(operatorName.split("_")[1]);
            operatorName = operatorName.split("_")[0];
        }

        switch (operatorName) {
            //GroupByThenOperators
            case "GroupByThenAvg":
                GroupByThenAvg gbtAvg = new GroupByThenAvg();
                return gbtAvg;
            case "GroupByThenMax":
                GroupByThenMax gbtMmax = new GroupByThenMax();
                return gbtMmax;
            case "GroupByThenMin":
                GroupByThenMin gbtMin = new GroupByThenMin();
                return gbtMin;
            case "GroupByThenCount":
                GroupByThenCount gbtCount = new GroupByThenCount();
                return gbtCount;
            case "GroupByThenStdev":
                GroupByThenStdev gbtStdev = new GroupByThenStdev();
                return gbtStdev;

            //BinaryOperators
            case "AddBinaryOperator":
                AddBinaryOperator abo = new AddBinaryOperator();
                return abo;
            case "SubtractBinaryOperator":
                SubtractBinaryOperator sbo = new SubtractBinaryOperator();
                return sbo;
            case "MultiplyBinaryOperator":
                MultiplyBinaryOperator mbo = new MultiplyBinaryOperator();
                return mbo;
            case "DivisionBinaryOperator":
                DivisionBinaryOperator dbo = new DivisionBinaryOperator();
                return dbo;

            //TimeBasedGroupByThen
            case "TimeBasedGroupByThenCountAndAvg":
                TimeBasedGroupByThenCountAndAvg tbgbycaa = new TimeBasedGroupByThenCountAndAvg(timeSpan);
                return tbgbycaa;
            case "TimeBasedGroupByThenCountAndCount":
                TimeBasedGroupByThenCountAndCount tbgbtccac = new TimeBasedGroupByThenCountAndCount(timeSpan);
                return tbgbtccac;
            case "TimeBasedGroupByThenCountAndMax":
                TimeBasedGroupByThenCountAndMax tbgbtcam = new TimeBasedGroupByThenCountAndMax(timeSpan);
                return tbgbtcam;
            case "TimeBasedGroupByThenCountAndMin":
                TimeBasedGroupByThenCountAndMin tbgbtcamm = new TimeBasedGroupByThenCountAndMin(timeSpan);
                return tbgbtcamm;
            case "TimeBasedGroupByThenCountAndStdev":
                TimeBasedGroupByThenCountAndStdev tbgbtcas = new TimeBasedGroupByThenCountAndStdev(timeSpan);
                return tbgbtcas;
            default:
                throw new Exception("unindentified unary operator: " + operatorName);
        }
    }

    /**
     * Used to obtain the requested classifier
     * @param classifier
     * @return
     * @throws Exception
     */
    public Classifier getClassifier(String classifier) throws Exception{
        switch (classifier) {
            case "J48":
                //more commonly known as C4.5
                J48 j48 = new J48();
                return j48;
            case "SVM":
                SMO svm = new SMO();
                return svm;
            case "RandomForest":
                RandomForest randomForest = new RandomForest();
                return randomForest;
            default:
                throw new Exception("unknown classifier");

        }
    }
}
