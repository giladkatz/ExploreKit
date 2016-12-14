package explorekit.Evaluation.FilterEvaluators;

import explorekit.Evaluation.ClassificationResults;
import explorekit.Evaluation.Evaluator;
import explorekit.data.Column;
import explorekit.data.ColumnInfo;
import explorekit.data.Dataset;
import explorekit.operators.OperatorAssignment;
import explorekit.operators.OperatorsAssignmentsManager;
import explorekit.operators.UnaryOperators.EqualRangeDiscretizerUnaryOperator;

import java.io.File;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

/**
 * Created by giladkatz on 20/02/2016.
 */
public abstract class FilterEvaluator implements Evaluator {

    public Evaluator.evaluatorType getType() {
        return Evaluator.evaluatorType.Filter;
    }

    //Unlike wrapper methods, we don't evaluate the entire dataset but rather a subset of attributes.
    //For this reason, we need to know what they are.
    protected List<ColumnInfo> analyzedColumns = new ArrayList<>();

    /**
     * Designed for datasets where an object is represented by multiple lines rather than one (for example, a user that is
     * represented by multiple activities). If an object is represented by multiple lines, some values have to remain identical
     * throught all lines. This function ensures that this is the case.
     * @param dataset
     * @param currentScore
     * @param oa
     * @param candidateAttribute
     * @return
     * @throws Exception
     */
    abstract double produceScoreWithDistinctValues(Dataset dataset, ClassificationResults currentScore, OperatorAssignment oa, ColumnInfo candidateAttribute) throws Exception;

    public void initFilterEvaluator(List<ColumnInfo> columnsToAnalyze) throws Exception{
        analyzedColumns = columnsToAnalyze;
    }

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

    public void recalculateDatasetBasedFeatures(Dataset analyzedDatasets, Properties properties) throws Exception {
        //Curretnly only needed for the ML-Based filter evaluatur
    }

    public void discretizeColumns(Dataset dataset, double[] bins) throws Exception {
        for (int i=0; i<analyzedColumns.size(); i++) {
            ColumnInfo ci = analyzedColumns.get(i);
            if (!ci.getColumn().getType().equals(Column.columnType.Discrete)) {
                EqualRangeDiscretizerUnaryOperator  discretizer = new EqualRangeDiscretizerUnaryOperator(bins);
                List<ColumnInfo> columns = new ArrayList<>();
                columns.add(ci);
                discretizer.processTrainingSet(dataset, columns, null);
                analyzedColumns.set(i, discretizer.generate(dataset, columns, null, false));
            }
        }
    }

    public void deleteBackgroundClassificationModel(Dataset dataset, Properties properties) throws Exception {}

    public abstract FilterEvaluator getCopy();

    public abstract boolean needToRecalculateScoreAtEachIteration();
}
