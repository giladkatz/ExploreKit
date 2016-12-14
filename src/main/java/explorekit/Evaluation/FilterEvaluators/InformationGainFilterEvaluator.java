package explorekit.Evaluation.FilterEvaluators;

import explorekit.Evaluation.ClassificationResults;
import explorekit.data.ColumnInfo;
import explorekit.data.Dataset;
import explorekit.data.DiscreteColumn;
import explorekit.operators.OperatorAssignment;

import java.util.HashMap;
import java.util.List;
import java.util.Properties;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Created by giladkatz on 20/02/2016.
 */
public class InformationGainFilterEvaluator extends FilterEvaluator {

    protected HashMap<List<Integer>, int[]> valuesPerKey = new HashMap<>();

    public InformationGainFilterEvaluator() {}

    public evaluatorScoringMethod getEvaluatorScoreingMethod() {
        return evaluatorScoringMethod.InformationGain;
    }

    public double produceScoreWithDistinctValues(Dataset dataset, ClassificationResults currentScore, OperatorAssignment oa, ColumnInfo candidateAttribute) throws Exception {
        try {
            valuesPerKey = new HashMap<>();
            ColumnInfo targetColumn = dataset.getTargetClassColumn();

            for (int i : dataset.getTestFoldsDistinctValRepresentatives()) {
                final int j = i;
                List<Integer> sourceValues = analyzedColumns.stream().map(c -> (Integer) c.getColumn().getValue(j)).collect(Collectors.toList());
                int targetValue = (int) targetColumn.getColumn().getValue(i);
                if (!valuesPerKey.containsKey(sourceValues)) {
                    valuesPerKey.put(sourceValues, new int[((DiscreteColumn) dataset.getTargetClassColumn().getColumn()).getNumOfPossibleValues()]);
                }
                valuesPerKey.get(sourceValues)[targetValue] += 1;
            }
            return calculateIG(dataset);
        }
        catch (Exception ex) {
            throw new Exception("failure to evaluate");
        }
    }

    public double produceScore(Dataset analyzedDatasets, ClassificationResults currentScore, Dataset completeDataset, OperatorAssignment oa, ColumnInfo candidateAttribute, Properties properties) throws Exception {

        if (candidateAttribute != null) {
            analyzedDatasets.addColumn(candidateAttribute);
        }

        //if any of the analyzed attribute is not discrete, it needs to be discretized
        double[] bins = new double[10];
        super.discretizeColumns(analyzedDatasets, bins);
        if (analyzedDatasets.getDistinctValueColumns() != null && analyzedDatasets.getDistinctValueColumns().size() > 0) {
            return produceScoreWithDistinctValues(analyzedDatasets, currentScore, oa, candidateAttribute);
        }

        valuesPerKey = new HashMap<>();
        ColumnInfo targetColumn = analyzedDatasets.getTargetClassColumn();

        //In filter evaluators we evaluate the test set, the same as we do in wrappers. The only difference here is that we
        //train and test on the test set directly, while in the wrappers we train a model on the training set and then apply on the test set
        for (int i =0; i<analyzedDatasets.getNumOfTestDatasetRows(); i++) {
            final int j = analyzedDatasets.getIndicesOfTestInstances().get(i);
            List<Integer> sourceValues = analyzedColumns.stream().map(c -> (Integer) c.getColumn().getValue(j)).collect(Collectors.toList());
            int targetValue = (int)targetColumn.getColumn().getValue(j);
            if (!valuesPerKey.containsKey(sourceValues)) {
                valuesPerKey.put(sourceValues, new int[((DiscreteColumn)analyzedDatasets.getTargetClassColumn().getColumn()).getNumOfPossibleValues()]);
            }
            valuesPerKey.get(sourceValues)[targetValue] += 1;
        }
        return calculateIG(analyzedDatasets);
    }

    private double calculateIG(Dataset dataset) {
        double IG = 0;
        for (int[] val: valuesPerKey.values()) {
            double numOfInstances = IntStream.of(val).sum();
            double tempIG = 0;
            for (int i=0; i<val.length; i++) {
                if (val[i] != 0) {
                    tempIG += -((val[i] / numOfInstances) * Math.log10(val[i] / numOfInstances));
                }
            }
            IG += (numOfInstances/dataset.getNumOfTrainingDatasetRows()) * tempIG;
        }
        return IG;
    }

    public FilterEvaluator getCopy() {
        return new InformationGainFilterEvaluator();
    }

    public boolean needToRecalculateScoreAtEachIteration() {
        return false;
    }
}
