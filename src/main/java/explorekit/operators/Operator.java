package explorekit.operators;

import explorekit.data.ColumnInfo;
import explorekit.data.Dataset;

import java.util.HashMap;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Created by giladkatz on 12/02/2016.
 */
public abstract class Operator {

    public enum operatorType { Unary, Binary, GroupByThen, TimeBasedGroupByThen }
    public enum outputType { Numeric, Discrete, Date }

    public abstract void processTrainingSet(Dataset dataset, List<ColumnInfo> sourceColumns, List<ColumnInfo> targetColumns) throws Exception;

    public abstract ColumnInfo generate(Dataset dataset, List<ColumnInfo> sourceColumns, List<ColumnInfo> targetColumns, boolean enforceDistinctVal) throws Exception;

    public abstract boolean isApplicable(Dataset dataset, List<ColumnInfo> sourceColumns, List<ColumnInfo> targetColumns);

    public abstract operatorType getType();

    public abstract outputType getOutputType();

    public abstract String getName();

    /**
     * Used to determine whether the values of a column are in accordance with the distinct val requirement.
     * @param dataset
     * @param evaluatedColumn
     * @return
     */
    public boolean isDistinctValEnforced(Dataset dataset, ColumnInfo evaluatedColumn) {
        if (dataset.getDistinctValueColumns().size() == 0) {
            return true;
        }

        HashMap<Object, Object> distinctValsDict = new HashMap<>();
        int numOfRows = dataset.getNumberOfRows();

        for (int i = 0; i < numOfRows; i++) {
            int j = dataset.getIndices().get(i);
            List<Object> sourceValues = dataset.getDistinctValueColumns().stream().map(c -> c.getColumn().getValue(j)).collect(Collectors.toList());
            if (!distinctValsDict.containsKey(sourceValues)) {
                distinctValsDict.put(sourceValues, evaluatedColumn.getColumn().getValue(j));
            } else {
                if (!distinctValsDict.get(sourceValues).equals(evaluatedColumn.getColumn().getValue(j))) {
                    return false;
                }
            }
        }
        return true;
    }
}
