package explorekit.operators.GroupByThenOperators;

import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ListMultimap;
import explorekit.data.Column;
import explorekit.data.ColumnInfo;
import explorekit.data.Dataset;
import explorekit.operators.Operator;

import java.util.List;
import java.util.stream.Collectors;

/**
 * Created by giladkatz on 12/02/2016.
 */
public abstract class GroupByThen extends Operator {
    protected ListMultimap<List<Integer>, Double> valuesPerKey = ArrayListMultimap.create();

    public void processTrainingSet(Dataset dataset, List<ColumnInfo> sourceColumns, List<ColumnInfo> targetColumns) {
        for (int i =0; i<dataset.getNumOfTrainingDatasetRows(); i++) {
            final int j = dataset.getIndicesOfTrainingInstances().get(i);
            List<Integer> sourceValues = sourceColumns.stream().map(c -> (Integer) c.getColumn().getValue(j)).collect(Collectors.toList());
            double targetValue = (double) targetColumns.get(0).getColumn().getValue(j);
            if (Double.isNaN(targetValue) || Double.isInfinite(targetValue)) {
                //don't do anything. If there are no other values, this will be taken care of by the "general" value
            }
            else {
                valuesPerKey.put(sourceValues, targetValue);
            }
        }
    }

    public boolean isApplicable(Dataset dataset, List<ColumnInfo> sourceColumns, List<ColumnInfo> targetColumns) {
        if (targetColumns.size() != 1)
            return false;
        for (ColumnInfo ci: sourceColumns) {
            if (!ci.getColumn().getType().equals(Column.columnType.Discrete)) {
                return false;
            }
        }
        return true;
    }

    public operatorType getType() {
        return operatorType.GroupByThen;
    }

    /**
     *Returns the attributes that are use in the creation of the new attirbute. The attributes are grouped by
     * "Source" and "Target", seperated by "_". Attributes in each group are separated by a semicolon.
     * @return
     */
    public String generateName(List<ColumnInfo> sourceColumns, List<ColumnInfo> targetColumns) {
        String sourceAtts = "Source(";
        for (ColumnInfo ci: sourceColumns) {
            sourceAtts = sourceAtts.concat(ci.getName());
            sourceAtts = sourceAtts.concat(";");
        }
        sourceAtts = sourceAtts.concat(")");

        String targetAtts = "Target(";
        for (ColumnInfo ci: targetColumns) {
            targetAtts = targetAtts.concat(ci.getName());
            targetAtts = targetAtts.concat(";");
        }
        targetAtts = targetAtts.concat(")");

        String finalString = "";
        finalString = finalString.concat(sourceAtts);
        finalString = finalString.concat("_");
        finalString = finalString.concat(targetAtts);
        return finalString;
    }
}
