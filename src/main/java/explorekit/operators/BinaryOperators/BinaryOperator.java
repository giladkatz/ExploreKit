package explorekit.operators.BinaryOperators;

import explorekit.data.Column;
import explorekit.data.ColumnInfo;
import explorekit.data.Dataset;
import explorekit.operators.Operator;

import java.util.List;

/**
 * Created by giladkatz on 22/02/2016.
 */
public abstract class BinaryOperator extends Operator {
    public boolean isApplicable(Dataset dataset, List<ColumnInfo> sourceColumns, List<ColumnInfo> targetColumns) {
        //if there are any target columns or if there is more than one source column, return false
        if (targetColumns.size() != 1 || sourceColumns.size() != 1) {
            return false;
        }
        if (!sourceColumns.get(0).getColumn().getType().equals(Column.columnType.Numeric) ||
                !targetColumns.get(0).getColumn().getType().equals(Column.columnType.Numeric)) {
            return false;
        }
        return true;
    }

    /**
     * For the current binary operators (which are arithmetic operations) this is not needed
     * @param dataset
     * @param sourceColumns
     * @param targetColumns
     */
    public void processTrainingSet(Dataset dataset, List<ColumnInfo> sourceColumns, List<ColumnInfo> targetColumns) {}

    public Operator.operatorType getType() {
        return operatorType.Binary;
    }

    public String generateName(List<ColumnInfo> sourceColumns, List<ColumnInfo> targetColumns) {
        String string = "(";
        string = string.concat(sourceColumns.get(0).getName());
        string = string.concat(";");
        string = string.concat(targetColumns.get(0).getName());
        string = string.concat(")");
        return string;
    }
}
