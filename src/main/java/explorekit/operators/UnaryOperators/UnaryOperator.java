package explorekit.operators.UnaryOperators;

import explorekit.data.ColumnInfo;
import explorekit.data.Dataset;
import explorekit.operators.Operator;

import java.util.List;

/**
 * Created by giladkatz on 20/02/2016.
 */
public abstract class UnaryOperator extends Operator {

    public boolean isApplicable(Dataset dataset, List<ColumnInfo> sourceColumns, List<ColumnInfo> targetColumns) {
        //if there are any target columns or if there is more than one source column, return false
        if (sourceColumns.size() != 1 || (targetColumns != null && targetColumns.size() != 0)) {
            return false;
        }
        else {
            return true;
        }
    }

    public Operator.operatorType getType() {
        return Operator.operatorType.Unary;
    }

    public abstract Operator.outputType requiredInputType();

    public abstract int getNumOfBins();
}
