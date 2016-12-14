package explorekit.operators.BinaryOperators;

import explorekit.data.ColumnInfo;
import explorekit.data.Dataset;
import explorekit.data.NumericColumn;
import explorekit.operators.Operator;

import java.util.List;

/**
 * Created by giladkatz on 05/03/2016.
 */
public class DivisionBinaryOperator extends BinaryOperator {

    @Override
    public ColumnInfo generate(Dataset dataset, List<ColumnInfo> sourceColumns, List<ColumnInfo> targetColumns, boolean enforceDistinctVal) {
        NumericColumn column = new NumericColumn(dataset.getNumOfInstancesPerColumn());

        int numOfRows = dataset.getNumOfTrainingDatasetRows() + dataset.getNumOfTestDatasetRows();
        NumericColumn sourceColumn = (NumericColumn)sourceColumns.get(0).getColumn();
        NumericColumn targetColumn = (NumericColumn)targetColumns.get(0).getColumn();

        for (int i=0; i<numOfRows; i++) {
            int j = dataset.getIndices().get(i);
            double val = ((double)sourceColumn.getValue(j)) / ((double)targetColumn.getValue(j));
            if (Double.isNaN(val) || Double.isInfinite(val)) {
                column.setValue(j, 0);
            }
            else {
                column.setValue(j, val);
            }
        }

        ColumnInfo newColumnInfo = new ColumnInfo(column, sourceColumns, targetColumns, this.getClass(), "Divide" + generateName(sourceColumns,targetColumns));
        if (enforceDistinctVal && !super.isDistinctValEnforced(dataset,newColumnInfo)) {
            return null;
        }
        return newColumnInfo;
    }

    public void processTrainingSet(Dataset dataset, List<ColumnInfo> sourceColumns, List<ColumnInfo> targetColumns) {

    }

    public Operator.operatorType getType() {
        return Operator.operatorType.Binary;
    }

    public Operator.outputType getOutputType() { return Operator.outputType.Numeric;}

    public String getName() {
        return "DivisionBinaryOperator";
    }
}
