package explorekit.operators.UnaryOperators;

import explorekit.data.*;

import java.io.InputStream;
import java.util.List;
import java.util.Properties;

/**
 * Created by giladkatz on 20/02/2016.
 */
public class EqualRangeDiscretizerUnaryOperator extends UnaryOperator {

    double[] upperBoundPerBin;

    public EqualRangeDiscretizerUnaryOperator(double[] upperBoundPerBin) throws Exception{
        this.upperBoundPerBin = upperBoundPerBin;
    }

    public boolean isApplicable(Dataset dataset, List<ColumnInfo> sourceColumns, List<ColumnInfo> targetColumns) {
        if (super.isApplicable(dataset, sourceColumns, targetColumns)) {
            if (sourceColumns.get(0).getColumn().getType().equals(Column.columnType.Numeric)) {
                return true;
            }
        }
        return false;
    }

    public void processTrainingSet(Dataset dataset, List<ColumnInfo> sourceColumns, List<ColumnInfo> targetColumns) {
        double minVal = Double.MAX_VALUE;
        double maxVal = Double.MIN_VALUE;

        ColumnInfo columnInfo = sourceColumns.get(0);
        for (int i =0; i<dataset.getNumOfTrainingDatasetRows(); i++) {
            int j = dataset.getIndicesOfTrainingInstances().get(i);
            double val = (Double)columnInfo.getColumn().getValue(i);
            if (!Double.isNaN(val) && !Double.isInfinite(val)) {
                minVal = Math.min(minVal, val);
                maxVal = Math.max(maxVal, val);
            }
            else {
                int x=5;
            }
        }
        double range = (maxVal-minVal)/upperBoundPerBin.length;
        double currentVal = minVal;
        for (int i=0; i<upperBoundPerBin.length; i++) {
            upperBoundPerBin[i] = currentVal + range;
            currentVal += range;
        }
    }

    public ColumnInfo generate(Dataset dataset, List<ColumnInfo> sourceColumns, List<ColumnInfo> targetColumns, boolean enforceDistinctVal) {
        try {
            DiscreteColumn column = new DiscreteColumn(dataset.getNumOfInstancesPerColumn(), upperBoundPerBin.length);
            //this is the number of rows we need to work on - not the size of the vector
            int numOfRows = dataset.getNumberOfRows();
            ColumnInfo columnInfo = sourceColumns.get(0);
            for (int i = 0; i < numOfRows; i++) {
                if (dataset.getIndices().size() == i) {
                    int x = 5;
                }
                int j = dataset.getIndices().get(i);
                int binIndex = GetBinIndex((double) columnInfo.getColumn().getValue(j));
                column.setValue(j, binIndex);
            }

            //now we generate the name of the new attribute
            String attString = "EqualRangeDiscretizer(";
            attString = attString.concat(columnInfo.getName());
            attString = attString.concat(")");

            return new ColumnInfo(column, sourceColumns, targetColumns, this.getClass(), attString);
        }
        catch (Exception ex) {
            System.out.println("error in EqualRangeDiscretizer:  " +  ex.getMessage());
            return null;
        }
    }

    private int GetBinIndex(double value) {
        for (int i=0; i<upperBoundPerBin.length; i++) {
            if (upperBoundPerBin[i] > value) {
                return i;
            }
        }
        return (upperBoundPerBin.length-1);
    }



    public outputType getOutputType() { return outputType.Discrete;}

    public outputType requiredInputType() {
        return outputType.Numeric;
    }

    public String getName() {
        return "EqualRangeDiscretizerUnaryOperator";
    }

    public int getNumOfBins() {
        return upperBoundPerBin.length;
    }
}
