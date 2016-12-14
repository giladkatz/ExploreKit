package explorekit.operators.UnaryOperators;

import explorekit.data.Column;
import explorekit.data.ColumnInfo;
import explorekit.data.Dataset;
import explorekit.data.NumericColumn;

import java.util.ArrayList;
import java.util.List;
import java.util.OptionalDouble;

/**
 * Created by giladkatz on 05/03/2016.
 */
public class StandardScoreUnaryOperator extends UnaryOperator{

    private double avg;
    private double stdev;

    public ColumnInfo generate(Dataset dataset, List<ColumnInfo> sourceColumns, List<ColumnInfo> targetColumns, boolean enforceDistinctVal) {
        NumericColumn column = new NumericColumn(dataset.getNumOfInstancesPerColumn());
        //this is the number of rows we need to work on - not the size of the vector
        int numOfRows = dataset.getNumberOfRows();
        ColumnInfo columnInfo = sourceColumns.get(0);
        for (int i=0; i<numOfRows; i++) {
            int j = dataset.getIndices().get(i);
            double standardScoreVal = getStandardScore((double) columnInfo.getColumn().getValue(j));
            if (Double.isNaN(standardScoreVal) || Double.isInfinite(standardScoreVal)) {

            }
            else {
                column.setValue(j, standardScoreVal);
            }
        }

        //now we generate the name of the new attribute
        String attString = "StandardScoreUnaryOperator(";
        attString = attString.concat(columnInfo.getName());
        attString = attString.concat(")");

        return new ColumnInfo(column, sourceColumns, targetColumns, this.getClass(), attString);
    }

    public double getStandardScore(double value) {
        if (stdev == 0) {
            return 0;
        }
        return (value - avg)/stdev;
    }

    public void processTrainingSet(Dataset dataset, List<ColumnInfo> sourceColumns, List<ColumnInfo> targetColumns) {
        List<Double> vals = new ArrayList<>();
        ColumnInfo columnInfo = sourceColumns.get(0);
        for (int i =0; i<dataset.getNumOfTrainingDatasetRows(); i++) {
            int j = dataset.getIndicesOfTrainingInstances().get(i);
            double val = (Double)columnInfo.getColumn().getValue(i);
            if (!Double.isNaN(val) && !Double.isInfinite(val)) {
                vals.add(val);
            }
        }
        OptionalDouble tempAvg = vals.stream().mapToDouble(a -> a).average();

        if (tempAvg.isPresent()) {
            avg = tempAvg.getAsDouble();
            double tempStdev = vals.stream().mapToDouble(a -> Math.pow(a-avg,2)).sum();
            stdev = Math.sqrt(tempStdev/vals.size());
        }
        else {
            System.out.println("no values in the attribute");
        }
    }

    public boolean isApplicable(Dataset dataset, List<ColumnInfo> sourceColumns, List<ColumnInfo> targetColumns) {
        if (super.isApplicable(dataset, sourceColumns, targetColumns)) {
            if (sourceColumns.get(0).getColumn().getType().equals(Column.columnType.Numeric)) {
                return true;
            }
        }
        return false;
    }

    public outputType getOutputType() { return outputType.Numeric;}

    public outputType requiredInputType() {
        return outputType.Numeric;
    }

    public String getName() {
        return "StandardScoreUnaryOperator";
    }

    //this is a normalizer, not a discretizer
    public int getNumOfBins() {
        return -1;
    }
}
