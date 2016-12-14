package explorekit.operators.TimeBasedGroupByThenOperators;

import explorekit.data.Column;
import explorekit.data.ColumnInfo;
import explorekit.data.Dataset;
import explorekit.data.NumericColumn;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Created by giladkatz on 18/04/2016.
 */
public class TimeBasedGroupByThenCountAndCount extends TimeBasedGroupByThen {

    private Map<List<Integer>, Double> countValuePerKey = new HashMap<>();
    private double missingValuesVal = 0;
    /**
     * Used to initialize the time window that the operator will be applied on
     * @param minutes
     */
    public TimeBasedGroupByThenCountAndCount(double minutes) {
        timeWindow = minutes;
    }

    public void processTrainingSet(Dataset dataset, List<ColumnInfo> sourceColumns, List<ColumnInfo> targetColumns) throws Exception {
        super.processTrainingSet(dataset, sourceColumns, targetColumns);
        for (List<Integer> sources : valuesPerKey.keySet()) {
            List<Double> numOfInstancesPerTimeWindow = new ArrayList<>();
            double numOfTimeWindowsPerVal = valuesPerKey.get(sources).size();
            countValuePerKey.put(sources,numOfTimeWindowsPerVal);
        }

        //the value for missing values would be the average of all teh averages
        OptionalDouble totalAverage = countValuePerKey.values().stream().mapToDouble(x -> x).average();
        missingValuesVal = totalAverage.getAsDouble();
    }

    @Override
    public ColumnInfo generate(Dataset dataset, List<ColumnInfo> sourceColumns, List<ColumnInfo> targetColumns, boolean enforceDistinctVal) throws Exception {
        //we begin by separating the date column from the other source columns
        ColumnInfo dateColumn = null;
        List<ColumnInfo> nonDateColumns = new ArrayList<>();
        for (ColumnInfo ci: sourceColumns) {
            if (ci.getColumn().getType() == Column.columnType.Date) {
                dateColumn = ci;
            }
            else {
                nonDateColumns.add(ci);
            }
        }

        try {
            NumericColumn column = new NumericColumn(dataset.getNumOfInstancesPerColumn());
            int numOfRows = dataset.getNumberOfRows();

            for (int i = 0; i < numOfRows; i++) {
                int j = dataset.getIndices().get(i);
                List<Integer> sourceValues = nonDateColumns.stream().map(c -> (Integer) c.getColumn().getValue(j)).collect(Collectors.toList());
                if (!countValuePerKey.containsKey(sourceValues)) {
                    column.setValue(j, missingValuesVal);
                } else {
                    column.setValue(j, countValuePerKey.get(sourceValues));
                }
            }

            //now we generate the name of the new attribute
            String attString = generateName(sourceColumns, targetColumns);
            String finalString = getName().concat(attString).concat(")");

            ColumnInfo newColumnInfo = new ColumnInfo(column, sourceColumns, targetColumns, this.getClass(), finalString);
            if (enforceDistinctVal && !super.isDistinctValEnforced(dataset, newColumnInfo)) {
                return null;
            }
            return newColumnInfo;
        }
        catch (Exception ex) {
            throw new Exception("failue to generate feature in TimeBasedGroupByThenCountAndCount: " + ex.getMessage());
        }
    }

    public boolean isApplicable(Dataset dataset, List<ColumnInfo> sourceColumns, List<ColumnInfo> targetColumns) {
        return (super.isApplicable(dataset, sourceColumns, targetColumns));
    }

    @Override
    public outputType getOutputType() {
        return outputType.Numeric;
    }

    @Override
    public String getName() {
        return "TimeBasedGroupByThenCountAndCount_" + timeWindow;
    }


}