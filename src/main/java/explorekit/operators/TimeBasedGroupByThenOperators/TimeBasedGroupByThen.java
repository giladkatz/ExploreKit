package explorekit.operators.TimeBasedGroupByThenOperators;

import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ListMultimap;
import explorekit.data.Column;
import explorekit.data.ColumnInfo;
import explorekit.data.Dataset;
import explorekit.data.DateColumn;
import explorekit.operators.Operator;

import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

/**
 * Created by giladkatz on 02/04/2016.
 */
public abstract class TimeBasedGroupByThen extends Operator {

    //The time window the operator will be applied on. Specified minutes
    protected double timeWindow;

    /**
     * For each unique combination of source values (the date not included), we hold lists of instances that are contained
     * within a specific time window
     */
    protected ListMultimap<List<Integer>, List<TimeBasedInstanceValue>> valuesPerKey = ArrayListMultimap.create();

    @Override
    public void processTrainingSet(Dataset dataset, List<ColumnInfo> sourceColumns, List<ColumnInfo> targetColumns) throws Exception {
        //we begin by separating the date column from the other source columns
        ColumnInfo dateColumn = null;
        List<ColumnInfo> nonDateColumns = new ArrayList<>();
        for (ColumnInfo ci: sourceColumns) {
            if (ci.getColumn().getType() == Column.columnType.Date) {
                if (dateColumn == null) {
                    dateColumn = ci;
                }
                else {
                    throw new Exception("More than one date column in operation");
                }
            }
            else {
                nonDateColumns.add(ci);
            }
        }

        //A not-very-elegant solution - a hashmap that connects the ordering by time to the current list of indices
        //By storing the indices in a HashMap we can quickly access if an index is included
        HashMap<Integer,Integer> indexToLocationMapping = new HashMap<>();
        for (int index : dataset.getIndices()) {
            indexToLocationMapping.put(index, indexToLocationMapping.size());
        }

        TreeMap<Date, List<Integer>> indicesByDate = ((DateColumn) dateColumn.getColumn()).getIndicesByDate();

        HashMap<List<Integer>, Date> currentDatesPerValue = new HashMap<>();
        HashMap<List<Integer>, List<TimeBasedInstanceValue>> currentWindowValues = new HashMap<>();
        for (Date currentDate: indicesByDate.keySet()) {
            for (int index : indicesByDate.get(currentDate)) {
                if (!indexToLocationMapping.containsKey(index)) {
                    continue;
                }
                //We obtain the source values and the target value
                List<Integer> sourceValues = nonDateColumns.stream().map(c -> (Integer) c.getColumn().getValue(index)).collect(Collectors.toList());
                Object targetValue = targetColumns.get(0).getColumn().getValue(index);
                TimeBasedInstanceValue tbiv = new TimeBasedInstanceValue(currentDate, targetValue);

                //now we need to determine where the new instance needs to be assigned - to an existing sliding time window or to a new one
                if (!currentDatesPerValue.containsKey(sourceValues)) {
                    currentDatesPerValue.put(sourceValues, currentDate);
                    currentWindowValues.put(sourceValues, new ArrayList<>());
                }

                //if the date of the current instance is greater than the bound of teh time window, process and clear it
                if (TimeDifferenceExceedsWindow(currentDate, currentDatesPerValue.get(sourceValues))) {
                    //add the items in the existing time window (if there are any) to the final Map
                    if (currentWindowValues.get(sourceValues).size() > 0) {
                        //add the current list to the final map
                        valuesPerKey.put(sourceValues, currentWindowValues.get(sourceValues));
                        //reset the list of the current window
                        currentWindowValues.get(sourceValues).clear();
                        //set a new beginning to the date window
                        currentDatesPerValue.put(sourceValues, currentDate);
                    }
                }

                //finally, add the new instance at the relevant place
                currentWindowValues.get(sourceValues).add(tbiv);

            }
        }
        //finally, we need to process all the items left in the currentWindowValues object
        for (List<Integer> sourceValues : currentWindowValues.keySet()) {
            valuesPerKey.put(sourceValues, currentWindowValues.get(sourceValues));
        }
    }


    /**
     * Used to determine if the date of the curernt analyzed instance is outside the current time window
     * @param currentDate
     * @param windowStartDate
     * @return
     */
    private boolean TimeDifferenceExceedsWindow(Date currentDate, Date windowStartDate) {
        long timeDiff = currentDate.getTime()-windowStartDate.getTime();
        long diffInMinutes = TimeUnit.MILLISECONDS.toMinutes(timeDiff);
        return (diffInMinutes > this.timeWindow);
    }

    @Override
    public boolean isApplicable(Dataset dataset, List<ColumnInfo> sourceColumns, List<ColumnInfo> targetColumns) {
        if (targetColumns.size() != 1 || sourceColumns.size() == 1)
            return false;
        if ((targetColumns.get(0).getColumn().getType() != Column.columnType.Discrete) &&
                (targetColumns.get(0).getColumn().getType() != Column.columnType.Numeric)) {
            return false;
        }

        int dateColumnsCounter = 0;
        for (ColumnInfo ci: sourceColumns) {
            if (ci.getColumn().getType().equals(Column.columnType.Numeric) || ci.getColumn().getType().equals(Column.columnType.String)) {
                return false;
            }
            if (ci.getColumn().getType().equals(Column.columnType.Date)) {
                dateColumnsCounter++;
            }
        }
        if (dateColumnsCounter != 1) {
            return false;
        }
        return true;
    }

    @Override
    public operatorType getType() {
        return operatorType.TimeBasedGroupByThen;
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
