package explorekit.search.AttributeRankersFilters;

import explorekit.Evaluation.ClassificationResults;
import explorekit.data.ColumnInfo;
import explorekit.data.Dataset;
import explorekit.operators.OperatorAssignment;
import explorekit.search.Search;

import java.util.Collections;
import java.util.List;

/**
 * Created by giladkatz on 27/03/2016.
 */
public class FilterScoreRanker extends AttributeRankerFilter {
    @Override
    public List<OperatorAssignment> rankAndFilter(List<OperatorAssignment> operatorAssignments, List<ColumnInfo> previousIterationChosenAttributes,
                                                  List<Dataset> datasets, List<ClassificationResults> currentScore) {
        Collections.sort(operatorAssignments, new FilterEvaluatorScoreComparator());
        Collections.reverse(operatorAssignments);
        return operatorAssignments;
    }
}
