package explorekit.search.AttributeRankersFilters;

import explorekit.Evaluation.ClassificationResults;
import explorekit.data.ColumnInfo;
import explorekit.data.Dataset;
import explorekit.operators.OperatorAssignment;

import java.util.Collections;
import java.util.List;

/**
 * Created by giladkatz on 27/03/2016.
 */
public class WrapperScoreRanker extends AttributeRankerFilter {
    @Override
    public List<OperatorAssignment> rankAndFilter(List<OperatorAssignment> operatorAssignments, List<ColumnInfo> previousIterationChosenAttributes, List<Dataset> datasets, List<ClassificationResults> currentScore) {
        Collections.sort(operatorAssignments, new WrapperEvaluatorScoreComparator());
        Collections.reverse(operatorAssignments);
        return operatorAssignments;
    }
}
