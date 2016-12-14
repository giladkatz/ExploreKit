package explorekit.search.AttributeRankersFilters;

import explorekit.Evaluation.ClassificationResults;
import explorekit.data.ColumnInfo;
import explorekit.data.Dataset;
import explorekit.operators.OperatorAssignment;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Created by giladkatz on 28/03/2016.
 */
public class FilterScoreWithExclusionsRanker extends AttributeRankerFilter{
    @Override
    /**
     * This ranker/filter is similar to the FilterScoreRanker class, but in addition to the sorting by the filter score
     * we also require that the attribute generated in this round will not be used in this iteration in the creation of
     * the new attribute. In addition, the attributes combinar that were used in the generation of the chosen attribute of the
     * previous iteration cannot be used again
     */
    public List<OperatorAssignment> rankAndFilter(List<OperatorAssignment> operatorAssignments, List<ColumnInfo> previousIterationChosenAttributes, List<Dataset> datasets, List<ClassificationResults> currentScore) {

        Collections.sort(operatorAssignments, new FilterEvaluatorScoreComparator());
        Collections.reverse(operatorAssignments);

        if (previousIterationChosenAttributes == null) {
            return operatorAssignments;
        }

        List<OperatorAssignment> indicesToDowngrade = new ArrayList<>();
        for (int i=0; i<operatorAssignments.size(); i++) {
            OperatorAssignment testedAssignment = operatorAssignments.get(i);
            //if the attribute that was chosen in the previous iteration is used either as source or target
            if ( (testedAssignment.getSources() != null && !Collections.disjoint(testedAssignment.getSources(),previousIterationChosenAttributes)) ||
                    (testedAssignment.getTragets() != null && !Collections.disjoint(testedAssignment.getTragets(),previousIterationChosenAttributes)) ) {
                indicesToDowngrade.add(testedAssignment);
                continue;
            }

            for (ColumnInfo previousIterationChosenAttribute: previousIterationChosenAttributes) {
                if ((testedAssignment.getSources() != null && testedAssignment.getSources().equals(previousIterationChosenAttribute.getSourceColumns()))
                        && (testedAssignment.getTragets() != null && testedAssignment.getTragets().equals(previousIterationChosenAttribute.getTargetColumns()))) {
                    indicesToDowngrade.add(testedAssignment);
                    continue;
                }
            }
        }

        //remove all the items from the list
        operatorAssignments.removeAll(indicesToDowngrade);

        //add all the items again at the end
        operatorAssignments.addAll(indicesToDowngrade);

        return operatorAssignments;
    }
}
