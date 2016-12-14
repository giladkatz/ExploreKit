package explorekit.search.AttributeRankersFilters;

import explorekit.Evaluation.ClassificationResults;
import explorekit.data.ColumnInfo;
import explorekit.data.Dataset;
import explorekit.operators.OperatorAssignment;

import java.util.Comparator;
import java.util.List;

/**
 * Created by giladkatz on 27/03/2016.
 */
public abstract class AttributeRankerFilter {
    /**
     * This abstract method contains all the objects that may be needed to rank and order the candidate attributes after scores
     * have been assigned by the FilterEvaluator object
     * @param operatorAssignments The candidate attributes
     * @param previousIterationChosenAttributes The attribute that has been chosen in the previous iteration
     * @param datasets The training folds used for the training
     * @param currentScore The probabilities assigned to each instance by the current classification
     * @return
     */
    public abstract List<OperatorAssignment> rankAndFilter(List<OperatorAssignment> operatorAssignments, List<ColumnInfo> previousIterationChosenAttributes,
                                                    List<Dataset> datasets, List<ClassificationResults> currentScore);

    /**
     * Used in the sorting of OperatorAssignment objects based on the score of the initial evaluator
     */
    public class FilterEvaluatorScoreComparator implements Comparator<OperatorAssignment> {
        public FilterEvaluatorScoreComparator(){
        }

        @Override
        public int compare(OperatorAssignment oa1, OperatorAssignment oa2) {
            if ((oa1.getFilterEvaluatorScore() - oa2.getFilterEvaluatorScore()) == 0) {
                return 0;
            }
            if ((oa1.getFilterEvaluatorScore() - oa2.getFilterEvaluatorScore()) > 0) {
                return 1;
            } else {
                return -1;
            }
        }
    }

    /**
     * Used in the sorting of OperatorAssignment objects based on the score of the initial evaluator
     */
    public class WrapperEvaluatorScoreComparator implements Comparator<OperatorAssignment> {
        public WrapperEvaluatorScoreComparator(){
        }

        @Override
        public int compare(OperatorAssignment oa1, OperatorAssignment oa2) {
            if ((oa1.getWrapperEvaluatorScore() - oa2.getWrapperEvaluatorScore()) == 0) {
                return 0;
            }
            if ((oa1.getWrapperEvaluatorScore() - oa2.getWrapperEvaluatorScore()) > 0) {
                return 1;
            } else {
                return -1;
            }
        }
    }
}


