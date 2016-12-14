package explorekit.operators;

import explorekit.data.ColumnInfo;
import explorekit.operators.UnaryOperators.UnaryOperator;

import java.util.List;

/**
 * Created by giladkatz on 02/03/2016.
 */
public class OperatorAssignment {
    private List<ColumnInfo> sourceCoolumns;
    private List<ColumnInfo> targetColumns;
    private Operator operator; //the operator that will be applied on the source and target columns
    private UnaryOperator secondaryOperator; //a discretizer/normalizer that will be applied on the product of the previous operator
    private double filterEvaluatorScore = 0;
    private double wrapperEvaluatorScore;

    public OperatorAssignment(List<ColumnInfo> sourceCoolumns, List<ColumnInfo> targetColumns, Operator operator, UnaryOperator secondaryOperator) {
        this.sourceCoolumns = sourceCoolumns;
        this.targetColumns = targetColumns;
        this.operator = operator;
        //this operator is to be applied AFTER the main operator is complete (serves as a discretizer/normalizer)
        this.secondaryOperator = secondaryOperator;
    }

    public String getName() {
        StringBuilder sb = new StringBuilder();
        sb.append("{Sources:[");
        for (ColumnInfo sCI : sourceCoolumns) {
            sb.append(sCI.getName());
            sb.append(",");
        }
        sb.append("];");
        sb.append("Targets:[");
        if (targetColumns != null) {
            for (ColumnInfo tCI : targetColumns) {
                sb.append(tCI.getName());
                sb.append(",");
            }
        }
        sb.append("];");
        sb.append(operator.getName());
        if (secondaryOperator != null) {
            sb.append(",");
            sb.append(secondaryOperator.getName());
        }
        sb.append("}");
        return sb.toString();
    }

    public List<ColumnInfo> getSources() {return this.sourceCoolumns;}
    public List<ColumnInfo> getTragets() {return this.targetColumns;}
    public Operator getOperator() {return this.operator;}
    public UnaryOperator getSecondaryOperator() {return this.secondaryOperator;}

    public double getFilterEvaluatorScore() {
        return this.filterEvaluatorScore;
    }

    public void setFilterEvaluatorScore(double score) {
        this.filterEvaluatorScore = score;
    }

    public double getWrapperEvaluatorScore() {
        return this.wrapperEvaluatorScore;
    }

    public void setWrapperEvaluatorScore(double score) {
        this.wrapperEvaluatorScore = score;
    }
}
