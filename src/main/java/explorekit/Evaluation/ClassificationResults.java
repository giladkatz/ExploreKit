package explorekit.Evaluation;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.TreeMap;

/**
 * Created by giladkatz on 07/03/2016.
 */
public class ClassificationResults {

    List<ClassificationItem> itemClassifications = new ArrayList<>();
    double auc;
    double logloss;
    TreeMap<Double,Double> tprFprValues;
    TreeMap<Double,Double> recallPrecisionValues;
    HashMap<Double,Double> fMeasureValuesPerRecall;

    public ClassificationResults(List<ClassificationItem> itemClassifications,  double auc, double logloss,
                                 TreeMap<Double,Double> tprFprValues, TreeMap<Double,Double> recallPrecisionValues,
                                 HashMap<Double,Double> fMeasureValuesPerRecall) {
        this.itemClassifications = itemClassifications;
        this.auc = auc;
        this.logloss = logloss;
        this.tprFprValues = tprFprValues;
        this.recallPrecisionValues = recallPrecisionValues;
        this.fMeasureValuesPerRecall = fMeasureValuesPerRecall;
    }

    public List<ClassificationItem> getItemClassifications() {
        return this.itemClassifications;
    }

    public double getAuc() {
        return this.auc;
    }
    public double getLogLoss() {return this.logloss; }
    public TreeMap<Double,Double> getTprFprValues() {return this.tprFprValues;}
    public TreeMap<Double,Double> getRecallPrecisionValues() { return this.recallPrecisionValues; }
    public HashMap<Double,Double> getFMeasureValuesPerRecall() { return this.fMeasureValuesPerRecall; }


}