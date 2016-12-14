package explorekit.Evaluation.MLFeatureExtraction;

import explorekit.data.Column;
import explorekit.data.ColumnInfo;
import explorekit.data.Dataset;
import explorekit.data.DiscreteColumn;
import explorekit.operators.UnaryOperators.EqualRangeDiscretizerUnaryOperator;
import explorekit.operators.UnaryOperators.UnaryOperator;
import org.apache.commons.math3.stat.inference.ChiSquareTest;
import org.apache.commons.math3.stat.inference.TTest;

import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

/**
 * Created by giladkatz on 06/05/2016.
 */
public class StatisticOperations{

    ChiSquareTest chiSquareTest = new ChiSquareTest();
    TTest tTest = new TTest();

    /**
     * The function reveives two lists of features and returns a list of each possible pairs Paired T-Test values
     * @param list1
     * @param list2
     * @return
     */
    public List<Double> calculatePairedTTestValues(List<ColumnInfo> list1, List<ColumnInfo> list2) throws Exception {
        List<Double> tTestValues = new ArrayList<>();
        for (ColumnInfo ci1 : list1) {
            if (ci1.getColumn().getType() != Column.columnType.Numeric) {
                throw new Exception("Unable to process non-numeric columns - list 1");
            }
            for (ColumnInfo ci2 : list2) {
                if (ci2.getColumn().getType() != Column.columnType.Numeric) {
                    throw new Exception("Unable to process non-numeric columns - list 2");
                }
                double testValue = Math.abs(tTest.pairedT((double[])ci1.getColumn().getValues(),(double[])ci2.getColumn().getValues()));
                if (!Double.isNaN(testValue)) {
                    tTestValues.add(testValue);
                }
            }
        }
        return tTestValues;
    }

    public List<Double> calculatePairedTTestValues(List<ColumnInfo> list1, ColumnInfo columnInfo) throws Exception {
        List<ColumnInfo> tempList = new ArrayList<>();
        tempList.add(columnInfo);
        return calculatePairedTTestValues(list1, tempList);
    }


    /**
     * Calculates the Chi-Square test values among all the possible combonation of elements in the two provided list.
     * Also supports numeruc attributes, a discretized versions of which will be used in the calculation.
     * @param list1
     * @param list2
     * @param dataset
     * @return
     * @throws Exception
     */
    public List<Double> calculateChiSquareTestValues(List<ColumnInfo> list1, List<ColumnInfo> list2, Dataset dataset, Properties properties) throws Exception {
        double[] bins = new double[Integer.parseInt(properties.getProperty("equalRangeDiscretizerBinsNumber"))];
        EqualRangeDiscretizerUnaryOperator erduo = new EqualRangeDiscretizerUnaryOperator(bins);
        List<Double> chiSquareValues = new ArrayList<>();

        for (ColumnInfo ci1 : list1) {
            if (ci1.getColumn().getType() != Column.columnType.Discrete && ci1.getColumn().getType() != Column.columnType.Numeric) {
                throw new Exception("unsupported column type");
            }
            for (ColumnInfo ci2 : list2) {
                if (ci2.getColumn().getType() != Column.columnType.Discrete && ci2.getColumn().getType() != Column.columnType.Numeric) {
                    throw new Exception("unsupported column type");
                }
                ColumnInfo tempColumn1;
                ColumnInfo tempColumn2;
                if (ci1.getColumn().getType() == Column.columnType.Numeric) {
                    tempColumn1 = discretizeNumericColumn(dataset, ci1,erduo, properties);
                }
                else {
                    tempColumn1 = ci1;
                }
                if (ci2.getColumn().getType() == Column.columnType.Numeric) {
                    tempColumn2 = discretizeNumericColumn(dataset, ci2,erduo, properties);
                }
                else {
                    tempColumn2 = ci2;
                }

                double chiSquareTestVal = chiSquareTest.chiSquare(
                        generateDiscreteAttributesCategoryIntersection((DiscreteColumn)tempColumn1.getColumn(),
                                (DiscreteColumn)tempColumn2.getColumn()));

                if (!Double.isNaN(chiSquareTestVal) && !Double.isInfinite(chiSquareTestVal)) {
                    chiSquareValues.add(chiSquareTestVal);
                }
            }
        }


        return chiSquareValues;
    }

    public List<Double> calculateChiSquareTestValues(List<ColumnInfo> list1, ColumnInfo columnInfo, Dataset dataset, Properties properties) throws Exception {
        List<ColumnInfo> tempList = new ArrayList<>();
        tempList.add(columnInfo);
        return calculateChiSquareTestValues(list1, tempList, dataset, properties);
    }

    /**
     * Receives a numeric column and returns its discretized version
     * @param dataset
     * @param columnInfo
     * @param discretizer
     * @return
     * @throws Exception
     */
    public ColumnInfo discretizeNumericColumn(Dataset dataset, ColumnInfo columnInfo, UnaryOperator discretizer, Properties properties) throws Exception {
        if (discretizer == null) {
            double[] bins = new double[Integer.parseInt(properties.getProperty("equalRangeDiscretizerBinsNumber"))];
            discretizer = new EqualRangeDiscretizerUnaryOperator(bins);
        }
        List<ColumnInfo> tempColumnsList = new ArrayList<>();
        tempColumnsList.add(columnInfo);
        discretizer.processTrainingSet(dataset,tempColumnsList,null);
        ColumnInfo discretizedAttribute = discretizer.generate(dataset,tempColumnsList,null,false);
        return discretizedAttribute;
    }

    public long[][] generateDiscreteAttributesCategoryIntersection(DiscreteColumn col1, DiscreteColumn col2) throws Exception {
        long[][] intersectionsMatrix = new long[col1.getNumOfPossibleValues()][col2.getNumOfPossibleValues()];
        int[] col1Values = (int[])col1.getValues();
        int[] col2Values = (int[])col2.getValues();

        if (col1Values.length != col2Values.length) {
            throw new Exception("Columns do not have the same number of instances");
        }

        for (int i=0; i<col1Values.length; i++) {
            intersectionsMatrix[col1Values[i]][col2Values[i]]++;
        }

        return intersectionsMatrix;
    }
}

//tTest.pairedT((double[]) oa.getSources().get(0).getColumn().getValues(), (double[]) oa.getTragets().get(0).getColumn().getValues());