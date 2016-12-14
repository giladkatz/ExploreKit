package explorekit.Evaluation.MLFeatureExtraction;

import explorekit.Evaluation.FilterEvaluators.InformationGainFilterEvaluator;
import explorekit.data.Column;
import explorekit.data.ColumnInfo;
import explorekit.data.Dataset;
import explorekit.data.DiscreteColumn;
import explorekit.operators.Operator;
import explorekit.operators.OperatorAssignment;
import explorekit.operators.OperatorsAssignmentsManager;
import explorekit.operators.UnaryOperators.UnaryOperator;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.math3.stat.inference.ChiSquareTest;
import org.apache.commons.math3.stat.inference.TTest;

import java.util.*;


/**
 * Created by giladkatz on 26/04/2016.
 */
public class OperatorAssignmentBasedAttributes {

    private int numOfSources;
    private int numOfNumericSources;
    private int numOfDiscreteSources;
    private int numOfDateSources;
    private int operatorTypeIdentifier; //The type of the operator: unary, binary etc.
    private int operatorIdentifier;
    private int discretizerInUse; //0 if none is used, otherwise the type of the discretizer (enumerated)
    private int normalizerInUse; //0 if none is used, otherwise the type of the normalizer (enumerated)

    //statistics on the values of discrete source attributes
    private double maxNumOfDiscreteSourceAttribtueValues;
    private double minNumOfDiscreteSourceAttribtueValues;
    private double avgNumOfDiscreteSourceAttribtueValues;
    private double stdevNumOfDiscreteSourceAttribtueValues;

    //atatistics on the values of the target attribute (currently for numeric values)
    private double maxValueOfNumericTargetAttribute;
    private double minValueOfNumericTargetAttribute;
    private double avgValueOfNumericTargetAttribute;
    private double stdevValueOfNumericTargetAttribute;

    //statistics on the value of the numeric source attribute (currently we only support cases where it's the only source attribute)
    private double maxValueOfNumericSourceAttribute;
    private double minValueOfNumericSourceAttribute;
    private double avgValueOfNumericSourceAttribute;
    private double stdevValueOfNumericSourceAttribute;

    //Paired-T amd Chi-Square tests on the source and target attributes
    private double chiSquareTestValueForSourceAttributes;
    private double pairedTTestValueForSourceAndTargetAttirbutes;  //used for numeric single source attirbute and numeric target

    private double maxChiSquareTsetForSourceAndTargetAttributes; //we discretize all the numeric attributes for this one
    private double minChiSquareTsetForSourceAndTargetAttributes;
    private double avgChiSquareTsetForSourceAndTargetAttributes;
    private double stdevChiSquareTsetForSourceAndTargetAttributes;

    //Calculate the similarity of the source attributes to other attibures in the dataset (discretuze all the numeric ones)
    private double maxChiSquareTestvalueForSourceDatasetAttributes;
    private double minChiSquareTestvalueForSourceDatasetAttributes;
    private double avgChiSquareTestvalueForSourceDatasetAttributes;
    private double stdevChiSquareTestvalueForSourceDatasetAttributes;


    ///////////////////////////////////////////////////////////////////////////////////////////////
    //statistics on the generated attributes
    private int isOutputDiscrete; //if  not, it's 0
    //If the generated attribute is discrete, count the number of possible values. If numeric, the value is set to 0
    private int numOfDiscreteValues;

    private double IGScore;

    //If the generated attribute is numeric, calculate the Paired T-Test statistics for it and the datasets's numeric attributes
    private double maxPairedTTestValuesForGeneratedAttributeAndDatasetNumericAttributes = -1;
    private double minPairedTTestValuesForGeneratedAttributeAndDatasetNumericAttributes = -1;
    private double avgPairedTTestValuesForGeneratedAttributeAndDatasetNumericAttributes = -1;
    private double stdevPairedTTestValuesForGeneratedAttributeAndDatasetNumericAttributes = -1;

    //The Chi-Squared test of the (discretized if needed) generate attribute and the dataset's discrete attributes
    private double maxChiSquaredTestValuesForGeneratedAttributeAndDatasetDiscreteAttributes;
    private double minChiSquaredTestValuesForGeneratedAttributeAndDatasetDiscreteAttributes;
    private double avgChiSquaredTestValuesForGeneratedAttributeAndDatasetDiscreteAttributes;
    private double stdevChiSquaredTestValuesForGeneratedAttributeAndDatasetDiscreteAttributes;

    //the Chi-Squared test of the (discretized if needed) generate attribute and ALL of the dataset's attributes (discrete and numeric)
    private double maxChiSquaredTestValuesForGeneratedAttributeAndAllDatasetAttributes;
    private double minChiSquaredTestValuesForGeneratedAttributeAndAllDatasetAttributes;
    private double avgChiSquaredTestValuesForGeneratedAttributeAndAllDatasetAttributes;
    private double stdevChiSquaredTestValuesForGeneratedAttributeAndAllDatasetAttributes;

    ///////////////////////////////////////////////////////////////////////////////////////////////

    private HashMap<Double,Double> probDiffScoreForTopMiscallasiffiedInstancesInMinorityClass = new HashMap<>();
    private HashMap<Double,Double> probDiffScoreForTopMiscallasiffiedInstancesInMajorityClass = new HashMap<>();


    //The statistical tests we're using
    ChiSquareTest chiSquareTest = new ChiSquareTest();
    TTest tTest = new TTest();

    /**
     * the statistics class object. Used for all calcualtions
     */
    StatisticOperations statisticOperations = new StatisticOperations();


    public HashMap<Integer,AttributeInfo> getOperatorAssignmentBasedAttributes(Dataset dataset, OperatorAssignment oa, ColumnInfo evaluatedAttribute, Properties properties) throws Exception {
        try {
            if (oa == null) {
                int x=5;
            }

            OperatorsAssignmentsManager oam = new OperatorsAssignmentsManager(properties);
            Dataset datasetReplica = dataset.replicateDataset();
            datasetReplica.addColumn(evaluatedAttribute);
            List<ColumnInfo> tempList = new ArrayList<>();
            tempList.add(evaluatedAttribute);

            //IGScore
            try {
                InformationGainFilterEvaluator igfe = new InformationGainFilterEvaluator();
                igfe.initFilterEvaluator(tempList);
                this.IGScore = igfe.produceScore(datasetReplica, null, dataset, null, null, properties);
            }
            catch (Exception ex) {
                int x=5;
            }

            //Calling the procedures that calculate the attributes of the OperatorAssignment obejct and the source and target attribtues
            try {
                ProcessOperatorAssignment(dataset, oa);
            }
            catch (Exception ex) {
                int x=5;
            }

            try {
                processSourceAndTargetAttributes(dataset, oa);
            }
            catch (Exception ex) {
                int x=5;
            }

            try {
                performStatisticalTestsOnSourceAndTargetAttributes(dataset, oa, properties);
            }
            catch (Exception ex) {
                int x=5;
            }

            try {
                performStatisticalTestOnOperatorAssignmentAndDatasetAtributes(dataset, oa, properties);
            }
            catch (Exception ex) {
                int x=5;
            }

            //Calling the procedures that calculate statistics on the candidate attribute
            try {
                processGeneratedAttribute(dataset, oa, evaluatedAttribute, properties);
            }
            catch (Exception ex) {
                int x=5;
            }

            HashMap<Integer, AttributeInfo> attributes = generateInstanceAttributesMap();

            return attributes;
        }
        catch (Exception ex) {
            return null;
        }
    }

    public HashMap<Integer,AttributeInfo> generateInstanceAttributesMap() {
        HashMap<Integer,AttributeInfo> attributes = new HashMap<>();

        try {
            AttributeInfo att1 = new AttributeInfo("numOfSources", Column.columnType.Numeric, numOfSources, -1);
            AttributeInfo att2 = new AttributeInfo("numOfNumericSources", Column.columnType.Numeric, numOfNumericSources, -1);
            AttributeInfo att3 = new AttributeInfo("numOfDiscreteSources", Column.columnType.Numeric, numOfDiscreteSources, -1);
            AttributeInfo att4 = new AttributeInfo("numOfDateSources", Column.columnType.Numeric, numOfDateSources, -1);
            AttributeInfo att5 = new AttributeInfo("operatorTypeIdentifier", Column.columnType.Discrete, operatorTypeIdentifier, 4);
            AttributeInfo att6 = new AttributeInfo("operatorIdentifier", Column.columnType.Discrete, operatorIdentifier, 30);
            AttributeInfo att7 = new AttributeInfo("discretizerInUse", Column.columnType.Discrete, discretizerInUse, 2);
            AttributeInfo att8 = new AttributeInfo("normalizerInUse", Column.columnType.Discrete, normalizerInUse, 2);
            AttributeInfo att9 = new AttributeInfo("maxNumOfDiscreteSourceAttribtueValues", Column.columnType.Numeric, maxNumOfDiscreteSourceAttribtueValues, -1);
            AttributeInfo att10 = new AttributeInfo("minNumOfDiscreteSourceAttribtueValues", Column.columnType.Numeric, minNumOfDiscreteSourceAttribtueValues, -1);
            AttributeInfo att11 = new AttributeInfo("avgNumOfDiscreteSourceAttribtueValues", Column.columnType.Numeric, avgNumOfDiscreteSourceAttribtueValues, -1);
            AttributeInfo att12 = new AttributeInfo("stdevNumOfDiscreteSourceAttribtueValues", Column.columnType.Numeric, stdevNumOfDiscreteSourceAttribtueValues, -1);
            AttributeInfo att13 = new AttributeInfo("maxValueOfNumericTargetAttribute", Column.columnType.Numeric, maxValueOfNumericTargetAttribute, -1);
            AttributeInfo att14 = new AttributeInfo("minValueOfNumericTargetAttribute", Column.columnType.Numeric, minValueOfNumericTargetAttribute, -1);
            AttributeInfo att15 = new AttributeInfo("avgValueOfNumericTargetAttribute", Column.columnType.Numeric, avgValueOfNumericTargetAttribute, -1);
            AttributeInfo att16 = new AttributeInfo("stdevValueOfNumericTargetAttribute", Column.columnType.Numeric, stdevValueOfNumericTargetAttribute, -1);
            AttributeInfo att17 = new AttributeInfo("maxValueOfNumericSourceAttribute", Column.columnType.Numeric, maxValueOfNumericSourceAttribute, -1);
            AttributeInfo att18 = new AttributeInfo("minValueOfNumericSourceAttribute", Column.columnType.Numeric, minValueOfNumericSourceAttribute, -1);
            AttributeInfo att19 = new AttributeInfo("avgValueOfNumericSourceAttribute", Column.columnType.Numeric, avgValueOfNumericSourceAttribute, -1);
            AttributeInfo att20 = new AttributeInfo("stdevValueOfNumericSourceAttribute", Column.columnType.Numeric, stdevValueOfNumericSourceAttribute, -1);
            AttributeInfo att21 = new AttributeInfo("chiSquareTestValueForSourceAttributes", Column.columnType.Numeric, chiSquareTestValueForSourceAttributes, -1);
            AttributeInfo att22 = new AttributeInfo("pairedTTestValueForSourceAndTargetAttirbutes", Column.columnType.Numeric, pairedTTestValueForSourceAndTargetAttirbutes, -1);
            AttributeInfo att23 = new AttributeInfo("maxChiSquareTsetForSourceAndTargetAttributes", Column.columnType.Numeric, maxChiSquareTsetForSourceAndTargetAttributes, -1);
            AttributeInfo att24 = new AttributeInfo("minChiSquareTsetForSourceAndTargetAttributes", Column.columnType.Numeric, minChiSquareTsetForSourceAndTargetAttributes, -1);
            AttributeInfo att25 = new AttributeInfo("avgChiSquareTsetForSourceAndTargetAttributes", Column.columnType.Numeric, avgChiSquareTsetForSourceAndTargetAttributes, -1);
            AttributeInfo att26 = new AttributeInfo("stdevChiSquareTsetForSourceAndTargetAttributes", Column.columnType.Numeric, stdevChiSquareTsetForSourceAndTargetAttributes, -1);
            AttributeInfo att27 = new AttributeInfo("maxChiSquareTestvalueForSourceDatasetAttributes", Column.columnType.Numeric, maxChiSquareTestvalueForSourceDatasetAttributes, -1);
            AttributeInfo att28 = new AttributeInfo("minChiSquareTestvalueForSourceDatasetAttributes", Column.columnType.Numeric, minChiSquareTestvalueForSourceDatasetAttributes, -1);
            AttributeInfo att29 = new AttributeInfo("avgChiSquareTestvalueForSourceDatasetAttributes", Column.columnType.Numeric, avgChiSquareTestvalueForSourceDatasetAttributes, -1);
            AttributeInfo att30 = new AttributeInfo("stdevChiSquareTestvalueForSourceDatasetAttributes", Column.columnType.Numeric, stdevChiSquareTestvalueForSourceDatasetAttributes, -1);
            AttributeInfo att31 = new AttributeInfo("isOutputDiscrete", Column.columnType.Discrete, isOutputDiscrete, 2);
            AttributeInfo att32 = new AttributeInfo("numOfDiscreteValues", Column.columnType.Numeric, numOfDiscreteValues, -1);
            AttributeInfo att33 = new AttributeInfo("numOfDiscreteValues", Column.columnType.Numeric, IGScore, -1);
            AttributeInfo att34 = new AttributeInfo("probDiffScore", Column.columnType.Numeric, -1, -1);
            AttributeInfo att35 = new AttributeInfo("maxPairedTTestValuesForGeneratedAttributeAndDatasetNumericAttributes", Column.columnType.Numeric, maxPairedTTestValuesForGeneratedAttributeAndDatasetNumericAttributes, -1);
            AttributeInfo att36 = new AttributeInfo("minPairedTTestValuesForGeneratedAttributeAndDatasetNumericAttributes", Column.columnType.Numeric, minPairedTTestValuesForGeneratedAttributeAndDatasetNumericAttributes, -1);
            AttributeInfo att37 = new AttributeInfo("avgPairedTTestValuesForGeneratedAttributeAndDatasetNumericAttributes", Column.columnType.Numeric, avgPairedTTestValuesForGeneratedAttributeAndDatasetNumericAttributes, -1);
            AttributeInfo att38 = new AttributeInfo("stdevPairedTTestValuesForGeneratedAttributeAndDatasetNumericAttributes", Column.columnType.Numeric, stdevPairedTTestValuesForGeneratedAttributeAndDatasetNumericAttributes, -1);
            AttributeInfo att39 = new AttributeInfo("maxChiSquaredTestValuesForGeneratedAttributeAndDatasetDiscreteAttributes", Column.columnType.Numeric, maxChiSquaredTestValuesForGeneratedAttributeAndDatasetDiscreteAttributes, -1);
            AttributeInfo att40 = new AttributeInfo("minChiSquaredTestValuesForGeneratedAttributeAndDatasetDiscreteAttributes", Column.columnType.Numeric, minChiSquaredTestValuesForGeneratedAttributeAndDatasetDiscreteAttributes, -1);
            AttributeInfo att41 = new AttributeInfo("avgChiSquaredTestValuesForGeneratedAttributeAndDatasetDiscreteAttributes", Column.columnType.Numeric, avgChiSquaredTestValuesForGeneratedAttributeAndDatasetDiscreteAttributes, -1);
            AttributeInfo att42 = new AttributeInfo("stdevChiSquaredTestValuesForGeneratedAttributeAndDatasetDiscreteAttributes", Column.columnType.Numeric, stdevChiSquaredTestValuesForGeneratedAttributeAndDatasetDiscreteAttributes, -1);
            AttributeInfo att43 = new AttributeInfo("maxChiSquaredTestValuesForGeneratedAttributeAndAllDatasetAttributes", Column.columnType.Numeric, maxChiSquaredTestValuesForGeneratedAttributeAndAllDatasetAttributes, -1);
            AttributeInfo att44 = new AttributeInfo("minChiSquaredTestValuesForGeneratedAttributeAndAllDatasetAttributes", Column.columnType.Numeric, minChiSquaredTestValuesForGeneratedAttributeAndAllDatasetAttributes, -1);
            AttributeInfo att45 = new AttributeInfo("avgChiSquaredTestValuesForGeneratedAttributeAndAllDatasetAttributes", Column.columnType.Numeric, avgChiSquaredTestValuesForGeneratedAttributeAndAllDatasetAttributes, -1);
            AttributeInfo att46 = new AttributeInfo("stdevChiSquaredTestValuesForGeneratedAttributeAndAllDatasetAttributes", Column.columnType.Numeric, stdevChiSquaredTestValuesForGeneratedAttributeAndAllDatasetAttributes, -1);

            attributes.put(attributes.size(), att1);
            attributes.put(attributes.size(), att2);
            attributes.put(attributes.size(), att3);
            attributes.put(attributes.size(), att4);
            attributes.put(attributes.size(), att5);
            attributes.put(attributes.size(), att6);
            attributes.put(attributes.size(), att7);
            attributes.put(attributes.size(), att8);
            attributes.put(attributes.size(), att9);
            attributes.put(attributes.size(), att10);
            attributes.put(attributes.size(), att11);
            attributes.put(attributes.size(), att12);
            attributes.put(attributes.size(), att13);
            attributes.put(attributes.size(), att14);
            attributes.put(attributes.size(), att15);
            attributes.put(attributes.size(), att16);
            attributes.put(attributes.size(), att17);
            attributes.put(attributes.size(), att18);
            attributes.put(attributes.size(), att19);
            attributes.put(attributes.size(), att20);
            attributes.put(attributes.size(), att21);
            attributes.put(attributes.size(), att22);
            attributes.put(attributes.size(), att23);
            attributes.put(attributes.size(), att24);
            attributes.put(attributes.size(), att25);
            attributes.put(attributes.size(), att26);
            attributes.put(attributes.size(), att27);
            attributes.put(attributes.size(), att28);
            attributes.put(attributes.size(), att29);
            attributes.put(attributes.size(), att30);
            attributes.put(attributes.size(), att31);
            attributes.put(attributes.size(), att32);
            attributes.put(attributes.size(), att33);
            attributes.put(attributes.size(), att34);
            attributes.put(attributes.size(), att35);
            attributes.put(attributes.size(), att36);
            attributes.put(attributes.size(), att37);
            attributes.put(attributes.size(), att38);
            attributes.put(attributes.size(), att39);
            attributes.put(attributes.size(), att40);
            attributes.put(attributes.size(), att41);
            attributes.put(attributes.size(), att42);
            attributes.put(attributes.size(), att43);
            attributes.put(attributes.size(), att44);
            attributes.put(attributes.size(), att45);
            attributes.put(attributes.size(), att46);
        }
        catch (Exception ex) {
            int x=5;
        }

        return attributes;
    }


    /**
     * Used to calculate statistics on the correlation of the generates attribute and the attributes of the dataset.
     * The attributes that were used to generate the feature are excluded.
     * @param dataset
     * @param generatedAttribute
     */
    private void processGeneratedAttribute(Dataset dataset, OperatorAssignment oa, ColumnInfo generatedAttribute, Properties properties) throws Exception {
        //IMPORTANT: make sure that the source and target attributes are not included in these calculations
        List<ColumnInfo> discreteColumns = dataset.getAllColumnsOfType(Column.columnType.Discrete, false);
        List<ColumnInfo> numericColumns = dataset.getAllColumnsOfType(Column.columnType.Numeric, false);

        //The paired T-Tests for the dataset's numeric attributes
        if (generatedAttribute.getColumn().getType() == Column.columnType.Numeric) {
            List<Double> pairedTTestScores = statisticOperations.calculatePairedTTestValues(filterOperatorAssignmentAttributes(numericColumns, oa), generatedAttribute);
            if (pairedTTestScores.size() > 0) {
                maxPairedTTestValuesForGeneratedAttributeAndDatasetNumericAttributes = pairedTTestScores.stream().mapToDouble(x -> x).max().getAsDouble();
                minPairedTTestValuesForGeneratedAttributeAndDatasetNumericAttributes = pairedTTestScores.stream().mapToDouble(x -> x).min().getAsDouble();
                avgPairedTTestValuesForGeneratedAttributeAndDatasetNumericAttributes = pairedTTestScores.stream().mapToDouble(x -> x).average().getAsDouble();
                double tempStdev = pairedTTestScores.stream().mapToDouble(a -> Math.pow(a - this.avgPairedTTestValuesForGeneratedAttributeAndDatasetNumericAttributes, 2)).sum();
                stdevPairedTTestValuesForGeneratedAttributeAndDatasetNumericAttributes = Math.sqrt(tempStdev / pairedTTestScores.size());
            } else {
                maxPairedTTestValuesForGeneratedAttributeAndDatasetNumericAttributes = 0;
                minPairedTTestValuesForGeneratedAttributeAndDatasetNumericAttributes = 0;
                avgPairedTTestValuesForGeneratedAttributeAndDatasetNumericAttributes = 0;
                stdevPairedTTestValuesForGeneratedAttributeAndDatasetNumericAttributes = 0;
            }
        }

        //The chi Squared test for the dataset's dicrete attribtues
        List<Double> chiSquareTestsScores = statisticOperations.calculateChiSquareTestValues(filterOperatorAssignmentAttributes(discreteColumns,oa),generatedAttribute,dataset, properties);
        if (chiSquareTestsScores.size() > 0) {
            maxChiSquaredTestValuesForGeneratedAttributeAndDatasetDiscreteAttributes = chiSquareTestsScores.stream().mapToDouble(x -> x).max().getAsDouble();
            minChiSquaredTestValuesForGeneratedAttributeAndDatasetDiscreteAttributes = chiSquareTestsScores.stream().mapToDouble(x -> x).max().getAsDouble();
            avgChiSquaredTestValuesForGeneratedAttributeAndDatasetDiscreteAttributes = chiSquareTestsScores.stream().mapToDouble(x -> x).max().getAsDouble();
            double tempStdev = chiSquareTestsScores.stream().mapToDouble(a -> Math.pow(a - this.avgChiSquaredTestValuesForGeneratedAttributeAndDatasetDiscreteAttributes, 2)).sum();
            stdevChiSquaredTestValuesForGeneratedAttributeAndDatasetDiscreteAttributes = Math.sqrt(tempStdev / chiSquareTestsScores.size());
        }
        else {
            maxChiSquaredTestValuesForGeneratedAttributeAndDatasetDiscreteAttributes = 0;
            minChiSquaredTestValuesForGeneratedAttributeAndDatasetDiscreteAttributes = 0;
            avgChiSquaredTestValuesForGeneratedAttributeAndDatasetDiscreteAttributes = 0;
            stdevChiSquaredTestValuesForGeneratedAttributeAndDatasetDiscreteAttributes = 0;
        }

        //The Chi Square test for ALL the dataset's attirbutes (Numeric attributes will be discretized)
        discreteColumns.addAll(numericColumns);
        List<Double> AllAttributesChiSquareTestsScores = statisticOperations.calculateChiSquareTestValues(filterOperatorAssignmentAttributes(discreteColumns,oa),generatedAttribute,dataset, properties);
        if (AllAttributesChiSquareTestsScores.size() > 0) {
            maxChiSquaredTestValuesForGeneratedAttributeAndAllDatasetAttributes = AllAttributesChiSquareTestsScores.stream().mapToDouble(x -> x).max().getAsDouble();
            minChiSquaredTestValuesForGeneratedAttributeAndAllDatasetAttributes = AllAttributesChiSquareTestsScores.stream().mapToDouble(x -> x).min().getAsDouble();
            avgChiSquaredTestValuesForGeneratedAttributeAndAllDatasetAttributes = AllAttributesChiSquareTestsScores.stream().mapToDouble(x -> x).average().getAsDouble();
            double tempStdev = AllAttributesChiSquareTestsScores.stream().mapToDouble(a -> Math.pow(a - this.avgChiSquaredTestValuesForGeneratedAttributeAndAllDatasetAttributes, 2)).sum();
            stdevChiSquaredTestValuesForGeneratedAttributeAndAllDatasetAttributes = Math.sqrt(tempStdev / AllAttributesChiSquareTestsScores.size());
        }
        else {
            maxChiSquaredTestValuesForGeneratedAttributeAndAllDatasetAttributes = 0;
            minChiSquaredTestValuesForGeneratedAttributeAndAllDatasetAttributes = 0;
            avgChiSquaredTestValuesForGeneratedAttributeAndAllDatasetAttributes = 0;
            stdevChiSquaredTestValuesForGeneratedAttributeAndAllDatasetAttributes =  0;
        }
    }


    private void performStatisticalTestOnOperatorAssignmentAndDatasetAtributes(Dataset dataset, OperatorAssignment oa, Properties properties) throws Exception {
        //first we put all the OA attributes (sources and targets) in one list. Numeric atts are discretized, other non-discretes are ignored
        List<ColumnInfo> columnsToAnalyze = new ArrayList<>();
        for (ColumnInfo ci: oa.getSources()) {
            if (ci.getColumn().getType() == Column.columnType.Discrete) {
                columnsToAnalyze.add(ci);
            }
            else {
                if (ci.getColumn().getType() == Column.columnType.Numeric) {
                    columnsToAnalyze.add(statisticOperations.discretizeNumericColumn(dataset,ci,null, properties));
                }
            }
        }
        if (oa.getTragets() != null) {
            for (ColumnInfo ci : oa.getTragets()) {
                if (ci.getColumn().getType() == Column.columnType.Discrete) {
                    columnsToAnalyze.add(ci);
                } else {
                    if (ci.getColumn().getType() == Column.columnType.Numeric) {
                        columnsToAnalyze.add(statisticOperations.discretizeNumericColumn(dataset, ci,null, properties));
                    }
                }
            }
        }

        //For each attribute in the list we created, we iterate over all the attributes in the dataset (all those that are not in the OA)
        List<Double> chiQuareTestValues = new ArrayList<>();
        for (ColumnInfo ci : columnsToAnalyze) {
            for (ColumnInfo datasetCI : dataset.getAllColumns(false)) {
                //if datasetCI is in the OA then skip
                if (oa.getSources().contains(datasetCI) || (oa.getTragets() != null && oa.getTragets().contains(datasetCI))) {
                    continue;
                }
                double chiSquareTestValue = 0;
                if (datasetCI.getColumn().getType() == Column.columnType.Date || datasetCI.getColumn().getType() == Column.columnType.String) {
                    continue;
                }
                if (datasetCI.getColumn().getType() == Column.columnType.Discrete) {
                    chiSquareTestValue = chiSquareTest.chiSquare(statisticOperations.generateDiscreteAttributesCategoryIntersection((DiscreteColumn)ci.getColumn(),(DiscreteColumn)datasetCI.getColumn()));
                }
                if (datasetCI.getColumn().getType() == Column.columnType.Numeric) {
                    ColumnInfo tempCI = statisticOperations.discretizeNumericColumn(dataset, datasetCI,null, properties);
                    chiSquareTestValue = chiSquareTest.chiSquare(statisticOperations.generateDiscreteAttributesCategoryIntersection((DiscreteColumn)ci.getColumn(),(DiscreteColumn)tempCI.getColumn()));
                }
                if (!Double.isNaN(chiSquareTestValue) && !Double.isInfinite(chiSquareTestValue)) {
                    chiQuareTestValues.add(chiSquareTestValue);
                }
            }
        }

        //now we calculate the max/min/avg/stdev
        if (chiQuareTestValues.size() > 0) {
            maxChiSquareTestvalueForSourceDatasetAttributes = chiQuareTestValues.stream().mapToDouble(x -> x).max().getAsDouble();
            minChiSquareTestvalueForSourceDatasetAttributes = chiQuareTestValues.stream().mapToDouble(x -> x).min().getAsDouble();
            avgChiSquareTestvalueForSourceDatasetAttributes = chiQuareTestValues.stream().mapToDouble(x -> x).average().getAsDouble();
            double tempStdev = chiQuareTestValues.stream().mapToDouble(a -> Math.pow(a - this.avgChiSquareTestvalueForSourceDatasetAttributes, 2)).sum();
            stdevChiSquareTestvalueForSourceDatasetAttributes = Math.sqrt(tempStdev / chiQuareTestValues.size());
        }
        else {
            maxChiSquareTestvalueForSourceDatasetAttributes = 0;
            minChiSquareTestvalueForSourceDatasetAttributes = 0;
            avgChiSquareTestvalueForSourceDatasetAttributes = 0;
            stdevChiSquareTestvalueForSourceDatasetAttributes = 0;
        }
    }

    private void performStatisticalTestsOnSourceAndTargetAttributes(Dataset dataset, OperatorAssignment oa, Properties properties) throws Exception {
        //Chi Square test for discrete source attributes
        this.chiSquareTestValueForSourceAttributes = 0;
        if (oa.getSources().size() == 2) {
            if (oa.getSources().get(0).getColumn().getType() == Column.columnType.Discrete && oa.getSources().get(1).getColumn().getType() == Column.columnType.Discrete) {
                DiscreteColumn dc1 = (DiscreteColumn)oa.getSources().get(0).getColumn();
                DiscreteColumn dc2 = (DiscreteColumn)oa.getSources().get(1).getColumn();

                double tempVal = chiSquareTest.chiSquare(statisticOperations.generateDiscreteAttributesCategoryIntersection(dc1,dc2));
                if (!Double.isNaN(tempVal) && !Double.isInfinite(tempVal)) {
                    this.chiSquareTestValueForSourceAttributes = tempVal;
                }
                else {
                    this.chiSquareTestValueForSourceAttributes = -1;
                }
            }
        }

        //Paired T-Test for numeric source and target
        this.pairedTTestValueForSourceAndTargetAttirbutes = 0;
        if (oa.getSources().size() == 1 && oa.getSources().get(0).getColumn().getType() == Column.columnType.Numeric && oa.getTragets() != null && oa.getTragets().size() == 1) {
            this.pairedTTestValueForSourceAndTargetAttirbutes = tTest.pairedTTest((double[]) oa.getSources().get(0).getColumn().getValues(), (double[]) oa.getTragets().get(0).getColumn().getValues());
        }

        //The chiSquare Test scores of all source and target attribtues (numeric atts are discretized, other non-discrete types are ignored)
        if (oa.getSources().size() == 1 && oa.getTragets() == null) {
            maxChiSquareTsetForSourceAndTargetAttributes = 0;
            minChiSquareTsetForSourceAndTargetAttributes = 0;
            avgChiSquareTsetForSourceAndTargetAttributes = 0;
            stdevChiSquareTsetForSourceAndTargetAttributes = 0;
        }
        else {
            List<ColumnInfo> columnsToAnalyze = new ArrayList<>();
            for (ColumnInfo ci: oa.getSources()) {
                if (ci.getColumn().getType() == Column.columnType.Discrete) {
                    columnsToAnalyze.add(ci);
                }
                else {
                    if (ci.getColumn().getType() == Column.columnType.Numeric) {
                        columnsToAnalyze.add(statisticOperations.discretizeNumericColumn(dataset,ci,null, properties));
                    }
                }
            }
            if (columnsToAnalyze.size() > 1) {
                List<Double> chiSquareTestValues = new ArrayList<>();
                for (int i=0; i< columnsToAnalyze.size()-1; i++) {
                    for (int j=i+1; j< columnsToAnalyze.size(); j++) {
                        double chiSquareTestVal = chiSquareTest.chiSquare(statisticOperations.generateDiscreteAttributesCategoryIntersection(
                                (DiscreteColumn)columnsToAnalyze.get(i).getColumn(), (DiscreteColumn)columnsToAnalyze.get(j).getColumn()));
                        if (!Double.isNaN(chiSquareTestVal) && !Double.isInfinite(chiSquareTestVal)) {
                            chiSquareTestValues.add(chiSquareTestVal);
                        }
                    }
                }
                if (chiSquareTestValues.size() > 0) {
                    maxChiSquareTsetForSourceAndTargetAttributes = chiSquareTestValues.stream().mapToDouble(x -> x).max().getAsDouble();
                    minChiSquareTsetForSourceAndTargetAttributes = chiSquareTestValues.stream().mapToDouble(x -> x).max().getAsDouble();
                    avgChiSquareTsetForSourceAndTargetAttributes = chiSquareTestValues.stream().mapToDouble(x -> x).average().getAsDouble();
                    double tempStdev = chiSquareTestValues.stream().mapToDouble(a -> Math.pow(a - this.avgChiSquareTsetForSourceAndTargetAttributes, 2)).sum();
                    stdevChiSquareTsetForSourceAndTargetAttributes = Math.sqrt(tempStdev / chiSquareTestValues.size());
                }
                else {
                    maxChiSquareTsetForSourceAndTargetAttributes = 0;
                    minChiSquareTsetForSourceAndTargetAttributes = 0;
                    avgChiSquareTsetForSourceAndTargetAttributes = 0;
                    stdevChiSquareTsetForSourceAndTargetAttributes = 0;
                }
            }
        }
    }


    private void processSourceAndTargetAttributes(Dataset dataset, OperatorAssignment oa) {
        //start by computing statistics on the discrete source attributes
        List<Double> sourceAttributesValuesList = new ArrayList<>();
        for (ColumnInfo sourceAttribute : oa.getSources()) {
            if (sourceAttribute.getColumn().getType() == Column.columnType.Discrete) {
                sourceAttributesValuesList.add((double)((DiscreteColumn)sourceAttribute.getColumn()).getNumOfPossibleValues());
            }
        }
        if (sourceAttributesValuesList.size() == 0) {
            maxNumOfDiscreteSourceAttribtueValues = 0;
            minNumOfDiscreteSourceAttribtueValues = 0;
            avgNumOfDiscreteSourceAttribtueValues = 0;
            stdevNumOfDiscreteSourceAttribtueValues = 0;
        }
        else {
            maxNumOfDiscreteSourceAttribtueValues = sourceAttributesValuesList.stream().mapToDouble(x->x).max().getAsDouble();
            minNumOfDiscreteSourceAttribtueValues = sourceAttributesValuesList.stream().mapToDouble(x->x).min().getAsDouble();
            avgNumOfDiscreteSourceAttribtueValues = sourceAttributesValuesList.stream().mapToDouble(x->x).average().getAsDouble();
            double tempStdev = sourceAttributesValuesList.stream().mapToDouble(a -> Math.pow(a - this.avgNumOfDiscreteSourceAttribtueValues, 2)).sum();
            this.stdevNumOfDiscreteSourceAttribtueValues = Math.sqrt(tempStdev / sourceAttributesValuesList.size());
        }

        //Statistics on numeric target attribute (we currently support a single attribute)
        if (oa.getTragets() == null || oa.getTragets().get(0).getColumn().getType() != Column.columnType.Numeric) {
            maxValueOfNumericTargetAttribute = 0;
            minValueOfNumericTargetAttribute = 0;
            avgValueOfNumericTargetAttribute = 0;
            stdevValueOfNumericTargetAttribute = 0;
        }
        else {
            //
            //List<Double> numericTargetAttributeValues = Arrays.asList((Double[])oa.getTragets().get(0).getColumn().getValues());
            Double[] arr = ArrayUtils.toObject((double[])oa.getTragets().get(0).getColumn().getValues());
            ArrayList<Double> numericTargetAttributeValues = new ArrayList<>(Arrays.asList(arr));
            maxValueOfNumericTargetAttribute = numericTargetAttributeValues.stream().mapToDouble(x->x).max().getAsDouble();
            minValueOfNumericTargetAttribute = numericTargetAttributeValues.stream().mapToDouble(x->x).min().getAsDouble();
            avgValueOfNumericTargetAttribute = numericTargetAttributeValues.stream().mapToDouble(x->x).average().getAsDouble();
            double tempStdev = numericTargetAttributeValues.stream().mapToDouble(a -> Math.pow(a - this.avgValueOfNumericTargetAttribute,2)).sum();
            this.stdevValueOfNumericTargetAttribute = Math.sqrt(tempStdev / numericTargetAttributeValues.size());
        }

        if (oa.getSources().get(0).getColumn().getType() != Column.columnType.Numeric) {
            maxValueOfNumericSourceAttribute = 0;
            minValueOfNumericSourceAttribute = 0;
            avgValueOfNumericSourceAttribute = 0;
            stdevValueOfNumericSourceAttribute = 0;
        }
        else {
            Double[] arr1 = ArrayUtils.toObject((double[])oa.getSources().get(0).getColumn().getValues());
            ArrayList<Double> numericSourceAttributeValues = new ArrayList<>(Arrays.asList(arr1));
            maxValueOfNumericSourceAttribute = numericSourceAttributeValues.stream().mapToDouble(x->x).max().getAsDouble();
            minValueOfNumericSourceAttribute = numericSourceAttributeValues.stream().mapToDouble(x->x).min().getAsDouble();
            avgValueOfNumericSourceAttribute = numericSourceAttributeValues.stream().mapToDouble(x->x).average().getAsDouble();
            double tempStdev = numericSourceAttributeValues.stream().mapToDouble(a -> Math.pow(a - this.avgValueOfNumericSourceAttribute,2)).sum();
            this.stdevValueOfNumericSourceAttribute = Math.sqrt(tempStdev / numericSourceAttributeValues.size());
        }


    }

    /**
     * Analyzes the characteristics of the OperatorAssignment object - the characteristics of the feature
     * that make up the object. Here we do not process the analyzed attribute itself.
     * @param dataset
     * @param oa
     */
    private void ProcessOperatorAssignment(Dataset dataset, OperatorAssignment oa) throws Exception {
        //numOfSources
        if (oa.getSources() != null) {
            this.numOfSources = oa.getSources().size();
        }
        else {
            this.numOfSources = 0;
        }

        //numOfNumericSources + numOfDiscreteSources + numOfDateSources
        this.numOfNumericSources = 0;
        if (oa.getSources() != null) {
            for (ColumnInfo ci : oa.getSources()) {
                if (ci.getColumn().getType() == Column.columnType.Numeric) {
                    this.numOfNumericSources++;
                }
                if (ci.getColumn().getType() == Column.columnType.Discrete) {
                    this.numOfDiscreteSources++;
                }
                if (ci.getColumn().getType() == Column.columnType.Date) {
                    this.numOfDateSources++;
                }
            }
        }

        //operatorType
        this.operatorTypeIdentifier = getOperatorTypeID(oa.getOperator().getType());

        //operatorName
        this.operatorIdentifier = GetOperatorIdentifier(oa.getOperator());

        //isOutputDiscrete
        if (oa.getSecondaryOperator() != null) {
            if ((oa.getSecondaryOperator().getOutputType().equals(Operator.outputType.Discrete))) {
                this.isOutputDiscrete = 1;
            }
            else {
                this.isOutputDiscrete = 0;
            }

        }
        else {
            if (oa.getOperator().getOutputType().equals(Operator.outputType.Discrete)) {
                this.isOutputDiscrete = 1;
            }
            else {
                this.isOutputDiscrete = 0;
            }
        }

        this.discretizerInUse = getDiscretizerID(oa.getSecondaryOperator());

        this.normalizerInUse = getNormalizerID(oa.getSecondaryOperator());

        this.numOfDiscreteValues = getNumOfNewAttributeDiscreteValues(oa);
    }

    public int getNumOfNewAttributeDiscreteValues(OperatorAssignment oa) {
        if (oa.getSecondaryOperator() != null) {
            return oa.getSecondaryOperator().getNumOfBins();
        }
        else {
            if (oa.getOperator().getOutputType() != Operator.outputType.Discrete) {
                return -1;
            }
            else {
                //currently the only operators which return a discrete value are the Unary.
                return ((UnaryOperator)oa.getOperator()).getNumOfBins();
            }
        }
    }

    /**
     * Returns an integer that represents the type of the operator in use
     * @param operatorType
     * @return
     * @throws Exception
     */
    private int getOperatorTypeID(Operator.operatorType operatorType) throws Exception {
        if (operatorType ==Operator.operatorType.Unary) {
            return 1;
        }
        if (operatorType ==Operator.operatorType.Binary) {
            return 2;
        }
        if (operatorType ==Operator.operatorType.GroupByThen) {
            return 3;
        }
        if (operatorType ==Operator.operatorType.TimeBasedGroupByThen) {
            return 4;
        }
        throw new Exception("Unrecognized operator type");
    }

    private int GetOperatorIdentifier(Operator operator) throws Exception {

        if (operator == null) {
            return 0;
        }

        String name = operator.getName();
        if (name.contains("_")) {
            name = name.substring(0, name.indexOf("_")-1);
        }

        switch (name) {
            case "EqualRangeDiscretizerUnaryOperator":
                return 1;
            case "DayOfWeekUnaryOperator":
                return 2;
            case "HourOfDayUnaryOperator":
                return 3;
            case "IsWeekendUnaryOperator":
                return 4;
            case "StandardScoreUnaryOperator":
                return 5;
            case "AddBinaryOperator":
                return 6;
            case "DivisionBinaryOperator":
                return 7;
            case "MultiplyBinaryOperator":
                return 8;
            case "SubtractBinaryOperator":
                return 9;
            case "GroupByThenAvg":
                return 10;
            case "GroupByThenCount":
                return 11;
            case "GroupByThenMax":
                return 12;
            case "GroupByThenMin":
                return 13;
            case "GroupByThenStdev":
                return 14;
            case "TimeBasedGroupByThenCountAndAvg":
                return 15;
            case "TimeBasedGroupByThenCountAndCount":
                return 16;
            case "TimeBasedGroupByThenCountAndMax":
                return 17;
            case "TimeBasedGroupByThenCountAndMin":
                return 18;
            case "TimeBasedGroupByThenCountAndStdev":
                return 19;
            default:
                throw new Exception("Unidentified operator in use");
        }
    }

    private int getDiscretizerID(UnaryOperator uo) {
        if (uo == null) {
            return 0;
        }
        switch (uo.getName()) {
            case "EqualRangeDiscretizerUnaryOperator":
                return 1;
            case "DayOfWeekUnaryOperator":
                return 2;
            case "HourOfDayUnaryOperator":
                return 3;
            case "IsWeekendUnaryOperator":
                return 4;
            default:
                //we can get here because even if the opertaor is not null, it may be a normalizer and not a discretizer
                return 0;
        }
    }

    private int getNormalizerID (UnaryOperator uo) {
        if (uo == null) {
            return 0;
        }
        switch (uo.getName()) {
            case "StandardScoreUnaryOperator":
                return 1;
            default:
                return 0;
        }
    }

    /**
     * Receives a list of attirbutes and an OperatorAssignment object. The function filters out the source and target
     * attribtues of the OperatorAssignmetn from the given list.
     * @param attributesList
     * @param operatorAssignment
     * @return
     */
    private List<ColumnInfo> filterOperatorAssignmentAttributes(List<ColumnInfo> attributesList, OperatorAssignment operatorAssignment) {
        List<ColumnInfo> listToReturn = new ArrayList<>();
        for (ColumnInfo ci : attributesList) {
            if (operatorAssignment.getSources().contains(ci)){
                continue;
            }
            if (operatorAssignment.getTragets() != null && operatorAssignment.getTragets().contains(ci)) {
                continue;
            }
            listToReturn.add(ci);
        }
        return listToReturn;
    }












}
