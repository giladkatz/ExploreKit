package explorekit.Evaluation.WrapperEvaluation;

import explorekit.Evaluation.ClassificationResults;
import explorekit.Evaluation.Evaluator;
import explorekit.data.ColumnInfo;
import explorekit.data.Dataset;
import explorekit.operators.OperatorAssignment;
import weka.classifiers.evaluation.Evaluation;

import java.io.InputStream;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.Properties;

/**
 * Created by giladkatz on 20/02/2016.
 */
public class AucWrapperEvaluator extends WrapperEvaluator {
    public double produceScore(Dataset analyzedDatasets, ClassificationResults currentScore, Dataset completeDataset, OperatorAssignment oa, ColumnInfo candidateAttribute, Properties properties) throws Exception {
        if (candidateAttribute != null) {
            analyzedDatasets.addColumn(candidateAttribute);
        }
        Evaluation evaluationResults = runClassifier(properties.getProperty("classifier"), analyzedDatasets.generateSet(true), analyzedDatasets.generateSet(false), properties);

        //in order to deal with multi-class datasets we calculate an average of all AUC scores (we may need to make this weighted)
        double auc = CalculateAUC(evaluationResults, analyzedDatasets);

        if (currentScore != null) {
            return auc - currentScore.getAuc();
        }
        else {
            return auc;
        }
    }

    public double produceScoreWithSampling(Dataset dataset, ClassificationResults currentScore,
                                    int numOfSamplesPerFold, int numOfInstancesInSample, int randomSeed, Properties properties) throws Exception {
        double avgAuc = 0;
        for (int i=0; i<numOfSamplesPerFold; i++) {
            //it's important to note that the training set is generated using the sampling. The random seed is (original seed+i) so that
            // we have different indices each time but also the same indices across training iterations
            Evaluation evaluationResults = runClassifier(properties.getProperty("classifier"), dataset.generateSetWithSampling(numOfInstancesInSample,randomSeed+i), dataset.generateSet(false), properties);
            double auc = CalculateAUC(evaluationResults, dataset);
            NumberFormat formatter = new DecimalFormat("#0.0000");
            auc = Double.parseDouble(formatter.format(auc));
            if (currentScore != null) {
                avgAuc += auc - currentScore.getAuc();
            }
            else {
                avgAuc += auc;
            }
        }
        return avgAuc/numOfSamplesPerFold;
    }

    public Evaluator.evaluatorScoringMethod getEvaluatorScoreingMethod() {
        return Evaluator.evaluatorScoringMethod.AUC;
    }

    public WrapperEvaluator getCopy() {
        return new AucWrapperEvaluator();
    }
}
