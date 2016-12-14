package explorekit.Evaluation.WrapperEvaluation;

import explorekit.Evaluation.ClassificationResults;
import explorekit.Evaluation.Evaluator;
import explorekit.data.ColumnInfo;
import explorekit.data.Dataset;
import explorekit.operators.OperatorAssignment;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.evaluation.Prediction;

import java.io.InputStream;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Locale;
import java.util.Properties;

/**
 * Created by giladkatz on 20/02/2016.
 */
public class LogLossWrapperEvaluator extends WrapperEvaluator {
    public double produceScore(Dataset analyzedDatasets, ClassificationResults currentScore, Dataset completeDataset, OperatorAssignment oa, ColumnInfo candidateAttribute, Properties properties) throws Exception {
        if (candidateAttribute != null) {
            analyzedDatasets.addColumn(candidateAttribute);
        }

        Evaluation evaluationResults = runClassifier(properties.getProperty("classifier"), analyzedDatasets.generateSet(true), analyzedDatasets.generateSet(false), properties);
        double logloss = CalculateLogLoss(evaluationResults, analyzedDatasets);

        if (currentScore != null) {
            return logloss - currentScore.getLogLoss();
        }
        else {
            return logloss;
        }
    }
    public double produceScoreWithSampling(Dataset dataset, ClassificationResults currentScore,
                                           int numOfSamplesPerFold, int numOfInstancesInSample, int randomSeed, Properties properties) throws Exception {
        double avgLogloss = 0;
        for (int i=0; i<numOfSamplesPerFold; i++) {
            //it's important to note that the training set is generated using the sampling. The random seed is (original seed+i) so that
            // we have different indices each time but also the same indices across training iterations
            Evaluation evaluationResults = runClassifier(properties.getProperty("classifier"), dataset.generateSetWithSampling(numOfInstancesInSample,randomSeed+i), dataset.generateSet(false), properties);
            double logloss = CalculateLogLoss(evaluationResults, dataset);
            NumberFormat formatter = new DecimalFormat("#0.0000");
            formatter = formatter.getInstance(Locale.US);

            logloss = Double.parseDouble(formatter.format(logloss));
            if (currentScore != null) {
                avgLogloss += logloss - currentScore.getLogLoss();
            }
            else {
                avgLogloss += logloss;
            }
        }
        return avgLogloss/numOfSamplesPerFold;

    }

    public Evaluator.evaluatorScoringMethod getEvaluatorScoreingMethod() {
        return Evaluator.evaluatorScoringMethod.LogLoss;
    }

    public WrapperEvaluator getCopy() {
        return new LogLossWrapperEvaluator();
    }
}