package explorekit.search;

import explorekit.Evaluation.FilterEvaluators.MLFilterEvaluator;
import explorekit.Evaluation.WrapperEvaluation.AucWrapperEvaluator;
import explorekit.Evaluation.FilterEvaluators.FilterEvaluator;
import explorekit.Evaluation.FilterEvaluators.InformationGainFilterEvaluator;
import explorekit.Evaluation.WrapperEvaluation.WrapperEvaluator;
import explorekit.data.Dataset;
import explorekit.search.AttributeRankersFilters.AttributeRankerFilter;
import explorekit.search.AttributeRankersFilters.FilterScoreRanker;
import explorekit.search.AttributeRankersFilters.FilterScoreWithExclusionsRanker;
import explorekit.search.AttributeRankersFilters.WrapperScoreRanker;

import java.io.FileWriter;
import java.io.InputStream;
import java.util.Date;
import java.util.Properties;

/**
 * Created by giladkatz on 11/02/2016.
 */
public abstract class Search {

    /**
     * Begins the generation and evaluation of the candidate attributes
     * @param dataset
     * @param runInfo
     * @throws Exception
     */
    public void run(Dataset dataset, String runInfo) throws Exception {}

    /**
     * Used to run Weka on the dataset and produce all relevant statistics.
     * IMPORTANT: we currently assume that the target class is discrete
     * @param dataset
     */
    protected void evaluateDataset(Dataset dataset) throws Exception{

    }

    /**
     * Returns the requested wrapper (initialized)
     * @param wrapperName
     * @return
     * @throws Exception
     */
    public WrapperEvaluator getWrapper(String wrapperName) throws Exception {
        switch(wrapperName) {
            case "AucWrapperEvaluator":
                AucWrapperEvaluator awe = new AucWrapperEvaluator();
                return awe;
            default:
                throw new Exception("Unidentified wrapper");
        }
    }

    /**
     * Returns the requested filter (initialized)
     * @param filterName
     * @return
     * @throws Exception
     */
    public FilterEvaluator getFilter(String filterName, Dataset dataset, Properties properties) throws Exception {
        switch (filterName) {
            case "InformationGainFilterEvaluator":
                InformationGainFilterEvaluator igfe = new InformationGainFilterEvaluator();
                return igfe;
            case "MLFilterEvaluator":
                MLFilterEvaluator mlfe = new MLFilterEvaluator(dataset, properties);
                return mlfe;
            default:
                throw new Exception("Unidentified evaluator");
        }
    }

    /**
     * Returns the requested ranker filter
     * @param rankerFilterName
     * @return
     * @throws Exception
     */
    public AttributeRankerFilter getRankerFilter(String rankerFilterName) throws Exception {
        switch (rankerFilterName) {
            case "FilterScoreRanker":
                FilterScoreRanker fsr = new FilterScoreRanker();
                return fsr;
            case "WrapperScoreRanker":
                WrapperScoreRanker wsr = new WrapperScoreRanker();
                return wsr;
            case "FilterScoreWithExclusionsRanker":
                FilterScoreWithExclusionsRanker fswer = new FilterScoreWithExclusionsRanker();
                return fswer;
            default:
                throw new Exception("Unidentified rankerFilter");
        }
    }

    public void writeFinalStatisticsToResultsFile(String datasetName, String runInfo, Date experimentStartTime, int totalNumOfWrapperEvaluations) throws Exception {
        Properties properties = new Properties();
        InputStream input = this.getClass().getClassLoader().getResourceAsStream("config.properties");
        properties.load(input);
        String filename= properties.getProperty("resultsFilePath") + datasetName + runInfo + ".csv";
        FileWriter fw = new FileWriter(filename,true);

        Date experimentEndTime = new Date();
        long diff = experimentEndTime.getTime() -experimentStartTime.getTime();
        long diffSeconds = diff / 1000 % 60;
        long diffMinutes = diff / (60 * 1000) % 60;
        long diffHours = diff / (60 * 60 * 1000);

        fw.write("\n");
        fw.write("Total Run Time: " + "\n");
        fw.write("Number of hours: ," + diffHours + "\n");
        fw.write("Number of minutes: ," + diffMinutes + "\n");
        fw.write("Number of seconds: ," + diffSeconds + "\n");
        fw.write("Total number of evaluated attribtues: " + totalNumOfWrapperEvaluations);
        fw.close();
    }
}
