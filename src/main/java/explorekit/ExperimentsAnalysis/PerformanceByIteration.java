package explorekit.ExperimentsAnalysis;

import java.io.*;

import static explorekit.operators.OperatorsAssignmentsManager.properties;

/**
 * Used to evaluate the perfomance of our approach as a function of the number of iterations
 */
public class PerformanceByIteration {

    public int[] getMaxPerformanceIterationIndex(String directory) throws Exception {
        File folder = new File(directory);
        File[] filesArray = folder.listFiles();

        int[] maxScoreByIndex = new int[15];

        for (int i=0; i<filesArray.length; i++) {
            double maxValue = 0;
            int maxValueIteration = 0;
            BufferedReader br = new BufferedReader(new FileReader(filesArray[i]));
            String line;

            int counter = 0;
            line = br.readLine();
            while ((line = br.readLine()) != null && counter < 16) {
                int quotationIndex = line.lastIndexOf("\"");
                if (quotationIndex != -1) {
                    line = line.substring(quotationIndex);
                }
                String[] splitLine = line.split(",");

                double currentVal = -1;
                try {
                    currentVal = Double.parseDouble(splitLine[2]);
                }
                catch (Exception ex) {
                    int x=5;
                }
                if (currentVal > maxValue) {
                    maxValue = currentVal;
                    maxValueIteration = counter;
                }
                counter++;
            }
            //if the maxIteration is 0, then we didn't have improvement over the baseline, and therefore we ignore that
            if (maxValueIteration > 0) {
                maxScoreByIndex[maxValueIteration-1]++;
            }
        }

        System.out.println("Results summary:");
        for (int i=0; i<maxScoreByIndex.length; i++) {
            System.out.println(i+1 + ":  " + maxScoreByIndex[i]);
        }

        return maxScoreByIndex;
    }
}
