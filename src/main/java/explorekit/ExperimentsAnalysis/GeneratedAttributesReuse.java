package explorekit.ExperimentsAnalysis;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

/**
 * Counts how many times a generated feature is used in later iterations
 */
public class GeneratedAttributesReuse {

    public HashMap<Integer,Integer> getNumberOfGeneratedAttributesReuses(String directory) throws Exception {

        File folder = new File(directory);
        File[] filesArray = folder.listFiles();

        HashMap<Integer,Integer> reusesMap = new HashMap<>();

        for (int i=0; i<filesArray.length; i++) {
            List<String> usedAttributes = new ArrayList<>();
            BufferedReader br = new BufferedReader(new FileReader(filesArray[i]));
            String line;
            int counter = 0;
            int reusesCoutner = 0;
            line = br.readLine();
            while ((line = br.readLine()) != null && counter < 15) {
                String att="";
                if (line.indexOf("\"") == -1) {
                    continue;
                }
                try {
                    att = line.substring(line.indexOf("\"") + 1, line.lastIndexOf("\""));
                    att = att.replace("{","");
                    att = att.replace("}","");
                }
                catch (Exception ex) {
                    int x=5;
                }
                if (att.length()<3) {
                    continue;
                }

                for (String usedAtt : usedAttributes) {
                    if (att.contains(usedAtt)) {
                        reusesCoutner++;
                        break;
                    }
                }

                usedAttributes.add(att);
                counter++;
            }
            if (!reusesMap.containsKey(reusesCoutner)) {
                reusesMap.put(reusesCoutner,0);
            }
            reusesMap.put(reusesCoutner, reusesMap.get(reusesCoutner) +1);
        }

        return reusesMap;
    }
}
